// npm-description: Speed up loading large Pi sessions by pruning superseded history.
// npm-keywords: session, session-management, session-file, session-loading, compaction, performance, history-pruning, cleanup

/**
 * Pi reads and scans the whole session file when it loads a session, including history
 * that compaction superseded and branches that were left behind, so long sessions open
 * more and more slowly. /session-trim rewrites the file without that history.
 *
 * A session file is JSONL: a header line, then one entry per line. Entries form a tree
 * through `id` and `parentId`. The last entry in the file is the current leaf, and its path
 * to the root is the active branch; branches left behind by /tree stay in the file. A
 * compaction entry summarizes the entries before it: the model context is the newest
 * compaction, the entries from its `firstKeptEntryId` on, and everything after it. Pi also
 * derives state from entries the context no longer shows, such as the tools, model and
 * thinking level to restore (see "Replayed state").
 *
 * The trim cuts the active branch at the first entry the second-to-last compaction kept
 * (see KEEP_GENERATIONS) and keeps everything from there verbatim. Of the older entries and
 * the other branches, only the state pi still reads is carried over; then all survivors are
 * re-linked into one chain. The file is rewritten only if everything pi derives from it
 * stays the same, and the session is re-opened afterwards.
 */

import { getCurrentSystemMessage, getCurrentTools, type SystemMessage } from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { closeSync, createReadStream, openSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";

const SESSION_VERSION = 3;
// Compaction generations whose history stays in full. 1 would leave only the newest summary
// and the entries it kept, so trimming right after a compaction would erase the visible history.
const KEEP_GENERATIONS = 2;

// One parsed line of the session file.
type Entry = Record<string, any>;

// Replayed state
//
// What pi derives from entries outside the context. Each kind is declared by how a later
// entry combines with an earlier one, which decides how it is carried across the cut: the
// latest entry per key wins (LATEST), or every entry adds to the state (FOLDS). Paths below
// are in github.com/earendil-works/pi/packages/coding-agent/src.

// Branch state is read along the active branch, file state across every entry in the file.
type Scope = "branch" | "file";

const isAssistant = (entry: Entry) => entry.type === "message" && entry.message?.role === "assistant";

// The latest entry per key wins, so the winning entries are carried over unchanged.
const LATEST: Record<Scope, ((entry: Entry) => string | undefined)[]> = {
  branch: [
    // The thinking level and model that buildSessionContext() reports:
    // getSessionContextSettings() in core/session-manager.ts. core/sdk.ts also checks that
    // the branch has a thinking level entry at all before restoring it.
    (entry) => (entry.type === "thinking_level_change" ? "thinking" : undefined),
    (entry) => (entry.type === "model_change" || isAssistant(entry) ? "model" : undefined),
    // The model pi selects on resume: getBranchSelection() in core/virtual-models.ts. A
    // virtual model is a router an extension registers, and responses name the physical
    // model it picked, so the latest physical response decides, unless the model_change
    // before it selected a virtual model.
    (entry) => (entry.type === "model_change" ? "model_change" : undefined),
    (entry) => (isAssistant(entry) && entry.message.api !== "pi-virtual" ? "response" : undefined),
    // The state each virtual model's router keeps: getVirtualModelState() in
    // core/virtual-models.ts.
    (entry) =>
      entry.type === "custom" && entry.customType === "pi.virtual-model-state"
        ? `virtual:${entry.data?.provider}/${entry.data?.modelId}`
        : undefined,
  ],
  // The session name and the label (bookmark) of each entry: getSessionName() and getLabel()
  // in core/session-manager.ts.
  file: [
    (entry) => (entry.type === "session_info" ? "name" : undefined),
    (entry) => (entry.type === "label" ? `label:${entry.targetId}` : undefined),
  ],
};

// Every entry adds to the state, applied in order from the root, so those before the cut are
// merged into the first of them. `read` mirrors what pi derives from the values, and `merge`
// returns one value that reads the same as all of them. Both kinds are branch state.
interface Fold<T> {
  field: "message" | "data";
  match: (entry: Entry) => boolean;
  read: (values: readonly T[]) => string;
  merge: (values: readonly T[]) => T;
}

// System messages: the session records the system prompt and the tools offered to the model
// as message entries with role "system". The first holds the whole prompt and tool list;
// later ones are patches that replace prompt sections or list toolsAdded/toolsRemoved. The
// model context replays only those after the newest compaction, whose entry stores a
// snapshot of the rest (`systemMessage`). Restoring the tools on resume and /tree replays
// every one on the branch instead: _restoreToolsFromTranscript() in core/agent-session.ts
// activates getCurrentTools() of them all, plus any extension tool they never mention, as
// newly installed. So losing one changes the tools.
//
// getCurrentSystemMessage() merges them the way pi builds that snapshot, but keeps only the
// current tools, so the removed ones are listed as toolsRemoved to stay mentioned.
const SYSTEM_MESSAGES: Fold<SystemMessage> = {
  field: "message",
  match: (entry) => entry.type === "message" && entry.message?.role === "system",
  // A branch without system messages keeps the current tools.
  read: (messages) =>
    messages.length === 0 ? "" : JSON.stringify([getCurrentTools(messages), [...mentionedTools(messages)].sort()]),
  merge: (messages) => {
    const folded: SystemMessage = getCurrentSystemMessage(messages) ?? { role: "system", content: "", timestamp: 0 };
    const current = new Set(folded.toolsAdded?.map((tool) => tool.name));
    const removed = [...mentionedTools(messages)].filter((name) => !current.has(name)).map((name) => ({ name }));
    return removed.length > 0 ? { ...folded, toolsRemoved: removed } : folded;
  },
};

function mentionedTools(messages: readonly SystemMessage[]): Set<string> {
  const tools = messages.flatMap((message) => [...(message.toolsAdded ?? []), ...(message.toolsRemoved ?? [])]);
  return new Set(tools.map((tool) => tool.name));
}

// codemode store: scripts keep values with store(), and each run that changes some appends a
// "codemode-store" custom entry with the keys it set and deleted. readCodemodeStore() in
// extensions/codemode/execute.ts applies them from the root, deletes before writes; it is not
// exported, so this mirrors it.
interface StoreWrite {
  set: Record<string, unknown>;
  delete: string[];
}

const CODEMODE_STORE: Fold<StoreWrite> = {
  field: "data",
  match: (entry) => entry.type === "custom" && entry.customType === "codemode-store" && isStoreWrite(entry.data),
  read: (writes) => JSON.stringify(replayStore(writes)),
  merge: (writes) => ({ set: replayStore(writes), delete: [] }),
};

function isStoreWrite(data: any): data is StoreWrite {
  return (
    typeof data?.set === "object" &&
    data.set !== null &&
    Array.isArray(data.delete) &&
    data.delete.every((key: unknown) => typeof key === "string")
  );
}

function replayStore(writes: readonly StoreWrite[]): Record<string, unknown> {
  const store = new Map<string, unknown>();
  for (const write of writes) {
    for (const key of write.delete) store.delete(key);
    for (const [key, value] of Object.entries(write.set)) store.set(key, value);
  }
  return Object.fromEntries(store);
}

const FOLDS: Fold<any>[] = [SYSTEM_MESSAGES, CODEMODE_STORE];

// Reading

// One session entry, reduced to what the trim needs. Bodies stay on disk, except the values
// a fold merges.
interface Meta {
  index: number; // position among the entries, which addresses it in the rewrite
  bytes: number;
  id: string;
  parentId: string | null;
  type: string;
  firstKeptEntryId?: string; // compaction
  latest: Record<Scope, string[]>; // the LATEST keys it writes
  fold?: Fold<any>; // the FOLDS kind it adds `value` to
  value?: unknown;
}

// The header gets index -1, so readSession() and writeTrimmed() number entries alike.
async function* sessionLines(file: string): AsyncGenerator<[line: string, entry: Entry, index: number]> {
  const input = createReadStream(file, { encoding: "utf8" });
  let index = -1;
  try {
    for await (const line of createInterface({ input, crlfDelay: Number.POSITIVE_INFINITY })) {
      if (!line.trim()) continue;
      try {
        const entry = JSON.parse(line);
        yield [line, entry, index++];
      } catch {
        // pi skips malformed lines on load; stay consistent with it.
      }
    }
  } finally {
    input.destroy();
  }
}

async function readSession(file: string): Promise<{ header: string; metas: Meta[] } | undefined> {
  let header: string | undefined;
  const metas: Meta[] = [];
  for await (const [line, entry, index] of sessionLines(file)) {
    if (index < 0) {
      if (entry.type !== "session" || entry.version !== SESSION_VERSION) return undefined;
      header = line;
      continue;
    }
    if (typeof entry.id !== "string") return undefined;
    const fold = FOLDS.find((candidate) => candidate.match(entry));
    const keys = (scope: Scope) => LATEST[scope].flatMap((keyOf) => keyOf(entry) ?? []);
    metas.push({
      index,
      bytes: Buffer.byteLength(line) + 1,
      id: entry.id,
      parentId: entry.parentId ?? null,
      type: entry.type,
      firstKeptEntryId: entry.firstKeptEntryId,
      latest: { branch: keys("branch"), file: keys("file") },
      fold,
      value: fold && entry[fold.field],
    });
  }
  return header === undefined ? undefined : { header, metas };
}

// Projections: what pi derives from the session, mirroring core/session-manager.ts and the
// replayed state declared above.

const isCompaction = (meta: Meta) => meta.type === "compaction";

// The entry types that shape the context: sessionEntryToContextMessages() turns these into
// messages, and buildSessionProjection() applies each context_edit to the entry it targets.
// The others (session_info, model_change, thinking_level_change, label, custom) are state or
// display entries, so moving or dropping them cannot change the context.
const PROJECTED_TYPES = new Set(["message", "custom_message", "branch_summary", "compaction", "context_edit"]);

// getBranch: pi restores the leaf as the last entry in file order.
function branchOf(metas: readonly Meta[]): Meta[] | undefined {
  const byId = new Map(metas.map((meta) => [meta.id, meta]));
  const seen = new Set<string>();
  const path: Meta[] = [];
  // A dangling parentId ends the branch, exactly as pi's own walk does; a cycle would
  // hang that walk, so treat it as a corrupt file instead of trimming it.
  let current: Meta | undefined = metas[metas.length - 1];
  while (current) {
    if (seen.has(current.id)) return undefined;
    seen.add(current.id);
    path.push(current);
    current = current.parentId ? byId.get(current.parentId) : undefined;
  }
  return path.reverse();
}

// buildContextEntries: the newest compaction, then its kept entries, then the rest. Kept
// system messages are left out, since the compaction's snapshot covers them.
function contextOf(path: readonly Meta[]): string {
  const visible: Meta[] = [];
  const compaction = path.filter(isCompaction).at(-1);
  if (!compaction) visible.push(...path);
  else {
    const index = path.indexOf(compaction);
    visible.push(compaction);
    let kept = false;
    for (let i = 0; i < index; i++) {
      if (path[i].id === compaction.firstKeptEntryId) kept = true;
      if (kept && path[i].fold !== SYSTEM_MESSAGES) visible.push(path[i]);
    }
    visible.push(...path.slice(index + 1));
  }
  return visible
    .filter((meta) => PROJECTED_TYPES.has(meta.type))
    .map((meta) => meta.id)
    .join(",");
}

function winners(metas: readonly Meta[], scope: Scope): Map<string, Meta> {
  const result = new Map<string, Meta>();
  for (const meta of metas) for (const key of meta.latest[scope]) result.set(key, meta);
  return result;
}

function latestOf(metas: readonly Meta[], scope: Scope): string {
  return JSON.stringify([...winners(metas, scope)].map(([key, meta]) => `${key}=${meta.id}`).sort());
}

function foldsOf(path: readonly Meta[]): string {
  const values = (fold: Fold<any>) => path.filter((meta) => meta.fold === fold).map((meta) => meta.value);
  return JSON.stringify(FOLDS.map((fold) => fold.read(values(fold))));
}

// Everything pi derives from a session, given its active branch and all of its entries.
function derived(branch: readonly Meta[], file: readonly Meta[]): string {
  return JSON.stringify([contextOf(branch), foldsOf(branch), latestOf(branch, "branch"), latestOf(file, "file")]);
}

// Planning

interface TrimPlan {
  chain: Meta[]; // survivors in file order: carried branch state, then the kept range
  tail: Meta[]; // carried file state, written last so it stays the latest write
  folded: Meta[]; // per fold, its first entry before the cut, rewritten to hold all of them
  total: number; // entries the file held when the plan was made
  remaining: number; // entries the rewritten file will hold
  bytes: number; // estimated size of those entries
}

// The first of a fold's entries, rewritten to hold the value of all of them.
function foldInto(fold: Fold<any>, members: readonly Meta[]): Meta {
  const [first] = members;
  const value = fold.merge(members.map((meta) => meta.value));
  const delta = Buffer.byteLength(JSON.stringify(value)) - Buffer.byteLength(JSON.stringify(first.value));
  return { ...first, value, bytes: first.bytes + delta };
}

// A plan, or why there is none.
function planTrim(metas: readonly Meta[]): TrimPlan | string {
  const path = branchOf(metas);
  if (!path) return "its entries form a cycle";
  const compactions = path.filter(isCompaction);
  if (compactions.length < KEEP_GENERATIONS) return `fewer than ${KEEP_GENERATIONS} compactions on this branch`;
  const cut = compactions[compactions.length - KEEP_GENERATIONS];
  const cutIndex = path.findIndex((meta) => meta.id === cut.firstKeptEntryId);
  if (cutIndex < 0) return "a compaction refers to an entry missing from this branch";

  // Carried over: the winners outside the kept range, branch state in front of the cut in its
  // original order and file state after everything, so each stays the latest write; and each
  // fold's entries before the cut, merged into one. The context sees none of them.
  const kept = path.slice(cutIndex);
  const inKept = new Set(kept.map((meta) => meta.id));
  const outside = (meta: Meta) => !inKept.has(meta.id);
  const byIndex = (a: Meta, b: Meta) => a.index - b.index;
  const older = path.slice(0, cutIndex);
  const folded = FOLDS.flatMap((fold) => {
    const members = older.filter((meta) => meta.fold === fold);
    return members.length > 0 ? [foldInto(fold, members)] : [];
  });
  // One entry can win several keys.
  const head = [...new Set(winners(path, "branch").values())].filter(outside).concat(folded).sort(byIndex);
  const tail = [...new Set(winners(metas, "file").values())].filter(outside).sort(byIndex);

  const chain = [...head, ...kept];
  const trimmed = [...chain, ...tail];
  // The rewritten file is one chain, so all of it is its active branch.
  if (derived(trimmed, trimmed) !== derived(path, metas)) return "trimming would change what pi restores from it";
  return {
    chain,
    tail,
    folded,
    total: metas.length,
    remaining: trimmed.length,
    bytes: trimmed.reduce((total, meta) => total + meta.bytes, 0),
  };
}

// Writing

// Write the survivors as one chain, re-linking parentIds. Ids never change, so entries pi
// appends later still link up.
async function writeTrimmed(file: string, header: string, plan: TrimPlan): Promise<void> {
  const keep = new Set(plan.chain.map((meta) => meta.index));
  const rewrites = new Map(plan.folded.map((meta) => [meta.index, meta]));
  const holdBack = new Set(plan.tail.map((meta) => meta.index));
  const held = new Map<number, Entry>();
  const temp = `${file}.trim`;
  try {
    const fd = openSync(temp, "w");
    let seen = 0;
    let parentId: string | null = null;
    const emit = (entry: Entry) => {
      entry.parentId = parentId;
      parentId = entry.id;
      writeFileSync(fd, `${JSON.stringify(entry)}\n`);
    };
    try {
      writeFileSync(fd, `${header}\n`);
      for await (const [, entry, index] of sessionLines(file)) {
        if (index < 0) continue;
        seen++;
        const rewrite = rewrites.get(index);
        if (rewrite?.fold) emit({ ...entry, [rewrite.fold.field]: rewrite.value });
        else if (keep.has(index)) emit(entry);
        else if (holdBack.has(index)) held.set(index, entry);
      }
      for (const meta of plan.tail) emit(held.get(meta.index)!);
    } finally {
      closeSync(fd);
    }
    // The plan addresses entries by line, so anything appended since it was made would be
    // dropped. Publish nothing and let the user re-run instead.
    if (seen !== plan.total) throw new Error("the session grew while trimming, so nothing was rewritten");
    renameSync(temp, file);
  } catch (error) {
    rmSync(temp, { force: true });
    throw error;
  }
}

// Command

interface Analysis {
  header: string;
  plan: TrimPlan;
  before: number; // current file size
  after: number; // estimated size once rewritten
  entries: number; // current entry count
}

// An analysis, or why there is nothing to trim.
async function analyze(file: string): Promise<Analysis | string> {
  const session = await readSession(file);
  if (!session) return "unsupported session format";
  const plan = planTrim(session.metas);
  if (typeof plan === "string") return plan;
  if (plan.remaining === session.metas.length) {
    return `nothing older than the last ${KEEP_GENERATIONS} compactions, and no abandoned branches`;
  }
  return {
    header: session.header,
    plan,
    before: statSync(file).size,
    after: plan.bytes + Buffer.byteLength(session.header) + 1,
    entries: session.metas.length,
  };
}

const formatSize = (bytes: number) =>
  bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(0)} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`;

export default function (pi: ExtensionAPI): void {
  pi.registerCommand("session-trim", {
    description: "Drop superseded pre-compaction history from this session",
    handler: async (_args, ctx) => {
      const file = ctx.sessionManager.getSessionFile();
      if (!file) {
        ctx.ui.notify("This session is not persisted, so there is nothing to trim.", "error");
        return;
      }
      try {
        // Never rewrite the file underneath a running turn.
        await ctx.waitForIdle();
        const analysis = await analyze(file);
        if (typeof analysis === "string") {
          ctx.ui.notify(`Session not trimmed: ${analysis}.`, "info");
          return;
        }

        const { header, plan, before, after, entries } = analysis;
        const reclaimed = formatSize(before - after);
        // The report doubles as the prompt: re-opening the session clears the transcript,
        // and the interactive "Resumed session" status would overwrite anything shown after.
        const report = `${formatSize(before)} → ${formatSize(after)}, ${entries} → ${plan.remaining} entries; history older than the last ${KEEP_GENERATIONS} compactions and all abandoned branches are dropped.`;
        if (!ctx.hasUI) {
          ctx.ui.notify(`Session trim would reclaim ${reclaimed}: ${report}`, "info");
          return;
        }
        if (!(await ctx.ui.confirm("Trim session", `Reclaim ${reclaimed}?\n${report}`))) return;

        await writeTrimmed(file, header, plan);
        // pi still holds the untrimmed entries in memory, and a later fork would write them
        // back, so re-open the file. switchSession() reloads it from disk.
        await ctx.switchSession(file);
      } catch (error) {
        ctx.ui.notify(`Session trim failed: ${(error as Error).message}`, "error");
      }
    },
  });
}
