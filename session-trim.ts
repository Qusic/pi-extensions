// npm-description: Speed up loading large Pi sessions by pruning superseded history.
// npm-keywords: session, session-management, session-file, session-loading, compaction, performance, history-pruning, cleanup

/**
 * Pi reads and scans the entire JSONL file when loading a session, including history
 * superseded by compaction and abandoned branches that are no longer replayed. Large,
 * long-running sessions can therefore become progressively slower to open.
 *
 * /session-trim drops entries older than the Nth-from-last compaction on the active
 * branch and every abandoned branch, then re-opens the session so Pi's in-memory entries
 * match the file. The trim is refused unless the projected context, resolved model and
 * thinking level, session name, and surviving labels are provably unchanged.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { closeSync, createReadStream, openSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";

// Schema

const SESSION_VERSION = 3;
// Generations of compaction to keep. 1 would leave only the summary and the current
// tail, so trimming right after a compaction would erase the visible history.
const KEEP_GENERATIONS = 2;

// One session entry, reduced to what the trim needs. Bodies stay on disk.
interface Meta {
  index: number;
  bytes: number;
  id: string;
  parentId: string | null;
  type: string;
  firstKeptEntryId?: string; // compaction
  targetId?: string; // label
  label?: string; // label
  name?: string; // session_info
  thinking?: string; // thinking_level_change
  model?: string; // model_change and assistant messages
}

interface TrimPlan {
  chain: Meta[]; // survivors in file order: rescued branch state, then the kept range
  tail: Meta[]; // file-scoped state winners, re-emitted last so they still win
  total: number; // entries the file held when the plan was made
  kept: number; // entries the rewritten file will hold
  bytes: number; // estimated size of those entries
}

// Reading

// Entry index is -1 for the header, so both passes number entries identically.
async function* sessionLines(file: string): AsyncGenerator<[line: string, entry: Record<string, any>, index: number]> {
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

// Read entry metadata without holding message bodies in memory.
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
    metas.push({
      index,
      bytes: Buffer.byteLength(line) + 1,
      id: entry.id,
      parentId: entry.parentId ?? null,
      type: entry.type,
      firstKeptEntryId: entry.firstKeptEntryId,
      targetId: entry.targetId,
      label: entry.label,
      name: entry.type === "session_info" ? entry.name : undefined,
      thinking: entry.type === "thinking_level_change" ? entry.thinkingLevel : undefined,
      model:
        entry.type === "model_change"
          ? `${entry.provider}/${entry.modelId}`
          : entry.type === "message" && entry.message?.role === "assistant"
            ? `${entry.message.provider}/${entry.message.model}`
            : undefined,
    });
  }
  return header === undefined ? undefined : { header, metas };
}

// Projections, mirroring pi's session-manager

const findLast = (metas: readonly Meta[], match: (meta: Meta) => boolean): Meta | undefined => {
  for (let i = metas.length - 1; i >= 0; i--) {
    if (match(metas[i])) return metas[i];
  }
  return undefined;
};

// sessionEntryToContextMessages: only these entry types become a message. The others
// (session_info, model_change, thinking_level_change, label, custom) are state or
// display entries, so moving or dropping them cannot change what pi replays.
const PROJECTED_TYPES = new Set(["message", "custom_message", "branch_summary", "compaction"]);

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

// buildContextEntries: the newest compaction, then its kept entries, then the rest.
function contextIds(path: readonly Meta[]): string {
  const visible: Meta[] = [];
  const compaction = findLast(path, (meta) => meta.type === "compaction");
  if (!compaction) visible.push(...path);
  else {
    const index = path.indexOf(compaction);
    visible.push(compaction);
    let kept = false;
    for (let i = 0; i < index; i++) {
      if (path[i].id === compaction.firstKeptEntryId) kept = true;
      if (kept) visible.push(path[i]);
    }
    visible.push(...path.slice(index + 1));
  }
  return visible
    .filter((meta) => PROJECTED_TYPES.has(meta.type))
    .map((meta) => meta.id)
    .join(",");
}

// getSessionContextSettings: last writer along the branch wins.
function settingsOf(path: readonly Meta[]): string {
  let thinking = "off";
  let model = "";
  for (const meta of path) {
    if (meta.thinking !== undefined) thinking = meta.thinking;
    if (meta.model !== undefined) model = meta.model;
  }
  return `${thinking}|${model}`;
}

// State pi resolves by walking the branch: the last writer wins.
const BRANCH_STATE = [(meta: Meta) => meta.thinking !== undefined, (meta: Meta) => meta.model !== undefined];

// State pi resolves by replaying the whole file, abandoned branches included:
// getSessionName takes the last session_info, getLabel the last write per target.
function fileStateWinners(metas: readonly Meta[]): Map<string, Meta> {
  const winners = new Map<string, Meta>();
  for (const meta of metas) {
    if (meta.type === "session_info") winners.set("name", meta);
    else if (meta.type === "label" && meta.targetId) winners.set(`label:${meta.targetId}`, meta);
  }
  return winners;
}

// The values those winners resolve to: an empty label clears the bookmark, and a label
// dies with its target, so callers pass the ids that survive the trim.
function fileStateOf(metas: readonly Meta[], targets: ReadonlySet<string>): string {
  return [...fileStateWinners(metas)]
    .filter(([key, meta]) => key === "name" || (meta.label && targets.has(meta.targetId!)))
    .map(([key, meta]) => `${key}=${meta.name ?? meta.label}`)
    .sort()
    .join(",");
}

// Planning

// The rewrite is only safe if every projection pi derives stays identical.
function preservesProjections(
  trimmed: readonly Meta[],
  path: readonly Meta[],
  metas: readonly Meta[],
  survivors: ReadonlySet<string>,
): boolean {
  return (
    contextIds(trimmed) === contextIds(path) &&
    settingsOf(trimmed) === settingsOf(path) &&
    fileStateOf(trimmed, survivors) === fileStateOf(metas, survivors)
  );
}

function planTrim(metas: readonly Meta[]): TrimPlan | undefined {
  const path = branchOf(metas);
  if (!path) return undefined;
  const compactions = path.filter((meta) => meta.type === "compaction");
  if (compactions.length < KEEP_GENERATIONS) return undefined;
  const cut = compactions[compactions.length - KEEP_GENERATIONS];
  const cutIndex = path.findIndex((meta) => meta.id === cut.firstKeptEntryId);
  if (cutIndex < 0) return undefined;

  // Everything from the cut on is kept verbatim. Only state pi replays has to be carried
  // over, and only when the entry that currently wins falls outside that range: branch
  // state goes in front of the cut so it stays on the path, file state goes last so it
  // stays the final write. Neither placement is visible to the compaction's context.
  const kept = path.slice(cutIndex);
  const inKept = new Set(kept.map((meta) => meta.id));
  const outside = (meta: Meta | undefined): meta is Meta => meta !== undefined && !inKept.has(meta.id);
  const byIndex = (a: Meta, b: Meta) => a.index - b.index;
  const head = BRANCH_STATE.map((defines) => findLast(path, defines))
    .filter(outside)
    .sort(byIndex);
  const tail = [...fileStateWinners(metas).values()].filter(outside).sort(byIndex);

  const chain = [...head, ...kept];
  const trimmed = [...chain, ...tail];
  const survivors = new Set(trimmed.map((meta) => meta.id));
  if (!preservesProjections(trimmed, path, metas, survivors)) return undefined;
  return {
    chain,
    tail,
    total: metas.length,
    kept: trimmed.length,
    bytes: trimmed.reduce((total, meta) => total + meta.bytes, 0),
  };
}

// Writing

// Re-chain survivors: ids never change, so entries appended later still link up.
async function writeTrimmed(file: string, header: string, plan: TrimPlan): Promise<void> {
  const keep = new Set(plan.chain.map((meta) => meta.index));
  const holdBack = new Set(plan.tail.map((meta) => meta.index));
  const held = new Map<number, Record<string, any>>();
  const temp = `${file}.trim`;
  try {
    const fd = openSync(temp, "w");
    let seen = 0;
    let parentId: string | null = null;
    const emit = (entry: Record<string, any>) => {
      entry.parentId = parentId;
      parentId = entry.id;
      writeFileSync(fd, `${JSON.stringify(entry)}\n`);
    };
    try {
      writeFileSync(fd, `${header}\n`);
      for await (const [, entry, index] of sessionLines(file)) {
        if (index < 0) continue;
        seen++;
        if (keep.has(index)) emit(entry);
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

async function analyze(file: string): Promise<Analysis | undefined> {
  const session = await readSession(file);
  if (!session) return undefined;
  const plan = planTrim(session.metas);
  // An already-trimmed session still plans cleanly; it just drops nothing.
  if (!plan || plan.kept === session.metas.length) return undefined;
  return {
    header: session.header,
    plan,
    before: statSync(file).size,
    after: plan.bytes + Buffer.byteLength(session.header) + 1,
    entries: session.metas.length,
  };
}

const size = (bytes: number) =>
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
        if (!analysis) {
          ctx.ui.notify("Nothing to trim: fewer than two compactions, or an unsupported session layout.", "info");
          return;
        }

        const { header, plan, before, after, entries } = analysis;
        const reclaimed = size(before - after);
        // The report doubles as the prompt: re-opening the session clears the transcript,
        // and the interactive "Resumed session" status would overwrite anything shown after.
        const report = `${size(before)} → ${size(after)}, ${entries} → ${plan.kept} entries; history older than the last ${KEEP_GENERATIONS} compactions and all abandoned branches are dropped.`;
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
