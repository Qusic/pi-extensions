// Reclaim disk space from the current compacted pi session.
// A compacted session keeps every superseded entry on disk even though pi only replays
// the newest compaction summary plus the entries it kept. /session-trim drops entries
// older than the Nth-from-last compaction on the active branch and every abandoned
// branch, then re-opens the session so pi's in-memory entries match the file. The trim
// is refused unless the projected context, the resolved model/thinking level, the
// session name, and the surviving labels are provably unchanged.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { createReadStream, renameSync, statSync, writeFileSync } from "node:fs";
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
  timestamp?: string;
  firstKeptEntryId?: string; // compaction
  targetId?: string; // label
  label?: string; // label
  name?: string; // session_info
  thinking?: string; // thinking_level_change
  model?: string; // model_change and assistant messages
}

interface TrimPlan {
  chain: Meta[]; // surviving entries, in file order
  labels: Meta[]; // effective labels whose own entry was dropped, re-emitted at the tail
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
      timestamp: entry.timestamp,
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

// getBranch: pi restores the leaf as the last entry in file order.
function branchOf(metas: readonly Meta[]): Meta[] {
  const byId = new Map(metas.map((meta) => [meta.id, meta]));
  const path: Meta[] = [];
  let current = metas[metas.length - 1];
  while (current) {
    path.push(current);
    current = current.parentId ? byId.get(current.parentId)! : undefined!;
  }
  return path.reverse();
}

// buildContextEntries: the newest compaction, then its kept entries, then the rest.
// Labels are excluded because they project to no message and are free to move.
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
    .filter((meta) => meta.type !== "label")
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

// getSessionName: the last session_info in file order, branch or not.
const nameOf = (metas: readonly Meta[]): string | undefined =>
  findLast(metas, (meta) => meta.type === "session_info")?.name;

// _buildIndex: labels are global state replayed over the whole file, last write wins.
// A label dies with its target, so callers pass the ids that survive the trim.
function labelsOn(metas: readonly Meta[], targets: ReadonlySet<string>): Meta[] {
  const byTarget = new Map<string, Meta>();
  for (const meta of metas) {
    if (meta.type !== "label" || !meta.targetId) continue;
    if (meta.label) byTarget.set(meta.targetId, meta);
    else byTarget.delete(meta.targetId);
  }
  return [...byTarget.values()].filter((meta) => targets.has(meta.targetId!));
}

const labelSignature = (labels: readonly Meta[]): string =>
  labels
    .map((meta) => `${meta.targetId}=${meta.label}`)
    .sort()
    .join(",");

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
    nameOf(trimmed) === nameOf(metas) &&
    labelSignature(labelsOn(trimmed, survivors)) === labelSignature(labelsOn(metas, survivors))
  );
}

function planTrim(metas: readonly Meta[]): TrimPlan | undefined {
  const path = branchOf(metas);
  const compactions = path.filter((meta) => meta.type === "compaction");
  if (compactions.length < KEEP_GENERATIONS) return undefined;
  const cut = compactions[compactions.length - KEEP_GENERATIONS];
  const cutIndex = path.findIndex((meta) => meta.id === cut.firstKeptEntryId);
  if (cutIndex < 0) return undefined;

  const kept = path.slice(cutIndex);
  const dropped = path.slice(0, cutIndex);
  // Settings are read off the branch, so rescue only what the kept range no longer states.
  const rescued: Meta[] = [];
  if (!kept.some((meta) => meta.thinking !== undefined)) {
    const meta = findLast(dropped, (entry) => entry.thinking !== undefined);
    if (meta) rescued.push(meta);
  }
  if (!kept.some((meta) => meta.model !== undefined)) {
    const meta = findLast(dropped, (entry) => entry.model !== undefined);
    if (meta) rescued.push(meta);
  }
  // Keeping exactly the file-order-last session_info reproduces getSessionName() no
  // matter which branch named the session.
  const name = findLast(metas, (meta) => meta.type === "session_info");
  if (name && !kept.includes(name)) rescued.push(name);
  rescued.sort((a, b) => a.index - b.index);

  // Labels stay where they are: a compaction may point at one, since findCutPoint()
  // extends the cut backwards over entries that produce no message.
  const core = [...rescued, ...kept].filter((meta) => meta.type !== "session_info" || meta === name);
  const coreIds = new Set(core.map((meta) => meta.id));
  // The leaf is never dropped: entries appended after the trim still point at it.
  const leaf = path[path.length - 1];
  const chain = core.filter((meta) => meta === leaf || meta.type !== "label" || coreIds.has(meta.targetId!));
  const survivors = new Set(chain.map((meta) => meta.id));
  // A surviving target whose winning label was dropped gets that label back at the tail.
  const labels = labelsOn(metas, survivors).filter((meta) => !survivors.has(meta.id));

  const trimmed = [...chain, ...labels];
  if (!preservesProjections(trimmed, path, metas, survivors)) return undefined;
  return { chain, labels, kept: trimmed.length, bytes: trimmed.reduce((total, meta) => total + meta.bytes, 0) };
}

// Writing

// Re-chain survivors: ids never change, so entries appended later still link up.
// Buffering is safe because the surviving range is what pi would replay anyway.
async function writeTrimmed(file: string, header: string, plan: TrimPlan): Promise<void> {
  const keep = new Set(plan.chain.map((meta) => meta.index));
  const lines: string[] = [header];
  let parentId: string | null = null;
  for await (const [, entry, index] of sessionLines(file)) {
    if (!keep.has(index)) continue;
    entry.parentId = parentId;
    parentId = entry.id;
    lines.push(JSON.stringify(entry));
  }
  for (const meta of plan.labels) {
    lines.push(
      JSON.stringify({
        type: "label",
        id: meta.id,
        parentId,
        timestamp: meta.timestamp,
        targetId: meta.targetId,
        label: meta.label,
      }),
    );
    parentId = meta.id;
  }
  const temp = `${file}.trim`;
  writeFileSync(temp, `${lines.join("\n")}\n`);
  renameSync(temp, file);
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
    after: plan.bytes + session.header.length + 1,
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
