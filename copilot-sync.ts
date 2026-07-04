// Sync GitHub Copilot context/output limits into models.json from your tenant's
// live CAPI /models. Run /copilot-sync anytime to refresh.
//
// Everything comes from pi at runtime, nothing hardcoded: getModels() is the real
// built-in catalog, getApiKeyAndHeaders() gives the fresh token + client headers,
// a built-in model's baseUrl is already rewritten to your tenant host, and
// modelRegistry.refresh() re-reads models.json so changes apply without a restart.
//
// Writes providers["github-copilot"].modelOverrides (contextWindow/maxTokens/
// thinkingLevelMap for models pi knows) and .models (entries for tenant models it
// doesn't). Other providers and top-level keys are preserved.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getModels } from "@earendil-works/pi-ai";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const PROVIDER = "github-copilot";

interface LiveModel {
  id: string;
  name?: string;
  vendor?: string;
  model_picker_enabled?: boolean;
  supported_endpoints?: string[];
  capabilities?: {
    type?: string;
    supports?: {
      vision?: boolean;
      thinking?: boolean;
      adaptive_thinking?: boolean;
      reasoning_effort?: string[];
      max_thinking_budget?: number;
      min_thinking_budget?: number;
    };
    limits?: { max_context_window_tokens?: number; max_prompt_tokens?: number; max_output_tokens?: number };
  };
}

type ThinkingMap = Record<string, string | null>;
type Override = { contextWindow?: number; maxTokens?: number; thinkingLevelMap?: ThinkingMap };
type Builtin = { id: string; api: string };

interface ModelInsert {
  id: string;
  name: string;
  api: string;
  baseUrl: string;
  reasoning: boolean;
  input: ("text" | "image")[];
  contextWindow: number;
  maxTokens: number;
  headers?: Record<string, string>;
  compat?: Record<string, boolean>;
  thinkingLevelMap?: ThinkingMap;
}

// github-copilot serves exactly these three APIs; a model's declared endpoints map
// to them (native endpoints preferred over the generic /chat/completions).
const ENDPOINT_API: Record<string, string> = {
  "/v1/messages": "anthropic-messages",
  "/responses": "openai-responses",
  "/chat/completions": "openai-completions",
};
// Which APIs send a thinking effort on Copilot; /chat/completions rejects it (we
// force supportsReasoningEffort:false and skip the thinking map there).
const usesEffort = (api: string) => api === "anthropic-messages" || api === "openai-responses";

const RANK: Record<string, number> = { minimal: 0, low: 1, medium: 2, high: 3, xhigh: 4, max: 5 };
const PI_LEVELS = ["minimal", "low", "medium", "high", "xhigh"];

const family = (id: string) => id.split(/[-.]/).slice(0, 2).join("-");

function agentDir(): string {
  const d = process.env.PI_CODING_AGENT_DIR;
  return d ? d.replace(/^~(\/|$)/, `${homedir()}$1`) : join(homedir(), ".pi", "agent");
}

// api for a tenant model pi doesn't know: prefer a native endpoint over the generic
// /chat/completions, else fall back to a same-family built-in or the vendor.
function resolveApi(m: LiveModel, builtins: readonly Builtin[]): string {
  const eps = m.supported_endpoints ?? [];
  for (const api of ["anthropic-messages", "openai-responses", "openai-completions"]) {
    if (eps.some((e) => ENDPOINT_API[e] === api)) return api;
  }
  const sib = builtins.find((b) => family(b.id) === family(m.id));
  if (sib) return sib.api;
  const v = (m.vendor ?? "").toLowerCase();
  return v.includes("anthropic") ? "anthropic-messages" : v.includes("openai") ? "openai-responses" : "openai-completions";
}

function contextWindowOf(cap: LiveModel["capabilities"]): number | undefined {
  const lim = cap?.limits;
  return lim?.max_context_window_tokens ?? ((lim?.max_prompt_tokens ?? 0) + (lim?.max_output_tokens ?? 0) || undefined);
}

// Copilot advertises reasoning two ways: effort levels (reasoning_effort/
// adaptive_thinking) on newer models, or a token budget (max/min_thinking_budget)
// on older ones -- check both or budget-style models look non-reasoning.
function isReasoning(m: LiveModel): boolean {
  const s = m.capabilities?.supports;
  return Boolean(s?.reasoning_effort?.length || s?.adaptive_thinking || s?.thinking || s?.max_thinking_budget || s?.min_thinking_budget);
}

// thinkingLevelMap for a model+api, or undefined if the api sends no effort or the
// model advertises none. Aligns pi's fixed levels to the model's reasoning_effort by
// count, anchored at the bottom (minimal -> lowest) so the top usable level reaches
// the model's highest effort (incl. "max"); extra top levels are hidden with null.
// off follows the API: allowed only if it lists a none/off effort, else disabled.
function thinkingMap(m: LiveModel, api: string): ThinkingMap | undefined {
  if (!usesEffort(api)) return undefined;
  const effort = m.capabilities?.supports?.reasoning_effort;
  const noneToken = effort?.find((e) => e === "none" || e === "off");
  let ladder = (effort ?? []).filter((e) => e !== "none" && e !== "off").sort((a, b) => (RANK[a] ?? 99) - (RANK[b] ?? 99));
  if (!ladder.length) return undefined;
  if (ladder.length > 5) ladder = ladder.slice(-5); // keep the top 5 so "max" stays reachable
  const map: ThinkingMap = {};
  PI_LEVELS.forEach((lvl, i) => (map[lvl] = i < ladder.length ? ladder[i] : null));
  map.off = noneToken ?? null; // non-null re-enables off (overriding a built-in off:null)
  return map;
}

// Override for a model pi already knows: only the fields the live API can supply
// reliably (context/output limits + thinking ladder). reasoning/input/compat/cost
// are left to pi's curated built-in.
function buildOverride(m: LiveModel, api: string): Override | undefined {
  const e: Override = {};
  const cw = contextWindowOf(m.capabilities);
  const out = m.capabilities?.limits?.max_output_tokens;
  if (Number.isFinite(cw)) e.contextWindow = cw;
  if (Number.isFinite(out)) e.maxTokens = out;
  const tlm = thinkingMap(m, api);
  if (tlm) e.thinkingLevelMap = tlm;
  return Object.keys(e).length ? e : undefined;
}

function buildInsert(m: LiveModel, api: string, base: string, clientHeaders?: Record<string, string>): ModelInsert {
  const supports = m.capabilities?.supports;
  const insert: ModelInsert = {
    id: m.id,
    name: m.name ?? m.id,
    api,
    baseUrl: base,
    reasoning: isReasoning(m),
    input: supports?.vision ? ["text", "image"] : ["text"],
    contextWindow: contextWindowOf(m.capabilities) ?? 128000,
    maxTokens: m.capabilities?.limits?.max_output_tokens ?? 4096,
    // Static Copilot client headers (User-Agent/Editor-*/Copilot-Integration-Id)
    // aren't inherited by custom models, so copy them or requests get rejected.
    headers: clientHeaders,
  };
  // pi auto-detects openai compat from baseUrl but doesn't know Copilot's host, so it
  // would wrongly default these to true; Copilot's /chat/completions rejects them
  // (built-ins hardcode all three false). Anthropic adaptive-thinking models need the
  // adaptive request format.
  if (api === "openai-completions") {
    insert.compat = { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false };
  } else if (api === "anthropic-messages" && supports?.adaptive_thinking) {
    insert.compat = { forceAdaptiveThinking: true };
  }
  const tlm = thinkingMap(m, api);
  if (tlm) insert.thinkingLevelMap = tlm;
  return insert;
}

function collectModels(live: LiveModel[], builtins: readonly Builtin[], base: string, clientHeaders?: Record<string, string>) {
  const builtinById = new Map(builtins.map((b) => [b.id, b]));
  const chat = live
    .filter((m) => m.capabilities?.type === "chat" && m.model_picker_enabled !== false)
    .sort((a, b) => a.id.localeCompare(b.id));

  const overrides: Record<string, Override> = {};
  const inserts: ModelInsert[] = [];
  for (const m of chat) {
    const builtin = builtinById.get(m.id);
    if (builtin) {
      const e = buildOverride(m, builtin.api);
      if (e) overrides[m.id] = e;
    } else {
      inserts.push(buildInsert(m, resolveApi(m, builtins), base, clientHeaders));
    }
  }
  return { overrides, inserts };
}

async function fetchLiveModels(base: string, headers: Record<string, string>, signal?: AbortSignal): Promise<LiveModel[]> {
  const res = await fetch(`${base}/models`, { headers, signal });
  if (!res.ok) {
    if (res.status === 401) throw new Error("Copilot token expired — send a message or /login, then retry.");
    throw new Error(`Copilot /models failed (${res.status}).`);
  }
  return ((await res.json()) as { data: LiveModel[] }).data;
}

// Merge into models.json, replacing only github-copilot's modelOverrides/models and
// preserving other providers and top-level keys.
function writeConfig(outFile: string, overrides: Record<string, Override>, inserts: ModelInsert[]): void {
  let cfg: { providers?: Record<string, Record<string, unknown>> } = {};
  if (existsSync(outFile)) {
    try {
      cfg = JSON.parse(readFileSync(outFile, "utf8").replace(/^\uFEFF/, ""));
    } catch {
      throw new Error("models.json is invalid JSON — aborting to avoid overwriting it.");
    }
  }
  cfg.providers ??= {};
  const prov = cfg.providers[PROVIDER] ?? {};
  prov.modelOverrides = overrides;
  if (inserts.length) prov.models = inserts;
  else delete prov.models;
  cfg.providers[PROVIDER] = prov;
  writeFileSync(outFile, JSON.stringify(cfg, null, 2) + "\n");
}

export default function (pi: ExtensionAPI): void {
  pi.registerCommand("copilot-sync", {
    description: "Sync GitHub Copilot context/output limits into models.json from your tenant's live /models",
    handler: async (_args, ctx) => {
      const builtins = getModels(PROVIDER);
      const builtinIds = new Set(builtins.map((m) => m.id));

      // A loaded built-in model carries pi's tenant-rewritten baseUrl + Copilot
      // client headers; getApiKeyAndHeaders then yields the fresh token.
      const probe = ctx.modelRegistry.getAll().find((m) => m.provider === PROVIDER && builtinIds.has(m.id));
      const auth = probe && (await ctx.modelRegistry.getApiKeyAndHeaders(probe));
      if (!probe || !auth?.ok || !auth.apiKey) {
        ctx.ui.notify("Not logged in to GitHub Copilot — run /login github-copilot first.", "error");
        return;
      }

      try {
        const authHeaders = { ...auth.headers, Authorization: `Bearer ${auth.apiKey}`, Accept: "application/json" };
        const live = await fetchLiveModels(probe.baseUrl, authHeaders, ctx.signal);
        const { overrides, inserts } = collectModels(live, builtins, probe.baseUrl, probe.headers);
        writeConfig(join(agentDir(), "models.json"), overrides, inserts);
        ctx.modelRegistry.refresh(); // re-reads models.json, applies now

        let msg = `Copilot synced: ${Object.keys(overrides).length} updated, ${inserts.length} added.`;
        if (inserts.length) msg += `\nAdded: ${inserts.map((i) => i.id).join(", ")}`;
        ctx.ui.notify(msg, "info");
      } catch (e) {
        ctx.ui.notify((e as Error).message, "error");
      }
    },
  });
}
