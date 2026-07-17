// Sync tenant-specific GitHub Copilot capabilities from live CAPI /models.
// Existing pi models get additive-only overrides: larger context/output limits and
// missing xhigh/max tiers. Unknown tenant models are added in full. Re-run after a pi
// upgrade so models newly built into pi move automatically from .models to overrides.
// Other providers and top-level models.json keys are preserved.

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
// pi's extension loader exposes pi-ai through compat; providers/all is not resolvable there.
import { getModels } from "@earendil-works/pi-ai/compat";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

// Schema and constants

const PROVIDER = "github-copilot";
// Mirrors pi-ai's Copilot /models request.
const COPILOT_API_VERSION = "2026-06-01";

interface LiveModel {
  id: string;
  name?: string;
  vendor?: string;
  model_picker_enabled?: boolean;
  supported_endpoints?: string[];
  policy?: { state?: string };
  capabilities?: {
    type?: string;
    supports?: {
      tool_calls?: boolean;
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
type ModelOverride = { contextWindow?: number; maxTokens?: number; thinkingLevelMap?: ThinkingMap };
type BuiltinModel = { id: string; api: string; contextWindow: number; maxTokens: number; thinkingLevelMap?: ThinkingMap };

interface CustomModel {
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

interface SyncPlan {
  overrides: Record<string, ModelOverride>;
  customModels: CustomModel[];
}

// Native endpoints first; generic Chat Completions last.
const ENDPOINTS = [
  ["/v1/messages", "anthropic-messages"],
  ["/responses", "openai-responses"],
  ["/chat/completions", "openai-completions"],
] as const;
const TIERS = ["low", "medium", "high", "xhigh", "max"] as const;
const OPTIONAL_TIERS = ["xhigh", "max"] as const;

// Live metadata interpretation

const family = (id: string) => id.split(/[-.]/).slice(0, 2).join("-");

function resolveApi(model: LiveModel, builtins: readonly BuiltinModel[]): string {
  const endpoints = new Set(model.supported_endpoints);
  for (const [endpoint, api] of ENDPOINTS) {
    if (endpoints.has(endpoint)) return api;
  }
  const sibling = builtins.find((item) => family(item.id) === family(model.id));
  if (sibling) return sibling.api;
  const vendor = model.vendor?.toLowerCase() ?? "";
  if (vendor.includes("anthropic")) return "anthropic-messages";
  if (vendor.includes("openai")) return "openai-responses";
  return "openai-completions";
}

function contextWindowOf(capabilities: LiveModel["capabilities"]): number | undefined {
  const limits = capabilities?.limits;
  if (limits?.max_context_window_tokens !== undefined) return limits.max_context_window_tokens;
  if (limits?.max_prompt_tokens !== undefined && limits.max_output_tokens !== undefined) {
    return limits.max_prompt_tokens + limits.max_output_tokens;
  }
  return undefined;
}

function isReasoning(model: LiveModel): boolean {
  const supports = model.capabilities?.supports;
  return Boolean(
    supports?.reasoning_effort?.length ||
      supports?.adaptive_thinking ||
      supports?.thinking ||
      supports?.max_thinking_budget ||
      supports?.min_thinking_budget,
  );
}

function reasoningEfforts(model: LiveModel, api: string): Set<string> | undefined {
  if (api === "openai-completions") return undefined;
  const efforts = model.capabilities?.supports?.reasoning_effort;
  return efforts?.length ? new Set(efforts) : undefined;
}

// Existing models only need explicit opt-in for pi's optional top tiers.
function deriveOverrideThinkingMap(model: LiveModel, api: string, builtinMap?: ThinkingMap): ThinkingMap | undefined {
  const offered = reasoningEfforts(model, api);
  if (!offered) return undefined;
  const additions: ThinkingMap = {};
  for (const level of OPTIONAL_TIERS) {
    if (offered.has(level) && typeof builtinMap?.[level] !== "string") additions[level] = level;
  }
  return Object.keys(additions).length ? additions : undefined;
}

// Unknown models need a complete map because they have no built-in defaults.
function deriveCustomThinkingMap(model: LiveModel, api: string): ThinkingMap | undefined {
  const offered = reasoningEfforts(model, api);
  if (!offered) return undefined;
  const map: ThinkingMap = {};
  if (api === "openai-responses") map.off = offered.has("none") ? "none" : offered.has("off") ? "off" : null;
  map.minimal = offered.has("minimal") ? "minimal" : (TIERS.find((level) => offered.has(level)) ?? null);
  for (const level of TIERS) map[level] = offered.has(level) ? level : null;
  return map;
}

// Sync plan construction

function buildModelOverride(model: LiveModel, builtin: BuiltinModel): ModelOverride | undefined {
  const override: ModelOverride = {};
  const contextWindow = contextWindowOf(model.capabilities);
  const maxTokens = model.capabilities?.limits?.max_output_tokens;
  if (contextWindow !== undefined && contextWindow > builtin.contextWindow) override.contextWindow = contextWindow;
  if (maxTokens !== undefined && maxTokens > builtin.maxTokens) override.maxTokens = maxTokens;
  const thinkingLevelMap = deriveOverrideThinkingMap(model, builtin.api, builtin.thinkingLevelMap);
  if (thinkingLevelMap) override.thinkingLevelMap = thinkingLevelMap;
  return Object.keys(override).length ? override : undefined;
}

function buildCustomModel(model: LiveModel, api: string, baseUrl: string, headers?: Record<string, string>): CustomModel {
  const supports = model.capabilities?.supports;
  const customModel: CustomModel = {
    id: model.id,
    name: model.name ?? model.id,
    api,
    baseUrl,
    reasoning: isReasoning(model),
    input: supports?.vision ? ["text", "image"] : ["text"],
    contextWindow: contextWindowOf(model.capabilities) ?? 128000,
    maxTokens: model.capabilities?.limits?.max_output_tokens ?? 4096,
    headers, // custom models do not inherit Copilot's required client headers
  };
  if (api === "openai-completions") {
    // Copilot rejects these OpenAI-compatible extensions.
    customModel.compat = { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false };
  } else if (api === "anthropic-messages" && supports?.adaptive_thinking) {
    customModel.compat = { forceAdaptiveThinking: true };
  }
  const thinkingLevelMap = deriveCustomThinkingMap(model, api);
  if (thinkingLevelMap) customModel.thinkingLevelMap = thinkingLevelMap;
  return customModel;
}

function isSelectableChat(model: LiveModel): boolean {
  return (
    model.capabilities?.type === "chat" &&
    model.model_picker_enabled === true &&
    model.policy?.state !== "disabled" &&
    model.capabilities.supports?.tool_calls !== false
  );
}

function buildSyncPlan(
  live: LiveModel[],
  builtins: readonly BuiltinModel[],
  baseUrl: string,
  headers?: Record<string, string>,
): SyncPlan {
  const builtinById = new Map(builtins.map((model) => [model.id, model]));
  const chat = live.filter(isSelectableChat).sort((a, b) => a.id.localeCompare(b.id));
  const plan: SyncPlan = { overrides: {}, customModels: [] };

  for (const model of chat) {
    const builtin = builtinById.get(model.id);
    if (builtin) {
      const override = buildModelOverride(model, builtin);
      if (override) plan.overrides[model.id] = override;
    } else {
      plan.customModels.push(buildCustomModel(model, resolveApi(model, builtins), baseUrl, headers));
    }
  }
  return plan;
}

// I/O

// OAuth baseUrl is request-scoped in current pi; the token carries the tenant endpoint.
function copilotBaseUrlFromToken(accessToken: string): string | undefined {
  const proxyHost = accessToken.match(/proxy-ep=([^;]+)/)?.[1];
  return proxyHost ? `https://${proxyHost.replace(/^proxy\./, "api.")}` : undefined;
}

function agentDir(): string {
  const configured = process.env.PI_CODING_AGENT_DIR;
  return configured ? configured.replace(/^~(\/|$)/, `${homedir()}$1`) : join(homedir(), ".pi", "agent");
}

async function fetchLiveModels(
  baseUrl: string,
  headers: Record<string, string>,
  signal?: AbortSignal,
): Promise<LiveModel[]> {
  const response = await fetch(`${baseUrl}/models`, { headers, signal });
  if (!response.ok) {
    if (response.status === 401) throw new Error("Copilot token expired — send a message or /login, then retry.");
    throw new Error(`Copilot /models failed (${response.status}).`);
  }
  const data = (await response.json()) as { data?: unknown };
  if (!Array.isArray(data.data)) throw new Error("Copilot /models returned an invalid response.");
  return data.data as LiveModel[];
}

// Replace only github-copilot's generated entries; preserve every other config key.
function writeConfig(outFile: string, { overrides, customModels }: SyncPlan): void {
  let cfg: { providers?: Record<string, Record<string, unknown>> } = {};
  if (existsSync(outFile)) {
    try {
      cfg = JSON.parse(readFileSync(outFile, "utf8").replace(/^\uFEFF/, ""));
    } catch {
      throw new Error("models.json is invalid JSON — aborting to avoid overwriting it.");
    }
  }
  cfg.providers ??= {};
  const provider = cfg.providers[PROVIDER] ?? {};
  if (Object.keys(overrides).length) provider.modelOverrides = overrides;
  else delete provider.modelOverrides;
  if (customModels.length) provider.models = customModels;
  else delete provider.models;
  if (Object.keys(provider).length) cfg.providers[PROVIDER] = provider;
  else delete cfg.providers[PROVIDER];
  writeFileSync(outFile, JSON.stringify(cfg, null, 2) + "\n");
}

// Extension entrypoint

export default function (pi: ExtensionAPI): void {
  pi.registerCommand("copilot-sync", {
    description: "Sync additive GitHub Copilot model capabilities from live /models",
    handler: async (_args, ctx) => {
      const builtins = getModels(PROVIDER);
      const builtinIds = new Set(builtins.map((m) => m.id));

      // Reuse a built-in model's required Copilot client headers.
      const probe = ctx.modelRegistry.getAll().find((model) => model.provider === PROVIDER && builtinIds.has(model.id));
      if (!probe) {
        ctx.ui.notify("GitHub Copilot models are unavailable — run /login github-copilot first.", "error");
        return;
      }

      try {
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(probe);
        if (!auth.ok || !auth.apiKey) {
          ctx.ui.notify("Not logged in to GitHub Copilot — run /login github-copilot first.", "error");
          return;
        }
        const baseUrl = copilotBaseUrlFromToken(auth.apiKey) ?? probe.baseUrl;
        const headers = {
          ...auth.headers,
          Authorization: `Bearer ${auth.apiKey}`,
          Accept: "application/json",
          "X-GitHub-Api-Version": COPILOT_API_VERSION,
        };
        const live = await fetchLiveModels(baseUrl, headers, ctx.signal);
        const plan = buildSyncPlan(live, builtins, baseUrl, probe.headers);
        writeConfig(join(agentDir(), "models.json"), plan);
        await ctx.modelRegistry.refresh();

        let msg = `Copilot synced: ${Object.keys(plan.overrides).length} updated, ${plan.customModels.length} added.`;
        if (plan.customModels.length) msg += `\nAdded: ${plan.customModels.map((model) => model.id).join(", ")}`;
        ctx.ui.notify(msg, "info");
      } catch (e) {
        ctx.ui.notify((e as Error).message, "error");
      }
    },
  });
}
