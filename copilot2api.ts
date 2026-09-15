// npm-description: Add copilot2api as a native Pi provider with live model discovery.
// npm-keywords: copilot2api, github-copilot, copilot, provider, pi-provider, models, model-discovery, llm, ai-gateway, openai-compatible, anthropic-compatible

/**
 * Integrates with Pi's /login flow and credential store for the API root and token.
 * Pi refreshes and persists the remote catalog automatically, while each model uses
 * its native Messages, Responses, or Chat endpoint.
 */

import {
  type ApiKeyCredential,
  type Model,
  type ThinkingLevelMap,
  anthropicMessagesApi,
  createProvider,
  getModels,
  openAICompletionsApi,
  openAIResponsesApi,
} from "@earendil-works/pi-ai/compat";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER = "copilot2api";
// Prefer native model protocols; copilot2api can translate when only a fallback exists.
const ENDPOINTS = [
  ["/messages", "anthropic-messages"],
  ["/responses", "openai-responses"],
  ["/chat/completions", "openai-completions"],
] as const;
type SupportedApi = (typeof ENDPOINTS)[number][1];

interface LiveModel {
  id: string;
  name?: string;
  vendor?: string;
  model_picker_enabled?: boolean;
  supported_endpoints?: string[] | null;
  policy?: { state?: string };
  capabilities?: {
    type?: string;
    supports?: {
      tool_calls?: boolean;
      vision?: boolean;
      adaptive_thinking?: boolean;
      reasoning_effort?: string[];
    };
    limits?: {
      max_context_window_tokens?: number;
      max_prompt_tokens?: number;
      max_output_tokens?: number;
    };
  };
}

const ZERO_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
const THINKING_LEVELS = ["low", "medium", "high", "xhigh", "max"] as const;
const COPILOT_MODELS = new Map(getModels("github-copilot").map((model) => [model.id, model]));

// Endpoint and authentication

function normalizeApiRoot(input: string): string {
  const value = input.trim();
  if (!value) throw new Error("API URL is required.");
  const url = new URL(value.includes("://") ? value : `http://${value}`);
  if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("API URL must use http or https.");
  if (url.username || url.password) throw new Error("API URL must not contain credentials.");
  if (url.search || url.hash) throw new Error("API URL must not contain a query or fragment.");
  url.pathname = url.pathname.replace(/\/+$/, "").replace(/\/models$/, "").replace(/\/v1$/, "");
  return url.toString().replace(/\/$/, "");
}

function credentialConfig(credential?: ApiKeyCredential): { apiRoot: string; token: string } | undefined {
  const apiRoot = credential?.env?.apiRoot;
  return credential?.key && apiRoot ? { apiRoot, token: credential.key } : undefined;
}

async function fetchLiveModels(apiRoot: string, token: string, signal: AbortSignal): Promise<LiveModel[]> {
  const response = await fetch(`${apiRoot}/v1/models`, {
    headers: { Accept: "application/json", Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok) {
    const detail = (await response.text()).trim().slice(0, 300);
    throw new Error(`copilot2api /v1/models failed (${response.status})${detail ? `: ${detail}` : ""}`);
  }
  const payload = (await response.json()) as { data?: unknown };
  if (!Array.isArray(payload.data)) throw new Error("copilot2api /v1/models returned an invalid response.");
  return payload.data as LiveModel[];
}

// Live catalog translation

function isSelectable(live: LiveModel): boolean {
  return (
    live.capabilities?.type === "chat" &&
    live.model_picker_enabled === true &&
    live.policy?.state !== "disabled" &&
    live.capabilities.supports?.tool_calls !== false
  );
}

function resolveApi(live: LiveModel): SupportedApi {
  const endpoints = new Set(
    (live.supported_endpoints ?? []).map((endpoint) => endpoint.trim().replace(/^\/v1/, "")),
  );
  for (const [endpoint, api] of ENDPOINTS) {
    if (endpoints.has(endpoint)) return api;
  }
  const inheritedApi = ENDPOINTS.find(([, api]) => api === COPILOT_MODELS.get(live.id)?.api)?.[1];
  if (inheritedApi) return inheritedApi;
  const vendor = live.vendor?.toLowerCase() ?? "";
  if (vendor.includes("anthropic")) return "anthropic-messages";
  if (vendor.includes("openai")) return "openai-responses";
  return "openai-completions";
}

function modelBaseUrl(apiRoot: string, api: SupportedApi): string {
  return api === "anthropic-messages" ? apiRoot : `${apiRoot}/v1`;
}

function contextWindowOf(live: LiveModel): number {
  const limits = live.capabilities?.limits;
  if (limits?.max_context_window_tokens !== undefined) return limits.max_context_window_tokens;
  if (limits?.max_prompt_tokens !== undefined && limits.max_output_tokens !== undefined) {
    return limits.max_prompt_tokens + limits.max_output_tokens;
  }
  return 128000;
}

function isReasoning(live: LiveModel): boolean {
  const supports = live.capabilities?.supports;
  // copilot2api rewrites legacy token-budget thinking to adaptive effort, so budget
  // metadata alone is not actionable unless the model advertises effort support.
  return Boolean(supports?.reasoning_effort?.length || supports?.adaptive_thinking);
}

function thinkingLevelMapOf(live: LiveModel, api: SupportedApi, inherited?: ThinkingLevelMap): ThinkingLevelMap | undefined {
  const efforts = live.capabilities?.supports?.reasoning_effort;
  if (!efforts?.length) return inherited;
  const offered = new Set(efforts);
  const map: ThinkingLevelMap = { ...inherited };
  if (api === "openai-responses") map.off = offered.has("none") ? "none" : offered.has("off") ? "off" : null;
  map.minimal = offered.has("minimal") ? "minimal" : offered.has("low") ? "low" : null;
  for (const level of THINKING_LEVELS) map[level] = offered.has(level) ? level : null;
  return map;
}

function toModel(live: LiveModel, apiRoot: string): Model<SupportedApi> {
  const inherited = COPILOT_MODELS.get(live.id);
  const api = resolveApi(live);
  const supports = live.capabilities?.supports;
  let compat: Model<SupportedApi>["compat"] = inherited?.api === api ? inherited.compat : undefined;
  if (api === "openai-completions") {
    compat = {
      ...compat,
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: Boolean(supports?.reasoning_effort?.length),
    };
  } else if (api === "anthropic-messages" && supports?.adaptive_thinking) {
    compat = { ...compat, forceAdaptiveThinking: true };
  }
  return {
    id: live.id,
    name: live.name ?? inherited?.name ?? live.id,
    api,
    provider: PROVIDER,
    baseUrl: modelBaseUrl(apiRoot, api),
    reasoning: isReasoning(live),
    thinkingLevelMap: thinkingLevelMapOf(live, api, inherited?.thinkingLevelMap),
    input: supports?.vision ? ["text", "image"] : ["text"],
    cost: inherited?.cost ?? ZERO_COST,
    contextWindow: contextWindowOf(live),
    maxTokens: live.capabilities?.limits?.max_output_tokens ?? inherited?.maxTokens ?? 16384,
    compat,
  };
}

function buildModels(live: LiveModel[], apiRoot: string): Model<SupportedApi>[] {
  return live
    .filter(isSelectable)
    .sort((a, b) => a.id.localeCompare(b.id))
    .map((model) => toModel(model, apiRoot));
}

// Provider

export function createCopilot2ApiProvider() {
  // Login already fetches a validated catalog. Expose it immediately (pi first performs
  // an offline credential sync), then let the following network refresh persist it.
  let loginCatalog: { apiRoot: string; models: Model<SupportedApi>[] } | undefined;
  const provider = createProvider<SupportedApi>({
    id: PROVIDER,
    name: "Copilot2API",
    auth: {
      apiKey: {
        name: "Copilot2API endpoint and token",
        async login(interaction) {
          const apiRoot = normalizeApiRoot(
            await interaction.prompt({
              type: "text",
              message: "API root URL",
              placeholder: "http://127.0.0.1:7777/api/<account-id>",
            }),
          );
          const token = (
            await interaction.prompt({ type: "secret", message: "API token" })
          ).trim();
          if (!token) throw new Error("API token is required.");
          interaction.notify({ type: "progress", message: "Checking endpoint and loading models..." });
          const models = buildModels(await fetchLiveModels(apiRoot, token, interaction.signal), apiRoot);
          if (!models.length) throw new Error("copilot2api returned no selectable chat models.");
          loginCatalog = { apiRoot, models };
          return { type: "api_key", key: token, env: { apiRoot } };
        },
        async check({ credential }) {
          return credentialConfig(credential) ? { type: "api_key", source: "stored credentials" } : undefined;
        },
        async resolve({ credential }) {
          const config = credentialConfig(credential);
          return config
            ? {
                auth: { apiKey: config.token },
                env: { apiRoot: config.apiRoot },
                source: "stored credentials",
              }
            : undefined;
        },
      },
    },
    models: [],
    async fetchModels(context) {
      const config = context.credential?.type === "api_key" ? credentialConfig(context.credential) : undefined;
      if (!config) throw new Error("Copilot2API is not configured — run /login copilot2api.");
      if (loginCatalog?.apiRoot === config.apiRoot) {
        const models = loginCatalog.models;
        loginCatalog = undefined;
        return models;
      }
      return buildModels(await fetchLiveModels(config.apiRoot, config.token, context.signal), config.apiRoot);
    },
    api: {
      "anthropic-messages": anthropicMessagesApi(),
      "openai-responses": openAIResponsesApi(),
      "openai-completions": openAICompletionsApi(),
    },
  });
  return { ...provider, getModels: () => loginCatalog?.models ?? provider.getModels() };
}

export default function (pi: ExtensionAPI): void {
  pi.registerProvider(createCopilot2ApiProvider());
}
