// npm-description: Add copilot2api as a native Pi provider with live model discovery.
// npm-keywords: copilot2api, github-copilot, copilot, provider, pi-provider, models, model-discovery, llm, ai-gateway, openai-compatible, anthropic-compatible

/**
 * Integrates with Pi's /login flow and credential store for the API root and token.
 * Pi refreshes and persists the remote catalog automatically. Chat models use their
 * native Messages, Responses, or Chat endpoint, decision models Copilot's decision
 * APIs, and image models the Images API.
 */

import {
  type ApiKeyCredential,
  type AssistantImages,
  type ClassifierApi,
  type ClassifierModel,
  type ImageApi,
  type ImageContent,
  type ImageModel,
  type ImagesContext,
  type ImagesOptions,
  type Model,
  type ThinkingLevelMap,
  anthropicMessagesApi,
  createProvider,
  isModelType,
  openAICompletionsApi,
  openAIResponsesApi,
} from "@earendil-works/pi-ai/compat";
import { builtinProviders, getBuiltinClassifierModels, getBuiltinModels } from "@earendil-works/pi-ai/providers/all";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const PROVIDER = "copilot2api";
// Prefer native model protocols; copilot2api can translate when only a fallback exists.
const CHAT_ENDPOINTS = [
  ["/messages", "anthropic-messages"],
  ["/responses", "openai-responses"],
  ["/chat/completions", "openai-completions"],
] as const;
type ChatApi = (typeof CHAT_ENDPOINTS)[number][1];
const CLASSIFIER_ENDPOINTS = [
  ["/decisions", "openai-decisions"],
  ["/systemone", "typesafe-system-one"],
] as const;
const IMAGES_API = "openai-images";
type CatalogModel = Model<ChatApi> | ClassifierModel<ClassifierApi> | ImageModel<ImageApi>;

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
const COPILOT_CHAT_MODELS = new Map(getBuiltinModels("github-copilot").map((model) => [model.id, model]));
const COPILOT_CLASSIFIER_MODELS = new Map(getBuiltinClassifierModels("github-copilot").map((model) => [model.id, model]));
// pi exposes its decision API clients to extensions only through its providers.
const copilotClassify = builtinProviders().find((provider) => provider.id === "github-copilot")?.classify;

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

async function requestError(path: string, response: Response): Promise<Error> {
  const detail = (await response.text()).trim().slice(0, 300);
  return new Error(`copilot2api ${path} failed (${response.status})${detail ? `: ${detail}` : ""}`);
}

async function fetchLiveModels(apiRoot: string, token: string, signal: AbortSignal): Promise<LiveModel[]> {
  const response = await fetch(`${apiRoot}/v1/models`, {
    headers: { Accept: "application/json", Authorization: `Bearer ${token}` },
    signal,
  });
  if (!response.ok) throw await requestError("/v1/models", response);
  const payload = (await response.json()) as { data?: unknown };
  if (!Array.isArray(payload.data)) throw new Error("copilot2api /v1/models returned an invalid response.");
  return payload.data as LiveModel[];
}

// Live catalog translation

function endpointsOf(live: LiveModel): Set<string> {
  return new Set((live.supported_endpoints ?? []).map((endpoint) => endpoint.trim().replace(/^\/v1/, "")));
}

// copilot2api serves every API under /v1. pi's Anthropic client and its Copilot decision clients add the /v1 themselves.
function baseUrlOf(apiRoot: string, api: string): string {
  return ["anthropic-messages", "openai-decisions", "typesafe-system-one"].includes(api) ? apiRoot : `${apiRoot}/v1`;
}

function resolveApi(live: LiveModel): ChatApi {
  const endpoints = endpointsOf(live);
  for (const [endpoint, api] of CHAT_ENDPOINTS) {
    if (endpoints.has(endpoint)) return api;
  }
  const inheritedApi = CHAT_ENDPOINTS.find(([, api]) => api === COPILOT_CHAT_MODELS.get(live.id)?.api)?.[1];
  if (inheritedApi) return inheritedApi;
  const vendor = live.vendor?.toLowerCase() ?? "";
  if (vendor.includes("anthropic")) return "anthropic-messages";
  if (vendor.includes("openai")) return "openai-responses";
  return "openai-completions";
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

function thinkingLevelMapOf(live: LiveModel, api: ChatApi, inherited?: ThinkingLevelMap): ThinkingLevelMap | undefined {
  const efforts = live.capabilities?.supports?.reasoning_effort;
  if (!efforts?.length) return inherited;
  const offered = new Set(efforts);
  const map: ThinkingLevelMap = { ...inherited };
  if (api === "openai-responses") map.off = offered.has("none") ? "none" : offered.has("off") ? "off" : null;
  map.minimal = offered.has("minimal") ? "minimal" : offered.has("low") ? "low" : null;
  for (const level of THINKING_LEVELS) map[level] = offered.has(level) ? level : null;
  return map;
}

// Strict flags as in pi's github-copilot catalog. Copilot's /responses treats tools without `strict` as strict.
function compatOf(live: LiveModel, api: ChatApi): Model<ChatApi>["compat"] {
  const supports = live.capabilities?.supports;
  switch (api) {
    case "anthropic-messages":
      return { supportsStrictTools: true, ...(supports?.adaptive_thinking && { forceAdaptiveThinking: true }) };
    case "openai-responses":
      return { supportsStrictMode: true };
    case "openai-completions":
      return {
        supportsStrictMode: true,
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: Boolean(supports?.reasoning_effort?.length),
      };
  }
}

function toChatModel(live: LiveModel, apiRoot: string): Model<ChatApi> | undefined {
  const supports = live.capabilities?.supports;
  // The model picker lists the chat models Copilot offers; other types never appear in it.
  if (!live.model_picker_enabled || supports?.tool_calls === false) return undefined;
  const inherited = COPILOT_CHAT_MODELS.get(live.id);
  const api = resolveApi(live);
  return {
    id: live.id,
    name: live.name ?? inherited?.name ?? live.id,
    api,
    provider: PROVIDER,
    baseUrl: baseUrlOf(apiRoot, api),
    reasoning: isReasoning(live),
    thinkingLevelMap: thinkingLevelMapOf(live, api, inherited?.thinkingLevelMap),
    input: supports?.vision ? ["text", "image"] : ["text"],
    cost: inherited?.cost ?? ZERO_COST,
    contextWindow: contextWindowOf(live),
    maxTokens: live.capabilities?.limits?.max_output_tokens ?? inherited?.maxTokens ?? 16384,
    compat: { ...(inherited?.api === api ? inherited.compat : undefined), ...compatOf(live, api) },
  };
}

function toClassifierModel(live: LiveModel, apiRoot: string): ClassifierModel<ClassifierApi> | undefined {
  const endpoints = endpointsOf(live);
  const api = CLASSIFIER_ENDPOINTS.find(([endpoint]) => endpoints.has(endpoint))?.[1];
  if (!api) return undefined;
  const builtin = COPILOT_CLASSIFIER_MODELS.get(live.id);
  return {
    type: "classifier",
    id: live.id,
    name: live.name ?? builtin?.name ?? live.id,
    api,
    provider: PROVIDER,
    baseUrl: baseUrlOf(apiRoot, api),
    input: builtin?.input ?? ["text"],
    cost: builtin?.cost ?? ZERO_COST,
    contextWindow: live.capabilities?.limits?.max_prompt_tokens ?? builtin?.contextWindow ?? 32768,
  };
}

function toImageModel(live: LiveModel, apiRoot: string): ImageModel<ImageApi> | undefined {
  const endpoints = endpointsOf(live);
  if (!endpoints.has("/images/generations")) return undefined;
  return {
    type: "image",
    id: live.id,
    name: live.name ?? live.id,
    api: IMAGES_API,
    provider: PROVIDER,
    baseUrl: baseUrlOf(apiRoot, IMAGES_API),
    input: endpoints.has("/images/edits") ? ["text", "image"] : ["text"],
    output: ["image"],
    cost: ZERO_COST,
  };
}

function toCatalogModel(live: LiveModel, apiRoot: string): CatalogModel | undefined {
  if (live.policy?.state === "disabled") return undefined;
  switch (live.capabilities?.type) {
    case "chat":
      return toChatModel(live, apiRoot);
    case "decisions":
      return toClassifierModel(live, apiRoot);
    case "image":
      return toImageModel(live, apiRoot);
  }
}

function buildModels(live: LiveModel[], apiRoot: string): CatalogModel[] {
  return live.flatMap((model) => toCatalogModel(model, apiRoot) ?? []).sort((a, b) => a.id.localeCompare(b.id));
}

// Images API

function imagesRequest(payload: Record<string, unknown>, images: ImageContent[]) {
  if (!images.length) {
    return { path: "images/generations", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  }
  const form = new FormData();
  for (const [key, value] of Object.entries(payload)) {
    form.append(key, typeof value === "string" ? value : JSON.stringify(value));
  }
  for (const [index, image] of images.entries()) {
    const file = new Blob([Buffer.from(image.data, "base64")], { type: image.mimeType });
    form.append("image[]", file, `image-${index}.${image.mimeType.split("/")[1]}`);
  }
  return { path: "images/edits", headers: {}, body: form };
}

async function generateImages(
  model: ImageModel<ImageApi>,
  context: ImagesContext,
  options?: ImagesOptions,
): Promise<AssistantImages> {
  const result: AssistantImages = {
    api: model.api,
    provider: model.provider,
    model: model.id,
    output: [],
    stopReason: "stop",
    timestamp: Date.now(),
  };
  try {
    if (!options?.apiKey) throw new Error(`No API key for provider: ${model.provider}`);
    const prompt = context.input.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
    const images = context.input.filter((block) => block.type === "image");
    const params = { model: model.id, prompt };
    const payload = ((await options.onPayload?.(params, model)) ?? params) as Record<string, unknown>;
    const request = imagesRequest(payload, images);
    const response = await (options.fetch ?? fetch)(`${model.baseUrl}/${request.path}`, {
      method: "POST",
      headers: { Authorization: `Bearer ${options.apiKey}`, ...request.headers },
      body: request.body,
      signal: options.signal,
    });
    await options.onResponse?.({ status: response.status, headers: Object.fromEntries(response.headers) }, model);
    if (!response.ok) throw await requestError(`/v1/${request.path}`, response);
    const body = (await response.json()) as {
      data?: { b64_json?: string }[];
      output_format?: string;
      usage?: { input_tokens?: number; output_tokens?: number };
    };
    const mimeType = `image/${body.output_format ?? "png"}`;
    for (const { b64_json: data } of body.data ?? []) {
      if (data) result.output.push({ type: "image", data, mimeType });
    }
    if (body.usage) {
      const { input_tokens: input = 0, output_tokens: output = 0 } = body.usage;
      const cost = { ...ZERO_COST, total: 0 };
      result.usage = { input, output, cacheRead: 0, cacheWrite: 0, totalTokens: input + output, cost };
    }
  } catch (error) {
    result.stopReason = options?.signal?.aborted ? "aborted" : "error";
    result.errorMessage = error instanceof Error ? error.message : String(error);
  }
  return result;
}

// Provider

export function createCopilot2ApiProvider() {
  // Login already fetches a validated catalog. Expose it immediately (pi first performs
  // an offline credential sync), then let the following network refresh persist it.
  let loginCatalog: { apiRoot: string; models: CatalogModel[] } | undefined;
  const provider = createProvider<ChatApi>({
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
          if (!models.some((model) => isModelType(model, "chat"))) {
            throw new Error("copilot2api returned no selectable chat models.");
          }
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
    classifiers: copilotClassify && {
      "openai-decisions": { classify: copilotClassify },
      "typesafe-system-one": { classify: copilotClassify },
    },
    images: { [IMAGES_API]: { generateImages } },
  });
  return {
    ...provider,
    getModels: () => loginCatalog?.models.filter((model) => isModelType(model, "chat")) ?? provider.getModels(),
    getAllModels: () => loginCatalog?.models ?? provider.getAllModels?.() ?? provider.getModels(),
  };
}

export default function (pi: ExtensionAPI): void {
  pi.registerProvider(createCopilot2ApiProvider());
}
