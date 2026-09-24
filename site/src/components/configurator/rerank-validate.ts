import type {
  ConfiguratorEmbeddingProviderOption,
  ConfiguratorRerankFormat,
} from "./types.ts";

// Shared <p> under the reranker inputs (single id in Configurator.astro).
export const RERANK_URL_ERROR_ID = "rerank-url-error";

const ABSOLUTE_URL_ERROR = "Use an absolute HTTP(S) URL or a relative path.";
const PROTOCOL_RELATIVE_ERROR =
  "Protocol-relative URLs are not supported — use an absolute HTTP(S) URL or a relative path.";
const INCOMPLETE_URL_ERROR =
  "That URL is incomplete — use a full HTTP(S) URL such as https://reranker.example.com/rerank.";
const RELATIVE_NO_BASE_URL_ERROR =
  "A relative reranker URL requires this retrieval preset to have a base URL.";

// An http(s)-like scheme that failed to parse is a typo'd URL, not a relative
// path — say so instead of returning the generic "absolute URL" message.
function incompleteSchemeUrlError(value: string): string | undefined {
  if (/^https?:/i.test(value)) return INCOMPLETE_URL_ERROR;
  if (/^[a-z][a-z\d+.-]*:/i.test(value)) return ABSOLUTE_URL_ERROR;
  return undefined;
}

export function rerankUrlValidationError(
  value: string,
  hasBaseUrl: boolean,
): string | undefined {
  if (!value) return undefined;
  // Protocol-relative URLs inherit the page scheme, so the fetch target is
  // ambiguous — reject them explicitly instead of treating them as relative.
  if (value.startsWith("//")) return PROTOCOL_RELATIVE_ERROR;
  try {
    const protocol = new URL(value).protocol;
    if (protocol === "http:" || protocol === "https:") return undefined;
    return ABSOLUTE_URL_ERROR;
  } catch {
    const schemeError = incompleteSchemeUrlError(value);
    if (schemeError) return schemeError;
  }
  return hasBaseUrl ? undefined : RELATIVE_NO_BASE_URL_ERROR;
}

// Exported so select-derived values can be narrowed without a cast.
export function isRerankFormat(
  value: string,
): value is ConfiguratorRerankFormat {
  return value === "cohere" || value === "tei" || value === "voyage";
}

export function rerankerState(
  retrieval: ConfiguratorEmbeddingProviderOption,
): "included" | "required" {
  return retrieval.reranker.included ? "included" : "required";
}

export function rerankerStatus(
  retrieval: ConfiguratorEmbeddingProviderOption,
): string {
  if (!retrieval.reranker.included) {
    return "Add an external reranker to enable research, web search, and fetch. Semantic search remains available without one.";
  }
  return retrieval.reranker.status ?? "Included by this preset.";
}

export function rerankerSummary(
  retrieval: ConfiguratorEmbeddingProviderOption,
): string {
  return rerankerState(retrieval) === "required"
    ? "Add external reranker"
    : "Customize reranker";
}

export interface EffectiveReranker {
  format: string;
  model: string;
  hasFormat: boolean;
  requiresModel: boolean;
  hasModel: boolean;
}

// Single source for effective reranker settings (explicit pick or preset
// default), so error text and aria-invalid flags cannot disagree.
function effectiveReranker(
  retrieval: ConfiguratorEmbeddingProviderOption,
  format: string,
  model: string,
): EffectiveReranker {
  const effectiveFormat = String(format || retrieval.config.rerank_format || "");
  const effectiveModel = String(model || retrieval.config.rerank_model || "");
  return {
    format: effectiveFormat,
    model: effectiveModel,
    hasFormat: isRerankFormat(effectiveFormat),
    requiresModel: effectiveFormat === "cohere" || effectiveFormat === "voyage",
    hasModel: Boolean(effectiveModel),
  };
}

export type RerankerErrorField = "url" | "format" | "model";

export interface RerankerValidation {
  message: string;
  field: RerankerErrorField;
}

// Returns the offending field so aria-invalid can target the exact input the
// user must fix, not a neighboring one.
export function rerankerValidationError(
  retrieval: ConfiguratorEmbeddingProviderOption,
  url: string,
  format: string,
  model: string,
): RerankerValidation | undefined {
  const urlError = rerankUrlValidationError(
    url,
    typeof retrieval.config.base_url === "string",
  );
  if (urlError) return { message: urlError, field: "url" };
  const effective = effectiveReranker(retrieval, format, model);
  if (!url && !effective.format && !effective.model) return undefined;
  // Deliberately stricter subset of embedding_config.validate_rerank_config:
  // (1) voyageai presets without a rerank URL still get the model-required
  // check (the backend returns early); (2) "auto" rerank_format is rejected
  // because the UI format field is a required <select> with explicit choices.
  // URL is checked before model-required here (backend reversed) — UX-intent:
  // surface the fixable endpoint field first. Reranking is implied by a model
  // or TEI format; without a preset base_url the backend rejects it unless the
  // provider reranks via SDK (VoyageAI).
  const rerankingImplied = effective.hasModel || effective.format === "tei";
  if (
    !url &&
    rerankingImplied &&
    !retrieval.config.base_url &&
    retrieval.config.provider !== "voyageai"
  ) {
    return {
      message: "This preset has no base URL — enter the reranker URL.",
      field: "url",
    };
  }
  if (!effective.hasFormat)
    return {
      message: "Choose TEI, Cohere, or Voyage for the external reranker.",
      field: "format",
    };
  return effective.requiresModel && !effective.hasModel
    ? { message: "A Cohere or Voyage reranker requires a model.", field: "model" }
    : undefined;
}
