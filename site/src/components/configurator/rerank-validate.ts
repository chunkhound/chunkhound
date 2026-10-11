import type {
  ConfiguratorEmbeddingProviderOption,
  ConfiguratorRerankFormat,
} from "./types.ts";

// Shared <p> under the reranker inputs (single id in Configurator.astro).
export const RERANK_URL_ERROR_ID = "rerank-url-error";

const ABSOLUTE_URL_ERROR = "Use an absolute HTTP(S) URL or a relative path.";
// Exported so tests can assert the rendered error equals this constant.
export const PROTOCOL_RELATIVE_ERROR =
  "Protocol-relative URLs are not supported — use an absolute HTTP(S) URL or a relative path.";
// Exported so tests can assert the rendered error equals this constant.
export const INCOMPLETE_URL_ERROR =
  "That URL is incomplete — use a full HTTP(S) URL such as https://reranker.example.com/rerank.";
// The backend classifies with a case-sensitive startswith, so an uppercase
// scheme parses in the browser but is not absolute to the backend.
// Exported for the same test reason as INCOMPLETE_URL_ERROR above.
export const SCHEME_CASE_ERROR =
  "Write the scheme in lowercase: http:// or https://.";
const RELATIVE_NO_BASE_URL_ERROR =
  "A relative reranker URL requires this retrieval preset to have a base URL.";

// An http(s)-like scheme that failed to parse is a typo'd URL, not a relative
// path — say so instead of returning the generic "absolute URL" message.
function incompleteSchemeUrlError(value: string): string | undefined {
  if (/^https?:/i.test(value)) return INCOMPLETE_URL_ERROR;
  if (/^[a-z][a-z\d+.-]*:/i.test(value)) return ABSOLUTE_URL_ERROR;
  return undefined;
}

// Returns undefined for canonical HTTP(S) URLs. For scheme forms that parse
// in the browser but miss the backend's case-sensitive startswith, returns
// SCHEME_CASE_ERROR for uppercase schemes ("HTTPS://x") or INCOMPLETE_URL_ERROR
// for structural faults ("http:/x") — so the emitted config always passes.
function classifySchemeParityError(
  value: string,
  protocol: string,
): string | undefined {
  if (value.startsWith("http://") || value.startsWith("https://"))
    return undefined;
  // "HTTPS://x" is a scheme-case fault; "http:/x" is a structural one.
  if (value.slice(0, protocol.length) !== protocol) return SCHEME_CASE_ERROR;
  return INCOMPLETE_URL_ERROR;
}

export function rerankUrlValidationError(
  value: string,
  hasBaseUrl: boolean,
): string | undefined {
  // The configurator trims the field before calling here (configurator.ts; the
  // emitted config trims in builders.ts configWithReranker too); trim again so
  // a direct caller cannot misread a padded URL as a scheme fault.
  value = value.trim();
  if (!value) return undefined;
  // Protocol-relative URLs inherit the page scheme, so the fetch target is
  // ambiguous — reject them explicitly instead of treating them as relative.
  if (value.startsWith("//")) return PROTOCOL_RELATIVE_ERROR;
  try {
    const protocol = new URL(value).protocol;
    if (protocol === "http:" || protocol === "https:") {
      return classifySchemeParityError(value, protocol);
    }
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

// Single source for the state label: the build-time pre-JS paint
// (Configurator.astro) and the runtime text updates (configurator.ts)
// must render the same string or the first JS repaint would flicker.
export function rerankerStateLabel(state: string): string {
  return state === "required" ? "Required dependency" : "";
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

/**
 * Mirrors embedding_config.validate_rerank_configuration. UI-only additions:
 * (1) the <select> only offers cohere/tei/voyage, so "auto" — a backend
 * runtime sentinel — is rejected here; (2) rerankUrlValidationError is
 * stricter on ambiguous URL forms (protocol-relative, empty host) and trims
 * surrounding whitespace before classifying. When reranking is implied but no
 * URL is given, the URL error fires first, matching the backend's
 * model_validator, which raises RERANK_BASE_URL_REQUIRED before the shared
 * validator reaches its model check.
 * Reranking is implied by a model or TEI format; without a preset base_url
 * the backend rejects it unless the provider reranks via SDK (VoyageAI).
 */
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
  // VoyageAI reranks via SDK: the backend accepts any format/model, so the
  // UI must not block a valid config the operator can't fix by picking a format.
  if (retrieval.config.provider === "voyageai" && !url) {
    return undefined;
  }
  return validateRerankerFields(effective);
}

// Format/model checks for configs that cleared the URL checks (URL present,
// or VoyageAI SDK bypass). Separated so rerankerValidationError stays a linear
// chain of early returns.
function validateRerankerFields(
  effective: EffectiveReranker,
): RerankerValidation | undefined {
  if (!effective.hasFormat)
    return {
      message: "Choose TEI, Cohere, or Voyage for the external reranker.",
      field: "format",
    };
  return effective.requiresModel && !effective.hasModel
    ? {
        message: "A Cohere or Voyage reranker requires a model.",
        field: "model",
      }
    : undefined;
}
