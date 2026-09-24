/**
 * Single source of truth for the ChunkHound engine's architecture.
 *
 * Drives the five-stage engine pipeline and the diagrams on the
 * /docs/architecture explanation page (why the engine is shaped this way).
 *
 * Hosts the structured facts (stages, phases, layers, roles, index phases) that
 * the page's diagrams render. The prose narrates these facts; the labels and
 * structure are defined only here, so a rename is a one-place edit.
 * Facts are code-verified against v6.0.0: the indexing engine is Rust — scanning,
 * diffing, scheduling, database writes, and compaction; Python still supplies
 * parsing and the embedding fallback, both slated to move to Rust.
 */

export type Stage = {
  icon: string;
  /** Short label shown in the pipeline overview. */
  title: string;
  /** Pipeline phase the stage belongs to; consecutive stages group visually. */
  phase: "Index" | "Exploration" | "Synthesis";
};

export const STAGES: Stage[] = [
  { icon: "ph-code", title: "Structural chunking", phase: "Index" },
  { icon: "ph-crosshair", title: "Exact + semantic retrieval", phase: "Exploration" },
  { icon: "ph-graph", title: "LLM-guided exploration", phase: "Exploration" },
  { icon: "ph-funnel", title: "Reranking + adaptive cutoff", phase: "Exploration" },
  { icon: "ph-arrows-merge", title: "Map-reduce synthesis", phase: "Synthesis" },
];

/**
 * Deep research is two decoupled phases. Exploration narrows any corpus the
 * same way (search, score, walk gaps); synthesis turns the converged result
 * set into one cited answer. Because they meet at that stable boundary, the
 * engine is corpus-agnostic. Sources are shown in SYSTEM_LAYERS, not here.
 */
export type ResearchMechanism = { title: string; detail: string };
export type ResearchPhase = {
  id: "exploration" | "synthesis";
  label: string;
  /** One-line role, paired with the label in the phase's band head. */
  caption: string;
  mechanisms: ResearchMechanism[];
};

export const RESEARCH_PHASES: ResearchPhase[] = [
  {
    id: "exploration",
    label: "Exploration",
    caption: "Every source is searched the same way.",
    mechanisms: [
      { title: "Semantic search", detail: "Finds code by meaning (HNSW kNN)." },
      { title: "Regex", detail: "Finds it by name, pattern, or symbol." },
      { title: "Rerank", detail: "Scores every candidate; keeps the elbow." },
      {
        title: "Utility LLM",
        detail: "Follow-ups, imports, and gaps until results plateau.",
      },
    ],
  },
  {
    id: "synthesis",
    label: "Synthesis",
    caption: "The converged results become one cited answer.",
    mechanisms: [
      { title: "Map-reduce", detail: "Cluster, summarize in parallel, merge." },
      {
        title: "Constants extraction",
        detail: "Exact config values lifted from chunk metadata; definite, no model.",
      },
      {
        title: "Ledger",
        detail: "Reconcile facts so every claim traces to a source.",
      },
      { title: "Synthesis LLM", detail: "Writes the one cited answer." },
    ],
  },
];

/** Layered view: clients call the engine; providers serve it; sources feed it. */
export type SystemNode = { title: string };
export type SystemLayer = {
  id: string;
  label: string;
  caption: string;
  nodes: SystemNode[];
};

export const SYSTEM_LAYERS: SystemLayer[] = [
  {
    id: "clients",
    label: "Clients",
    caption: "One contract, two front doors.",
    nodes: [
      { title: "AI agent (MCP)" },
      { title: "CLI / shell" },
    ],
  },
  {
    id: "engine",
    label: "ChunkHound (local process)",
    caption: "Owns the index, the watcher, and the research path.",
    nodes: [
      { title: "Research service" },
      { title: "Indexing coordinator" },
      { title: "DuckDB or LanceDB index" },
    ],
  },
  {
    id: "providers",
    label: "Model providers (local or remote)",
    caption: "Embeddings, reranking, and synthesis.",
    nodes: [
      { title: "Embedding provider" },
      { title: "Reranker" },
      { title: "Utility LLM" },
      { title: "Synthesis LLM" },
    ],
  },
  {
    id: "sources",
    label: "Sources",
    caption: "Persistent code plus request-scoped git and web.",
    nodes: [
      { title: "Repository files" },
      { title: "Git history" },
      { title: "Web pages" },
    ],
  },
];

/** The two LLM roles and the volume of work each absorbs. */
export type LlmRole = { name: string; calls: string };

export const LLM_ROLES: LlmRole[] = [
  {
    name: "Utility model",
    calls: "Many small calls",
  },
  {
    name: "Synthesis model",
    calls: "One expensive pass",
  },
];

/** The real index build, phase by phase. */
export type IndexPhase = {
  title: string;
  detail: string;
  /** Compute owning the phase today; rendered as the figure's ownership tag. */
  language: "Rust" | "Python" | "Rust + Python";
};

export const INDEX_PHASES: IndexPhase[] = [
  { title: "Discover", detail: "A parallel ignore walker finds files on every core.", language: "Rust" },
  { title: "Detect changes", detail: "Rust diffs the filesystem against the stored index; only changes proceed.", language: "Rust" },
  { title: "Parse ∥ store", detail: "Rust streams batches into the database while Python tree-sitter parses them on a process pool.", language: "Rust + Python" },
  { title: "Embed", detail: "Native HTTP embedding (OpenAI-compatible, VoyageAI); other providers fall back to Python.", language: "Rust" },
  { title: "Compact + rebuild", detail: "An atomic DB swap reclaims space; HNSW indexes rebuild once.", language: "Rust" },
];

