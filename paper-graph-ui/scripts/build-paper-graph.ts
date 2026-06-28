#!/usr/bin/env node
// Scan src/content/papers/**/*.md, build graph.json with nodes, year buckets,
// category metadata, and inter-paper edges (same-category + keyword cross-links).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const PAPERS_DIR = path.join(ROOT, 'src/content/papers');
const OUT = path.join(ROOT, 'src/lib/graph.generated.json');
const NEON = path.join(ROOT, 'src/lib/neon-metadata.generated.json');

type CategoryMeta = {
  title: string;
  icon: string;
  color: string;
  blurb: string;
};

type DomainBridge = [string, string, string];

type Topic = {
  id: string;
  label: string;
  kw: string[];
};

type MarkdownFile = {
  full: string;
  rel: string;
};

type YearMonth = {
  year: number;
  month: number;
};

type NullableYearMonth = {
  year: number;
  month: number | null;
};

type NeonMetadata = {
  id?: string;
  title?: string;
  abstract?: string;
  affiliations?: Record<string, unknown> | null;
  arxiv_comment?: string;
  authors?: string[];
  categories?: string[];
  cited_by_count?: number;
  doi?: string;
  fwci?: number;
  github?: string;
  github_stars?: number;
  interested?: number;
  is_only_important_because_of_company?: boolean;
  org_fullname?: string;
  organization?: string;
  primary_category?: string;
  published?: string;
  score?: number;
  score_reason?: string;
  similar_paper?: string;
  tag_categories_v2?: string[];
  tag_category_v2?: string;
  tag_confidence?: number;
  tag_reason?: string;
  upvotes?: number;
  _has_summary_file?: boolean;
  _summaryless_scope?: boolean;
  [key: string]: unknown;
};

type NeonMetadataById = Record<string, NeonMetadata>;

type Paper = {
  id: string;
  arxivId: string | null;
  title: string;
  category: string;
  slug: string;
  year: number;
  month: number | null;
  isArxiv: boolean;
  topics: string[];
  tokens: string[];
  relativePath: string | null;
  wordCount: number;
  readTimeMin: number | null;
  hasSummary: boolean;
  score: number | null;
  similarPaper: string | null;
  scoreReason: string | null;
  citedByCount: number | null;
  fwci: number | null;
  doi: string | null;
  published: string | null;
  abstract: string | null;
  tagCategoryV2: string | null;
  authors: string[] | null;
  organization: string | null;
  primaryCategory: string | null;
  upvotes: number | null;
  github: string | null;
  githubStars: number | null;
  arxivComment: string | null;
  companyOnly: boolean;
  tagCategories: string[];
};

type Edge = {
  source: string;
  target: string;
  type: string;
  weight: number;
  via?: string;
  topic?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function loadNeon(): NeonMetadataById {
  if (!fs.existsSync(NEON)) {
    console.warn(
      'No Neon metadata cache; run `uv run python scripts/pull-neon-metadata.py` for richer data.'
    );
    return {};
  }
  try {
    const parsed: unknown = JSON.parse(fs.readFileSync(NEON, 'utf8'));
    return isRecord(parsed) ? (parsed as NeonMetadataById) : {};
  } catch (err) {
    console.warn('Failed to read Neon metadata:', err instanceof Error ? err.message : String(err));
    return {};
  }
}

// Loose title equivalence so similar_paper (free-text title) can be matched
// to a graph node. Drops punctuation, lowercases, collapses whitespace.
function normalizeTitle(t: unknown): string {
  return String(t || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

const CATEGORY_META: Record<string, CategoryMeta> = {
  agents: {
    title: 'Agents',
    icon: 'sparkle',
    color: '#aa99ff',
    blurb: 'Agentic systems, tool use, and autonomous reasoning.',
  },
  alignment: {
    title: 'Alignment',
    icon: 'handshake',
    color: '#00E599',
    blurb: 'RLHF, DPO, preference learning, and safety tuning.',
  },
  architecture: {
    title: 'Architecture',
    icon: 'puzzle',
    color: '#259df4',
    blurb: 'Transformers, attention variants, and model design.',
  },
  code: {
    title: 'Code',
    icon: 'code',
    color: '#f0f075',
    blurb: 'Code generation and code-LLM systems.',
  },
  'context-optimization': {
    title: 'Context Optimization',
    icon: 'find-replace',
    color: '#ffa64c',
    blurb: 'KV cache compression, long context, and efficient attention.',
  },
  data: {
    title: 'Data',
    icon: 'data',
    color: '#f0f075',
    blurb: 'Data curation and dataset construction.',
  },
  diffusion: {
    title: 'Diffusion',
    icon: 'sparkle',
    color: '#ff4c79',
    blurb: 'Diffusion models and generative processes.',
  },
  'distributed-training': {
    title: 'Distributed Training',
    icon: 'network',
    color: '#259df4',
    blurb: 'Multi-GPU and multi-node training systems.',
  },
  evaluation: {
    title: 'Evaluation',
    icon: 'metrics',
    color: '#aa99ff',
    blurb: 'Benchmarks, leaderboards, and eval methodologies.',
  },
  'inference-optimization': {
    title: 'Inference Optimization',
    icon: 'autoscaling',
    color: '#00E599',
    blurb: 'Speculative decoding, quantization at inference, and runtime tricks.',
  },
  'llm-systems': {
    title: 'LLM Systems',
    icon: 'gear',
    color: '#259df4',
    blurb: 'End-to-end systems for serving and orchestrating LLMs.',
  },
  'low-precision': {
    title: 'Low Precision',
    icon: 'binary-code',
    color: '#ffa64c',
    blurb: 'FP8, FP4, quantization, and mixed-precision training.',
  },
  moe: {
    title: 'Mixture of Experts',
    icon: 'split-branch',
    color: '#aa99ff',
    blurb: 'MoE routing, sparsity, and expert architectures.',
  },
  multimodal: {
    title: 'Multimodal',
    icon: 'cards',
    color: '#ff4c79',
    blurb: 'Vision-language, audio, and multi-modal fusion.',
  },
  pretraining: {
    title: 'Pretraining',
    icon: 'database',
    color: '#00E599',
    blurb: 'Base model pretraining recipes, scaling laws, and corpora.',
  },
  prompting: {
    title: 'Prompting',
    icon: 'sparkle',
    color: '#f0f075',
    blurb: 'Prompt engineering, CoT, in-context learning.',
  },
  reasoning: {
    title: 'Reasoning',
    icon: 'research',
    color: '#ffa64c',
    blurb: 'CoT, chain-of-thought, test-time compute.',
  },
  retrieval: {
    title: 'Retrieval',
    icon: 'search',
    color: '#259df4',
    blurb: 'Embeddings, RAG, and vector retrieval.',
  },
  'rl-training': {
    title: 'RL Training',
    icon: 'trend-up',
    color: '#00E599',
    blurb: 'RLHF, GRPO, RLVR, and reward modeling.',
  },
  safety: {
    title: 'Safety',
    icon: 'lock-landscape',
    color: '#ff4c79',
    blurb: 'Jailbreak resistance, alignment, and red-teaming.',
  },
  'scaling-laws': {
    title: 'Scaling Laws',
    icon: 'trend-up',
    color: '#ffa64c',
    blurb: 'Compute/data/parameter scaling, Chinchilla, and compute-optimal training.',
  },
  serving: {
    title: 'Serving',
    icon: 'globe',
    color: '#259df4',
    blurb: 'vLLM, SGLang, TensorRT-LLM, and inference servers.',
  },
  'training-methods': {
    title: 'Training Methods',
    icon: 'gear',
    color: '#aa99ff',
    blurb: 'Optimizers, schedulers, and training infrastructure.',
  },
  uncategorized: {
    title: 'Uncategorized',
    icon: 'cards',
    color: '#94979E',
    blurb: 'Papers awaiting categorization.',
  },
  vision: { title: 'Vision', icon: 'screen', color: '#ff4c79', blurb: 'Computer vision and ViT.' },
};

// Domain bridges — which categories are adjacent topically. Used both for
// the high-level domain graph and for cross-domain edge inference.
const DOMAIN_BRIDGES: DomainBridge[] = [
  ['agents', 'llm-systems', 'Agent systems'],
  ['agents', 'rl-training', 'RL-driven agents'],
  ['agents', 'prompting', 'Tool prompting'],
  ['agents', 'reasoning', 'Agent reasoning'],
  ['agents', 'evaluation', 'Agent benchmarks'],
  ['agents', 'code', 'Code agents'],
  ['alignment', 'rl-training', 'RLHF lineage'],
  ['alignment', 'safety', 'Safety alignment'],
  ['architecture', 'pretraining', 'Foundation arch'],
  ['architecture', 'inference-optimization', 'Arch-aware inference'],
  ['architecture', 'moe', 'Sparse architectures'],
  ['architecture', 'context-optimization', 'Long-context arch'],
  ['context-optimization', 'inference-optimization', 'KV cache shared'],
  ['context-optimization', 'serving', 'Cache-aware serving'],
  ['inference-optimization', 'serving', 'Serving stack'],
  ['inference-optimization', 'low-precision', 'Quantized inference'],
  ['inference-optimization', 'distributed-training', 'Parallel inference'],
  ['low-precision', 'pretraining', 'Mixed-precision pretraining'],
  ['low-precision', 'distributed-training', 'FP8 training'],
  ['moe', 'pretraining', 'MoE pretraining'],
  ['moe', 'serving', 'MoE serving'],
  ['multimodal', 'vision', 'Vision-language'],
  ['multimodal', 'pretraining', 'Multimodal pretraining'],
  ['multimodal', 'diffusion', 'Diffusion vision'],
  ['pretraining', 'training-methods', 'Pretraining recipes'],
  ['pretraining', 'data', 'Pretraining data'],
  ['pretraining', 'scaling-laws', 'Compute-optimal pretraining'],
  ['scaling-laws', 'architecture', 'Arch scaling'],
  ['scaling-laws', 'training-methods', 'Scaling-aware training'],
  ['prompting', 'reasoning', 'Prompted reasoning'],
  ['prompting', 'evaluation', 'Prompt evaluation'],
  ['reasoning', 'rl-training', 'RL for reasoning'],
  ['reasoning', 'evaluation', 'Reasoning evals'],
  ['retrieval', 'agents', 'Retrieval agents'],
  ['retrieval', 'prompting', 'RAG prompting'],
  ['rl-training', 'reasoning', 'RLVR reasoning'],
  ['rl-training', 'alignment', 'Preference RL'],
  ['training-methods', 'distributed-training', 'Scaled training'],
];

// Topic keywords to cluster papers into "topic threads" within / across
// categories. First match wins for the primary topic; all matches recorded.
const TOPICS: Topic[] = [
  {
    id: 'tools',
    label: 'Tool use & function calling',
    kw: [
      'toolformer',
      'gorilla',
      'function call',
      'tool use',
      'tool-use',
      'function-calling',
      'mcp',
      'agentbench',
    ],
  },
  {
    id: 'cot',
    label: 'Chain-of-thought reasoning',
    kw: [
      'chain-of-thought',
      'chain of thought',
      'cot ',
      'self-consistency',
      'tree of thought',
      'tot ',
    ],
  },
  {
    id: 'rlhf',
    label: 'RLHF / preference learning',
    kw: ['rlhf', 'preference', 'dpo', 'ppo for', 'reward model', 'human feedback'],
  },
  {
    id: 'grpo',
    label: 'GRPO / RLVR group methods',
    kw: ['grpo', 'rlvr', 'group relative', 'group reward', 'verifiable reward'],
  },
  {
    id: 'spec-decoding',
    label: 'Speculative decoding',
    kw: ['speculative decoding', 'eagle', 'medusa', 'draft model', 'self-speculative', 'mtp'],
  },
  {
    id: 'kv-cache',
    label: 'KV cache / paged attention',
    kw: [
      'kv cache',
      'kv-cache',
      'paged attention',
      'pagedattention',
      'radix cache',
      'prefix cache',
      'hicache',
    ],
  },
  {
    id: 'moe',
    label: 'Mixture of Experts',
    kw: [
      'mixture of experts',
      'moe',
      'mixture-of-experts',
      'expert routing',
      'sparse expert',
      'deepseekmoe',
    ],
  },
  {
    id: 'mamba',
    label: 'State-space / Mamba / linear attention',
    kw: ['mamba', 'state space', 'state-space', 'ssm ', 'linear attention', 'rwkv', 'retnet'],
  },
  {
    id: 'long-context',
    label: 'Long context',
    kw: [
      'long context',
      'long-context',
      'longbench',
      '1m context',
      'context extension',
      'rope scaling',
      'yarn',
    ],
  },
  {
    id: 'fp8',
    label: 'FP8 / low-precision',
    kw: ['fp8', 'fp4', 'int8', 'int4', 'quantization', 'mxfp', 'nvfp', 'gptq', 'awq'],
  },
  {
    id: 'flash',
    label: 'Flash attention',
    kw: ['flashattention', 'flash attention', 'flash-attention', 'flashinfer'],
  },
  {
    id: 'scaling-laws',
    label: 'Scaling laws',
    kw: ['scaling law', 'chinchilla', 'compute-optimal', 'kaplan'],
  },
  {
    id: 'agents-coding',
    label: 'Coding agents',
    kw: ['swe-bench', 'swe bench', 'codeact', 'opendevin', 'devin', 'cursor agent', 'code agent'],
  },
  {
    id: 'distillation',
    label: 'Distillation',
    kw: ['distillation', 'distill ', 'minitron', 'student model'],
  },
  {
    id: 'transformer-core',
    label: 'Transformer foundations',
    kw: [
      'attention is all you need',
      'transformer-xl',
      'bert',
      'gpt-2',
      'gpt-3',
      'gpt-4',
      'roberta',
      'electra',
    ],
  },
  {
    id: 'pretrain-recipe',
    label: 'Pretraining recipes',
    kw: [
      'olmo',
      'llama',
      'qwen',
      'mistral',
      'mixtral',
      'gemma',
      'phi-',
      'deepseek-v',
      'deepseek v',
      'minicpm',
      'pythia',
      'palm',
    ],
  },
  {
    id: 'rag',
    label: 'Retrieval-augmented generation',
    kw: ['retrieval-augmented', 'retrieval augmented', 'rag ', 'colbert', 'rerank'],
  },
  {
    id: 'vision-lm',
    label: 'Vision-language models',
    kw: [
      'llava',
      'visual instruction',
      'molmo',
      'clip ',
      'vit ',
      'vit-',
      'qwen2-vl',
      'qwen2.5-vl',
      'gemini',
    ],
  },
  {
    id: 'reasoning-rl',
    label: 'Reasoning via RL',
    kw: [
      'deepseek-r1',
      'r1 ',
      'o1 ',
      'o3 ',
      'reasoning model',
      'rstar',
      'mathstral',
      'reasoning rl',
    ],
  },
  {
    id: 'serving-systems',
    label: 'Serving systems',
    kw: [
      'vllm',
      'sglang',
      'tensorrt-llm',
      'sarathi',
      'llumnix',
      'distserve',
      'splitwise',
      'mooncake',
      'nanoflow',
      'punica',
      's-lora',
    ],
  },
  {
    id: 'self-improve',
    label: 'Self-improvement & synthetic data',
    kw: [
      'self-instruct',
      'self-improve',
      'self-reward',
      'self-play',
      'self-training',
      'synthetic data',
      'self-rewarding',
    ],
  },
  {
    id: 'world-models',
    label: 'World models',
    kw: ['world model', 'world-model', 'dreamer', 'genie'],
  },
  {
    id: 'multimodal-gen',
    label: 'Multimodal generation / diffusion',
    kw: ['stable diffusion', 'dall-e', 'dall e', 'sora', 'flow matching', 'rectified flow'],
  },
];

function listMarkdownFiles(dir: string, base = ''): MarkdownFile[] {
  const out: MarkdownFile[] = [];
  const items = fs.readdirSync(dir, { withFileTypes: true });
  for (const item of items) {
    const full = path.join(dir, item.name);
    const rel = path.posix.join(base, item.name);
    if (item.isDirectory()) {
      out.push(...listMarkdownFiles(full, rel));
    } else if (item.name.endsWith('.md')) {
      out.push({ full, rel });
    }
  }
  return out;
}

// Strip filename to a slug usable in URLs. Preserves arxiv id prefix if present.
function makeSlug(filename: string): string {
  return filename.replace(/\.md$/, '');
}

function parseArxivId(text: string, filename: string): string | null {
  // 1) from arxiv.org URL anywhere in body (most reliable)
  const url = text.match(/arxiv\.org\/abs\/(\d{4}\.\d{4,6})/i);
  if (url) return url[1];
  // 2) from "ArXiv:" marker, with optional ** wrapping
  const m = text.match(/ArXiv:\**\s*\[?(\d{4}\.\d{4,6})\]?/i);
  if (m) return m[1];
  // 3) from filename prefix
  const fm = filename.match(/^(\d{4}\.\d{4,6})/);
  if (fm) return fm[1];
  return null;
}

function parseTitle(text: string, filename: string): string {
  // first H1
  const m = text.match(/^#\s+(.+?)\s*$/m);
  if (m) return m[1].trim();
  // fallback: filename without arxiv prefix
  return filename
    .replace(/^\d{4}\.\d{4,6}-?/, '')
    .replace(/\.md$/, '')
    .replace(/-/g, ' ');
}

function yearFromArxivId(arxivId: string | null): YearMonth | null {
  if (!arxivId) return null;
  const yy = parseInt(arxivId.slice(0, 2), 10);
  const mm = parseInt(arxivId.slice(2, 4), 10);
  if (Number.isNaN(yy) || Number.isNaN(mm)) return null;
  // arxiv IDs use YYMM format starting in April 2007.
  // YY < 7 means 20YY (so 26 → 2026), >= 7 means 20YY too (07 → 2007).
  // Effectively always 2000 + YY for the new-style IDs (post-2007 only).
  return { year: 2000 + yy, month: mm };
}

// Year inference for classical/non-arxiv papers from common titles.
const CLASSICAL_YEARS: Record<string, number> = {
  attentionisallyouneed: 2017,
  bert: 2018,
  gpt2: 2019,
  gpt3: 2020,
  alphago: 2016,
  alphazero: 2017,
  alphafold: 2020,
  deepresiduallearningforimagerecognition: 2015,
  imagenet: 2012,
  alexnet: 2012,
  vgg: 2014,
  inception: 2014,
  word2vec: 2013,
  seq2seq: 2014,
  dropout: 2014,
  batchnorm: 2015,
  adam: 2014,
  generativeadversarialnetworks: 2014,
  generativeadversarialnets: 2014,
  variationalautoencoder: 2013,
  longshorttermmemory: 1997,
  retnet: 2023,
  rwkv: 2023,
  mamba: 2023,
  flashattention: 2022,
  deeplearning: 2015,
  randomsearchforhyperparameter: 2012,
};

function inferClassicalYear(title: string): number | null {
  const norm = title.toLowerCase().replace(/[^a-z0-9]/g, '');
  for (const [key, year] of Object.entries(CLASSICAL_YEARS)) {
    if (norm.includes(key)) return year;
  }
  return null;
}

function detectTopics(title: string): string[] {
  const lc = title.toLowerCase();
  const matches: string[] = [];
  for (const t of TOPICS) {
    if (t.kw.some((k) => lc.includes(k))) matches.push(t.id);
  }
  return matches;
}

// Extract significant words from a title for cross-paper similarity.
const STOPWORDS = new Set([
  'a',
  'an',
  'the',
  'of',
  'for',
  'and',
  'or',
  'in',
  'on',
  'with',
  'to',
  'from',
  'is',
  'are',
  'be',
  'by',
  'as',
  'at',
  'this',
  'that',
  'these',
  'those',
  'we',
  'our',
  'their',
  'its',
  'it',
  'into',
  'via',
  'using',
  'using',
  'use',
  'a',
  'using',
  'using',
  'data',
  'model',
  'models',
  'language',
  'paper',
  'study',
  'an',
  'approach',
  'method',
  'methods',
  'system',
  'systems',
  'large',
  'small',
  'efficient',
  'efficiently',
  'scaling',
  'scaled',
  'new',
  'novel',
  'better',
  'best',
  'fast',
  'faster',
  'fastest',
  'simple',
  'simpler',
  'simplest',
]);

function tokenize(title: string): string[] {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\-\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

function cleanString(value: unknown): string | null {
  if (value == null) return null;
  const text = String(value).trim();
  return text.length > 0 ? text : null;
}

function normalizeCategorySlug(value: unknown): string | null {
  const text = cleanString(value);
  if (!text) return null;
  const slug = text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return slug.length > 0 ? slug : null;
}

function tagCategoriesFromMeta(
  meta: NeonMetadata | null | undefined,
  fallbackCategory: string | null = null,
  includeFallback = false
): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  const add = (value: unknown): void => {
    const slug = normalizeCategorySlug(value);
    if (!slug || seen.has(slug)) return;
    seen.add(slug);
    out.push(slug);
  };

  if (includeFallback) add(fallbackCategory);

  const raw = meta?.tag_categories_v2;
  if (Array.isArray(raw)) {
    for (const category of raw) add(category);
  }
  add(meta?.tag_category_v2);

  if (out.length === 0) add(fallbackCategory);
  return out;
}

function categoryFromMetadata(meta: NeonMetadata): string {
  // Only let a paper's raw arxiv `primary_category` (e.g. cs.CV → `cs-cv`)
  // become its bucket if that slug is a CURATED category. Otherwise fold it
  // into `uncategorized` so the broad org-paper backfill (which carries every
  // arxiv primary category, incl. physics/bio/math) does not spray ~46 raw
  // arxiv-code buckets across the category nav. Curated tags still win.
  const fromPrimary = normalizeCategorySlug(meta?.primary_category);
  return (
    (fromPrimary && fromPrimary in CATEGORY_META ? fromPrimary : null) ??
    tagCategoriesFromMeta(meta)[0] ??
    'uncategorized'
  );
}

function slugFromMetadataPaper(arxivId: string | null, title: string): string {
  const id = cleanString(arxivId);
  if (id) return id;
  const titleSlug = normalizeCategorySlug(title);
  return titleSlug ?? 'paper';
}

function yearMonthFromMetadata(
  meta: NeonMetadata,
  arxivId: string | null,
  title: string
): NullableYearMonth {
  if (meta?.published) {
    const date = new Date(meta.published);
    if (!Number.isNaN(date.getTime())) {
      return { year: date.getUTCFullYear(), month: date.getUTCMonth() + 1 };
    }
  }
  const ymd = yearFromArxivId(arxivId);
  return {
    year: ymd?.year ?? inferClassicalYear(title) ?? 2000,
    month: ymd?.month ?? null,
  };
}

function main(): void {
  if (!fs.existsSync(PAPERS_DIR)) {
    console.error(`papers dir not found at ${PAPERS_DIR}`);
    process.exit(1);
  }

  const files = listMarkdownFiles(PAPERS_DIR);
  const papers: Paper[] = [];
  const categoryCounts = new Map<string, number>();
  const neon = loadNeon();
  const neonHits = { found: 0, missing: 0 };

  for (const { full, rel } of files) {
    const parts = rel.split('/');
    if (parts.length < 2) continue; // ignore docs/index.md
    if (parts[parts.length - 1] === 'index.md') continue;

    const category = parts[0];
    const filename = parts[parts.length - 1];

    const text = fs.readFileSync(full, 'utf8');
    const arxivId = parseArxivId(text, filename);
    const title = parseTitle(text, filename);
    const slug = makeSlug(filename);
    const ymd = yearFromArxivId(arxivId);

    const meta: NeonMetadata | null = arxivId ? (neon[arxivId] ?? null) : null;
    if (arxivId) {
      if (meta) neonHits.found++;
      else neonHits.missing++;
    }

    // Prefer Neon's `published` date when present — it's authoritative.
    let year: number | undefined;
    let month: number | null | undefined;
    if (meta?.published) {
      const d = new Date(meta.published);
      if (!Number.isNaN(d.getTime())) {
        year = d.getUTCFullYear();
        month = d.getUTCMonth() + 1;
      }
    }
    if (!year) {
      year = ymd?.year ?? inferClassicalYear(title) ?? 2000;
      month = ymd?.month ?? null;
    }

    const topics = detectTopics(title);
    const tokens = tokenize(title);

    // Prefer Neon's title (cleaner) when available, but fall back to the
    // markdown H1 if the Neon row has none.
    const finalTitle = meta?.title || title;

    // Per-paper read-time estimate from word count of the body. 200 wpm
    // covers technical reading on a phone.
    const wordCount = text
      .replace(/[`{}$]/g, ' ')
      .split(/\s+/)
      .filter(Boolean).length;
    const readTimeMin = Math.max(1, Math.round(wordCount / 200));

    papers.push({
      id: `${category}/${slug}`,
      arxivId,
      title: finalTitle,
      category,
      slug,
      year,
      month: month ?? null,
      isArxiv: Boolean(arxivId),
      topics,
      tokens,
      relativePath: rel,
      wordCount,
      readTimeMin,
      hasSummary: true,
      // enriched from Neon (all optional; ui must handle nulls)
      score: meta?.score ?? null,
      similarPaper: meta?.similar_paper ?? null,
      scoreReason: meta?.score_reason ?? null,
      citedByCount: meta?.cited_by_count ?? null,
      fwci: meta?.fwci ?? null,
      doi: meta?.doi ?? null,
      published: meta?.published ?? null,
      abstract: meta?.abstract ?? null,
      tagCategoryV2: meta?.tag_category_v2 ?? null,
      authors: meta?.authors ?? null,
      organization: meta?.org_fullname || meta?.organization || null,
      primaryCategory: meta?.primary_category ?? null,
      upvotes: meta?.upvotes ?? null,
      github: meta?.github ?? null,
      githubStars: meta?.github_stars ?? null,
      arxivComment: meta?.arxiv_comment ?? null,
      // True when interested=1 was set just because the paper is from a
      // tracked company (Qwen/DeepSeek/Moonshot/ByteDance/NVIDIA/etc.)
      // rather than picked on merit. UI filter "merit only" hides these.
      companyOnly: Boolean(meta?.is_only_important_because_of_company),
      // Multi-tag from V4-Pro. Always a non-empty array — if Neon hasn't
      // multi-tagged this paper yet, fall back to its filesystem category.
      // Routes that "list papers in category X" filter where X is IN
      // tagCategories, so a paper can show up in multiple category indexes.
      tagCategories: tagCategoriesFromMeta(meta, category),
    });
    // categoryCounts is incremented once per (paper, category) pair — so a
    // multi-tagged paper bumps each of its categories' counts. The home page
    // + the /c/[category] page filters surface papers via tagCategories now,
    // not the filesystem category, so the counts must match.
    const lastPaper = papers[papers.length - 1]!;
    for (const cat of lastPaper.tagCategories) {
      categoryCounts.set(cat, (categoryCounts.get(cat) ?? 0) + 1);
    }
  }

  if (Object.keys(neon).length > 0) {
    console.log(`  Neon metadata: ${neonHits.found} matched, ${neonHits.missing} missing`);
  }

  // Build edges:
  // 1) same-category chronological: prev paper in same category = "parent"
  // 2) topic-thread: connect papers sharing a topic, prev chronological
  // 3) domain-bridge: between categories
  // 4) token similarity within category: capture sequel-style relationships

  // Sort papers within each category by year then month
  const byCategory = new Map<string, Paper[]>();
  for (const p of papers) {
    if (!byCategory.has(p.category)) byCategory.set(p.category, []);
    byCategory.get(p.category)!.push(p);
  }
  for (const arr of byCategory.values()) {
    arr.sort((a, b) => (a.year !== b.year ? a.year - b.year : (a.month ?? 0) - (b.month ?? 0)));
  }

  const edges: Edge[] = [];

  // LLM-judged similar-paper edges: the Neon scorer wrote a free-text title
  // for each paper's nearest neighbor. Resolve by normalized-title match.
  const titleIndex = new Map<string, Paper>();
  for (const p of papers) titleIndex.set(normalizeTitle(p.title), p);
  let similarHits = 0;
  for (const p of papers) {
    if (!p.similarPaper) continue;
    const match = titleIndex.get(normalizeTitle(p.similarPaper));
    if (!match || match.id === p.id) continue;
    // Direction: similar paper → this paper (the older one usually predates
    // the newer one; orient by year so the dep tree walks backward in time).
    const [older, newer] = (
      match.year !== p.year ? match.year < p.year : (match.month ?? 0) < (p.month ?? 0)
    )
      ? [match, p]
      : [p, match];
    edges.push({
      source: older.id,
      target: newer.id,
      type: 'llm-similar',
      weight: 5,
      via: p.similarPaper,
    });
    similarHits++;
  }
  if (similarHits > 0) console.log(`  llm-similar edges: ${similarHits}`);

  // Chronological backbone within category — chain each paper to the previous
  // one in the same category (capped to keep the graph readable).
  for (const arr of byCategory.values()) {
    for (let i = 1; i < arr.length; i++) {
      edges.push({
        source: arr[i - 1].id,
        target: arr[i].id,
        type: 'category-chronology',
        weight: 1,
      });
    }
  }

  // Topic threads: group by topic, chain chronologically across categories.
  const byTopic = new Map<string, Paper[]>();
  for (const p of papers) {
    for (const t of p.topics) {
      if (!byTopic.has(t)) byTopic.set(t, []);
      byTopic.get(t)!.push(p);
    }
  }
  for (const arr of byTopic.values()) {
    arr.sort((a, b) => (a.year !== b.year ? a.year - b.year : (a.month ?? 0) - (b.month ?? 0)));
    for (let i = 1; i < arr.length; i++) {
      // Skip if same-category (already in backbone)
      if (arr[i - 1].category === arr[i].category) continue;
      edges.push({
        source: arr[i - 1].id,
        target: arr[i].id,
        type: 'topic',
        topic: arr[i].topics.find((t) => arr[i - 1].topics.includes(t)),
        weight: 2,
      });
    }
  }

  // Token-similarity edges within category: pairs sharing 2+ significant
  // tokens. Cap each paper at top-3 neighbors to control density.
  for (const arr of byCategory.values()) {
    for (let i = 0; i < arr.length; i++) {
      const scored: { other: Paper; overlap: number }[] = [];
      for (let j = 0; j < arr.length; j++) {
        if (i === j) continue;
        const setI = new Set(arr[i].tokens);
        const overlap = arr[j].tokens.filter((t) => setI.has(t)).length;
        if (overlap >= 2) scored.push({ other: arr[j], overlap });
      }
      scored.sort((a, b) => b.overlap - a.overlap);
      for (const { other, overlap } of scored.slice(0, 3)) {
        edges.push({
          source: arr[i].id,
          target: other.id,
          type: 'similarity',
          weight: overlap,
        });
      }
    }
  }

  const summaryArxivIds = new Set(
    papers.map((p) => p.arxivId).filter((id): id is string => Boolean(id))
  );
  const paperIds = new Set(papers.map((p) => p.id));
  let summarylessAdded = 0;

  for (const [metadataId, meta] of Object.entries(neon)) {
    if (meta?._summaryless_scope !== true) continue;
    const arxivId = cleanString(meta.id) ?? cleanString(metadataId);
    if (arxivId && summaryArxivIds.has(arxivId)) continue;

    const title = cleanString(meta.title) ?? arxivId ?? 'Untitled paper';
    const category = categoryFromMetadata(meta);
    const baseSlug = slugFromMetadataPaper(arxivId, title);
    let slug = baseSlug;
    let id = `${category}/${slug}`;
    let suffix = 2;
    while (paperIds.has(id)) {
      slug = `${baseSlug}-${suffix}`;
      id = `${category}/${slug}`;
      suffix += 1;
    }

    const { year, month } = yearMonthFromMetadata(meta, arxivId, title);
    const tagCategories = tagCategoriesFromMeta(meta, category, true);

    papers.push({
      id,
      arxivId,
      title,
      category,
      slug,
      year,
      month,
      isArxiv: Boolean(arxivId?.match(/^\d{4}\.\d{4,6}$/)),
      topics: detectTopics(title),
      tokens: [],
      relativePath: null,
      wordCount: 0,
      readTimeMin: null,
      hasSummary: false,
      score: meta?.score ?? null,
      similarPaper: meta?.similar_paper ?? null,
      scoreReason: meta?.score_reason ?? null,
      citedByCount: meta?.cited_by_count ?? null,
      fwci: meta?.fwci ?? null,
      doi: meta?.doi ?? null,
      published: meta?.published ?? null,
      abstract: meta?.abstract ?? null,
      tagCategoryV2: meta?.tag_category_v2 ?? null,
      authors: meta?.authors ?? null,
      organization: meta?.org_fullname || meta?.organization || null,
      primaryCategory: meta?.primary_category ?? null,
      upvotes: meta?.upvotes ?? null,
      github: meta?.github ?? null,
      githubStars: meta?.github_stars ?? null,
      arxivComment: meta?.arxiv_comment ?? null,
      companyOnly: Boolean(meta?.is_only_important_because_of_company),
      tagCategories,
    });

    paperIds.add(id);
    if (arxivId) summaryArxivIds.add(arxivId);
    for (const cat of tagCategories) {
      categoryCounts.set(cat, (categoryCounts.get(cat) ?? 0) + 1);
    }
    summarylessAdded++;
  }
  if (summarylessAdded > 0) console.log(`  summary-less nodes: ${summarylessAdded}`);

  // Build category metadata
  const categories: Record<string, CategoryMeta & { slug: string; count: number }> = {};
  for (const [slug, count] of categoryCounts.entries()) {
    categories[slug] = {
      slug,
      ...(CATEGORY_META[slug] ?? {
        title: slug.replace(/-/g, ' '),
        icon: 'cards',
        color: '#94979E',
        blurb: '',
      }),
      count,
    };
  }

  // Year buckets
  const yearBuckets: Record<string, string[]> = {};
  for (const p of papers) {
    const y = String(p.year);
    if (!yearBuckets[y]) yearBuckets[y] = [];
    yearBuckets[y].push(p.id);
  }

  // Strip tokens off the wire — we only needed them locally to build edges.
  const wirePapers = papers.map(({ tokens: _, ...rest }) => rest);
  const summaryCount = papers.filter((p) => p.hasSummary !== false).length;
  const summarylessCount = papers.length - summaryCount;

  const out = {
    generatedAt: new Date().toISOString(),
    counts: {
      papers: papers.length,
      summaries: summaryCount,
      summarylessPapers: summarylessCount,
      categories: Object.keys(categories).length,
      edges: edges.length,
    },
    categories,
    domainBridges: DOMAIN_BRIDGES.map(([a, b, label]) => ({ a, b, label })),
    topics: TOPICS.map(({ id, label }) => ({ id, label })),
    papers: wirePapers,
    edges,
    yearBuckets,
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(out, null, 2));

  console.log(`Wrote ${OUT}`);
  console.log(`  papers: ${papers.length}`);
  console.log(`  summaries: ${summaryCount}`);
  console.log(`  summary-less: ${summarylessCount}`);
  console.log(`  categories: ${Object.keys(categories).length}`);
  console.log(`  edges: ${edges.length}`);
  const years = Object.keys(yearBuckets).sort();
  console.log(`  year span: ${years[0]} → ${years[years.length - 1]}`);
}

main();
