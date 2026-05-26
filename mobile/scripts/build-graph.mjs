#!/usr/bin/env node
// Mirror of paper-graph-ui/scripts/build-paper-graph.mjs adapted for mobile.
// Reads ../paper-graph-ui/src/content/papers/**/*.md
// Emits src/lib/graph.generated.json (baked into the bundle).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const MOBILE_ROOT = path.resolve(__dirname, '..');
const PAPERS_DIR = path.resolve(__dirname, '../../paper-graph-ui/src/content/papers');
const NEON = path.resolve(__dirname, '../../paper-graph-ui/src/lib/neon-metadata.generated.json');
const OUT = path.join(MOBILE_ROOT, 'src/lib/graph.generated.json');
const ASSETS_OUT = path.join(MOBILE_ROOT, 'assets/papers');
const ASSET_MAP_OUT = path.join(MOBILE_ROOT, 'src/lib/paper-asset-map.ts');

function loadNeon() {
  if (!fs.existsSync(NEON)) {
    console.warn('No Neon metadata cache found; enriched metadata will be missing.');
    return {};
  }
  try {
    return JSON.parse(fs.readFileSync(NEON, 'utf8'));
  } catch (err) {
    console.warn('Failed to read Neon metadata:', err.message);
    return {};
  }
}

function normalizeTitle(t) {
  return String(t || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

const CATEGORY_META = {
  agents: { title: 'Agents', color: '#aa99ff', blurb: 'Agentic systems, tool use, and autonomous reasoning.' },
  alignment: { title: 'Alignment', color: '#00E599', blurb: 'RLHF, DPO, preference learning, and safety tuning.' },
  architecture: { title: 'Architecture', color: '#259df4', blurb: 'Transformers, attention variants, and model design.' },
  code: { title: 'Code', color: '#f0f075', blurb: 'Code generation and code-LLM systems.' },
  'context-optimization': { title: 'Context Optimization', color: '#ffa64c', blurb: 'KV cache compression, long context, and efficient attention.' },
  data: { title: 'Data', color: '#f0f075', blurb: 'Data curation and dataset construction.' },
  diffusion: { title: 'Diffusion', color: '#ff4c79', blurb: 'Diffusion models and generative processes.' },
  'distributed-training': { title: 'Distributed Training', color: '#259df4', blurb: 'Multi-GPU and multi-node training systems.' },
  evaluation: { title: 'Evaluation', color: '#aa99ff', blurb: 'Benchmarks, leaderboards, and eval methodologies.' },
  'inference-optimization': { title: 'Inference Optimization', color: '#00E599', blurb: 'Speculative decoding, quantization at inference, and runtime tricks.' },
  'llm-systems': { title: 'LLM Systems', color: '#259df4', blurb: 'End-to-end systems for serving and orchestrating LLMs.' },
  'low-precision': { title: 'Low Precision', color: '#ffa64c', blurb: 'FP8, FP4, quantization, and mixed-precision training.' },
  moe: { title: 'Mixture of Experts', color: '#aa99ff', blurb: 'MoE routing, sparsity, and expert architectures.' },
  multimodal: { title: 'Multimodal', color: '#ff4c79', blurb: 'Vision-language, audio, and multi-modal fusion.' },
  pretraining: { title: 'Pretraining', color: '#00E599', blurb: 'Base model pretraining recipes, scaling laws, and corpora.' },
  prompting: { title: 'Prompting', color: '#f0f075', blurb: 'Prompt engineering, CoT, in-context learning.' },
  reasoning: { title: 'Reasoning', color: '#ffa64c', blurb: 'CoT, chain-of-thought, test-time compute.' },
  retrieval: { title: 'Retrieval', color: '#259df4', blurb: 'Embeddings, RAG, and vector retrieval.' },
  'rl-training': { title: 'RL Training', color: '#00E599', blurb: 'RLHF, GRPO, RLVR, and reward modeling.' },
  safety: { title: 'Safety', color: '#ff4c79', blurb: 'Jailbreak resistance, alignment, and red-teaming.' },
  'scaling-laws': { title: 'Scaling Laws', color: '#ffa64c', blurb: 'Compute/data/parameter scaling, Chinchilla, and compute-optimal training.' },
  serving: { title: 'Serving', color: '#259df4', blurb: 'vLLM, SGLang, TensorRT-LLM, and inference servers.' },
  'training-methods': { title: 'Training Methods', color: '#aa99ff', blurb: 'Optimizers, schedulers, and training infrastructure.' },
  uncategorized: { title: 'Uncategorized', color: '#94979E', blurb: 'Papers awaiting categorization.' },
  vision: { title: 'Vision', color: '#ff4c79', blurb: 'Computer vision and ViT.' },
};

const DOMAIN_BRIDGES = [
  ['agents', 'llm-systems', 'Agent systems'],
  ['agents', 'rl-training', 'RL-driven agents'],
  ['agents', 'prompting', 'Tool prompting'],
  ['agents', 'reasoning', 'Agent reasoning'],
  ['alignment', 'rl-training', 'RLHF lineage'],
  ['alignment', 'safety', 'Safety alignment'],
  ['architecture', 'pretraining', 'Foundation arch'],
  ['architecture', 'inference-optimization', 'Arch-aware inference'],
  ['architecture', 'moe', 'Sparse architectures'],
  ['context-optimization', 'inference-optimization', 'KV cache shared'],
  ['inference-optimization', 'serving', 'Serving stack'],
  ['inference-optimization', 'low-precision', 'Quantized inference'],
  ['low-precision', 'pretraining', 'Mixed-precision pretraining'],
  ['moe', 'serving', 'MoE serving'],
  ['multimodal', 'vision', 'Vision-language'],
  ['pretraining', 'data', 'Pretraining data'],
  ['pretraining', 'scaling-laws', 'Compute-optimal pretraining'],
  ['reasoning', 'rl-training', 'RL for reasoning'],
  ['retrieval', 'agents', 'Retrieval agents'],
  ['rl-training', 'alignment', 'Preference RL'],
  ['training-methods', 'distributed-training', 'Scaled training'],
];

const TOPICS = [
  { id: 'tools', label: 'Tool use & function calling', kw: ['toolformer', 'gorilla', 'function call', 'tool use', 'tool-use', 'function-calling', 'mcp', 'agentbench'] },
  { id: 'cot', label: 'Chain-of-thought reasoning', kw: ['chain-of-thought', 'chain of thought', 'cot ', 'self-consistency', 'tree of thought', 'tot '] },
  { id: 'rlhf', label: 'RLHF / preference learning', kw: ['rlhf', 'preference', 'dpo', 'ppo for', 'reward model', 'human feedback'] },
  { id: 'grpo', label: 'GRPO / RLVR group methods', kw: ['grpo', 'rlvr', 'group relative', 'group reward', 'verifiable reward'] },
  { id: 'spec-decoding', label: 'Speculative decoding', kw: ['speculative decoding', 'eagle', 'medusa', 'draft model', 'self-speculative', 'mtp'] },
  { id: 'kv-cache', label: 'KV cache / paged attention', kw: ['kv cache', 'kv-cache', 'paged attention', 'pagedattention', 'radix cache', 'prefix cache', 'hicache'] },
  { id: 'moe', label: 'Mixture of Experts', kw: ['mixture of experts', 'moe', 'mixture-of-experts', 'expert routing', 'sparse expert', 'deepseekmoe'] },
  { id: 'mamba', label: 'State-space / Mamba / linear attention', kw: ['mamba', 'state space', 'state-space', 'ssm ', 'linear attention', 'rwkv', 'retnet'] },
  { id: 'long-context', label: 'Long context', kw: ['long context', 'long-context', 'longbench', '1m context', 'context extension', 'rope scaling', 'yarn'] },
  { id: 'fp8', label: 'FP8 / low-precision', kw: ['fp8', 'fp4', 'int8', 'int4', 'quantization', 'mxfp', 'nvfp', 'gptq', 'awq'] },
  { id: 'flash', label: 'Flash attention', kw: ['flashattention', 'flash attention', 'flash-attention', 'flashinfer'] },
  { id: 'scaling-laws', label: 'Scaling laws', kw: ['scaling law', 'chinchilla', 'compute-optimal', 'kaplan'] },
  { id: 'agents-coding', label: 'Coding agents', kw: ['swe-bench', 'swe bench', 'codeact', 'opendevin', 'devin', 'cursor agent', 'code agent'] },
  { id: 'distillation', label: 'Distillation', kw: ['distillation', 'distill ', 'minitron', 'student model'] },
  { id: 'transformer-core', label: 'Transformer foundations', kw: ['attention is all you need', 'transformer-xl', 'bert', 'gpt-2', 'gpt-3', 'gpt-4', 'roberta', 'electra'] },
  { id: 'pretrain-recipe', label: 'Pretraining recipes', kw: ['olmo', 'llama', 'qwen', 'mistral', 'mixtral', 'gemma', 'phi-', 'deepseek-v', 'deepseek v', 'minicpm', 'pythia', 'palm'] },
  { id: 'rag', label: 'Retrieval-augmented generation', kw: ['retrieval-augmented', 'retrieval augmented', 'rag ', 'colbert', 'rerank'] },
  { id: 'vision-lm', label: 'Vision-language models', kw: ['llava', 'visual instruction', 'molmo', 'clip ', 'vit ', 'vit-', 'qwen2-vl', 'qwen2.5-vl', 'gemini'] },
  { id: 'reasoning-rl', label: 'Reasoning via RL', kw: ['deepseek-r1', 'r1 ', 'o1 ', 'o3 ', 'reasoning model', 'rstar', 'mathstral', 'reasoning rl'] },
  { id: 'serving-systems', label: 'Serving systems', kw: ['vllm', 'sglang', 'tensorrt-llm', 'sarathi', 'llumnix', 'distserve', 'splitwise', 'mooncake', 'nanoflow', 'punica', 's-lora'] },
];

function listMarkdownFiles(dir, base = '') {
  const out = [];
  if (!fs.existsSync(dir)) return out;
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

function makeSlug(filename) {
  return filename.replace(/\.md$/, '');
}

function parseArxivId(text, filename) {
  const url = text.match(/arxiv\.org\/abs\/(\d{4}\.\d{4,6})/i);
  if (url) return url[1];
  const m = text.match(/ArXiv:\**\s*\[?(\d{4}\.\d{4,6})\]?/i);
  if (m) return m[1];
  const fm = filename.match(/^(\d{4}\.\d{4,6})/);
  if (fm) return fm[1];
  return null;
}

function parseTitle(text, filename) {
  const m = text.match(/^#\s+(.+?)\s*$/m);
  if (m) return m[1].trim();
  return filename
    .replace(/^\d{4}\.\d{4,6}-?/, '')
    .replace(/\.md$/, '')
    .replace(/-/g, ' ');
}

function yearFromArxivId(arxivId) {
  if (!arxivId) return null;
  const yy = parseInt(arxivId.slice(0, 2), 10);
  const mm = parseInt(arxivId.slice(2, 4), 10);
  if (Number.isNaN(yy) || Number.isNaN(mm)) return null;
  return { year: 2000 + yy, month: mm };
}

const CLASSICAL_YEARS = {
  attentionisallyouneed: 2017, bert: 2018, gpt2: 2019, gpt3: 2020,
  alphago: 2016, alphazero: 2017, alphafold: 2020,
  deepresiduallearningforimagerecognition: 2015,
  imagenet: 2012, alexnet: 2012, vgg: 2014, inception: 2014,
  word2vec: 2013, seq2seq: 2014, dropout: 2014, batchnorm: 2015,
  adam: 2014, generativeadversarialnetworks: 2014,
  generativeadversarialnets: 2014, variationalautoencoder: 2013,
  longshorttermmemory: 1997, retnet: 2023, rwkv: 2023, mamba: 2023,
  flashattention: 2022, deeplearning: 2015,
  randomsearchforhyperparameter: 2012,
};

function inferClassicalYear(title) {
  const norm = title.toLowerCase().replace(/[^a-z0-9]/g, '');
  for (const [key, year] of Object.entries(CLASSICAL_YEARS)) {
    if (norm.includes(key)) return year;
  }
  return null;
}

function detectTopics(title) {
  const lc = title.toLowerCase();
  const matches = [];
  for (const t of TOPICS) {
    if (t.kw.some((k) => lc.includes(k))) matches.push(t.id);
  }
  return matches;
}

const STOPWORDS = new Set([
  'a', 'an', 'the', 'of', 'for', 'and', 'or', 'in', 'on', 'with', 'to', 'from',
  'is', 'are', 'be', 'by', 'as', 'at', 'this', 'that', 'these', 'those', 'we',
  'our', 'their', 'its', 'it', 'into', 'via', 'using', 'use', 'data', 'model',
  'models', 'language', 'paper', 'study', 'approach', 'method', 'methods',
  'system', 'systems', 'large', 'small', 'efficient', 'efficiently', 'scaling',
  'scaled', 'new', 'novel', 'better', 'best', 'fast', 'faster', 'fastest',
  'simple', 'simpler', 'simplest',
]);

function tokenize(title) {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9\-\s]/g, ' ')
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

function parsePitch(text) {
  // Extract the ## 🎯 Pitch section (short abstract shown on cards)
  const m = text.match(/##\s*🎯\s*Pitch\s*\n+([\s\S]*?)(?=\n##|\n---|\n#|$)/);
  if (m) return m[1].trim().replace(/\*\*/g, '').slice(0, 400);
  return null;
}

function main() {
  if (!fs.existsSync(PAPERS_DIR)) {
    console.error(`papers dir not found at ${PAPERS_DIR}`);
    process.exit(1);
  }

  const files = listMarkdownFiles(PAPERS_DIR);
  const papers = [];
  const categoryCounts = new Map();
  const neon = loadNeon();
  let neonFound = 0, neonMissing = 0;

  // Sync md files into assets/papers/ so Metro bundles them. We rsync rather
  // than symlink because Metro chokes on symlinks in asset paths.
  // The asset-map collected below feeds src/lib/paper-asset-map.ts so
  // useMarkdownContent can require() each .md (Metro resolves at bundle time).
  //
  // IMPORTANT: nuke the existing assets/papers/ tree before copying. After a
  // T2-style re-categorization run, the source moves files between category
  // dirs but assets/papers/ would still have the stale copies at the old
  // paths — bloating the EAS upload by ~2x and confusing Metro with files
  // that aren't in the asset map.
  fs.rmSync(ASSETS_OUT, { recursive: true, force: true });
  fs.mkdirSync(ASSETS_OUT, { recursive: true });
  const assetMap = []; // [{ key, relAssetPath }]

  for (const { full, rel } of files) {
    const parts = rel.split('/');
    if (parts.length < 2) continue;
    if (parts[parts.length - 1] === 'index.md') continue;

    const category = parts[0];
    const filename = parts[parts.length - 1];

    const text = fs.readFileSync(full, 'utf8');
    const arxivId = parseArxivId(text, filename);
    const title = parseTitle(text, filename);
    const slug = makeSlug(filename);

    // Copy md into bundled assets and record path for the asset map.
    const destDir = path.join(ASSETS_OUT, category);
    fs.mkdirSync(destDir, { recursive: true });
    const destPath = path.join(destDir, filename);
    fs.copyFileSync(full, destPath);
    assetMap.push({ key: `${category}/${slug}`, rel: `${category}/${filename}` });
    const ymd = yearFromArxivId(arxivId);
    const pitch = parsePitch(text);

    const meta = (arxivId && neon[arxivId]) || null;
    if (arxivId) { if (meta) neonFound++; else neonMissing++; }

    let year, month;
    if (meta?.published) {
      const d = new Date(meta.published);
      if (!Number.isNaN(d.getTime())) { year = d.getUTCFullYear(); month = d.getUTCMonth() + 1; }
    }
    if (!year) {
      year = ymd?.year ?? inferClassicalYear(title) ?? 2000;
      month = ymd?.month ?? null;
    }

    const topics = detectTopics(title);
    const tokens = tokenize(title);

    const finalTitle = meta?.title || title;
    const wordCount = text.replace(/[`{}$]/g, ' ').split(/\s+/).filter(Boolean).length;
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
      pitch,
      relativePath: rel,
      wordCount,
      readTimeMin,
      score: meta?.score ?? null,
      similarPaper: meta?.similar_paper ?? null,
      scoreReason: meta?.score_reason ?? null,
      tagCategoryV2: meta?.tag_category_v2 ?? null,
      authors: meta?.authors ?? null,
      organization: meta?.org_fullname || meta?.organization || null,
      primaryCategory: meta?.primary_category ?? null,
      upvotes: meta?.upvotes ?? null,
      github: meta?.github ?? null,
      githubStars: meta?.github_stars ?? null,
    });
    categoryCounts.set(category, (categoryCounts.get(category) ?? 0) + 1);
  }

  console.log(`  Neon metadata: ${neonFound} matched, ${neonMissing} missing`);

  // Build edges
  const byCategory = new Map();
  for (const p of papers) {
    if (!byCategory.has(p.category)) byCategory.set(p.category, []);
    byCategory.get(p.category).push(p);
  }
  for (const arr of byCategory.values()) {
    arr.sort((a, b) => a.year !== b.year ? a.year - b.year : (a.month ?? 0) - (b.month ?? 0));
  }

  const edges = [];

  const titleIndex = new Map();
  for (const p of papers) titleIndex.set(normalizeTitle(p.title), p);
  let similarHits = 0;
  for (const p of papers) {
    if (!p.similarPaper) continue;
    const match = titleIndex.get(normalizeTitle(p.similarPaper));
    if (!match || match.id === p.id) continue;
    const [older, newer] =
      (match.year !== p.year ? match.year < p.year : (match.month ?? 0) < (p.month ?? 0))
        ? [match, p] : [p, match];
    edges.push({ source: older.id, target: newer.id, type: 'llm-similar', weight: 5 });
    similarHits++;
  }
  console.log(`  llm-similar edges: ${similarHits}`);

  for (const arr of byCategory.values()) {
    for (let i = 1; i < arr.length; i++) {
      edges.push({ source: arr[i - 1].id, target: arr[i].id, type: 'category-chronology', weight: 1 });
    }
  }

  const byTopic = new Map();
  for (const p of papers) {
    for (const t of p.topics) {
      if (!byTopic.has(t)) byTopic.set(t, []);
      byTopic.get(t).push(p);
    }
  }
  for (const arr of byTopic.values()) {
    arr.sort((a, b) => a.year !== b.year ? a.year - b.year : (a.month ?? 0) - (b.month ?? 0));
    for (let i = 1; i < arr.length; i++) {
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

  const categories = {};
  for (const [slug, count] of categoryCounts.entries()) {
    categories[slug] = {
      slug,
      ...(CATEGORY_META[slug] ?? { title: slug.replace(/-/g, ' '), color: '#94979E', blurb: '' }),
      count,
    };
  }

  const yearBuckets = {};
  for (const p of papers) {
    const y = String(p.year);
    if (!yearBuckets[y]) yearBuckets[y] = [];
    yearBuckets[y].push(p.id);
  }

  const wirePapers = papers.map(({ tokens: _, ...rest }) => rest);

  const out = {
    generatedAt: new Date().toISOString(),
    counts: { papers: papers.length, categories: Object.keys(categories).length, edges: edges.length },
    categories,
    domainBridges: DOMAIN_BRIDGES.map(([a, b, label]) => ({ a, b, label })),
    topics: TOPICS.map(({ id, label }) => ({ id, label })),
    papers: wirePapers,
    edges,
    yearBuckets,
  };

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(out));
  console.log(`Wrote ${OUT}`);
  console.log(`  papers: ${papers.length}, categories: ${Object.keys(categories).length}, edges: ${edges.length}`);
  const years = Object.keys(yearBuckets).sort();
  console.log(`  year span: ${years[0]} → ${years[years.length - 1]}`);

  // Emit src/lib/paper-asset-map.ts. Static require() entries so Metro
  // statically resolves them and bundles each .md as an asset. The runtime
  // hook (useMarkdownContent) reads the require map and fetches via expo-asset.
  assetMap.sort((a, b) => a.key.localeCompare(b.key));
  const mapLines = assetMap
    .map(({ key, rel }) => `  ${JSON.stringify(key)}: require('../../assets/papers/${rel}'),`)
    .join('\n');
  const mapSource = [
    '// AUTO-GENERATED by scripts/build-graph.mjs — DO NOT EDIT.',
    '// Maps "<category>/<slug>" to the Metro-bundled require() handle for the .md asset.',
    '// Runtime: useMarkdownContent.ts calls Asset.fromModule(PAPER_ASSETS[key]).',
    '',
    'export const PAPER_ASSETS: Record<string, number> = {',
    mapLines,
    '};',
    '',
  ].join('\n');
  fs.writeFileSync(ASSET_MAP_OUT, mapSource);
  console.log(`Wrote ${ASSET_MAP_OUT}  (${assetMap.length} paper assets)`);
}

main();
