// Server-only data access layer for paper graph + markdown bodies.
// graph.generated.json is built by scripts/build-paper-graph.mjs.
import 'server-only';

import fs from 'node:fs';
import path from 'node:path';
import { cache } from 'react';

import graphData from './graph.generated.json';

const ROOT = process.cwd();
const PAPERS_DIR = path.join(ROOT, 'src/content/papers');

// Block path traversal: every requested path must resolve inside PAPERS_DIR.
function safeJoin(base, ...parts) {
  const resolved = path.resolve(base, ...parts);
  if (!resolved.startsWith(path.resolve(base) + path.sep)) {
    throw new Error('Path traversal attempt');
  }
  return resolved;
}

export const GRAPH = graphData;

export const listPapers = cache(() => GRAPH.papers);

export const hasGeneratedSummary = (paper) => paper?.hasSummary !== false;

export const listCategories = cache(() =>
  Object.values(GRAPH.categories).sort((a, b) => b.count - a.count)
);

export const getCategory = cache((slug) => GRAPH.categories[slug] ?? null);

export const listPapersInCategory = cache((slug) => {
  // Multi-tag aware: a paper appears in category X if X is in its
  // tagCategories array. Falls back to single-cat (filesystem category)
  // for papers without multi-tag data.
  const all = GRAPH.papers.filter((p) => {
    const cats = Array.isArray(p.tagCategories) && p.tagCategories.length > 0
      ? p.tagCategories
      : [p.category];
    return cats.includes(slug);
  });
  all.sort((a, b) =>
    a.year !== b.year ? b.year - a.year : (b.month ?? 0) - (a.month ?? 0)
  );
  return all;
});

export const getPaper = cache((category, slug) => {
  const id = `${category}/${slug}`;
  return GRAPH.papers.find((p) => p.id === id) ?? null;
});

export const readPaperBody = cache((category, slug) => {
  try {
    const full = safeJoin(PAPERS_DIR, category, `${slug}.md`);
    return fs.readFileSync(full, 'utf8');
  } catch {
    return null;
  }
});

// Outgoing + incoming edges for a single paper, useful for the tree view.
export const getPaperEdges = cache((id) => {
  const outgoing = [];
  const incoming = [];
  for (const e of GRAPH.edges) {
    if (e.source === id) outgoing.push(e);
    if (e.target === id) incoming.push(e);
  }
  return { outgoing, incoming };
});

// Walk backwards through the chronological/topic chain to build a dependency
// tree rooted at the given paper. Each node visited at most once. Depth cap
// prevents runaway traversal across topic threads.
export function buildDependencyTree(rootId, maxDepth = 4) {
  const paperById = new Map(
    GRAPH.papers.filter(hasGeneratedSummary).map((p) => [p.id, p])
  );
  const incomingByTarget = new Map();
  for (const e of GRAPH.edges) {
    if (!paperById.has(e.source) || !paperById.has(e.target)) continue;
    if (!incomingByTarget.has(e.target)) incomingByTarget.set(e.target, []);
    incomingByTarget.get(e.target).push(e);
  }

  const visited = new Set();

  function recurse(id, depth) {
    if (visited.has(id) || depth > maxDepth) return null;
    visited.add(id);
    const paper = paperById.get(id);
    if (!paper) return null;
    const incoming = incomingByTarget.get(id) ?? [];
    // Sort by year descending so closest predecessors first, then by edge type
    // priority (category > topic > similarity).
    const priority = {
      'llm-similar': 0,
      'category-chronology': 1,
      topic: 2,
      similarity: 3,
    };
    incoming.sort((a, b) => {
      const pa = priority[a.type] ?? 9;
      const pb = priority[b.type] ?? 9;
      if (pa !== pb) return pa - pb;
      const py = paperById.get(b.source)?.year ?? 0;
      const ay = paperById.get(a.source)?.year ?? 0;
      return py - ay;
    });
    const children = [];
    for (const e of incoming) {
      const child = recurse(e.source, depth + 1);
      if (child) children.push({ edge: e, ...child });
    }
    return { paper, children };
  }

  return recurse(rootId, 0);
}

// Forward-walk: papers that this one influenced (descendants by category /
// topic chain).
export function buildInfluenceTree(rootId, maxDepth = 4) {
  const paperById = new Map(
    GRAPH.papers.filter(hasGeneratedSummary).map((p) => [p.id, p])
  );
  const outgoingBySource = new Map();
  for (const e of GRAPH.edges) {
    if (!paperById.has(e.source) || !paperById.has(e.target)) continue;
    if (!outgoingBySource.has(e.source)) outgoingBySource.set(e.source, []);
    outgoingBySource.get(e.source).push(e);
  }

  const visited = new Set();

  function recurse(id, depth) {
    if (visited.has(id) || depth > maxDepth) return null;
    visited.add(id);
    const paper = paperById.get(id);
    if (!paper) return null;
    const outgoing = outgoingBySource.get(id) ?? [];
    const priority = {
      'llm-similar': 0,
      'category-chronology': 1,
      topic: 2,
      similarity: 3,
    };
    outgoing.sort((a, b) => {
      const pa = priority[a.type] ?? 9;
      const pb = priority[b.type] ?? 9;
      if (pa !== pb) return pa - pb;
      const ay = paperById.get(a.target)?.year ?? 0;
      const by = paperById.get(b.target)?.year ?? 0;
      return ay - by;
    });
    const children = [];
    for (const e of outgoing) {
      const child = recurse(e.target, depth + 1);
      if (child) children.push({ edge: e, ...child });
    }
    return { paper, children };
  }

  return recurse(rootId, 0);
}

// All papers grouped by year, sorted desc, with categories inlined for rendering.
export const yearTimeline = cache(() => {
  const groups = new Map();
  for (const p of GRAPH.papers) {
    if (!groups.has(p.year)) groups.set(p.year, []);
    groups.get(p.year).push(p);
  }
  const arr = [...groups.entries()]
    .sort((a, b) => b[0] - a[0])
    .map(([year, papers]) => ({
      year,
      papers: papers.sort((a, b) => {
        if ((b.month ?? 0) !== (a.month ?? 0)) return (b.month ?? 0) - (a.month ?? 0);
        return a.title.localeCompare(b.title);
      }),
    }));
  return arr;
});

// Aggregate counts per (year, category) for the calendar heatmap on home.
export const yearCategoryMatrix = cache(() => {
  const out = new Map();
  for (const p of GRAPH.papers) {
    const k = `${p.year}|${p.category}`;
    out.set(k, (out.get(k) ?? 0) + 1);
  }
  return out;
});

export const searchPapers = cache((query) => {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return GRAPH.papers.filter(
    (p) =>
      p.title.toLowerCase().includes(q) ||
      p.id.toLowerCase().includes(q) ||
      p.category.toLowerCase().includes(q) ||
      (p.arxivId && p.arxivId.includes(q))
  );
});
