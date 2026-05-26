import graphJson from './graph.generated.json';
import type { GraphData, Paper, CategoryMeta } from './types';

// The graph is baked at build time by scripts/build-graph.mjs
export const graph = graphJson as unknown as GraphData;

export function listCategories(): CategoryMeta[] {
  return Object.values(graph.categories).sort((a, b) => b.count - a.count);
}

export function getPapersForCategory(category: string): Paper[] {
  return graph.papers
    .filter((p) => p.category === category)
    .sort((a, b) => b.year - a.year || (b.month ?? 0) - (a.month ?? 0));
}

export function getPaper(category: string, slug: string): Paper | undefined {
  return graph.papers.find((p) => p.category === category && p.slug === slug);
}

export function getPaperById(id: string): Paper | undefined {
  return graph.papers.find((p) => p.id === id);
}

export function getPapersForYear(year: number): Paper[] {
  return graph.papers
    .filter((p) => p.year === year)
    .sort((a, b) => (b.month ?? 0) - (a.month ?? 0));
}

export function getAvailableYears(): number[] {
  return Object.keys(graph.yearBuckets)
    .map(Number)
    .filter((y) => y > 2000)
    .sort((a, b) => b - a);
}

export function getPapersForTopic(topicId: string): Paper[] {
  return graph.papers
    .filter((p) => p.topics.includes(topicId))
    .sort((a, b) => b.year - a.year || (b.month ?? 0) - (a.month ?? 0));
}

export function getAdjacentPapers(paper: Paper): { prev: Paper | null; next: Paper | null } {
  const catPapers = getPapersForCategory(paper.category);
  // catPapers sorted newest-first; for nav we want chronological prev/next
  const idx = catPapers.findIndex((p) => p.id === paper.id);
  return {
    prev: idx < catPapers.length - 1 ? catPapers[idx + 1] : null,
    next: idx > 0 ? catPapers[idx - 1] : null,
  };
}

export function getRelatedPapers(paper: Paper): Paper[] {
  const related = new Set<string>();
  for (const edge of graph.edges) {
    if (edge.source === paper.id) related.add(edge.target);
    else if (edge.target === paper.id) related.add(edge.source);
  }
  return Array.from(related)
    .map((id) => getPaperById(id))
    .filter((p): p is Paper => p != null && p.id !== paper.id)
    .slice(0, 8);
}
