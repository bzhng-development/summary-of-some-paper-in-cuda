import Fuse from 'fuse.js';
import { graph } from './graph';
import type { Paper } from './types';

let fuseInstance: Fuse<Paper> | null = null;

function getFuse(): Fuse<Paper> {
  if (!fuseInstance) {
    fuseInstance = new Fuse(graph.papers, {
      keys: [
        { name: 'title', weight: 0.7 },
        { name: 'category', weight: 0.15 },
        { name: 'organization', weight: 0.1 },
        { name: 'arxivId', weight: 0.05 },
      ],
      threshold: 0.35,
      includeScore: true,
      minMatchCharLength: 2,
    });
  }
  return fuseInstance;
}

export function searchPapers(query: string, limit = 30): Paper[] {
  if (!query.trim()) return [];
  const results = getFuse().search(query, { limit });
  return results.map((r) => r.item);
}
