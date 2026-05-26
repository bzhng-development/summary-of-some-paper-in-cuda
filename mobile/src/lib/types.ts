export interface Paper {
  id: string; // "category/slug"
  arxivId: string | null;
  title: string;
  category: string;
  slug: string;
  year: number;
  month: number | null;
  isArxiv: boolean;
  topics: string[];
  pitch: string | null;
  relativePath: string;
  wordCount: number;
  readTimeMin: number;
  score: number | null;
  similarPaper: string | null;
  scoreReason: string | null;
  tagCategoryV2: string | null;
  authors: string[] | null;
  organization: string | null;
  primaryCategory: string | null;
  upvotes: number | null;
  github: string | null;
  githubStars: number | null;
}

export interface CategoryMeta {
  slug: string;
  title: string;
  color: string;
  blurb: string;
  count: number;
}

export interface Edge {
  source: string;
  target: string;
  type: 'llm-similar' | 'category-chronology' | 'topic' | 'similarity';
  weight: number;
  topic?: string;
  via?: string;
}

export interface TopicMeta {
  id: string;
  label: string;
}

export interface DomainBridge {
  a: string;
  b: string;
  label: string;
}

export interface GraphData {
  generatedAt: string;
  counts: { papers: number; categories: number; edges: number };
  categories: Record<string, CategoryMeta>;
  domainBridges: DomainBridge[];
  topics: TopicMeta[];
  papers: Paper[];
  edges: Edge[];
  yearBuckets: Record<string, string[]>;
}
