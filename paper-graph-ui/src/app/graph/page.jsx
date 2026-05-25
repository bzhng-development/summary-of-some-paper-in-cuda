import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import GradientLabel from 'components/shared/gradient-label/gradient-label';
import Heading from 'components/shared/heading/heading';

import { GRAPH, listCategories } from 'lib/papers';

import RoadmapCanvas from './_components/roadmap-canvas';

export const metadata = { title: 'Roadmap — Paper Graph' };

// All page data is computed server-side and passed as plain serializable
// props to the client roadmap. The client never imports lib/papers (which
// is server-only).
const GraphPage = () => {
  const categories = listCategories();

  // Compact each paper down to only the fields the canvas needs. This
  // keeps the wire payload small (the full graph.json is ~1 MB; this is
  // closer to ~120 KB).
  const papers = GRAPH.papers.map((p) => ({
    id: p.id,
    category: p.category,
    slug: p.slug,
    title: p.title,
    year: p.year,
    month: p.month,
    score: p.score ?? null,
    organization: p.organization ?? null,
    readTimeMin: p.readTimeMin ?? null,
    arxivId: p.arxivId ?? null,
  }));

  // Only the LLM-similar + category-chronology edges form the lineage
  // backbone. Topic + token-similarity edges are too noisy at paper-level.
  const edges = GRAPH.edges
    .filter((e) => e.type === 'llm-similar' || e.type === 'category-chronology')
    .map((e) => ({ source: e.source, target: e.target, type: e.type }));

  const cats = categories.map((c) => ({
    slug: c.slug,
    title: c.title,
    color: c.color,
    count: c.count,
  }));

  return (
    <main>
      <Container size="1280" className="px-4 pt-6 pb-16 sm:px-3">
        <header className="relative mb-6 overflow-hidden rounded-2xl p-6 sm:p-4">
          <GradientBorder />
          <div className="relative flex flex-col gap-3">
            <GradientLabel theme="green" className="self-start">
              {papers.length} papers · {cats.length} domains
            </GradientLabel>
            <Heading tag="h1" size="md-new" theme="white" className="tracking-tight sm:!text-3xl">
              Lineage Roadmap
            </Heading>
            <p className="t-sm max-w-2xl text-gray-new-70">
              Every paper plotted by year (←→) and domain (↑↓). Curves connect
              papers in the same lineage — direct LLM-judged similarity in
              green, chronological in-domain succession in grey. Tap a dot to
              open that paper.
            </p>
          </div>
        </header>

        <RoadmapCanvas papers={papers} edges={edges} categories={cats} />
      </Container>
    </main>
  );
};

export default GraphPage;
