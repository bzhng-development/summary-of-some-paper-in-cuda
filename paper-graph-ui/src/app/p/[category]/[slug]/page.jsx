import { notFound } from 'next/navigation';

import MarkRead from 'app/_components/mark-read';
import QueueButton from 'app/_components/queue-button';
import RecordVisit from 'app/_components/record-visit';
import ScrollTracker from 'app/_components/scroll-tracker';
import Breadcrumbs from 'components/pages/doc/breadcrumbs/breadcrumbs';
import Tag from 'components/pages/doc/tag/tag';
import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import GradientCard from 'components/shared/gradient-card/gradient-card';
import GradientLabel from 'components/shared/gradient-label/gradient-label';
import Heading from 'components/shared/heading/heading';
import TableOfContents from 'components/shared/table-of-contents/table-of-contents';

import { extractToc } from 'lib/headings';
import {
  GRAPH,
  buildDependencyTree,
  buildInfluenceTree,
  getCategory,
  getPaper,
  listPapersInCategory,
  readPaperBody,
} from 'lib/papers';

import AutoAdvance from './_components/auto-advance';
import DependencyTree from './_components/dependency-tree';
import PaperBody from './_components/paper-body';
import PaperInteractions from './_components/paper-interactions';
import RelatedPapers from './_components/related-papers';
import SwipeNav from './_components/swipe-nav';

// Pre-render every paper at build time. The graph is static; nothing here
// hits the network at request time.
export async function generateStaticParams() {
  return GRAPH.papers.map((p) => ({ category: p.category, slug: p.slug }));
}

export async function generateMetadata({ params }) {
  const { category, slug } = await params;
  const decodedSlug = decodeURIComponent(slug);
  const decodedCategory = decodeURIComponent(category);
  const paper = getPaper(decodedCategory, decodedSlug);
  return { title: paper ? `${paper.title} — Paper Graph` : 'Paper not found' };
}

// Strip the paper object down to JSON-serializable fields before handing
// off to client islands. Server-only fields like `tokens` never cross the
// boundary.
function toClientShape(paper) {
  return {
    id: paper.id,
    category: paper.category,
    slug: paper.slug,
    title: paper.title,
    year: paper.year,
    readTimeMin: paper.readTimeMin ?? null,
  };
}

const PaperPage = async ({ params }) => {
  const { category, slug } = await params;
  const decodedSlug = decodeURIComponent(slug);
  const decodedCategory = decodeURIComponent(category);
  const paper = getPaper(decodedCategory, decodedSlug);
  if (!paper) notFound();
  const body = readPaperBody(decodedCategory, decodedSlug);
  if (body == null) notFound();
  const cat = getCategory(decodedCategory);
  const toc = extractToc(body);
  const depTree = buildDependencyTree(paper.id, 3);
  const influence = buildInfluenceTree(paper.id, 2);

  const siblings = listPapersInCategory(decodedCategory);
  const idx = siblings.findIndex((p) => p.id === paper.id);
  const prev = idx > 0 ? siblings[idx - 1] : null;
  const next = idx < siblings.length - 1 ? siblings[idx + 1] : null;

  const prevHref = prev
    ? `/p/${encodeURIComponent(prev.category)}/${encodeURIComponent(prev.slug)}`
    : null;
  const nextHref = next
    ? `/p/${encodeURIComponent(next.category)}/${encodeURIComponent(next.slug)}`
    : null;

  const clientPaper = toClientShape(paper);

  return (
    <main>
      <RecordVisit paper={clientPaper} />
      <ScrollTracker paper={clientPaper} />
      <SwipeNav prevHref={prevHref} nextHref={nextHref} />
      <Container size="1280" className="px-4 pt-6 pb-16 sm:px-3 sm:pt-4">
        <Breadcrumbs
          className="mb-4"
          baseUrl="/"
          breadcrumbs={[
            { title: 'Domains', slug: '/' },
            { title: cat?.title ?? decodedCategory, slug: `/c/${decodedCategory}` },
            { title: paper.title, slug: null },
          ]}
        />

        <div className="flex gap-10 lg:gap-6 lt:flex-col">
          <article className="min-w-0 flex-1">
            <header className="relative mb-6 overflow-hidden rounded-2xl p-6 sm:p-4">
              <GradientBorder />
              <div className="relative flex flex-wrap items-start justify-between gap-4">
                <div className="flex min-w-0 flex-col gap-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <span
                      className="inline-block h-2 w-2 rounded-full"
                      style={{ backgroundColor: cat?.color ?? '#94979E' }}
                      aria-hidden
                    />
                    <a href={`/c/${paper.category}`} className="block">
                      <Tag label={cat?.title ?? paper.category} size="sm" />
                    </a>
                    <GradientLabel theme="gray">{paper.year}</GradientLabel>
                    {paper.readTimeMin ? (
                      <span className="t-sm inline-flex items-center gap-1 font-mono text-xs text-gray-new-60">
                        <span aria-hidden>⏱</span>
                        ~{paper.readTimeMin} min
                      </span>
                    ) : null}
                    {paper.arxivId ? (
                      <a
                        href={`https://arxiv.org/abs/${paper.arxivId}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="t-sm font-mono text-gray-new-60 transition-colors hover:text-primary-1"
                      >
                        arXiv:{paper.arxivId}
                      </a>
                    ) : null}
                  </div>
                  <Heading
                    tag="h1"
                    size="sm"
                    theme="white"
                    className="!font-sans tracking-tight"
                  >
                    {paper.title}
                  </Heading>
                  {paper.topics.length > 0 ? (
                    <div className="flex flex-wrap gap-1.5">
                      {paper.topics.map((t) => {
                        const meta = GRAPH.topics.find((x) => x.id === t);
                        return (
                          <a key={t} href={`/topic/${t}`} className="block">
                            <Tag label={meta?.label ?? t} size="sm" />
                          </a>
                        );
                      })}
                    </div>
                  ) : null}
                  <PaperMetaLine paper={paper} />
                </div>
                <div className="flex flex-col items-end gap-2">
                  <MarkRead paperId={paper.id} />
                  <QueueButton paper={clientPaper} />
                </div>
              </div>
            </header>

            <div className="relative overflow-hidden rounded-2xl p-6 sm:p-4">
              <GradientBorder />
              <div className="prose-doc post-content prose relative max-w-none dark:prose-invert xs:prose-code:break-words">
                <PaperInteractions paperId={paper.id}>
                  <PaperBody markdown={body} />
                </PaperInteractions>
              </div>
            </div>

            <div className="mt-10 flex flex-col gap-8">
              <DependencyTree
                title="Dependency tree"
                description="Papers this one builds on, walking backwards through the same domain and shared topics."
                tree={depTree}
              />
              {influence && influence.children.length > 0 ? (
                <DependencyTree
                  title="What it influenced"
                  description="Later papers in the same chain. Direction reversed."
                  tree={influence}
                  direction="forward"
                />
              ) : null}
              <RelatedPapers prev={prev} next={next} category={decodedCategory} />
              <AutoAdvance
                currentPaperId={paper.id}
                fallbackNextHref={prevHref /* siblings are newest-first; "next paper" in our timeline = older = prev in array */}
                fallbackNextTitle={prev?.title ?? null}
              />
            </div>
          </article>

          {toc.length > 0 ? (
            <aside className="w-[220px] shrink-0 lt:w-full">
              <div className="sticky top-20 lt:static">
                <GradientCard className="p-4">
                  <TableOfContents items={toc} />
                </GradientCard>
              </div>
            </aside>
          ) : null}
        </div>
      </Container>
    </main>
  );
};

const SCORE_COLOR = (s) => {
  if (s == null) return '#71717A';
  if (s >= 9) return '#00E599';
  if (s >= 7) return '#34D59A';
  if (s >= 5) return '#f0f075';
  return '#71717A';
};

const PaperMetaLine = ({ paper }) => {
  const authorList = Array.isArray(paper.authors) ? paper.authors : null;
  const authorsShort = authorList
    ? authorList.slice(0, 3).join(', ') + (authorList.length > 3 ? ` +${authorList.length - 3}` : '')
    : null;

  const bits = [];
  if (authorsShort) bits.push({ key: 'authors', label: authorsShort });
  if (paper.organization) bits.push({ key: 'org', label: paper.organization });
  if (paper.upvotes) bits.push({ key: 'upvotes', label: `↑ ${paper.upvotes}` });
  if (paper.githubStars)
    bits.push({ key: 'stars', label: `★ ${paper.githubStars.toLocaleString()}` });

  return (
    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
      {paper.score != null ? (
        <span
          className="inline-flex items-center gap-1 rounded-md border px-2 py-0.5 font-mono text-[11px] tracking-wider uppercase"
          style={{ borderColor: `${SCORE_COLOR(paper.score)}55`, color: SCORE_COLOR(paper.score) }}
          title={paper.scoreReason ?? undefined}
        >
          score {paper.score}
        </span>
      ) : null}
      {bits.map((b, i) => (
        <span
          key={b.key}
          className={`t-sm font-mono text-xs text-gray-new-60 ${i > 0 ? 'border-l border-gray-new-20 pl-3' : ''}`}
        >
          {b.label}
        </span>
      ))}
      {paper.github ? (
        <a
          href={`https://github.com/${paper.github}`}
          target="_blank"
          rel="noopener noreferrer"
          className="t-sm font-mono text-xs text-gray-new-60 underline decoration-gray-new-30 underline-offset-2 transition-colors hover:text-primary-1"
        >
          {paper.github}
        </a>
      ) : null}
    </div>
  );
};

export default PaperPage;
