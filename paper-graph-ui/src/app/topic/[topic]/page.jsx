import Link from 'next/link';
import { notFound } from 'next/navigation';

import Breadcrumbs from 'components/pages/doc/breadcrumbs/breadcrumbs';
import Tag from 'components/pages/doc/tag/tag';
import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import Heading from 'components/shared/heading/heading';

import { GRAPH, listCategories } from 'lib/papers';

import ProgressStrip from '../../_components/progress-strip';
import ReadDot from '../../_components/read-dot';

export async function generateStaticParams() {
  return GRAPH.topics.map((t) => ({ topic: t.id }));
}

export async function generateMetadata({ params }) {
  const { topic } = await params;
  const t = GRAPH.topics.find((x) => x.id === topic);
  return { title: `${t?.label ?? topic} — Topic Thread` };
}

const TopicPage = async ({ params }) => {
  const { topic } = await params;
  const meta = GRAPH.topics.find((x) => x.id === topic);
  if (!meta) notFound();
  const papers = GRAPH.papers
    .filter((p) => p.topics.includes(topic))
    .sort((a, b) =>
      a.year !== b.year ? b.year - a.year : (b.month ?? 0) - (a.month ?? 0)
    );
  const cats = new Map(listCategories().map((c) => [c.slug, c]));

  return (
    <main>
      <Container size="1280" className="px-4 pt-6 pb-16 sm:px-3">
        <Breadcrumbs
          className="mb-4"
          baseUrl="/"
          breadcrumbs={[
            { title: 'Topics', slug: '/' },
            { title: meta.label, slug: null },
          ]}
        />
        <header className="relative mb-6 overflow-hidden rounded-2xl p-6 sm:p-4">
          <GradientBorder />
          <div className="relative flex flex-col gap-3">
            <Heading
              tag="h1"
              size="md-new"
              theme="white"
              className="tracking-tight sm:!text-3xl"
            >
              {meta.label}
            </Heading>
            <p className="t-sm text-gray-new-70">
              {papers.length} papers in this thread, across {new Set(papers.map((p) => p.category)).size} domains.
            </p>
            <ProgressStrip paperIds={papers.map((p) => p.id)} />
          </div>
        </header>

        <ul className="flex flex-col gap-1.5">
          {papers.map((p) => {
            const cat = cats.get(p.category);
            return (
              <li key={p.id}>
                <Link
                  href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
                  className="group flex items-baseline gap-3 rounded-lg border border-transparent px-2 py-1.5 transition-colors hover:border-gray-new-15 hover:bg-gray-new-10"
                >
                  <span className="t-sm w-12 shrink-0 font-mono text-gray-new-50 tabular-nums">
                    {p.year}
                  </span>
                  <span
                    className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                    style={{ backgroundColor: cat?.color ?? '#94979E' }}
                    aria-hidden
                  />
                  <span className="min-w-0 flex-1">
                    <span className="t-sm flex items-center gap-2 text-white group-hover:text-primary-1">
                      <span className="truncate">{p.title}</span>
                      <ReadDot paperId={p.id} />
                    </span>
                    <span className="t-sm mt-0.5 block truncate font-mono text-xs text-gray-new-50">
                      {cat?.title ?? p.category}
                    </span>
                  </span>
                  <Tag label={cat?.title ?? p.category} size="sm" />
                </Link>
              </li>
            );
          })}
        </ul>
      </Container>
    </main>
  );
};

export default TopicPage;
