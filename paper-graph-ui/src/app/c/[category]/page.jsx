import Link from 'next/link';
import { notFound } from 'next/navigation';

import Breadcrumbs from 'components/pages/doc/breadcrumbs/breadcrumbs';
import Tag from 'components/pages/doc/tag/tag';
import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import GradientLabel from 'components/shared/gradient-label/gradient-label';
import Heading from 'components/shared/heading/heading';

import { GRAPH, getCategory, listPapersInCategory } from 'lib/papers';

import CompletionRing from '../../_components/completion-ring';
import { PaperListClient } from '../../_components/paper-list-client';
import ProgressStrip from '../../_components/progress-strip';

export async function generateStaticParams() {
  return Object.keys(GRAPH.categories).map((slug) => ({ category: slug }));
}

export async function generateMetadata({ params }) {
  const { category } = await params;
  const cat = getCategory(category);
  return { title: `${cat?.title ?? category} — Paper Graph` };
}

const CategoryPage = async ({ params }) => {
  const { category } = await params;
  const cat = getCategory(category);
  if (!cat) notFound();

  const papers = listPapersInCategory(category);
  const summaryPapers = papers.filter((p) => p.hasSummary !== false);

  // Related domains via bridges
  const related = GRAPH.domainBridges
    .filter((b) => b.a === category || b.b === category)
    .map((b) => ({
      slug: b.a === category ? b.b : b.a,
      label: b.label,
    }));

  return (
    <main>
      <Container size="1280" className="px-4 pt-6 pb-16 sm:px-3 sm:pt-4">
        <Breadcrumbs
          className="mb-4"
          baseUrl="/"
          breadcrumbs={[
            { title: 'Domains', slug: '/' },
            { title: cat.title, slug: null },
          ]}
        />
        <header className="relative mb-6 overflow-hidden rounded-2xl p-6 sm:p-4">
          <GradientBorder />
          <div className="relative flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: cat.color }}
                aria-hidden
              />
              <GradientLabel theme="green" className="">
                {cat.count} papers
              </GradientLabel>
            </div>
            <div className="flex flex-wrap items-baseline gap-4">
              <Heading tag="h1" size="md-new" theme="white" className="tracking-tight sm:!text-3xl">
                {cat.title}
              </Heading>
              <CompletionRing
                paperIds={summaryPapers.map((p) => p.id)}
                size={32}
                label={`${cat.title} progress`}
              />
            </div>
            <p className="t-sm max-w-2xl text-gray-new-70">{cat.blurb}</p>
            <ProgressStrip paperIds={summaryPapers.map((p) => p.id)} />
            {related.length > 0 ? (
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <span className="t-sm font-mono tracking-[0.18em] text-gray-new-50 uppercase">
                  Related
                </span>
                {related.map((r) => (
                  <Link key={r.slug + r.label} href={`/c/${r.slug}`} className="block">
                    <Tag label={GRAPH.categories[r.slug]?.title ?? r.slug} size="sm" />
                  </Link>
                ))}
              </div>
            ) : null}
          </div>
        </header>

        <PaperListClient scope={{ type: 'category', slug: category }} />
      </Container>
    </main>
  );
};

export default CategoryPage;
