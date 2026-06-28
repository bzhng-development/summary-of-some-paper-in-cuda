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
import { buildPaperSearchText } from '../../_components/paper-filter-utils';
import PaperListMeta from '../../_components/paper-list-meta';
import ProgressStrip from '../../_components/progress-strip';
import ReadDot from '../../_components/read-dot';

export async function generateStaticParams() {
  return Object.keys(GRAPH.categories).map((slug) => ({ category: slug }));
}

export async function generateMetadata({ params }) {
  const { category } = await params;
  const cat = getCategory(category);
  return { title: `${cat?.title ?? category} — Paper Graph` };
}

const MONTH_NAMES = [
  '',
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

const CategoryPage = async ({ params }) => {
  const { category } = await params;
  const cat = getCategory(category);
  if (!cat) notFound();

  const papers = listPapersInCategory(category);
  const summaryPapers = papers.filter((p) => p.hasSummary !== false);
  const yearBuckets = new Map();
  for (const p of papers) {
    const y = p.year;
    if (!yearBuckets.has(y)) yearBuckets.set(y, []);
    yearBuckets.get(y).push(p);
  }
  const years = [...yearBuckets.keys()].sort((a, b) => b - a);

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

        <div className="flex flex-col gap-10 sm:gap-8">
          {years.map((y) => (
            <section
              key={y}
              data-paper-group="true"
              data-has-summary={
                yearBuckets.get(y).some((p) => p.hasSummary !== false) ? 'true' : 'false'
              }
            >
              <header className="mb-3 flex items-baseline justify-between border-b border-gray-new-15 pb-2">
                <h2 className="font-sans text-2xl font-medium text-white sm:text-xl">{y}</h2>
                <span className="t-sm font-mono tracking-[0.2em] text-gray-new-50 uppercase">
                  {yearBuckets.get(y).length}
                </span>
              </header>
              <ul className="flex flex-col gap-1.5">
                {yearBuckets.get(y).map((p) => (
                  <li
                    key={p.id}
                    data-paper-row="true"
                    data-company-only={p.companyOnly ? 'true' : 'false'}
                    data-has-summary={p.hasSummary === false ? 'false' : 'true'}
                    data-org={p.organization ?? ''}
                    data-search-text={buildPaperSearchText(p)}
                  >
                    <Link
                      href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
                      className="group flex items-baseline gap-3 rounded-lg border border-transparent px-2 py-1.5 transition-colors hover:border-gray-new-15 hover:bg-gray-new-10"
                    >
                      <span className="t-sm w-10 shrink-0 font-mono text-gray-new-50 tabular-nums">
                        {p.month ? MONTH_NAMES[p.month] : '—'}
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="t-sm flex items-center gap-2 leading-tight text-white group-hover:text-primary-1">
                          <span className="truncate">{p.title}</span>
                          <ReadDot paperId={p.id} />
                        </span>
                        <PaperListMeta paper={p} />
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </Container>
    </main>
  );
};

export default CategoryPage;
