import Link from 'next/link';

import Container from 'components/shared/container/container';
import Heading from 'components/shared/heading/heading';

import { yearTimeline, listCategories } from 'lib/papers';

import ReadDot from '../_components/read-dot';

export const metadata = {
  title: 'Timeline — Paper Graph',
};

const MONTH_NAMES = [
  '', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const TimelinePage = () => {
  const rows = yearTimeline();
  const catBySlug = new Map(listCategories().map((c) => [c.slug, c]));

  return (
    <main>
      <Container size="1280" className="px-4 pt-10 pb-16 sm:px-3 sm:pt-6">
        <header className="mb-8 flex flex-col gap-3">
          <Heading tag="h1" size="md-new" theme="white" className="tracking-tight sm:!text-3xl">
            Timeline
          </Heading>
          <p className="t-sm max-w-2xl text-gray-new-70">
            Every paper grouped by publication year. Tap a row to open the
            summary; the bullet dot lights up once you mark it read.
          </p>
        </header>

        {/* Sticky year nav */}
        <nav className="-mx-3 mb-6 overflow-x-auto px-3 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
          <ul className="flex gap-2">
            {rows.map((row) => (
              <li key={row.year}>
                <a
                  href={`#${row.year}`}
                  className="block rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wide text-gray-new-70 transition-colors hover:border-primary-1/40 hover:text-white"
                >
                  {row.year}
                  <span className="ml-2 text-gray-new-50">{row.papers.length}</span>
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex flex-col gap-12 sm:gap-10">
          {rows.map((row) => (
            <YearBlock key={row.year} row={row} catBySlug={catBySlug} />
          ))}
        </div>
      </Container>
    </main>
  );
};

const YearBlock = ({ row, catBySlug }) => (
  <section id={String(row.year)} className="scroll-mt-20">
    <header className="mb-4 flex items-baseline justify-between gap-3 border-b border-gray-new-15 pb-2">
      <h2 className="font-sans text-3xl font-medium tracking-tight text-white sm:text-2xl">
        {row.year}
      </h2>
      <span className="t-sm font-mono tracking-[0.2em] text-gray-new-50 uppercase">
        {row.papers.length} papers
      </span>
    </header>
    <ul className="flex flex-col gap-1.5">
      {row.papers.map((p) => {
        const cat = catBySlug.get(p.category);
        const color = cat?.color ?? '#94979E';
        return (
          <li key={p.id} data-company-only={p.companyOnly ? 'true' : 'false'}>
            <Link
              href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
              className="group flex items-baseline gap-3 rounded-lg border border-transparent px-2 py-1.5 transition-colors hover:border-gray-new-15 hover:bg-gray-new-10"
            >
              <span className="t-sm w-12 shrink-0 font-mono text-gray-new-50 tabular-nums">
                {p.month ? MONTH_NAMES[p.month] : '—'}
              </span>
              <span
                className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="min-w-0 flex-1">
                <span className="t-sm flex items-center gap-2 leading-tight text-white group-hover:text-primary-1">
                  <span className="truncate">{p.title}</span>
                  <ReadDot paperId={p.id} />
                </span>
                <span className="t-sm mt-0.5 flex items-center gap-x-2 truncate font-mono text-xs text-gray-new-50">
                  <span className="truncate">{cat?.title ?? p.category}</span>
                  {p.score != null ? (
                    <span className="shrink-0">· s{p.score}</span>
                  ) : null}
                  {p.organization ? (
                    <span className="shrink-0 truncate">· {p.organization}</span>
                  ) : null}
                </span>
              </span>
            </Link>
          </li>
        );
      })}
    </ul>
  </section>
);

export default TimelinePage;
