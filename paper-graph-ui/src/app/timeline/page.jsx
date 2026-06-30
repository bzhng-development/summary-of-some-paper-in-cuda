import Link from 'next/link';

import Container from 'components/shared/container/container';
import Heading from 'components/shared/heading/heading';

import { yearTimeline } from 'lib/papers';

import { PaperListClient } from '../_components/paper-list-client';

export const metadata = {
  title: 'Timeline — Paper Graph',
};

const TimelinePage = () => {
  const rows = yearTimeline();

  return (
    <main>
      <Container size="1280" className="px-4 pt-10 pb-16 sm:px-3 sm:pt-6">
        <header className="mb-8 flex flex-col gap-3">
          <Heading tag="h1" size="md-new" theme="white" className="tracking-tight sm:!text-3xl">
            Timeline
          </Heading>
          <p className="t-sm max-w-2xl text-gray-new-70">
            Every paper grouped by publication year. Tap a row to open the summary; the bullet dot
            lights up once you mark it read.
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

        <PaperListClient scope={{ type: 'all' }} />
      </Container>
    </main>
  );
};

export default TimelinePage;
