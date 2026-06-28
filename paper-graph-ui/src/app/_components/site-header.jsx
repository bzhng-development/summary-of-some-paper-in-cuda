import Link from 'next/link';

import Container from 'components/shared/container/container';

import { listCompanyOptions } from 'lib/papers';

import CompanyFilter from './company-filter';
import MeritOnlyToggle from './merit-only-toggle';
import NoSummaryToggle from './no-summary-toggle';
import PaperSearch from './paper-search';
import StreakCounter from './streak-counter';

const SiteHeader = () => {
  const companyOptions = listCompanyOptions();

  return (
    <header className="sticky top-0 z-40 border-b border-gray-new-15 bg-black-pure/85 backdrop-blur">
      <Container
        size="1280"
        className="flex min-h-14 flex-wrap items-center justify-between gap-3 px-4 py-2 sm:px-3"
      >
        <Link
          href="/"
          className="flex items-center gap-2 text-[14px] font-medium tracking-tight text-white transition-opacity hover:opacity-80"
        >
          <span className="inline-block h-2 w-2 rounded-full bg-primary-1" aria-hidden />
          <span>Paper Graph</span>
        </Link>
        <nav className="t-sm flex flex-wrap items-center justify-end gap-x-3 gap-y-2 text-gray-new-70">
          <StreakCounter />
          <PaperSearch />
          <CompanyFilter options={companyOptions} />
          <MeritOnlyToggle />
          <NoSummaryToggle />
          <Link href="/timeline" className="transition-colors hover:text-white">
            Timeline
          </Link>
          <Link href="/graph" className="transition-colors hover:text-white">
            Graph
          </Link>
        </nav>
      </Container>
    </header>
  );
};

export default SiteHeader;
