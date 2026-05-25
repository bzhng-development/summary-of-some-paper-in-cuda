import Link from 'next/link';

import Container from 'components/shared/container/container';

import StreakCounter from './streak-counter';

const SiteHeader = () => (
  <header className="sticky top-0 z-40 border-b border-gray-new-15 bg-black-pure/85 backdrop-blur">
    <Container size="1280" className="flex h-14 items-center justify-between gap-3 px-4 sm:px-3">
      <Link
        href="/"
        className="flex items-center gap-2 text-[14px] font-medium tracking-tight text-white transition-opacity hover:opacity-80"
      >
        <span className="inline-block h-2 w-2 rounded-full bg-primary-1" aria-hidden />
        <span>Paper Graph</span>
      </Link>
      <nav className="t-sm flex items-center gap-x-4 text-gray-new-70 sm:gap-x-3">
        <StreakCounter />
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

export default SiteHeader;
