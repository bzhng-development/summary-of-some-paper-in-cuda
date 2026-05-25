import Link from 'next/link';

import Container from 'components/shared/container/container';
import Heading from 'components/shared/heading/heading';

const NotFound = () => (
  <main>
    <Container size="1280" className="px-4 pt-20 pb-16 sm:px-3">
      <div className="mx-auto flex max-w-xl flex-col items-start gap-4">
        <span className="font-mono text-xs tracking-[0.22em] text-gray-new-50 uppercase">
          404
        </span>
        <Heading tag="h1" size="md-new" theme="white" className="tracking-tight">
          Not in the graph
        </Heading>
        <p className="t-base text-gray-new-70">
          The paper or page you tried to open doesn't exist. Head back to the
          home or browse the timeline.
        </p>
        <div className="flex flex-wrap gap-3">
          <Link
            href="/"
            className="rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-primary-1/50 hover:text-primary-1"
          >
            ← Home
          </Link>
          <Link
            href="/timeline"
            className="rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-primary-1/50 hover:text-primary-1"
          >
            Timeline
          </Link>
        </div>
      </div>
    </Container>
  </main>
);

export default NotFound;
