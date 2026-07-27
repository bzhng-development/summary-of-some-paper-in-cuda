import Container from 'components/shared/container/container';
import Heading from 'components/shared/heading/heading';

import { PaperListClient } from '../_components/paper-list-client';

export const metadata = {
  title: 'Timeline — Paper Graph',
};

// The year nav lives inside PaperListClient so its counts and clickable years
// track the active filter (merit/company/search) instead of the full corpus,
// and clicking a year expands the list far enough to actually scroll to it.
const TimelinePage = () => (
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

      <PaperListClient scope={{ type: 'all' }} />
    </Container>
  </main>
);

export default TimelinePage;
