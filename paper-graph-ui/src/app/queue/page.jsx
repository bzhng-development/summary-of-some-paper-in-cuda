import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import GradientLabel from 'components/shared/gradient-label/gradient-label';
import Heading from 'components/shared/heading/heading';

import QueueList from './_components/queue-list';

export const metadata = { title: 'Up Next — Paper Graph' };

// Static shell rendered server-side; the actual queue contents come from
// localStorage, so the QueueList client component handles hydration and
// renders the list once mounted.
const QueuePage = () => (
  <main>
    <Container size="1280" className="px-4 pt-6 pb-16 sm:px-3">
      <header className="relative mb-6 overflow-hidden rounded-2xl p-6 sm:p-4">
        <GradientBorder />
        <div className="relative flex flex-col gap-3">
          <GradientLabel theme="green" className="self-start">
            Up Next
          </GradientLabel>
          <Heading
            tag="h1"
            size="md-new"
            theme="white"
            className="tracking-tight sm:!text-3xl"
          >
            Reading queue
          </Heading>
          <p className="t-sm max-w-2xl text-gray-new-70">
            Papers you've queued from anywhere on the site. Open the first to
            get started — auto-advance will keep you moving through the list.
          </p>
        </div>
      </header>

      <QueueList />
    </Container>
  </main>
);

export default QueuePage;
