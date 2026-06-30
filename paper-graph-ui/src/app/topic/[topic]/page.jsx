import { notFound } from 'next/navigation';

import Breadcrumbs from 'components/pages/doc/breadcrumbs/breadcrumbs';
import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import Heading from 'components/shared/heading/heading';

import { GRAPH } from 'lib/papers';

import { PaperListClient } from '../../_components/paper-list-client';
import ProgressStrip from '../../_components/progress-strip';

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
    .sort((a, b) => (a.year !== b.year ? b.year - a.year : (b.month ?? 0) - (a.month ?? 0)));
  const summaryPapers = papers.filter((p) => p.hasSummary !== false);

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
            <Heading tag="h1" size="md-new" theme="white" className="tracking-tight sm:!text-3xl">
              {meta.label}
            </Heading>
            <p className="t-sm text-gray-new-70">
              {papers.length} papers in this thread, across{' '}
              {new Set(papers.map((p) => p.category)).size} domains.
            </p>
            <ProgressStrip paperIds={summaryPapers.map((p) => p.id)} />
          </div>
        </header>

        <PaperListClient scope={{ type: 'topic', id: topic }} />
      </Container>
    </main>
  );
};

export default TopicPage;
