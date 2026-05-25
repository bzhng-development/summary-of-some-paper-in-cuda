import Link from 'next/link';

import Callout from 'components/pages/doc/callout/callout';
import DetailIconCards from 'components/pages/doc/detail-icon-cards/detail-icon-cards';
import Tag from 'components/pages/doc/tag/tag';
import Container from 'components/shared/container/container';
import GradientBorder from 'components/shared/gradient-border/gradient-border';
import GradientCard from 'components/shared/gradient-card/gradient-card';
import GradientLabel from 'components/shared/gradient-label/gradient-label';
import Heading from 'components/shared/heading/heading';
import HintText from 'components/shared/hint-text/hint-text';
import MegaLink from 'components/shared/mega-link/mega-link';
import Tooltip from 'components/shared/tooltip/tooltip';

import { GRAPH, listCategories, listPapers, yearTimeline } from 'lib/papers';

import DomainProgress from './_components/domain-progress';
import RecentReads from './_components/recent-reads';
import SurpriseMe from './_components/surprise-me';
import WhatsNext from './_components/whats-next';

export const metadata = {
  title: 'Paper Graph — LLMs, Systems & RL',
};

// Server component. All client components below receive plain JSON-safe
// props derived here once per request (cached by React.cache).
const Home = () => {
  const categories = listCategories();
  const timeline = yearTimeline();
  const yearsCovered = timeline.length;
  const newestPaper = timeline[0]?.papers[0];
  const papers = listPapers();

  // Compact paper shape for the client islands (SurpriseMe, WhatsNext).
  const clientPapers = papers.map((p) => ({
    id: p.id,
    category: p.category,
    slug: p.slug,
    title: p.title,
    year: p.year,
    score: p.score ?? null,
    readTimeMin: p.readTimeMin ?? null,
    topics: p.topics ?? [],
  }));

  // Pre-group paper IDs by domain so the client CompletionRing only needs
  // the read set, not the full papers list.
  const domainGroups = categories
    .map((c) => ({
      slug: c.slug,
      title: c.title,
      color: c.color,
      paperIds: papers.filter((p) => p.category === c.slug).map((p) => p.id),
    }))
    .sort((a, b) => b.paperIds.length - a.paperIds.length);

  const tiles = [
    { id: 'tile-papers', value: GRAPH.counts.papers, label: 'papers' },
    { id: 'tile-cats', value: GRAPH.counts.categories, label: 'domains' },
    { id: 'tile-years', value: yearsCovered, label: 'years' },
  ];

  return (
    <main>
      <Container size="1280" className="px-4 pt-10 pb-16 sm:px-3 sm:pt-6">
        <header className="relative overflow-hidden rounded-2xl p-8 sm:p-5 xs:p-4">
          <GradientBorder />
          <div className="relative grid grid-cols-[1fr_auto] items-end gap-10 lt:grid-cols-1 lt:items-start lt:gap-6">
            <div className="flex flex-col gap-5 sm:gap-3">
              <GradientLabel theme="green" className="self-start">
                {GRAPH.counts.papers} summaries
              </GradientLabel>
              <Heading
                tag="h1"
                size="md-new"
                theme="white"
                className="tracking-tighter sm:!text-3xl"
              >
                Paper Graph
              </Heading>
              <HintText
                className="t-base block max-w-xl leading-snug text-gray-new-70 sm:text-sm"
                text="A *reading graph* of paper summaries across LLMs, RL training, inference systems and architectures — built for mobile commutes."
                tooltip="Tap any card to drill in. Use the timeline or graph views to navigate by year or topic."
                tooltipId="hero-tagline"
                tooltipPlace="top"
              />
            </div>
            <div className="grid grid-cols-3 gap-3 lt:w-full sm:gap-2">
              {tiles.map((t) => (
                <StatTile key={t.id} {...t} />
              ))}
            </div>
          </div>
          <Tooltip anchorSelect=".hero-stat-anchor" place="top" />
        </header>

        <div className="my-6 flex flex-wrap items-center gap-3 sm:my-4">
          <SurpriseMe papers={clientPapers} />
          <Link
            href="/timeline"
            className="rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-primary-1/50 hover:text-primary-1"
          >
            Timeline →
          </Link>
          <Link
            href="/graph"
            className="rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-primary-1/50 hover:text-primary-1"
          >
            Roadmap →
          </Link>
        </div>

        {newestPaper ? (
          <MegaLink
            className="my-6 sm:my-4"
            tag="MOST RECENT"
            title={newestPaper.title}
            url={`/p/${encodeURIComponent(newestPaper.category)}/${encodeURIComponent(newestPaper.slug)}`}
          />
        ) : null}

        <Callout title="How to use this">
          The <strong>Timeline</strong> is the fastest way to scan what landed
          recently. The <strong>Roadmap</strong> shows lineage between papers;
          tap any dot to open its summary. Everything is mobile-first — swipe
          left/right on a paper page to walk in-domain.
        </Callout>

        <WhatsNext papers={clientPapers} />

        <RecentReads />

        <section className="mt-12 sm:mt-10">
          <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
            Browse by domain
          </h2>
          <p className="t-sm mb-4 max-w-2xl text-gray-new-70">
            Each domain is a sub-collection of papers, sorted newest first. Tap
            a card to open the timeline for that domain.
          </p>
          <DetailIconCards compact>
            {categories.map((c) => (
              <a
                key={c.slug}
                href={`/c/${c.slug}`}
                description={`${c.count} paper${c.count === 1 ? '' : 's'} — ${c.blurb}`}
                icon={c.icon}
              >
                {c.title}
              </a>
            ))}
          </DetailIconCards>
        </section>

        <DomainProgress groups={domainGroups} />

        <section className="mt-14 sm:mt-10">
          <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
            Years covered
          </h2>
          <GradientCard className="p-5 sm:p-4">
            <div className="flex flex-wrap gap-2">
              {timeline.map((row) => (
                <Link
                  key={row.year}
                  href={`/timeline#${row.year}`}
                  className="group flex flex-col gap-0.5 rounded-lg border border-gray-new-20 bg-gray-new-10 px-3 py-2 transition-colors hover:border-primary-1/50 hover:bg-gray-new-15"
                >
                  <span className="font-mono text-xs tracking-wide text-gray-new-60 group-hover:text-primary-1">
                    {row.year}
                  </span>
                  <span className="text-lg font-medium text-white">
                    {row.papers.length}
                  </span>
                </Link>
              ))}
            </div>
          </GradientCard>
        </section>

        <section className="mt-14 sm:mt-10">
          <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
            Topic threads
          </h2>
          <p className="t-sm mb-4 max-w-2xl text-gray-new-70">
            Cross-domain threads of papers that share a topic — RLHF, MoE, KV
            caches, agents, and more.
          </p>
          <div className="flex flex-wrap gap-2">
            {GRAPH.topics.map((t) => (
              <Link key={t.id} href={`/topic/${t.id}`} className="block">
                <Tag label={t.label} size="md" />
              </Link>
            ))}
          </div>
        </section>
      </Container>
    </main>
  );
};

const StatTile = ({ value, label }) => (
  <div className="hero-stat-anchor relative flex min-w-0 flex-col gap-1 rounded-lg border border-gray-new-20 bg-gray-new-10 px-4 py-3 text-right transition-colors hover:border-gray-new-30 sm:px-3 sm:py-2.5 xs:px-2.5 xs:py-2">
    <span className="font-sans text-3xl leading-none font-medium tracking-tighter text-white sm:text-2xl xs:text-xl">
      {value}
    </span>
    <span className="t-sm truncate font-mono tracking-[0.18em] text-gray-new-50 uppercase xs:text-[10px] xs:tracking-[0.12em]">
      {label}
    </span>
  </div>
);

export default Home;
