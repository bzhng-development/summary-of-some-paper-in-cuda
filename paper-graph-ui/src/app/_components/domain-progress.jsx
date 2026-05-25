import GradientCard from 'components/shared/gradient-card/gradient-card';

import CompletionRing from './completion-ring';

// Compact grid of domain → completion ring. Server passes the paperIds
// per domain (pre-grouped) so the client island only needs the read set.
const DomainProgress = ({ groups }) => (
  <section className="mt-12 sm:mt-10">
    <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
      Your progress
    </h2>
    <GradientCard className="p-4 sm:p-3">
      <ul className="grid grid-cols-2 gap-3 sm:grid-cols-1">
        {groups.map((g) => (
          <li
            key={g.slug}
            className="flex items-center gap-3 rounded-md border border-gray-new-15 bg-gray-new-10/40 px-3 py-2"
          >
            <span
              className="inline-block h-2 w-2 shrink-0 rounded-full"
              style={{ backgroundColor: g.color }}
              aria-hidden
            />
            <a
              href={`/c/${g.slug}`}
              className="t-sm flex-1 truncate text-white hover:text-primary-1"
            >
              {g.title}
            </a>
            <CompletionRing paperIds={g.paperIds} label={`${g.title} progress`} />
          </li>
        ))}
      </ul>
    </GradientCard>
  </section>
);

export default DomainProgress;
