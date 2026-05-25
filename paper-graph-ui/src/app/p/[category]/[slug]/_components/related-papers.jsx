import DetailIconCards from 'components/pages/doc/detail-icon-cards/detail-icon-cards';

// In-domain prev/next pair, rendered as DetailIconCards to match the rest
// of the design system.
const RelatedPapers = ({ prev, next, category }) => {
  const cards = [
    prev ? (
      <a
        key="prev"
        href={`/p/${encodeURIComponent(category)}/${encodeURIComponent(prev.slug)}`}
        description={`← Newer in ${category} · ${prev.year}`}
        icon="respond-arrow"
      >
        {prev.title}
      </a>
    ) : null,
    next ? (
      <a
        key="next"
        href={`/p/${encodeURIComponent(category)}/${encodeURIComponent(next.slug)}`}
        description={`Older in ${category} · ${next.year} →`}
        icon="respond-arrow"
      >
        {next.title}
      </a>
    ) : null,
  ].filter(Boolean);

  if (cards.length === 0) return null;

  return (
    <section aria-label="Adjacent papers in domain">
      <h2 className="t-sm mb-3 font-mono tracking-[0.2em] text-gray-new-50 uppercase">
        Adjacent in domain
      </h2>
      <DetailIconCards compact>{cards}</DetailIconCards>
    </section>
  );
};

export default RelatedPapers;
