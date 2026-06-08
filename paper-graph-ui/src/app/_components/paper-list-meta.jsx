const cleanText = (value) => {
  if (value == null) return null;
  const text = String(value).trim();
  return text.length > 0 ? text : null;
};

const toFiniteNumber = (value) => {
  if (value == null || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
};

const formatPublishedDate = (value) => {
  const text = cleanText(value);
  if (!text) return null;
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) return text;
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    timeZone: 'UTC',
  }).format(date);
};

export const NoSummaryMarker = ({ className = '' }) => (
  <span
    className={`shrink-0 rounded border border-gray-new-30 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-gray-new-60 uppercase ${className}`}
  >
    no summary yet
  </span>
);

const PaperListMeta = ({ paper, categoryTitle = null }) => {
  const bits = [];
  const citedByCount = toFiniteNumber(paper.citedByCount);
  const published = formatPublishedDate(paper.published);

  if (categoryTitle) bits.push(categoryTitle);
  else if (paper.category) bits.push(paper.category);
  if (paper.arxivId) bits.push(paper.arxivId);
  if (paper.organization) bits.push(paper.organization);
  if (citedByCount != null) bits.push(`${citedByCount.toLocaleString()} cites`);
  if (published) bits.push(published);
  if (paper.score != null) bits.push(`score ${paper.score}`);
  if (paper.readTimeMin) bits.push(`~${paper.readTimeMin} min`);

  return (
    <span className="t-sm mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-gray-new-50">
      {paper.hasSummary === false ? <NoSummaryMarker /> : null}
      {bits.map((bit) => (
        <span key={bit} className="truncate">
          {bit}
        </span>
      ))}
    </span>
  );
};

export default PaperListMeta;
