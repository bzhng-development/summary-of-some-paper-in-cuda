'use client';

import { useReadSet } from 'hooks/use-read-set';

// Segmented progress strip: one cell per paper in the given list, colored
// by read status. Used on the topic page so a topic-thread feels like a
// curriculum.
const ProgressStrip = ({ paperIds }) => {
  const { set, hydrated } = useReadSet();
  if (!paperIds || paperIds.length === 0) return null;
  const done = paperIds.reduce((acc, id) => acc + (set.has(id) ? 1 : 0), 0);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
          Progress
        </span>
        <span className="font-mono text-[11px] tabular-nums text-gray-new-60">
          {done} of {paperIds.length}
        </span>
      </div>
      <div
        className={`flex h-2 w-full gap-[2px] overflow-hidden rounded-sm ${
          hydrated ? '' : 'opacity-40'
        }`}
        aria-label={`${done} of ${paperIds.length} papers read`}
      >
        {paperIds.map((id) => (
          <span
            key={id}
            className={`flex-1 ${
              set.has(id) ? 'bg-primary-1/80' : 'bg-gray-new-15'
            }`}
          />
        ))}
      </div>
    </div>
  );
};

export default ProgressStrip;
