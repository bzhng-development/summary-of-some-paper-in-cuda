'use client';

import { useMemo } from 'react';

import { useReadSet } from 'hooks/use-read-set';

// Small Activity-style SVG ring showing read/total for a set of paper IDs.
// Server passes the list of paper IDs in this group (a category, a topic,
// the whole site); the ring computes the read fraction client-side.
const CompletionRing = ({ paperIds, size = 36, label }) => {
  const { set, hydrated } = useReadSet();
  const stats = useMemo(() => {
    if (!paperIds || paperIds.length === 0) return { done: 0, total: 0 };
    const done = paperIds.reduce((acc, id) => acc + (set.has(id) ? 1 : 0), 0);
    return { done, total: paperIds.length };
  }, [paperIds, set]);

  const { done, total } = stats;
  const pct = total === 0 ? 0 : done / total;
  const r = size / 2 - 3;
  const c = size / 2;
  const circumference = 2 * Math.PI * r;
  const dash = `${pct * circumference} ${circumference}`;

  return (
    <span
      className="inline-flex items-center gap-1.5"
      title={label ?? `${done} of ${total} read`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        aria-hidden
        className={hydrated ? '' : 'opacity-40'}
      >
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="3"
        />
        <circle
          cx={c}
          cy={c}
          r={r}
          fill="none"
          stroke="#00E599"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={dash}
          transform={`rotate(-90 ${c} ${c})`}
        />
      </svg>
      <span className="font-mono text-[10px] tabular-nums text-gray-new-60">
        {done}/{total}
      </span>
    </span>
  );
};

export default CompletionRing;
