'use client';

import { useReadSet } from 'hooks/use-read-set';

const ReadDot = ({ paperId, className = '' }) => {
  const { has } = useReadSet();
  if (!has(paperId)) return null;
  return (
    <span
      title="Read"
      aria-label="Read"
      className={`inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-primary-1 ${className}`}
    />
  );
};

export default ReadDot;
