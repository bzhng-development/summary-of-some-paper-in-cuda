'use client';

import Link from 'next/link';

import { useQueue } from 'hooks/use-queue';

// Floating bottom-right pill that surfaces queue depth and links to /queue.
// Renders nothing until hydrated to avoid SSR/CSR count mismatch.
const QueuePill = () => {
  const { list, hydrated } = useQueue();

  if (!hydrated || list.length === 0) return null;

  const total = list.reduce((acc, p) => acc + (p.readTimeMin ?? 0), 0);

  return (
    <Link
      href="/queue"
      className="fixed right-4 bottom-[calc(1rem+env(safe-area-inset-bottom))] z-30 inline-flex items-center gap-2 rounded-full border border-primary-1/50 bg-black-new/85 px-4 py-2 font-mono text-xs tracking-wider text-primary-1 uppercase shadow-lg shadow-black/40 backdrop-blur transition-all hover:bg-black-new/95"
      aria-label={`Reading queue: ${list.length} papers, about ${total} minutes`}
    >
      <span aria-hidden>↑</span>
      <span>Up Next · {list.length}</span>
      {total > 0 ? (
        <span className="text-[10px] text-primary-1/70">~{total}min</span>
      ) : null}
    </Link>
  );
};

export default QueuePill;
