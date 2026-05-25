'use client';

import { useQueue } from 'hooks/use-queue';

// Add-to-queue / remove-from-queue toggle button. Used on paper pages and
// in the recent-reads / digest lists.
const QueueButton = ({ paper, size = 'md' }) => {
  const { ids, hydrated, enqueue, remove } = useQueue();
  const queued = ids.has(paper.id);

  const sizeCls = size === 'sm' ? 'h-7 px-2 text-xs' : 'h-9 px-3 text-sm';

  return (
    <button
      type="button"
      onClick={(ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        if (queued) remove(paper.id);
        else enqueue(paper);
      }}
      aria-pressed={queued}
      className={`${sizeCls} inline-flex items-center gap-1.5 rounded-md border font-mono tracking-wide uppercase transition-colors ${
        queued
          ? 'border-orange-300/60 bg-orange-300/15 text-orange-200 hover:bg-orange-300/25'
          : 'border-gray-new-20 bg-gray-new-10 text-gray-new-70 hover:border-gray-new-30 hover:text-white'
      } ${hydrated ? '' : 'opacity-0'}`}
    >
      <span aria-hidden className="text-base leading-none">
        {queued ? '✓' : '+'}
      </span>
      <span>{queued ? 'Queued' : 'Queue'}</span>
    </button>
  );
};

export default QueueButton;
