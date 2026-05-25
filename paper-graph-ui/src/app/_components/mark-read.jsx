'use client';

import { useReadSet } from 'hooks/use-read-set';

const MarkRead = ({ paperId, size = 'md' }) => {
  const { has, toggle, hydrated } = useReadSet();
  const done = has(paperId);

  const sizeCls = size === 'sm' ? 'h-7 px-2 text-xs' : 'h-9 px-3 text-sm';

  return (
    <button
      type="button"
      onClick={() => toggle(paperId)}
      aria-pressed={done}
      className={`${sizeCls} inline-flex items-center gap-1.5 rounded-md border font-mono tracking-wide uppercase transition-colors ${
        done
          ? 'border-primary-1/60 bg-primary-1/15 text-primary-1 hover:bg-primary-1/25'
          : 'border-gray-new-20 bg-gray-new-10 text-gray-new-70 hover:border-gray-new-30 hover:text-white'
      } ${hydrated ? '' : 'opacity-0'}`}
    >
      <span aria-hidden className="text-base leading-none">
        {done ? '✓' : '○'}
      </span>
      <span>{done ? 'Read' : 'Mark read'}</span>
    </button>
  );
};

export default MarkRead;
