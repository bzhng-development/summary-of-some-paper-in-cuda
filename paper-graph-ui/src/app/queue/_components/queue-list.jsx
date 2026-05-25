'use client';

import Link from 'next/link';

import { useQueue } from 'hooks/use-queue';

const QueueList = () => {
  const { list, hydrated, remove, clear, peek } = useQueue();

  if (!hydrated) {
    return <p className="t-sm font-mono text-gray-new-50">Loading…</p>;
  }

  if (list.length === 0) {
    return (
      <div className="rounded-lg border border-gray-new-15 bg-gray-new-10 p-6 text-center">
        <p className="t-base mb-3 text-gray-new-70">
          Your queue is empty.
        </p>
        <Link
          href="/timeline"
          className="inline-block rounded-md border border-primary-1/40 bg-primary-1/10 px-3 py-1.5 font-mono text-xs tracking-wider text-primary-1 uppercase transition-colors hover:bg-primary-1/20"
        >
          Browse timeline →
        </Link>
      </div>
    );
  }

  const next = peek();
  const totalMin = list.reduce((acc, p) => acc + (p.readTimeMin ?? 0), 0);

  return (
    <>
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="t-sm font-mono tracking-wider text-gray-new-50 uppercase">
          {list.length} papers · ~{totalMin} min total
        </p>
        <div className="flex gap-2">
          {next ? (
            <Link
              href={`/p/${encodeURIComponent(next.category)}/${encodeURIComponent(next.slug)}`}
              className="inline-flex items-center gap-1.5 rounded-md border border-primary-1/40 bg-primary-1/10 px-3 py-1.5 font-mono text-xs tracking-wider text-primary-1 uppercase transition-colors hover:bg-primary-1/20"
            >
              Start →
            </Link>
          ) : null}
          <button
            type="button"
            onClick={() => {
              if (window.confirm(`Clear ${list.length} queued papers?`)) clear();
            }}
            className="rounded-md border border-gray-new-20 bg-gray-new-10 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-gray-new-30 hover:text-white"
          >
            Clear
          </button>
        </div>
      </div>

      <ol className="flex flex-col gap-1.5">
        {list.map((p, i) => (
          <li key={p.id}>
            <div className="group flex items-baseline gap-3 rounded-lg border border-gray-new-15 bg-gray-new-10/60 px-3 py-2.5 transition-colors hover:border-gray-new-25">
              <span className="t-sm w-6 shrink-0 font-mono text-gray-new-50 tabular-nums">
                {String(i + 1).padStart(2, '0')}
              </span>
              <Link
                href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
                className="min-w-0 flex-1"
              >
                <span className="t-sm block truncate text-white group-hover:text-primary-1">
                  {p.title}
                </span>
                <span className="mt-0.5 block font-mono text-[11px] text-gray-new-50">
                  {p.category} · {p.year}
                  {p.readTimeMin ? ` · ~${p.readTimeMin} min` : ''}
                </span>
              </Link>
              <button
                type="button"
                onClick={() => remove(p.id)}
                className="rounded p-1 font-mono text-xs text-gray-new-50 transition-colors hover:text-red-400"
                aria-label={`Remove ${p.title} from queue`}
              >
                ✕
              </button>
            </div>
          </li>
        ))}
      </ol>
    </>
  );
};

export default QueueList;
