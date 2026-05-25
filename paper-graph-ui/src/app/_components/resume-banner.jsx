'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

import { STORAGE_KEYS } from 'lib/storage-keys';

// Reads the last-paper + scroll-ratio from localStorage and shows a sticky
// "Continue reading" banner on non-paper routes when the last session was
// interrupted before 80% read.
//
// Recording happens in RecordVisit (mount) + a scroll listener on the paper
// page; this component just consumes.
const ResumeBanner = () => {
  const pathname = usePathname();
  const [snapshot, setSnapshot] = useState(null);
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEYS.resume);
      if (raw) setSnapshot(JSON.parse(raw));
    } catch {
      // ignore
    }
  }, []);

  // Hide on the resumed paper itself.
  const onSamePaper =
    snapshot && pathname === `/p/${encodeURIComponent(snapshot.category)}/${encodeURIComponent(snapshot.slug)}`;
  // Hide on the paper detail flow in general (any /p/*) — banner is for
  // navigational moments (home, timeline, category).
  const onPaperRoute = pathname?.startsWith('/p/');

  if (
    !snapshot ||
    dismissed ||
    onSamePaper ||
    onPaperRoute ||
    snapshot.scrollRatio >= 0.8
  ) {
    return null;
  }

  const pct = Math.round((snapshot.scrollRatio ?? 0) * 100);
  const remaining = Math.max(1, 100 - pct);

  return (
    <div className="sticky top-14 z-30 border-b border-primary-1/30 bg-primary-1/10 backdrop-blur">
      <div className="mx-auto flex max-w-[1280px] items-center gap-3 px-4 py-2 sm:px-3">
        <Link
          href={`/p/${encodeURIComponent(snapshot.category)}/${encodeURIComponent(snapshot.slug)}`}
          className="min-w-0 flex-1 truncate text-sm text-white hover:text-primary-1"
        >
          <span className="font-mono text-[10px] tracking-wider text-primary-1/80 uppercase">
            Continue ·
          </span>{' '}
          {snapshot.title}
        </Link>
        <span className="hidden shrink-0 font-mono text-[10px] tracking-wider text-gray-new-60 uppercase sm:inline">
          {remaining}% left
        </span>
        <button
          type="button"
          onClick={() => setDismissed(true)}
          className="rounded p-1 font-mono text-xs text-gray-new-50 transition-colors hover:text-white"
          aria-label="Dismiss resume banner"
        >
          ✕
        </button>
      </div>
    </div>
  );
};

export default ResumeBanner;
