'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useState } from 'react';

const KEY = 'pg.read.v1';

// Picks a random unread paper with score >= 8 (or any unread if pool < 5).
// Pure client-side: no backend, no network on click.
const SurpriseMe = ({ papers }) => {
  const router = useRouter();
  const [hydrated, setHydrated] = useState(false);
  const [pool, setPool] = useState(0);

  // When "Merit only" toggle is on, exclude papers we marked interested=1
  // just because they're from a tracked company (Qwen/DeepSeek/etc.).
  const meritOnlyKey = 'pg.merit-only.v1';

  const isMeritOnly = () => {
    try {
      return localStorage.getItem(meritOnlyKey) === '1';
    } catch {
      return false;
    }
  };

  const passesMeritFilter = (p) => !isMeritOnly() || !p.companyOnly;

  useEffect(() => {
    setHydrated(true);
    try {
      const raw = localStorage.getItem(KEY);
      const read = new Set(raw ? JSON.parse(raw) : []);
      const high = papers.filter(
        (p) => (p.score ?? 0) >= 8 && !read.has(p.id) && passesMeritFilter(p)
      );
      setPool(
        high.length > 0
          ? high.length
          : papers.filter((p) => !read.has(p.id) && passesMeritFilter(p)).length
      );
    } catch {
      setPool(papers.length);
    }
  }, [papers]);

  const go = () => {
    let read = new Set();
    try {
      const raw = localStorage.getItem(KEY);
      read = new Set(raw ? JSON.parse(raw) : []);
    } catch {}
    let candidates = papers.filter(
      (p) => (p.score ?? 0) >= 8 && !read.has(p.id) && passesMeritFilter(p)
    );
    if (candidates.length < 5) {
      candidates = papers.filter((p) => !read.has(p.id) && passesMeritFilter(p));
    }
    if (candidates.length === 0) candidates = papers.filter(passesMeritFilter);
    if (candidates.length === 0) candidates = papers;
    const pick = candidates[Math.floor(Math.random() * candidates.length)];
    router.push(`/p/${encodeURIComponent(pick.category)}/${encodeURIComponent(pick.slug)}`);
  };

  return (
    <button
      type="button"
      onClick={go}
      className={`group inline-flex items-center gap-2 rounded-md border border-primary-1/40 bg-primary-1/10 px-3 py-1.5 font-mono text-xs tracking-wider text-primary-1 uppercase transition-all hover:bg-primary-1/20 ${
        hydrated ? '' : 'opacity-0'
      }`}
    >
      <span aria-hidden>✦</span>
      <span>Surprise me</span>
      <span className="text-[10px] text-primary-1/60">({pool})</span>
    </button>
  );
};

export default SurpriseMe;
