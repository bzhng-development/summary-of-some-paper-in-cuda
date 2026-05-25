'use client';

import { useEffect } from 'react';

const KEY = 'pg.recent.v1';
const STREAK_KEY = 'pg.streak.v1';
const MAX = 16;

function todayIso() {
  const d = new Date();
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
}

function dayBefore(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() - 1);
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
}

// Records the current paper into localStorage's recent-reads list AND bumps
// the daily streak counter on first visit each UTC day. Mounted on every
// paper page; client-only so it doesn't affect server rendering.
const RecordVisit = ({ paper }) => {
  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      const list = raw ? JSON.parse(raw) : [];
      const trimmed = [
        { id: paper.id, category: paper.category, slug: paper.slug, title: paper.title, year: paper.year },
        ...list.filter((p) => p.id !== paper.id),
      ].slice(0, MAX);
      localStorage.setItem(KEY, JSON.stringify(trimmed));
    } catch {
      // ignore
    }
    try {
      const t = todayIso();
      const sraw = localStorage.getItem(STREAK_KEY);
      const s = sraw ? JSON.parse(sraw) : { last: null, count: 0 };
      if (s.last !== t) {
        const next =
          s.last && dayBefore(t) === s.last ? (s.count || 0) + 1 : 1;
        localStorage.setItem(
          STREAK_KEY,
          JSON.stringify({ last: t, count: next })
        );
        window.dispatchEvent(new CustomEvent('pg:streak-changed'));
      }
    } catch {
      // ignore
    }
  }, [paper.id, paper.category, paper.slug, paper.title, paper.year]);
  return null;
};

export default RecordVisit;
