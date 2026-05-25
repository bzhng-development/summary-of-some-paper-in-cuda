'use client';

import { useEffect, useState } from 'react';

const KEY = 'pg.streak.v1';

function today() {
  const d = new Date();
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
}

function dayBefore(iso) {
  const d = new Date(iso + 'T00:00:00Z');
  d.setUTCDate(d.getUTCDate() - 1);
  return `${d.getUTCFullYear()}-${String(d.getUTCMonth() + 1).padStart(2, '0')}-${String(d.getUTCDate()).padStart(2, '0')}`;
}

// Lightweight streak: increments once per UTC day on first paper-page visit.
// All state in localStorage. The component itself only renders the count;
// the actual increment is triggered by RecordVisit through a shared event.
const StreakCounter = () => {
  const [hydrated, setHydrated] = useState(false);
  const [count, setCount] = useState(0);

  useEffect(() => {
    function refresh() {
      try {
        const raw = localStorage.getItem(KEY);
        const parsed = raw ? JSON.parse(raw) : { last: null, count: 0 };
        const t = today();
        if (parsed.last === t) {
          setCount(parsed.count);
        } else if (parsed.last && dayBefore(t) === parsed.last) {
          setCount(parsed.count);
        } else {
          setCount(0);
        }
      } catch {
        setCount(0);
      }
    }
    refresh();
    setHydrated(true);
    window.addEventListener('pg:streak-changed', refresh);
    window.addEventListener('storage', refresh);
    return () => {
      window.removeEventListener('pg:streak-changed', refresh);
      window.removeEventListener('storage', refresh);
    };
  }, []);

  if (!hydrated || count === 0) return null;

  return (
    <span
      className="inline-flex items-center gap-1 rounded-md border border-orange-400/30 bg-orange-400/10 px-2 py-0.5 font-mono text-[11px] tracking-wider text-orange-300 uppercase"
      title={`${count}-day reading streak`}
    >
      <span aria-hidden>🔥</span>
      <span className="tabular-nums">{count}</span>
    </span>
  );
};

export default StreakCounter;
