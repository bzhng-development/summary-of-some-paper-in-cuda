'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

const KEY = 'pg.recent.v1';

// Read history surfaced on the homepage. Tracks last 8 paper IDs visited,
// stored in localStorage. Hydrates client-side so SSR stays static.
const RecentReads = () => {
  const [recent, setRecent] = useState([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setRecent(JSON.parse(raw));
    } catch {
      // ignore
    }
    setHydrated(true);
  }, []);

  if (!hydrated || recent.length === 0) return null;

  return (
    <section className="mt-10">
      <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
        Recently opened
      </h2>
      <ul className="flex flex-col gap-1.5">
        {recent.slice(0, 8).map((p) => (
          <li key={p.id}>
            <Link
              href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
              className="group flex items-baseline justify-between gap-3 rounded-lg border border-gray-new-15 bg-gray-new-10/60 px-3 py-2 transition-colors hover:border-primary-1/40 hover:bg-gray-new-10"
            >
              <span className="t-sm truncate text-white group-hover:text-primary-1">
                {p.title}
              </span>
              <span className="t-sm font-mono text-gray-new-50">{p.year}</span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
};

export default RecentReads;
