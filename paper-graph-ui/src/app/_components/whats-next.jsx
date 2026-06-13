'use client';

import Link from 'next/link';
import { useMemo } from 'react';

import { useReadSet } from 'hooks/use-read-set';
import { STORAGE_KEYS } from 'lib/storage-keys';
import { useStorageState } from 'hooks/use-storage-state';

import QueueButton from './queue-button';

// Picks 3 unread papers most likely to interest the user. Heuristic:
//   score(p) = 2*paperScore + 3*topicOverlapWithRecentReads + recencyBonus
//   - paperScore: Neon LLM-judged 1..10
//   - topicOverlap: |paper.topics ∩ recentReadTopics|
//   - recencyBonus: 2 if published in current year, 1 if last year, 0 else
//
// All computation is client-side using the full papers list passed from
// the server (already trimmed to small shape). Pure derivation — no
// requests, no re-render storms.

const NOW = new Date().getUTCFullYear();

const WhatsNext = ({ papers }) => {
  const { set: readSet, hydrated: readHydrated } = useReadSet();
  const [recent, , recentHydrated] = useStorageState(STORAGE_KEYS.recent, [], null);

  const ranked = useMemo(() => {
    if (!papers || papers.length === 0) return [];
    const papersById = new Map(papers.map((p) => [p.id, p]));

    // Aggregate topics across recently-opened papers (most recent first).
    const recentTopics = new Map(); // topic → weight (newer = higher)
    recent.forEach((r, i) => {
      const full = papersById.get(r.id);
      if (!full) return;
      const w = Math.max(1, 8 - i); // recent items weigh more
      for (const t of full.topics ?? []) {
        recentTopics.set(t, (recentTopics.get(t) ?? 0) + w);
      }
    });

    const out = [];
    for (const p of papers) {
      if (readSet.has(p.id)) continue;
      if (recent.some((r) => r.id === p.id)) continue;
      const sc = p.score ?? 0;
      let overlap = 0;
      for (const t of p.topics ?? []) {
        overlap += recentTopics.get(t) ?? 0;
      }
      const recencyBonus = p.year === NOW ? 2 : p.year === NOW - 1 ? 1 : 0;
      const score = sc * 2 + overlap * 3 + recencyBonus;
      if (score > 0) out.push({ p, score });
    }
    out.sort((a, b) => b.score - a.score);
    return out.slice(0, 3).map((x) => x.p);
  }, [papers, readSet, recent]);

  if (!readHydrated || !recentHydrated) return null;
  if (ranked.length === 0) return null;
  if (recent.length === 0) return null; // need history before we recommend

  return (
    <section className="mt-12 sm:mt-10">
      <h2 className="t-sm mb-3 font-mono tracking-[0.22em] text-gray-new-50 uppercase">
        What's next
      </h2>
      <p className="t-sm mb-3 max-w-2xl text-gray-new-70">
        Picked from unread papers that share topics with what you've read recently, weighted by
        score and recency.
      </p>
      <ul className="flex flex-col gap-2">
        {ranked.map((p) => (
          <li
            key={p.id}
            className="group flex flex-wrap items-center gap-3 rounded-lg border border-gray-new-15 bg-gray-new-10/60 px-3 py-2.5 transition-colors hover:border-primary-1/30"
          >
            <Link
              href={`/p/${encodeURIComponent(p.category)}/${encodeURIComponent(p.slug)}`}
              className="min-w-0 flex-1"
            >
              <span className="t-sm block truncate text-white group-hover:text-primary-1">
                {p.title}
              </span>
              <span className="mt-0.5 block truncate font-mono text-[11px] text-gray-new-50">
                {p.category} · {p.year}
                {p.score != null ? ` · score ${p.score}` : ''}
                {p.readTimeMin ? ` · ~${p.readTimeMin} min` : ''}
              </span>
            </Link>
            <QueueButton paper={p} size="sm" />
          </li>
        ))}
      </ul>
    </section>
  );
};

export default WhatsNext;
