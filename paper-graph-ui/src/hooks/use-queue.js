'use client';

import { useCallback, useMemo } from 'react';

import { STORAGE_EVENTS, STORAGE_KEYS } from 'lib/storage-keys';

import { useStorageState } from './use-storage-state';

// Up-Next reading queue. Order-preserving array of compact paper refs.
// Each entry: { id, category, slug, title, year, readTimeMin }.
const MAX_QUEUE = 50;

export function useQueue() {
  const [list, setList, hydrated] = useStorageState(
    STORAGE_KEYS.queue,
    [],
    STORAGE_EVENTS.queue
  );

  const ids = useMemo(() => new Set(list.map((p) => p.id)), [list]);

  const enqueue = useCallback(
    (paper) => {
      setList((prev) => {
        if (prev.some((p) => p.id === paper.id)) return prev;
        const trimmed = [
          ...prev,
          {
            id: paper.id,
            category: paper.category,
            slug: paper.slug,
            title: paper.title,
            year: paper.year,
            readTimeMin: paper.readTimeMin ?? null,
          },
        ].slice(-MAX_QUEUE);
        return trimmed;
      });
    },
    [setList]
  );

  const remove = useCallback(
    (id) => setList((prev) => prev.filter((p) => p.id !== id)),
    [setList]
  );

  const clear = useCallback(() => setList([]), [setList]);

  const peek = useCallback(() => list[0] ?? null, [list]);

  return { list, ids, hydrated, enqueue, remove, clear, peek };
}
