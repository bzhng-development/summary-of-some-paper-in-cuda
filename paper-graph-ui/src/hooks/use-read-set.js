'use client';

import { useCallback, useMemo } from 'react';

import { STORAGE_EVENTS, STORAGE_KEYS } from 'lib/storage-keys';

import { useStorageState } from './use-storage-state';

// Treats the read-status localStorage entry as a Set, but persists as an array
// so it round-trips through JSON cleanly. Returns the set, has(), toggle().
export function useReadSet() {
  const [list, setList, hydrated] = useStorageState(
    STORAGE_KEYS.read,
    [],
    STORAGE_EVENTS.read
  );

  const set = useMemo(() => new Set(list), [list]);

  const has = useCallback((id) => set.has(id), [set]);

  const toggle = useCallback(
    (id) => {
      setList((prev) => {
        const s = new Set(prev);
        if (s.has(id)) s.delete(id);
        else s.add(id);
        return [...s];
      });
    },
    [setList]
  );

  return { set, list, has, toggle, hydrated };
}
