'use client';

import { useCallback, useEffect, useRef, useState } from 'react';

// Shared client hook for JSON-encoded localStorage state with:
// - SSR-safe initial render (returns `initial` until mounted)
// - cross-tab sync via `storage` event
// - same-tab sync via a custom event name
// - guards against quota / privacy exceptions
//
// Returns [value, setValue, hydrated]. setValue accepts a value or updater
// function, mirroring useState.
export function useStorageState(key, initial, eventName) {
  const [value, setValue] = useState(initial);
  const [hydrated, setHydrated] = useState(false);
  const lastSerialized = useRef(null);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(key);
      if (raw != null) {
        const parsed = JSON.parse(raw);
        setValue(parsed);
        lastSerialized.current = raw;
      }
    } catch {
      // ignore corrupted entries
    }
    setHydrated(true);

    function refresh() {
      try {
        const raw = localStorage.getItem(key);
        if (raw !== lastSerialized.current) {
          lastSerialized.current = raw;
          setValue(raw == null ? initial : JSON.parse(raw));
        }
      } catch {
        // ignore
      }
    }
    function onStorage(e) {
      if (e.key === key) refresh();
    }
    window.addEventListener('storage', onStorage);
    if (eventName) window.addEventListener(eventName, refresh);
    return () => {
      window.removeEventListener('storage', onStorage);
      if (eventName) window.removeEventListener(eventName, refresh);
    };
    // initial is intentionally not in deps — it's the SSR fallback only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, eventName]);

  const setAndPersist = useCallback(
    (next) => {
      setValue((prev) => {
        const resolved = typeof next === 'function' ? next(prev) : next;
        try {
          const serialized = JSON.stringify(resolved);
          localStorage.setItem(key, serialized);
          lastSerialized.current = serialized;
        } catch {
          // ignore quota / privacy
        }
        if (eventName) window.dispatchEvent(new CustomEvent(eventName));
        return resolved;
      });
    },
    [key, eventName]
  );

  return [value, setAndPersist, hydrated];
}
