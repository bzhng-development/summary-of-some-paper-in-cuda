'use client';

import { useEffect, useRef } from 'react';

import { STORAGE_KEYS } from 'lib/storage-keys';

// Mounted on each paper detail page. Throttles scroll events to ~5/s and
// persists {category, slug, title, scrollRatio, savedAt} to localStorage
// so ResumeBanner can pick up where the user left off.
const ScrollTracker = ({ paper }) => {
  const lastSaved = useRef(0);

  useEffect(() => {
    function compute() {
      const doc = document.documentElement;
      const total = Math.max(1, doc.scrollHeight - window.innerHeight);
      return Math.min(1, Math.max(0, window.scrollY / total));
    }

    function persist() {
      try {
        localStorage.setItem(
          STORAGE_KEYS.resume,
          JSON.stringify({
            category: paper.category,
            slug: paper.slug,
            title: paper.title,
            scrollRatio: compute(),
            savedAt: Date.now(),
          })
        );
      } catch {
        // ignore quota/privacy
      }
    }

    let frame = null;
    function onScroll() {
      const now = Date.now();
      if (now - lastSaved.current < 200) return;
      lastSaved.current = now;
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(persist);
    }

    function onLeave() {
      persist();
    }

    // Record an initial snapshot in case the user navigates away before
    // scrolling at all.
    persist();

    window.addEventListener('scroll', onScroll, { passive: true });
    document.addEventListener('visibilitychange', onLeave);
    window.addEventListener('pagehide', onLeave);
    return () => {
      window.removeEventListener('scroll', onScroll);
      document.removeEventListener('visibilitychange', onLeave);
      window.removeEventListener('pagehide', onLeave);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [paper.category, paper.slug, paper.title]);

  return null;
};

export default ScrollTracker;
