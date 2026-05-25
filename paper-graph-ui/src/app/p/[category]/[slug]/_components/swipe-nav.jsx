'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';

// Horizontal swipe → prev/next paper. Vertical swipe is NOT bound to any
// navigation — any vertical action conflicts with normal mobile scrolling
// and was causing accidental "back" navigation after scrolling a few times.
//
// Hard guards against false positives during scroll:
//   1. Swipe must be quick (< 350ms touchstart-to-touchend).
//   2. Horizontal travel must be ≥ 120 px.
//   3. Horizontal travel must dominate vertical by ≥ 2.5x.
//   4. The page must NOT have scrolled during the gesture — if window.scrollY
//      changed at all between touchstart and touchend, the user was scrolling,
//      not swiping. This is the key check.
//   5. Touches that begin on interactive ancestors are ignored entirely.
const SwipeNav = ({ prevHref, nextHref }) => {
  const router = useRouter();
  const start = useRef(null);
  const [flash, setFlash] = useState(null);

  useEffect(() => {
    function isInteractive(el) {
      if (!el) return false;
      return Boolean(
        el.closest(
          'a, button, input, textarea, select, [role="button"], [contenteditable="true"], pre, code, svg, .pg-no-swipe'
        )
      );
    }

    function onTouchStart(e) {
      if (e.touches.length !== 1) {
        start.current = null;
        return;
      }
      if (isInteractive(e.target)) {
        start.current = null;
        return;
      }
      const t = e.touches[0];
      start.current = {
        x: t.clientX,
        y: t.clientY,
        t: Date.now(),
        scrollY: window.scrollY,
      };
    }

    function onTouchMove(e) {
      // Multi-touch (e.g. pinch) cancels.
      if (e.touches.length !== 1) start.current = null;
    }

    function onTouchEnd(e) {
      if (!start.current) return;
      const s = start.current;
      start.current = null;
      const t = e.changedTouches[0];
      const dx = t.clientX - s.x;
      const dy = t.clientY - s.y;
      const adx = Math.abs(dx);
      const ady = Math.abs(dy);
      const dt = Date.now() - s.t;
      const scrollDelta = Math.abs(window.scrollY - s.scrollY);

      // Page scrolled during gesture → user was scrolling, not swiping.
      if (scrollDelta > 4) return;
      // Too slow → drag, not flick.
      if (dt > 350) return;
      // Must be a strong horizontal swipe.
      if (adx < 120) return;
      if (adx < ady * 2.5) return;

      if (dx < 0 && nextHref) {
        setFlash('next');
        router.push(nextHref);
      } else if (dx > 0 && prevHref) {
        setFlash('prev');
        router.push(prevHref);
      }
    }

    window.addEventListener('touchstart', onTouchStart, { passive: true });
    window.addEventListener('touchmove', onTouchMove, { passive: true });
    window.addEventListener('touchend', onTouchEnd, { passive: true });
    window.addEventListener('touchcancel', () => {
      start.current = null;
    });
    return () => {
      window.removeEventListener('touchstart', onTouchStart);
      window.removeEventListener('touchmove', onTouchMove);
      window.removeEventListener('touchend', onTouchEnd);
    };
  }, [prevHref, nextHref, router]);

  if (!flash) return null;
  const label = flash === 'next' ? 'Next paper →' : '← Previous paper';
  return (
    <div className="pointer-events-none fixed inset-x-0 top-1/2 z-50 -translate-y-1/2 text-center">
      <span className="inline-block rounded-full bg-black-new/85 px-4 py-2 font-mono text-xs tracking-wider text-primary-1 uppercase shadow-lg backdrop-blur">
        {label}
      </span>
    </div>
  );
};

export default SwipeNav;
