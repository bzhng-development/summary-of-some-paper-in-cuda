'use client';

import { useEffect, useRef, useState } from 'react';

// Wraps an overflowing child and paints a left/right fade gradient that
// indicates more content is scrollable in that direction. Used for code
// blocks, tables, and math display rows on narrow viewports.
const OverflowFade = ({ className = '', children }) => {
  const ref = useRef(null);
  const [fade, setFade] = useState({ left: false, right: false });

  useEffect(() => {
    const el = ref.current;
    if (!el) return undefined;
    // The scrollable element is the first child (the <pre>/<table>) when
    // we render this as a positioning wrapper.
    const scroller = el.firstElementChild;
    if (!scroller) return undefined;
    function compute() {
      const left = scroller.scrollLeft > 4;
      const right =
        scroller.scrollWidth - scroller.clientWidth - scroller.scrollLeft > 4;
      setFade((f) => (f.left === left && f.right === right ? f : { left, right }));
    }
    compute();
    scroller.addEventListener('scroll', compute, { passive: true });
    const ro = new ResizeObserver(compute);
    ro.observe(scroller);
    return () => {
      scroller.removeEventListener('scroll', compute);
      ro.disconnect();
    };
  }, []);

  return (
    <div ref={ref} className={`pg-overflow-fade relative ${className}`}>
      {children}
      {fade.left ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-0 left-0 w-6 rounded-l-md bg-gradient-to-r from-black-fog to-transparent"
        />
      ) : null}
      {fade.right ? (
        <span
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-6 rounded-r-md bg-gradient-to-l from-black-fog to-transparent"
        />
      ) : null}
    </div>
  );
};

export default OverflowFade;
