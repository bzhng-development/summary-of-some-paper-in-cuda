'use client';

import { useEffect, useState } from 'react';

const KEY = 'pg.show-summaryless.v1';

const NoSummaryToggle = () => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      setShow(window.localStorage.getItem(KEY) === '1');
    } catch {
      /* localStorage blocked */
    }
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.body.classList.toggle('pg-show-summaryless', show);
    try {
      window.localStorage.setItem(KEY, show ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [show]);

  return (
    <button
      type="button"
      onClick={() => setShow((value) => !value)}
      title={show ? 'Hide papers without summaries.' : 'Show papers without summaries.'}
      className={`rounded border px-2 py-1 text-[11px] tracking-wider uppercase transition-colors ${
        show
          ? 'border-primary-1 bg-primary-1/15 text-primary-1'
          : 'border-gray-new-30 text-gray-new-70 hover:border-gray-new-50 hover:text-white'
      }`}
    >
      No summaries
    </button>
  );
};

export default NoSummaryToggle;
