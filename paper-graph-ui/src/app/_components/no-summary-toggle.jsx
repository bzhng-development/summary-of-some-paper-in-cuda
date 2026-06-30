'use client';

import { useEffect, useState } from 'react';

import { SUMMARYLESS_FILTER_KEY, dispatchPaperFilters } from './paper-filter-utils';

const NoSummaryToggle = () => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SUMMARYLESS_FILTER_KEY) === '1';
      queueMicrotask(() => setShow(stored));
      dispatchPaperFilters({ showSummaryless: stored });
    } catch {
      /* localStorage blocked */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(SUMMARYLESS_FILTER_KEY, show ? '1' : '0');
    } catch {
      /* ignore */
    }
    dispatchPaperFilters({ showSummaryless: show });
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
