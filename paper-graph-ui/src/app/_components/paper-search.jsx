'use client';

import { useEffect, useState } from 'react';

import { PAPER_FILTER_EVENT, applyPaperFilters, updateUrlParam } from './paper-filter-utils';

const getInitialQuery = () => {
  if (typeof window === 'undefined') return '';
  return new URLSearchParams(window.location.search).get('q') ?? '';
};

const PaperSearch = () => {
  const [query, setQuery] = useState('');
  const [count, setCount] = useState({ visible: 0, total: 0 });

  useEffect(() => {
    const initial = getInitialQuery();
    const nextCount = applyPaperFilters({ query: initial });
    queueMicrotask(() => {
      setQuery(initial);
      setCount(nextCount);
    });
  }, []);

  useEffect(() => {
    const onFilterChange = (event) => {
      setCount({
        visible: event.detail.visible,
        total: event.detail.total,
      });
    };
    window.addEventListener(PAPER_FILTER_EVENT, onFilterChange);
    return () => window.removeEventListener(PAPER_FILTER_EVENT, onFilterChange);
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setCount(applyPaperFilters({ query }));
      updateUrlParam('q', query.trim());
    }, 150);
    return () => window.clearTimeout(handle);
  }, [query]);

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="paper-search" className="sr-only">
        Search papers
      </label>
      <input
        id="paper-search"
        type="search"
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search papers"
        aria-label="Search papers"
        className="h-7 w-44 rounded border border-gray-new-30 bg-transparent px-2 font-mono text-[11px] tracking-wide text-white outline-none transition-colors placeholder:text-gray-new-50 hover:border-gray-new-50 focus:border-primary-1 sm:w-36"
        autoComplete="off"
      />
      {count.total > 0 ? (
        <span
          className="font-mono text-[11px] tracking-wider whitespace-nowrap text-gray-new-50 uppercase"
          title={`${count.visible} of ${count.total} papers visible`}
        >
          {count.visible} papers
        </span>
      ) : null}
    </div>
  );
};

export default PaperSearch;
