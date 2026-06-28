'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import { applyPaperFilters, updateUrlParam } from './paper-filter-utils';

const KEY = 'pg.company-filter.v1';

const getInitialOrg = (options) => {
  if (typeof window === 'undefined') return '';
  const validOrgs = new Set(options.map((option) => option.org));
  const params = new URLSearchParams(window.location.search);
  const urlOrg = params.get('org')?.trim();
  if (urlOrg && validOrgs.has(urlOrg)) return urlOrg;

  try {
    const stored = window.localStorage.getItem(KEY)?.trim();
    if (stored && validOrgs.has(stored)) return stored;
  } catch {
    /* localStorage blocked */
  }
  return '';
};

const CompanyFilter = ({ options }) => {
  const [selected, setSelected] = useState('');
  const [filter, setFilter] = useState('');
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    const initial = getInitialOrg(options);
    queueMicrotask(() => setSelected(initial));
    applyPaperFilters({ org: initial });
    updateUrlParam('org', initial);
  }, [options]);

  useEffect(() => {
    const onPointerDown = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, []);

  const filteredOptions = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return options;
    return options.filter((option) => option.org.toLowerCase().includes(q));
  }, [filter, options]);

  const selectedOption = options.find((option) => option.org === selected);
  const buttonLabel = selectedOption ? selectedOption.org : 'All companies';

  const selectOrg = (org) => {
    setSelected(org);
    setOpen(false);
    setFilter('');
    applyPaperFilters({ org });
    updateUrlParam('org', org);
    try {
      if (org) window.localStorage.setItem(KEY, org);
      else window.localStorage.removeItem(KEY);
    } catch {
      /* ignore */
    }
  };

  return (
    <div ref={rootRef} className="relative">
      <label htmlFor="company-filter-search" className="sr-only">
        Filter by company
      </label>
      <button
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls="company-filter-options"
        onClick={() => setOpen((value) => !value)}
        className={`max-w-48 truncate rounded border px-2 py-1 text-[11px] tracking-wider uppercase transition-colors ${
          selected
            ? 'border-primary-1 bg-primary-1/15 text-primary-1'
            : 'border-gray-new-30 text-gray-new-70 hover:border-gray-new-50 hover:text-white'
        }`}
        title={
          selectedOption
            ? `${selectedOption.count} papers from ${selectedOption.org}`
            : 'Show all companies'
        }
      >
        {buttonLabel}
      </button>
      {open ? (
        <div className="absolute right-0 top-full z-50 mt-2 w-72 rounded-lg border border-gray-new-20 bg-black-pure p-2 shadow-2xl shadow-black/40">
          <input
            id="company-filter-search"
            type="search"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
            placeholder="Search companies"
            className="mb-2 h-8 w-full rounded border border-gray-new-20 bg-gray-new-10 px-2 font-mono text-xs text-white outline-none transition-colors placeholder:text-gray-new-50 focus:border-primary-1"
            autoComplete="off"
          />
          <div
            id="company-filter-options"
            role="listbox"
            aria-label="Company filter options"
            className="max-h-72 overflow-y-auto pr-1"
          >
            <button
              type="button"
              role="option"
              aria-selected={!selected}
              onClick={() => selectOrg('')}
              className="flex w-full items-center justify-between gap-3 rounded px-2 py-1.5 text-left text-xs text-gray-new-70 transition-colors hover:bg-gray-new-10 hover:text-white"
            >
              <span>All companies</span>
              <span className="font-mono text-gray-new-50">
                {options.reduce((sum, option) => sum + option.count, 0)}
              </span>
            </button>
            {filteredOptions.map((option) => (
              <button
                key={option.org}
                type="button"
                role="option"
                aria-selected={selected === option.org}
                onClick={() => selectOrg(option.org)}
                className={`flex w-full items-center justify-between gap-3 rounded px-2 py-1.5 text-left text-xs transition-colors ${
                  selected === option.org
                    ? 'bg-primary-1/15 text-primary-1'
                    : 'text-gray-new-70 hover:bg-gray-new-10 hover:text-white'
                }`}
              >
                <span className="truncate">{option.org}</span>
                <span className="font-mono text-gray-new-50">{option.count}</span>
              </button>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
};

export default CompanyFilter;
