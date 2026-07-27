'use client';

import Link from 'next/link';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { useReadSet } from 'hooks/use-read-set';
import paperIndex from 'lib/paper-index.generated.json';

import { filterRecords } from './paper-filter';
import {
  PAPER_FILTER_EVENT,
  dispatchPaperFilters,
  readPaperFilterState,
} from './paper-filter-utils';

const PAGE_SIZE = 400;

const CATEGORY_META = {
  agents: { title: 'Agents', color: '#aa99ff' },
  alignment: { title: 'Alignment', color: '#00E599' },
  architecture: { title: 'Architecture', color: '#259df4' },
  code: { title: 'Code', color: '#f0f075' },
  'context-optimization': { title: 'Context Optimization', color: '#ffa64c' },
  data: { title: 'Data', color: '#f0f075' },
  diffusion: { title: 'Diffusion', color: '#ff4c79' },
  'distributed-training': { title: 'Distributed Training', color: '#259df4' },
  evaluation: { title: 'Evaluation', color: '#aa99ff' },
  'inference-optimization': { title: 'Inference Optimization', color: '#00E599' },
  'llm-systems': { title: 'LLM Systems', color: '#259df4' },
  'low-precision': { title: 'Low Precision', color: '#ffa64c' },
  moe: { title: 'Mixture of Experts', color: '#aa99ff' },
  multimodal: { title: 'Multimodal', color: '#ff4c79' },
  pretraining: { title: 'Pretraining', color: '#00E599' },
  prompting: { title: 'Prompting', color: '#f0f075' },
  reasoning: { title: 'Reasoning', color: '#ffa64c' },
  retrieval: { title: 'Retrieval', color: '#259df4' },
  'rl-training': { title: 'RL Training', color: '#00E599' },
  safety: { title: 'Safety', color: '#ff4c79' },
  'scaling-laws': { title: 'Scaling Laws', color: '#ffa64c' },
  serving: { title: 'Serving', color: '#259df4' },
  'training-methods': { title: 'Training Methods', color: '#aa99ff' },
  uncategorized: { title: 'Uncategorized', color: '#94979E' },
  vision: { title: 'Vision', color: '#ff4c79' },
};

const MONTH_NAMES = [
  '',
  'Jan',
  'Feb',
  'Mar',
  'Apr',
  'May',
  'Jun',
  'Jul',
  'Aug',
  'Sep',
  'Oct',
  'Nov',
  'Dec',
];

const categoryMeta = (slug) => CATEGORY_META[slug] ?? { title: slug, color: '#94979E' };

const formatDate = (record) => {
  if (record.m) return `${MONTH_NAMES[record.m]} ${record.y}`;
  return String(record.y);
};

const groupByYear = (records) => {
  const groups = [];
  let currentYear = null;
  let current = null;

  for (const record of records) {
    if (record.y !== currentYear) {
      currentYear = record.y;
      current = { year: record.y, records: [] };
      groups.push(current);
    }
    current.records.push(record);
  }

  return groups;
};

// Per-year offsets into the (year-desc sorted) filtered array. `end` is the
// exclusive index just past the year's last record — expanding the visible
// window to `end` guarantees the year's whole section is rendered so an anchor
// jump can land on it. Counts here are the *filtered* counts, so the year nav
// always agrees with what the list actually shows.
const buildYearNav = (records) => {
  const nav = [];
  let current = null;
  records.forEach((record, index) => {
    if (!current || current.year !== record.y) {
      current = { year: record.y, count: 0, start: index, end: index + 1 };
      nav.push(current);
    }
    current.count += 1;
    current.end = index + 1;
  });
  return nav;
};

const RecordMeta = ({ record, showCategory }) => {
  const bits = [];
  if (showCategory) bits.push(categoryMeta(record.c).title);
  if (record.o) bits.push(record.o);
  bits.push(formatDate(record));
  if (record.au) bits.push(record.au);

  return (
    <span className="t-sm mt-0.5 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 font-mono text-xs text-gray-new-50">
      {record.hs === 0 ? (
        <span className="shrink-0 rounded border border-gray-new-30 px-1.5 py-0.5 font-mono text-[10px] tracking-wider text-gray-new-60 uppercase">
          no summary yet
        </span>
      ) : null}
      {bits.map((bit, index) => (
        <span key={`${index}-${bit}`} className="truncate">
          {bit}
        </span>
      ))}
    </span>
  );
};

const PaperRow = ({ record, readSet, showCategory = false, leading = null }) => {
  const cat = categoryMeta(record.c);
  const isRead = readSet.has(record.id);

  return (
    <li>
      <Link
        href={`/p/${encodeURIComponent(record.c)}/${encodeURIComponent(record.s)}`}
        className="group flex items-baseline gap-3 rounded-lg border border-transparent px-2 py-1.5 transition-colors hover:border-gray-new-15 hover:bg-gray-new-10"
      >
        {leading ? (
          <span className="t-sm w-12 shrink-0 font-mono text-gray-new-50 tabular-nums">
            {leading}
          </span>
        ) : null}
        <span
          className="mt-1.5 inline-block h-2 w-2 shrink-0 rounded-full"
          style={{ backgroundColor: cat.color }}
          aria-hidden
        />
        <span className="min-w-0 flex-1">
          <span className="t-sm flex items-center gap-2 leading-tight text-white group-hover:text-primary-1">
            <span className="truncate">{record.t}</span>
            {isRead ? (
              <span
                title="Read"
                aria-label="Read"
                className="inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-primary-1"
              />
            ) : null}
          </span>
          <RecordMeta record={record} showCategory={showCategory} />
        </span>
      </Link>
    </li>
  );
};

const YearNav = ({ entries, activeYear, onJump }) => (
  <nav className="sticky top-14 z-30 -mx-3 mb-6 overflow-x-auto border-b border-gray-new-15 bg-black-pure/90 px-3 py-2 backdrop-blur [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
    <ul className="flex gap-2">
      {entries.map((entry) => {
        const active = entry.year === activeYear;
        return (
          <li key={entry.year}>
            <button
              type="button"
              onClick={() => onJump(entry.year, true)}
              aria-current={active ? 'true' : undefined}
              className={`block rounded-md border px-3 py-1.5 font-mono text-xs tracking-wide whitespace-nowrap transition-colors ${
                active
                  ? 'border-primary-1 bg-primary-1/15 text-primary-1'
                  : 'border-gray-new-20 bg-gray-new-10 text-gray-new-70 hover:border-primary-1/40 hover:text-white'
              }`}
            >
              {entry.year}
              <span className={active ? 'ml-2 text-primary-1/70' : 'ml-2 text-gray-new-50'}>
                {entry.count.toLocaleString()}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  </nav>
);

// Parse a `#2025`-style timeline hash into a plausible publication year.
const parseYearHash = () => {
  if (typeof window === 'undefined') return null;
  const raw = window.location.hash.replace('#', '').trim();
  if (!/^\d{4}$/.test(raw)) return null;
  const year = Number(raw);
  return year >= 1990 && year <= 2100 ? year : null;
};

export const PaperListClient = ({ scope }) => {
  const isTimeline = scope?.type === 'all';
  const { set: readSet } = useReadSet();
  const [filterState, setFilterState] = useState(() => readPaperFilterState());
  const [visibleLimit, setVisibleLimit] = useState(PAGE_SIZE);
  const [activeYear, setActiveYear] = useState(null);
  const filterRef = useRef(filterState);
  const sentinelRef = useRef(null);
  const yearNavRef = useRef([]);

  useEffect(() => {
    filterRef.current = filterState;
  }, [filterState]);

  useEffect(() => {
    dispatchPaperFilters(readPaperFilterState());

    // Each filter control (search, company, merit, no-summary) re-broadcasts its
    // state on mount, and PaperSearch re-fires a debounced dispatch ~150ms after
    // load. Resetting pagination on every one of those redundant events clobbers
    // an in-flight "Load more" or hash-jump expansion — the section a deep link
    // just scrolled to vanishes. Only reset when a filter value actually changed.
    const onFilterChange = (event) => {
      if (event.detail?.total !== undefined) return;
      const current = filterRef.current;
      const next = { ...current, ...event.detail };
      const changed = ['org', 'query', 'sourceMode', 'showSummaryless'].some(
        (key) => next[key] !== current[key]
      );
      if (!changed) return;
      filterRef.current = next;
      setFilterState(next);
      setVisibleLimit(PAGE_SIZE);
    };

    window.addEventListener(PAPER_FILTER_EVENT, onFilterChange);
    return () => window.removeEventListener(PAPER_FILTER_EVENT, onFilterChange);
  }, []);

  const filtered = useMemo(
    () => filterRecords(paperIndex, { ...filterState, scope }),
    [filterState, scope]
  );
  const yearNav = useMemo(() => (isTimeline ? buildYearNav(filtered) : []), [filtered, isTimeline]);
  const visible = filtered.slice(0, visibleLimit);
  const hasMore = visible.length < filtered.length;
  const grouped = scope?.type === 'topic' ? null : groupByYear(visible);

  useEffect(() => {
    yearNavRef.current = yearNav;
  }, [yearNav]);

  useEffect(() => {
    window.dispatchEvent(
      new CustomEvent(PAPER_FILTER_EVENT, {
        detail: {
          ...filterState,
          total: filtered.length,
          visible: filtered.length,
        },
      })
    );
  }, [filterState, filtered.length]);

  // Jump to a year: expand the paginated window so the whole year renders (older
  // years live thousands of rows deep, past the window, so the anchor wouldn't
  // exist), then scroll to its section. The section paints a frame or two after
  // the state update, so poll across a few frames rather than reading the DOM
  // synchronously. Stable identity (reads the latest nav via a ref) so the deep
  // link effect below can depend on it without re-firing on every filter change.
  const jumpToYear = useCallback((year, smooth) => {
    const entry = yearNavRef.current.find((item) => item.year === year);
    if (!entry) return; // no rows for this year under the active filter
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', `#${year}`);
    }
    setVisibleLimit((value) => Math.max(value, entry.end));
    let frames = 0;
    const scroll = () => {
      const target = document.getElementById(String(year));
      if (target) {
        target.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
      } else if (frames++ < 30) {
        requestAnimationFrame(scroll);
      }
    };
    requestAnimationFrame(scroll);
  }, []);

  // Honor a `/timeline#2025` deep link on load, and respond to later hash edits
  // (e.g. the "Years covered" links on the home page after a client nav).
  useEffect(() => {
    if (!isTimeline) return;
    const applyHash = () => {
      const year = parseYearHash();
      if (year != null) jumpToYear(year, false);
    };
    applyHash();
    window.addEventListener('hashchange', applyHash);
    return () => window.removeEventListener('hashchange', applyHash);
  }, [isTimeline, jumpToYear]);

  // Auto-load the next page as the sentinel nears the viewport — infinite scroll
  // instead of a "Load more" button. One PAGE_SIZE grow per scroll-to-bottom:
  // the freshly added rows push the sentinel far below the pre-load margin, so it
  // stops until the reader scrolls again. Keeps the DOM bounded (the full "All"
  // view is ~21.5k rows / ~130k nodes) without ever showing a button.
  useEffect(() => {
    if (!hasMore || typeof IntersectionObserver === 'undefined') return;
    const sentinel = sentinelRef.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisibleLimit((value) => value + PAGE_SIZE);
        }
      },
      { rootMargin: '0px 0px 800px 0px' }
    );
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [hasMore, filtered.length]);

  // Highlight the year currently under the sticky nav.
  useEffect(() => {
    if (!isTimeline || typeof IntersectionObserver === 'undefined') return;
    const sections = Array.from(document.querySelectorAll('[data-year-section]'));
    if (sections.length === 0) return;
    const visibleYears = new Set();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const year = Number(entry.target.getAttribute('data-year-section'));
          if (entry.isIntersecting) visibleYears.add(year);
          else visibleYears.delete(year);
        }
        if (visibleYears.size > 0) setActiveYear(Math.max(...visibleYears));
      },
      { rootMargin: '-120px 0px -70% 0px', threshold: 0 }
    );
    for (const section of sections) observer.observe(section);
    return () => observer.disconnect();
  }, [isTimeline, visible.length, filtered.length]);

  return (
    <section aria-live="polite">
      {isTimeline && yearNav.length > 0 ? (
        <YearNav entries={yearNav} activeYear={activeYear} onJump={jumpToYear} />
      ) : null}

      <div className="mb-4 flex items-center justify-between gap-3 border-b border-gray-new-15 pb-2">
        <p className="t-sm font-mono tracking-[0.18em] text-gray-new-50 uppercase">
          {filtered.length.toLocaleString()} papers
        </p>
        {filtered.length > 0 ? (
          <p className="t-sm text-gray-new-60">
            Showing {visible.length.toLocaleString()} of {filtered.length.toLocaleString()}
          </p>
        ) : null}
      </div>

      {filtered.length === 0 ? (
        <div className="rounded-lg border border-gray-new-15 px-4 py-8 text-center">
          <p className="t-sm text-gray-new-60">No papers match the current filters.</p>
        </div>
      ) : grouped ? (
        <div className="flex flex-col gap-10 sm:gap-8">
          {grouped.map((group) => (
            <section
              key={group.year}
              id={isTimeline ? String(group.year) : undefined}
              data-year-section={isTimeline ? group.year : undefined}
              className="scroll-mt-28"
            >
              <header className="mb-3 flex items-baseline justify-between border-b border-gray-new-15 pb-2">
                <h2 className="font-sans text-2xl font-medium text-white sm:text-xl">
                  {group.year}
                </h2>
                <span className="t-sm font-mono tracking-[0.2em] text-gray-new-50 uppercase">
                  {group.records.length}
                </span>
              </header>
              <ul className="flex flex-col gap-1.5">
                {group.records.map((record) => (
                  <PaperRow
                    key={record.id}
                    record={record}
                    readSet={readSet}
                    leading={record.m ? MONTH_NAMES[record.m] : '-'}
                    showCategory={isTimeline}
                  />
                ))}
              </ul>
            </section>
          ))}
        </div>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {visible.map((record) => (
            <PaperRow
              key={record.id}
              record={record}
              readSet={readSet}
              leading={record.y}
              showCategory
            />
          ))}
        </ul>
      )}

      {hasMore ? (
        <div
          ref={sentinelRef}
          className="mt-8 flex justify-center py-4 font-mono text-[11px] tracking-wider text-gray-new-50 uppercase"
          aria-hidden
        >
          Loading {(filtered.length - visible.length).toLocaleString()} more…
        </div>
      ) : null}
    </section>
  );
};
