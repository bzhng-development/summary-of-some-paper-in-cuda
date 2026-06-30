'use client';

import Link from 'next/link';
import { useEffect, useMemo, useState } from 'react';

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
      {bits.map((bit) => (
        <span key={bit} className="truncate">
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

export const PaperListClient = ({ scope }) => {
  const { set: readSet } = useReadSet();
  const [filterState, setFilterState] = useState(() => readPaperFilterState());
  const [visibleLimit, setVisibleLimit] = useState(PAGE_SIZE);

  useEffect(() => {
    dispatchPaperFilters(readPaperFilterState());

    const onFilterChange = (event) => {
      if (event.detail?.total !== undefined) return;
      setFilterState((current) => ({
        ...current,
        ...event.detail,
      }));
      setVisibleLimit(PAGE_SIZE);
    };

    window.addEventListener(PAPER_FILTER_EVENT, onFilterChange);
    return () => window.removeEventListener(PAPER_FILTER_EVENT, onFilterChange);
  }, []);

  const filtered = useMemo(
    () => filterRecords(paperIndex, { ...filterState, scope }),
    [filterState, scope]
  );
  const visible = filtered.slice(0, visibleLimit);
  const hasMore = visible.length < filtered.length;
  const grouped = scope?.type === 'topic' ? null : groupByYear(visible);

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

  return (
    <section aria-live="polite">
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
            <section key={group.year} id={scope?.type === 'all' ? String(group.year) : undefined}>
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
                    showCategory={scope?.type === 'all'}
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
        <div className="mt-8 flex justify-center">
          <button
            type="button"
            onClick={() => setVisibleLimit((value) => value + PAGE_SIZE)}
            className="rounded border border-gray-new-30 px-4 py-2 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:border-primary-1 hover:text-white"
          >
            Load more
          </button>
        </div>
      ) : null}
    </section>
  );
};
