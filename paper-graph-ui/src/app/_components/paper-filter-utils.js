export const PAPER_FILTER_EVENT = 'pg-paper-filters-change';

export const SOURCE_FILTER_KEY = 'pg.paper-filter.v2';
export const LEGACY_SOURCE_FILTER_KEY = 'pg.merit-only.v1';
export const SUMMARYLESS_FILTER_KEY = 'pg.show-summaryless.v1';
export const COMPANY_FILTER_KEY = 'pg.company-filter.v1';

const SOURCE_MODES = new Set(['all', 'merit', 'company']);

const normalizeQuery = (value) => String(value ?? '').trim();

const readStorage = (key) => {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
};

const readSourceMode = () => {
  const stored = readStorage(SOURCE_FILTER_KEY);
  if (stored && SOURCE_MODES.has(stored)) return stored;
  if (readStorage(LEGACY_SOURCE_FILTER_KEY) === '1') return 'merit';
  return 'merit';
};

export const readPaperFilterState = () => {
  if (typeof window === 'undefined') {
    return { org: '', query: '', sourceMode: 'merit', showSummaryless: false };
  }

  const params = new URLSearchParams(window.location.search);
  const urlOrg = normalizeQuery(params.get('org'));
  const storedOrg = normalizeQuery(readStorage(COMPANY_FILTER_KEY));

  return {
    org: urlOrg || storedOrg,
    query: params.get('q') ?? '',
    sourceMode: readSourceMode(),
    showSummaryless: readStorage(SUMMARYLESS_FILTER_KEY) === '1',
  };
};

export const updateUrlParam = (key, value) => {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  const nextValue = normalizeQuery(value);
  if (nextValue) url.searchParams.set(key, nextValue);
  else url.searchParams.delete(key);
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
};

export const dispatchPaperFilters = (partial = {}) => {
  if (typeof window === 'undefined') return {};
  const detail = {
    ...readPaperFilterState(),
    ...partial,
  };
  detail.query = normalizeQuery(detail.query);
  detail.org = normalizeQuery(detail.org);
  window.dispatchEvent(new CustomEvent(PAPER_FILTER_EVENT, { detail }));
  return detail;
};
