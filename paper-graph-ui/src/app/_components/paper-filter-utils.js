export const PAPER_FILTER_EVENT = 'pg-paper-filters-change';

const ROW_SELECTOR = '[data-paper-row="true"]';
const GROUP_SELECTOR = '[data-paper-group="true"]';

const normalizeText = (value) =>
  String(value ?? '')
    .trim()
    .toLowerCase();

const getStoredFilter = (name) => document.body.dataset[name] ?? '';

const setStoredFilter = (name, value) => {
  if (value) document.body.dataset[name] = value;
  else delete document.body.dataset[name];
};

const rowPassesSourceFilter = (row) => {
  if (document.body.classList.contains('pg-merit-only')) {
    return row.dataset.companyOnly !== 'true';
  }
  if (document.body.classList.contains('pg-company-only')) {
    return row.dataset.companyOnly !== 'false';
  }
  return true;
};

const rowPassesSummaryFilter = (row) =>
  document.body.classList.contains('pg-show-summaryless') || row.dataset.hasSummary !== 'false';

export const updateUrlParam = (key, value) => {
  if (typeof window === 'undefined') return;
  const url = new URL(window.location.href);
  if (value) url.searchParams.set(key, value);
  else url.searchParams.delete(key);
  window.history.replaceState(window.history.state, '', `${url.pathname}${url.search}${url.hash}`);
};

export const applyPaperFilters = ({ org, query } = {}) => {
  if (typeof document === 'undefined') return { visible: 0, total: 0 };

  if (org !== undefined) setStoredFilter('pgOrg', org);
  if (query !== undefined) setStoredFilter('pgQuery', normalizeText(query));

  const activeOrg = getStoredFilter('pgOrg');
  const activeQuery = getStoredFilter('pgQuery');
  const rows = [...document.querySelectorAll(ROW_SELECTOR)];
  let visible = 0;

  for (const row of rows) {
    const passesOrg = !activeOrg || row.dataset.org === activeOrg;
    const passesQuery = !activeQuery || (row.dataset.searchText ?? '').includes(activeQuery);
    const passes =
      passesOrg && passesQuery && rowPassesSourceFilter(row) && rowPassesSummaryFilter(row);

    if (passes) {
      delete row.dataset.paperFilterHidden;
      visible += 1;
    } else {
      row.dataset.paperFilterHidden = 'true';
    }
  }

  for (const group of document.querySelectorAll(GROUP_SELECTOR)) {
    const groupRows = [...group.querySelectorAll(ROW_SELECTOR)];
    const hasVisibleRow = groupRows.some((row) => row.dataset.paperFilterHidden !== 'true');
    if (hasVisibleRow) delete group.dataset.paperFilterHidden;
    else group.dataset.paperFilterHidden = 'true';
  }

  const detail = { org: activeOrg, query: activeQuery, total: rows.length, visible };
  window.dispatchEvent(new CustomEvent(PAPER_FILTER_EVENT, { detail }));
  return detail;
};

export const buildPaperSearchText = (paper) =>
  normalizeText(
    [paper.title, paper.organization, ...(paper.authors ?? [])].filter(Boolean).join(' ')
  );
