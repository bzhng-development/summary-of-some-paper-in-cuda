const normalizeText = (value) =>
  String(value ?? '')
    .trim()
    .toLowerCase();

const recordMatchesScope = (record, scope) => {
  if (!scope || scope.type === 'all') return true;
  if (scope.type === 'category') {
    return record.c === scope.slug || record.tc?.includes(scope.slug);
  }
  if (scope.type === 'topic') {
    return record.tp?.includes(scope.id) ?? false;
  }
  return true;
};

const recordMatchesSource = (record, sourceMode) => {
  if (sourceMode === 'merit') return record.co !== 1;
  if (sourceMode === 'company') return record.co === 1;
  return true;
};

export const buildRecordSearchText = (record) =>
  normalizeText([record.t, record.o, record.au].filter(Boolean).join(' '));

export const filterRecords = (
  records,
  { scope = { type: 'all' }, org = '', query = '', sourceMode = 'all', showSummaryless = false } =
    {}
) => {
  const activeOrg = String(org ?? '').trim();
  const activeQuery = normalizeText(query);
  // An explicit narrowing intent — a chosen company or a search query — searches
  // the FULL corpus. The merit/company toggle and the "no summaries" toggle only
  // shape the default browse view. Without this bypass, picking a company (whose
  // papers are almost all company-only + summary-less) or searching returns ~0
  // rows and the feature looks broken: e.g. "deepseek" matches 49 papers but only
  // 3 survive the default merit+summary gate.
  const explicitIntent = Boolean(activeOrg) || Boolean(activeQuery);
  const out = [];

  for (const record of records) {
    if (!recordMatchesScope(record, scope)) continue;
    if (!explicitIntent) {
      if (!recordMatchesSource(record, sourceMode)) continue;
      if (!showSummaryless && record.hs === 0) continue;
    }
    if (activeOrg && record.o !== activeOrg) continue;
    if (activeQuery && !buildRecordSearchText(record).includes(activeQuery)) continue;
    out.push(record);
  }

  return out;
};
