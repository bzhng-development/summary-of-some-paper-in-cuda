'use client';

import { useEffect, useState } from 'react';

import {
  LEGACY_SOURCE_FILTER_KEY,
  SOURCE_FILTER_KEY,
  dispatchPaperFilters,
} from './paper-filter-utils';

// Header toggle that cycles paper visibility between three mutually exclusive
// modes:
//   merit   — hide tracked-company-only picks (Qwen/DeepSeek/Moonshot/etc.).
//             DEFAULT for new visitors.
//   all     — show every interested=1 paper.
//   company — hide hand-picked merit papers, show only tracked-company picks.
//
// State persists in localStorage so the preference survives navigation and
// reload. No server round-trip.
const MODES = ['all', 'merit', 'company'];
const LABEL = { all: 'All', merit: 'Merit only', company: 'Company only' };
const TITLE = {
  all: 'Showing all interested papers. Click to show merit-curated picks only.',
  merit:
    'Showing merit-curated picks only. Click to show tracked-company-only picks (Qwen/DeepSeek/MSR/etc.).',
  company: 'Showing tracked-company-only picks. Click to show all interested papers.',
};

const MeritOnlyToggle = () => {
  const [mode, setMode] = useState('merit');

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SOURCE_FILTER_KEY);
      if (stored && MODES.includes(stored)) {
        queueMicrotask(() => setMode(stored));
        dispatchPaperFilters({ sourceMode: stored });
        return;
      }
      // Migrate the old binary v1 key so users don't lose their preference.
      const legacy = window.localStorage.getItem(LEGACY_SOURCE_FILTER_KEY);
      if (legacy === '1') {
        queueMicrotask(() => setMode('merit'));
        dispatchPaperFilters({ sourceMode: 'merit' });
      }
    } catch {
      /* localStorage blocked */
    }
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(SOURCE_FILTER_KEY, mode);
    } catch {
      /* ignore */
    }
    dispatchPaperFilters({ sourceMode: mode });
  }, [mode]);

  const cycle = () => {
    setMode((m) => MODES[(MODES.indexOf(m) + 1) % MODES.length]);
  };

  return (
    <button
      type="button"
      onClick={cycle}
      title={TITLE[mode]}
      className={`rounded border px-2 py-1 text-[11px] uppercase tracking-wider transition-colors ${
        mode === 'all'
          ? 'border-gray-new-30 text-gray-new-70 hover:border-gray-new-50 hover:text-white'
          : 'border-primary-1 bg-primary-1/15 text-primary-1'
      }`}
    >
      {LABEL[mode]}
    </button>
  );
};

export default MeritOnlyToggle;
