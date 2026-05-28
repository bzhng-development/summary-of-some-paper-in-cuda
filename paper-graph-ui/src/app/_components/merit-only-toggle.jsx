'use client';

import { useEffect, useState } from 'react';

// Header toggle that cycles paper visibility between three mutually exclusive
// modes:
//   all     — show every interested=1 paper (default).
//   merit   — hide tracked-company-only picks (Qwen/DeepSeek/Moonshot/etc.).
//             body gains class `pg-merit-only`, CSS hides
//             `[data-company-only="true"]`.
//   company — hide hand-picked merit papers, show only tracked-company picks.
//             body gains class `pg-company-only`, CSS hides
//             `[data-company-only="false"]`.
//
// State persists in localStorage so the preference survives navigation and
// reload. No server round-trip.
const KEY = 'pg.paper-filter.v2';
const LEGACY_KEY = 'pg.merit-only.v1';

const MODES = ['all', 'merit', 'company'];
const LABEL = { all: 'All', merit: 'Merit only', company: 'Company only' };
const TITLE = {
  all: 'Showing all interested papers. Click to show merit-curated picks only.',
  merit:
    'Showing merit-curated picks only. Click to show tracked-company-only picks (Qwen/DeepSeek/MSR/etc.).',
  company:
    'Showing tracked-company-only picks. Click to show all interested papers.',
};

const MeritOnlyToggle = () => {
  const [mode, setMode] = useState('all');

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(KEY);
      if (stored && MODES.includes(stored)) {
        setMode(stored);
        return;
      }
      // Migrate the old binary v1 key so users don't lose their preference.
      const legacy = window.localStorage.getItem(LEGACY_KEY);
      if (legacy === '1') setMode('merit');
    } catch {
      /* localStorage blocked */
    }
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.body.classList.toggle('pg-merit-only', mode === 'merit');
    document.body.classList.toggle('pg-company-only', mode === 'company');
    try {
      window.localStorage.setItem(KEY, mode);
    } catch {
      /* ignore */
    }
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
