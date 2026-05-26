'use client';

import { useEffect, useState } from 'react';

const KEY = 'pg.merit-only.v1';

// Toggle that hides "tracked-company" papers (papers where interested=1 was set
// just because the paper is from Qwen/DeepSeek/Moonshot/ByteDance/etc. rather
// than picked on merit). When ON, body gets `pg-merit-only` and CSS hides
// any element with `data-company-only="true"`.
//
// State persists in localStorage and the CSS class is applied on mount so the
// preference survives navigation and reload. No server round-trip.
const MeritOnlyToggle = () => {
  const [enabled, setEnabled] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(KEY);
      if (stored === '1') setEnabled(true);
    } catch {
      /* localStorage blocked */
    }
  }, []);

  useEffect(() => {
    if (typeof document === 'undefined') return;
    document.body.classList.toggle('pg-merit-only', enabled);
    try {
      window.localStorage.setItem(KEY, enabled ? '1' : '0');
    } catch {
      /* ignore */
    }
  }, [enabled]);

  return (
    <button
      type="button"
      onClick={() => setEnabled((v) => !v)}
      aria-pressed={enabled}
      title={
        enabled
          ? 'Showing merit-curated picks only. Click to also show tracked-company papers (Qwen/DeepSeek/etc.).'
          : 'Showing all interested papers. Click to hide tracked-company-only picks.'
      }
      className={`rounded border px-2 py-1 text-[11px] uppercase tracking-wider transition-colors ${
        enabled
          ? 'border-primary-1 bg-primary-1/15 text-primary-1'
          : 'border-gray-new-30 text-gray-new-70 hover:border-gray-new-50 hover:text-white'
      }`}
    >
      {enabled ? 'Merit only' : 'All'}
    </button>
  );
};

export default MeritOnlyToggle;
