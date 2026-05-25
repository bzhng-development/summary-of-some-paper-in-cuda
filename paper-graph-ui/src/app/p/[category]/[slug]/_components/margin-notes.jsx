'use client';

import { useCallback, useEffect, useId, useState } from 'react';

import { STORAGE_EVENTS, STORAGE_KEYS } from 'lib/storage-keys';

// Margin notes: per-paper map of {headingSlug → text}. We attach a small
// "+ note" button next to every h2/h3 in the rendered markdown, and a
// drawer at the bottom of the page that lists all notes for the paper and
// lets you edit/delete them.
//
// Heading slugs are stable because react-markdown + rehype-slug emits id=...
// on every heading, matching its own slug algorithm.

function readAll() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.notes);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeAll(map) {
  try {
    localStorage.setItem(STORAGE_KEYS.notes, JSON.stringify(map));
    window.dispatchEvent(new CustomEvent(STORAGE_EVENTS.notes));
  } catch {
    // ignore
  }
}

const MarginNotes = ({ paperId, containerRef }) => {
  const [notes, setNotes] = useState({}); // { headingSlug → { text, title } }
  const [editing, setEditing] = useState(null); // { slug, title, draft }
  const editorId = useId();

  const refresh = useCallback(() => {
    const all = readAll();
    setNotes(all[paperId] ?? {});
  }, [paperId]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Inject "+ note" buttons next to every h2/h3 inside the container.
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return undefined;

    const headings = [...root.querySelectorAll('h2[id], h3[id], h4[id]')];
    const cleanups = [];
    for (const h of headings) {
      const slug = h.getAttribute('id');
      if (!slug) continue;
      if (h.querySelector('.pg-note-add')) continue;
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className =
        'pg-note-add pg-no-swipe ml-2 inline-flex h-5 w-5 items-center justify-center rounded border border-gray-new-20 bg-gray-new-10 align-middle font-mono text-[11px] leading-none text-gray-new-50 transition-colors hover:border-primary-1/50 hover:text-primary-1';
      btn.setAttribute('aria-label', `Add note for ${h.textContent}`);
      btn.textContent = '+';
      const onClick = (ev) => {
        ev.preventDefault();
        ev.stopPropagation();
        const title = h.textContent?.replace(/^#+\s*/, '').trim() || slug;
        setEditing({ slug, title, draft: notes[slug]?.text ?? '' });
      };
      btn.addEventListener('click', onClick);
      h.appendChild(btn);
      cleanups.push(() => {
        btn.removeEventListener('click', onClick);
        if (btn.parentNode) btn.parentNode.removeChild(btn);
      });
    }

    return () => {
      for (const fn of cleanups) fn();
    };
  }, [containerRef, notes]);

  const save = useCallback(() => {
    if (!editing) return;
    const all = readAll();
    const pmap = { ...(all[paperId] ?? {}) };
    const draft = editing.draft.trim();
    if (draft.length === 0) {
      delete pmap[editing.slug];
    } else {
      pmap[editing.slug] = {
        text: draft,
        title: editing.title,
        updatedAt: Date.now(),
      };
    }
    if (Object.keys(pmap).length === 0) delete all[paperId];
    else all[paperId] = pmap;
    writeAll(all);
    refresh();
    setEditing(null);
  }, [editing, paperId, refresh]);

  const handleRemove = useCallback(
    (slug) => {
      const all = readAll();
      const pmap = { ...(all[paperId] ?? {}) };
      delete pmap[slug];
      if (Object.keys(pmap).length === 0) delete all[paperId];
      else all[paperId] = pmap;
      writeAll(all);
      refresh();
    },
    [paperId, refresh]
  );

  const noteList = Object.entries(notes);

  return (
    <>
      {editing ? (
        <div
          className="pg-no-swipe fixed inset-0 z-40 flex items-end justify-center bg-black/40 backdrop-blur-sm sm:items-center"
          onClick={() => setEditing(null)}
        >
          <div
            className="w-full max-w-lg rounded-t-2xl border border-gray-new-20 bg-black-new p-4 shadow-xl sm:rounded-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <header className="mb-2 flex items-baseline justify-between gap-3">
              <span className="font-mono text-[10px] tracking-wider text-gray-new-50 uppercase">
                Note on
              </span>
              <button
                type="button"
                onClick={() => setEditing(null)}
                aria-label="Close"
                className="rounded p-1 text-gray-new-50 hover:text-white"
              >
                ✕
              </button>
            </header>
            <p className="t-sm mb-3 truncate text-white">{editing.title}</p>
            <label htmlFor={editorId} className="sr-only">
              Note text
            </label>
            <textarea
              id={editorId}
              value={editing.draft}
              onChange={(e) => setEditing({ ...editing, draft: e.target.value })}
              rows={6}
              autoFocus
              placeholder="Jot a thought…"
              className="t-sm w-full resize-y rounded-md border border-gray-new-20 bg-gray-new-10 p-3 text-white placeholder-gray-new-50 focus:border-primary-1/50 focus:outline-none"
            />
            <div className="mt-3 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setEditing(null)}
                className="rounded-md border border-gray-new-20 px-3 py-1.5 font-mono text-xs tracking-wider text-gray-new-70 uppercase transition-colors hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={save}
                className="rounded-md border border-primary-1/60 bg-primary-1/15 px-3 py-1.5 font-mono text-xs tracking-wider text-primary-1 uppercase transition-colors hover:bg-primary-1/25"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {noteList.length > 0 ? (
        <section className="mt-8">
          <h2 className="t-sm mb-3 font-mono tracking-[0.2em] text-gray-new-50 uppercase">
            Your notes · {noteList.length}
          </h2>
          <ul className="flex flex-col gap-2">
            {noteList.map(([slug, n]) => (
              <li
                key={slug}
                className="group flex flex-col gap-1 rounded-lg border border-orange-300/20 bg-orange-300/5 px-3 py-2.5"
              >
                <div className="flex items-baseline justify-between gap-2">
                  <a
                    href={`#${slug}`}
                    className="t-sm flex-1 truncate font-medium text-orange-100 hover:text-orange-50"
                  >
                    {n.title}
                  </a>
                  <button
                    type="button"
                    onClick={() => setEditing({ slug, title: n.title, draft: n.text })}
                    className="rounded p-1 font-mono text-xs text-gray-new-50 transition-colors hover:text-orange-200"
                    aria-label="Edit note"
                  >
                    ✎
                  </button>
                  <button
                    type="button"
                    onClick={() => handleRemove(slug)}
                    className="rounded p-1 font-mono text-xs text-gray-new-50 transition-colors hover:text-red-400"
                    aria-label="Remove note"
                  >
                    ✕
                  </button>
                </div>
                <p className="t-sm whitespace-pre-wrap text-gray-new-80">{n.text}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </>
  );
};

export default MarginNotes;
