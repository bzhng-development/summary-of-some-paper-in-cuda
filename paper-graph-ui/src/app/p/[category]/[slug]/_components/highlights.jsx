'use client';

import { useCallback, useEffect, useId, useRef, useState } from 'react';

import { STORAGE_EVENTS, STORAGE_KEYS } from 'lib/storage-keys';

// Local-first highlight system. Persists {paperId → [highlight]} in
// localStorage. Each highlight stores a normalized text snippet plus the
// surrounding 30 chars on each side so we can re-anchor it after the
// markdown re-renders without relying on fragile DOM offsets.
//
// Highlighting strategy:
//   1. On selection within `containerRef`, show a tiny floating "Highlight"
//      button near the selection.
//   2. Save → walk the container's text nodes, find the snippet, wrap in a
//      <mark> via Range surroundContents-equivalent (we re-find the text
//      and inject spans).
//   3. On mount, re-apply all stored highlights for this paper.

const COLOR = 'rgba(0, 229, 153, 0.28)';

function readAll() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.highlights);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function writeAll(map) {
  try {
    localStorage.setItem(STORAGE_KEYS.highlights, JSON.stringify(map));
    window.dispatchEvent(new CustomEvent(STORAGE_EVENTS.highlights));
  } catch {
    // ignore
  }
}

// Walks all text nodes inside root, accumulating offsets, and returns the
// (startNode, startOffset, endNode, endOffset) for the first occurrence of
// `snippet` near the given surrounding context. Returns null if not found.
function findRange(root, snippet, before, after) {
  if (!snippet) return null;
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentNode && !node.parentNode.closest('mark[data-pg-hl]')
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_REJECT,
  });
  const chunks = [];
  let full = '';
  while (walker.nextNode()) {
    const n = walker.currentNode;
    chunks.push({ node: n, start: full.length, end: full.length + n.nodeValue.length });
    full += n.nodeValue;
  }
  const needle = `${before}${snippet}${after}`;
  let i = full.indexOf(needle);
  if (i === -1) {
    i = full.indexOf(snippet);
    if (i === -1) return null;
  } else {
    i += before.length;
  }
  const startGlobal = i;
  const endGlobal = i + snippet.length;
  let startNode = null;
  let startOff = 0;
  let endNode = null;
  let endOff = 0;
  for (const c of chunks) {
    if (!startNode && startGlobal >= c.start && startGlobal <= c.end) {
      startNode = c.node;
      startOff = startGlobal - c.start;
    }
    if (endGlobal >= c.start && endGlobal <= c.end) {
      endNode = c.node;
      endOff = endGlobal - c.start;
      break;
    }
  }
  if (!startNode || !endNode) return null;
  const range = document.createRange();
  try {
    range.setStart(startNode, startOff);
    range.setEnd(endNode, endOff);
  } catch {
    return null;
  }
  return range;
}

function applyHighlight(range, id) {
  // Split into per-text-node wraps so we never cross element boundaries.
  const fragments = [];
  const iter = document.createTreeWalker(range.commonAncestorContainer, NodeFilter.SHOW_TEXT);
  while (iter.nextNode()) {
    const node = iter.currentNode;
    if (!range.intersectsNode(node)) continue;
    const start = node === range.startContainer ? range.startOffset : 0;
    const end = node === range.endContainer ? range.endOffset : node.nodeValue.length;
    if (end <= start) continue;
    fragments.push({ node, start, end });
  }
  for (const { node, start, end } of fragments.reverse()) {
    const text = node.nodeValue;
    const before = text.slice(0, start);
    const middle = text.slice(start, end);
    const after = text.slice(end);
    const parent = node.parentNode;
    if (!parent) continue;
    const mark = document.createElement('mark');
    mark.setAttribute('data-pg-hl', id);
    mark.style.backgroundColor = COLOR;
    mark.style.color = 'inherit';
    mark.style.borderRadius = '2px';
    mark.style.padding = '0 1px';
    mark.textContent = middle;
    if (after) parent.insertBefore(document.createTextNode(after), node.nextSibling);
    parent.insertBefore(mark, node.nextSibling);
    if (before) node.nodeValue = before;
    else parent.removeChild(node);
  }
}

function removeHighlight(root, id) {
  const marks = root.querySelectorAll(`mark[data-pg-hl="${id}"]`);
  for (const m of marks) {
    const parent = m.parentNode;
    if (!parent) continue;
    while (m.firstChild) parent.insertBefore(m.firstChild, m);
    parent.removeChild(m);
    parent.normalize();
  }
}

const Highlights = ({ paperId, containerRef }) => {
  const [list, setList] = useState([]);
  const [popup, setPopup] = useState(null); // { x, y, range, snippet, before, after }
  const buttonId = useId();
  const popupRef = useRef(null);

  // Load + re-apply existing highlights when the container is mounted.
  const reapply = useCallback(() => {
    const root = containerRef.current;
    if (!root) return;
    // strip prior marks
    for (const m of root.querySelectorAll('mark[data-pg-hl]')) {
      const parent = m.parentNode;
      if (!parent) continue;
      while (m.firstChild) parent.insertBefore(m.firstChild, m);
      parent.removeChild(m);
      parent.normalize();
    }
    const all = readAll();
    const items = all[paperId] ?? [];
    setList(items);
    for (const h of items) {
      const range = findRange(root, h.snippet, h.before ?? '', h.after ?? '');
      if (range) applyHighlight(range, h.id);
    }
  }, [paperId, containerRef]);

  useEffect(() => {
    reapply();
  }, [reapply]);

  // Selection listener — show the floating "Highlight" button.
  useEffect(() => {
    const root = containerRef.current;
    if (!root) return undefined;
    function onUp() {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) {
        setPopup(null);
        return;
      }
      const range = sel.getRangeAt(0);
      if (!root.contains(range.commonAncestorContainer)) {
        setPopup(null);
        return;
      }
      const snippet = sel.toString().trim();
      if (snippet.length < 3 || snippet.length > 600) {
        setPopup(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      // grab a small context window for re-anchoring
      const fullText = root.innerText || '';
      const idx = fullText.indexOf(snippet);
      const before = idx > 0 ? fullText.slice(Math.max(0, idx - 30), idx) : '';
      const after = idx >= 0 ? fullText.slice(idx + snippet.length, idx + snippet.length + 30) : '';
      setPopup({
        x: rect.left + rect.width / 2,
        y: rect.top - 8,
        snippet,
        before,
        after,
      });
    }
    document.addEventListener('mouseup', onUp);
    document.addEventListener('touchend', onUp);
    return () => {
      document.removeEventListener('mouseup', onUp);
      document.removeEventListener('touchend', onUp);
    };
  }, [containerRef]);

  // Dismiss popup on outside click.
  useEffect(() => {
    if (!popup) return undefined;
    function onDown(e) {
      if (popupRef.current && !popupRef.current.contains(e.target)) {
        setPopup(null);
      }
    }
    document.addEventListener('pointerdown', onDown);
    return () => document.removeEventListener('pointerdown', onDown);
  }, [popup]);

  const save = useCallback(() => {
    if (!popup) return;
    const id = `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const next = {
      id,
      snippet: popup.snippet,
      before: popup.before,
      after: popup.after,
      createdAt: Date.now(),
    };
    const all = readAll();
    all[paperId] = [...(all[paperId] ?? []), next];
    writeAll(all);
    setPopup(null);
    setList(all[paperId]);
    // Re-apply so the new mark renders.
    requestAnimationFrame(reapply);
    // Clear the browser selection so the popup closes cleanly.
    window.getSelection()?.removeAllRanges();
  }, [popup, paperId, reapply]);

  const handleRemove = useCallback(
    (id) => {
      const all = readAll();
      all[paperId] = (all[paperId] ?? []).filter((h) => h.id !== id);
      if (all[paperId].length === 0) delete all[paperId];
      writeAll(all);
      setList(all[paperId] ?? []);
      const root = containerRef.current;
      if (root) removeHighlight(root, id);
    },
    [paperId, containerRef]
  );

  return (
    <>
      {popup ? (
        <div
          ref={popupRef}
          className="pg-no-swipe fixed z-50"
          style={{
            left: `${popup.x}px`,
            top: `${popup.y}px`,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <button
            type="button"
            id={buttonId}
            onClick={save}
            className="inline-flex items-center gap-1.5 rounded-md border border-primary-1/60 bg-black-new px-3 py-1.5 font-mono text-xs tracking-wider text-primary-1 uppercase shadow-lg"
          >
            <span aria-hidden>✎</span>
            Highlight
          </button>
        </div>
      ) : null}
      {list.length > 0 ? <SavedHighlights items={list} onRemove={handleRemove} /> : null}
    </>
  );
};

const SavedHighlights = ({ items, onRemove }) => (
  <section className="mt-8">
    <h2 className="t-sm mb-3 font-mono tracking-[0.2em] text-gray-new-50 uppercase">
      Your highlights · {items.length}
    </h2>
    <ul className="flex flex-col gap-2">
      {items.map((h) => (
        <li
          key={h.id}
          className="group flex items-start gap-3 rounded-lg border border-primary-1/20 bg-primary-1/5 px-3 py-2"
        >
          <span className="t-sm flex-1 italic text-gray-new-80">"{h.snippet}"</span>
          <button
            type="button"
            onClick={() => onRemove(h.id)}
            className="rounded p-1 font-mono text-xs text-gray-new-50 transition-colors hover:text-red-400"
            aria-label="Remove highlight"
          >
            ✕
          </button>
        </li>
      ))}
    </ul>
  </section>
);

export default Highlights;
