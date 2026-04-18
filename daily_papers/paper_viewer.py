#!/usr/bin/env python3
"""
paper_viewer.py — Generate a compact HTML viewer for scored HF daily papers.

Reads all_scored.json, groups papers by date (extracted from arxiv_id),
and renders an interactive HTML page where you can click papers to select them.
Selected paper titles are copied to clipboard via a button.

Usage:
    python paper_viewer.py --json ./papers_out/all_scored.json --out ./papers_out/viewer.html
    python paper_viewer.py --json ./papers_out/all_scored.json --out ./papers_out/viewer.html --min-score 5
"""

from __future__ import annotations

import argparse
import json
import html
import re
from pathlib import Path


def arxiv_id_to_date(arxiv_id: str) -> str:
    """Extract approximate submission date from arxiv id (YYMM prefix)."""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if m:
        yy, mm = m.groups()
        return f"20{yy}-{mm}"
    return "unknown"


def generate_html(papers: list[dict], min_score: int = 0) -> str:
    papers = [p for p in papers if p.get("score", 0) >= min_score]
    papers.sort(key=lambda p: (-p.get("score", 0),))

    # Group by YYMM date
    groups: dict[str, list[dict]] = {}
    for p in papers:
        key = arxiv_id_to_date(p["arxiv_id"])
        groups.setdefault(key, []).append(p)

    # Sort groups by date descending
    sorted_groups = sorted(groups.items(), key=lambda x: x[0], reverse=True)

    paper_rows = []
    for date_key, group in sorted_groups:
        group.sort(key=lambda p: -p.get("score", 0))
        rows = []
        for p in group:
            title_esc = html.escape(p["title"])
            reason_esc = html.escape(p.get("reason", ""))
            similar_esc = html.escape(p.get("similar_paper", "NONE"))
            arxiv_id = html.escape(p["arxiv_id"])
            score = p.get("score", 0)
            upvotes = p.get("upvotes", 0)
            gh = p.get("github") or ""
            gh_stars = p.get("github_stars") or 0
            # Format authors with affiliations
            affiliations = p.get("affiliations") or {}
            author_parts = []
            for name in p.get("authors", [])[:3]:
                affs = affiliations.get(name)
                if affs:
                    author_parts.append(f"{name} ({', '.join(affs)})")
                else:
                    author_parts.append(name)
            if len(p.get("authors", [])) > 3:
                author_parts.append(f"+{len(p['authors']) - 3}")
            authors = ", ".join(author_parts)

            org = p.get("org_fullname") or p.get("organization") or ""
            categories = ", ".join(p.get("categories", []))
            tag_category = p.get("tag_category") or ""
            comment = p.get("arxiv_comment") or ""
            journal = p.get("journal_ref") or ""

            is_interested = p.get("_interested", False)
            score_class = "high" if score >= 8 else "mid" if score >= 5 else "low"
            interested_class = " interested" if is_interested else ""

            gh_link = ""
            if gh:
                gh_link = f'<a href="{html.escape(gh)}" target="_blank" class="gh-link" onclick="event.stopPropagation()">code({gh_stars}⭐)</a>'

            similar_html = (
                f"<div class='similar'>Similar: {similar_esc}</div>" if similar_esc and similar_esc != "NONE" else ""
            )
            org_html = f'<span class="org">{html.escape(org)}</span>' if org else ""
            cats_html = f'<span class="cats">{html.escape(categories)}</span>' if categories else ""
            tag_html = f'<span class="tag-cat">{html.escape(tag_category)}</span>' if tag_category else ""
            comment_html = f'<div class="comment">{html.escape(comment)}</div>' if comment else ""
            journal_html = f'<span class="journal">{html.escape(journal)}</span>' if journal else ""
            interested_badge = '<span class="interested-badge">interested</span>' if is_interested else ""
            # NOTE: the per-row embedded paper data used to live here as
            # `data-paper='<~3KB JSON>'`, which ballooned the HTML to ~40 MB
            # and made initial parse + scroll janky. It now lives in a single
            # `<script type="application/json" id="papers-data">` block below,
            # keyed by arxiv_id. Rows carry only the minimum dataset needed
            # for filtering (authors, reason, tag, score, interested flag).
            row = (
                f'<div class="paper {score_class}{interested_class}" data-title="{title_esc}" data-arxiv="{arxiv_id}" data-authors="{html.escape(authors)}" data-tag="{html.escape(tag_category)}" data-score="{score}" data-reason="{reason_esc}" onclick="toggleSelect(this)">\n'
                f'  <div class="paper-header">\n'
                f'    <span class="score">{score}/10</span>\n'
                f"    {tag_html}\n"
                f'    <span class="title">{title_esc}</span>\n'
                f'    <span class="upvotes">{upvotes}↑</span>\n'
                f"    {interested_badge}\n"
                f"  </div>\n"
                f'  <div class="paper-meta">\n'
                f'    <span class="authors">{html.escape(authors)}</span>\n'
                f"    {org_html}\n"
                f"    {cats_html}\n"
                f"    {journal_html}\n"
                f'    <a href="https://arxiv.org/abs/{arxiv_id}" target="_blank" class="arxiv-link" onclick="event.stopPropagation()">arXiv</a>\n'
                f"    {gh_link}\n"
                f"  </div>\n"
                f"  {comment_html}\n"
                f'  <div class="paper-reason">{reason_esc}</div>\n'
                f"  {similar_html}\n"
                f"</div>"
            )
            rows.append(row)

        paper_rows.append(
            f'<div class="date-group"><h2>{date_key} ({len(group)} papers)</h2>\n' + "\n".join(rows) + "\n</div>"
        )

    all_rows = "\n".join(paper_rows)
    total = len(papers)

    # Collect unique tag categories for filter dropdown
    all_tags = sorted({p.get("tag_category") for p in papers if p.get("tag_category")})
    tag_options = "\n".join(f'      <option value="{html.escape(t)}">{html.escape(t)}</option>' for t in all_tags)

    # Build the single paper-data blob that used to live on each row.
    # Strips the fields the server doesn't need and keys the rest by arxiv_id
    # so markInterested() can look up everything with one O(1) lookup instead
    # of parsing an HTML attribute per row.
    paper_blob = {
        p["arxiv_id"]: {
            k: v
            for k, v in p.items()
            if k not in ("_interested", "reason", "similar_paper", "score")
        }
        for p in papers
    }
    paper_blob_json = json.dumps(paper_blob, ensure_ascii=True, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Paper Viewer ({total} papers)</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: #f8f7f4;
    color: #2d2d2d;
    padding: 1rem;
    max-width: 960px;
    margin: 0 auto;
  }}
  .toolbar {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #f8f7f4;
    padding: 0.75rem 0;
    border-bottom: 1px solid #e0ddd5;
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex-wrap: wrap;
  }}
  .toolbar button {{
    padding: 0.4rem 1rem;
    border: 1px solid #c5c0b0;
    border-radius: 6px;
    background: #fff;
    cursor: pointer;
    font-size: 0.85rem;
    transition: background 0.15s;
  }}
  .toolbar button:hover {{ background: #eee; }}
  .toolbar .count {{ font-size: 0.85rem; color: #666; }}
  .toolbar input {{
    padding: 0.4rem 0.7rem;
    border: 1px solid #c5c0b0;
    border-radius: 6px;
    font-size: 0.85rem;
    width: 150px;
  }}
  .filter-row {{
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }}
  .filter-row label {{ font-size: 0.8rem; color: #666; }}
  .filter-row select {{
    padding: 0.3rem 0.5rem;
    border: 1px solid #c5c0b0;
    border-radius: 6px;
    font-size: 0.8rem;
  }}
  h2 {{
    font-size: 1rem;
    color: #555;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e0ddd5;
  }}
  .paper {{
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    border-radius: 6px;
    border: 1.5px solid transparent;
    cursor: pointer;
    /* Scope transitions to the two properties that actually change. */
    transition: background-color 0.12s, border-color 0.12s;
    background: #fff;
    /* Skip layout/paint for offscreen rows. Huge scroll win at 12k rows. */
    content-visibility: auto;
    contain-intrinsic-size: 0 80px;
  }}
  .paper:hover {{ background: #f0eee8; }}
  .paper.selected {{
    border-color: #5b8a72;
    background: #eef5f0;
  }}
  .paper-header {{
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
  }}
  .score {{
    font-weight: 700;
    font-size: 0.8rem;
    min-width: 3rem;
    text-align: center;
    padding: 0.1rem 0.3rem;
    border-radius: 4px;
  }}
  .high .score {{ background: #d4edda; color: #155724; }}
  .mid .score {{ background: #fff3cd; color: #856404; }}
  .low .score {{ background: #f0eded; color: #888; }}
  .title {{ font-weight: 600; font-size: 0.9rem; flex: 1; }}
  .upvotes {{ font-size: 0.8rem; color: #888; white-space: nowrap; }}
  .paper-meta {{
    font-size: 0.78rem;
    color: #777;
    margin-top: 0.2rem;
    display: flex;
    gap: 0.6rem;
    align-items: center;
  }}
  .paper-meta a {{
    color: #5b8a72;
    text-decoration: none;
  }}
  .paper-meta a:hover {{ text-decoration: underline; }}
  .org {{
    background: #e8e0f0;
    color: #5b3d8f;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 500;
    white-space: nowrap;
  }}
  .cats {{
    color: #888;
    font-size: 0.72rem;
    font-style: italic;
  }}
  .tag-cat {{
    background: #dbeafe;
    color: #1e40af;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.7rem;
    font-weight: 600;
    white-space: nowrap;
  }}
  .journal {{
    background: #d4edda;
    color: #155724;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.72rem;
    font-weight: 500;
    white-space: nowrap;
  }}
  .comment {{
    font-size: 0.73rem;
    color: #997a00;
    margin-top: 0.15rem;
    line-height: 1.3;
  }}
  .paper-reason {{
    font-size: 0.78rem;
    color: #666;
    margin-top: 0.2rem;
    line-height: 1.3;
  }}
  .similar {{
    font-size: 0.75rem;
    color: #888;
    font-style: italic;
    margin-top: 0.15rem;
  }}
  .interested {{
    opacity: 0.5;
  }}
  .interested:hover {{ opacity: 0.8; }}
  .interested-badge {{
    background: #2563eb;
    color: #fff;
    padding: 0.1rem 0.4rem;
    border-radius: 3px;
    font-size: 0.65rem;
    font-weight: 600;
    white-space: nowrap;
  }}
  .toolbar .btn-primary {{
    background: #2563eb;
    color: #fff;
    border-color: #2563eb;
  }}
  .toolbar .btn-primary:hover {{ background: #1d4ed8; }}
  .hidden {{ display: none !important; }}
  .toast {{
    position: fixed;
    bottom: 2rem;
    left: 50%;
    transform: translateX(-50%);
    background: #333;
    color: #fff;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    font-size: 0.85rem;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }}
  .toast.show {{ opacity: 1; }}
</style>
</head>
<body>

<div class="toolbar">
  <button class="btn-primary" onclick="markInterested()">Mark Interested</button>
  <button onclick="copySelected()">Copy</button>
  <button onclick="clearAll()">Clear</button>
  <span class="count" id="sel-count">0 selected</span>
  <input type="text" id="search" placeholder="Title..." oninput="filterPapers()">
  <input type="text" id="search-author" placeholder="Author..." oninput="filterPapers()">
  <input type="text" id="search-desc" placeholder="Description..." oninput="filterPapers()">
  <span style="border-left:1px solid #c5c0b0;height:1.5rem"></span>
  <input type="text" id="add-url" placeholder="arxiv/HF URL or ID..." style="width:220px">
  <button onclick="addPaper()">Add to DB</button>
  <div class="filter-row">
    <label>Min score:</label>
    <select id="min-score" onchange="filterPapers()">
      <option value="0">All</option>
      <option value="3">3+</option>
      <option value="5">5+</option>
      <option value="6">6+</option>
      <option value="7">7+</option>
      <option value="8">8+</option>
    </select>
  </div>
  <div class="filter-row">
    <label>Category:</label>
    <select id="tag-filter" onchange="filterPapers()">
      <option value="">All</option>
{tag_options}
    </select>
  </div>
</div>

{all_rows}

<script type="application/json" id="papers-data">{paper_blob_json}</script>

<div class="toast" id="toast"></div>

<script>
// Parse the embedded JSON blob once. This replaces the previous per-row
// `data-paper` attribute that ballooned the HTML to ~40 MB and made initial
// parse + hover repaints unbearable.
const PAPERS_BY_ID = JSON.parse(document.getElementById('papers-data').textContent);

// Build a flat filter index once, at load. filterPapers() used to call
// querySelector and read .dataset on every keystroke across 12k rows — this
// pre-walks the DOM once and stores the minimum scalars we need.
const ROWS = Array.from(document.querySelectorAll('.paper')).map(el => ({{
  el,
  title: el.dataset.title.toLowerCase(),
  authors: (el.dataset.authors || '').toLowerCase(),
  reason: (el.dataset.reason || '').toLowerCase(),
  score: parseInt(el.dataset.score, 10) || 0,
  tag: el.dataset.tag || '',
}}));
const DATE_GROUPS = Array.from(document.querySelectorAll('.date-group'));

function toggleSelect(el) {{
  el.classList.toggle('selected');
  updateCount();
}}

function updateCount() {{
  const n = document.querySelectorAll('.paper.selected').length;
  document.getElementById('sel-count').textContent = n + ' selected';
}}

function markInterested() {{
  const items = document.querySelectorAll('.paper.selected:not(.interested)');
  if (!items.length) {{ showToast('Nothing to mark (all selected already interested)'); return; }}
  const papers = Array.from(items).map(el => {{
    const full = PAPERS_BY_ID[el.dataset.arxiv];
    return full || {{arxiv_id: el.dataset.arxiv, title: el.dataset.title}};
  }});
  fetch('/interested', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{papers}})
  }}).then(r => r.json()).then(data => {{
    const n = data.marked ? data.marked.length : 0;
    showToast('Marked ' + n + ' as interested');
    items.forEach(el => {{
      el.classList.add('interested');
      el.classList.remove('selected');
      const hdr = el.querySelector('.paper-header');
      if (hdr && !hdr.querySelector('.interested-badge')) {{
        const badge = document.createElement('span');
        badge.className = 'interested-badge';
        badge.textContent = 'interested';
        hdr.appendChild(badge);
      }}
    }});
    updateCount();
  }}).catch(err => {{
    showToast('Error: ' + err.message);
  }});
}}

function copySelected() {{
  const items = document.querySelectorAll('.paper.selected');
  if (!items.length) {{ showToast('Nothing selected'); return; }}
  const lines = Array.from(items).map(el => {{
    const title = el.dataset.title;
    const arxiv = el.dataset.arxiv;
    return title + ' (https://arxiv.org/abs/' + arxiv + ')';
  }});
  navigator.clipboard.writeText(lines.join('\\n')).then(() => {{
    showToast('Copied ' + items.length + ' papers');
  }});
}}

function clearAll() {{
  document.querySelectorAll('.paper.selected').forEach(el => el.classList.remove('selected'));
  updateCount();
}}

// filterPapers runs against the pre-built ROWS cache (no per-row DOM reads)
// and batches the visibility writes into a single animation frame. Debounced
// below so three search boxes don't stampede the main thread per keystroke.
let _filterRafId = 0;
function filterPapersNow() {{
  const q = document.getElementById('search').value.toLowerCase();
  const qAuthor = document.getElementById('search-author').value.toLowerCase();
  const qDesc = document.getElementById('search-desc').value.toLowerCase();
  const minScore = parseInt(document.getElementById('min-score').value, 10);
  const tagFilter = document.getElementById('tag-filter').value;

  // Compute match bits synchronously, apply writes under rAF.
  const writes = new Array(ROWS.length);
  for (let i = 0; i < ROWS.length; i++) {{
    const r = ROWS[i];
    const match = (!q || r.title.includes(q))
      && (!qAuthor || r.authors.includes(qAuthor))
      && (!qDesc || r.reason.includes(qDesc))
      && r.score >= minScore
      && (!tagFilter || r.tag === tagFilter);
    writes[i] = match;
  }}
  cancelAnimationFrame(_filterRafId);
  _filterRafId = requestAnimationFrame(() => {{
    let visibleGroup = new Map();
    for (let i = 0; i < ROWS.length; i++) {{
      const r = ROWS[i];
      const match = writes[i];
      r.el.classList.toggle('hidden', !match);
      // Track per-date-group visibility by the row's parent.
      if (match) {{
        const g = r.el.parentElement;
        visibleGroup.set(g, (visibleGroup.get(g) || 0) + 1);
      }}
    }}
    for (const g of DATE_GROUPS) {{
      g.classList.toggle('hidden', (visibleGroup.get(g) || 0) === 0);
    }}
  }});
}}

let _filterDebounce = 0;
function filterPapers() {{
  clearTimeout(_filterDebounce);
  _filterDebounce = setTimeout(filterPapersNow, 120);
}}

function addPaper() {{
  const input = document.getElementById('add-url');
  const url = input.value.trim();
  if (!url) {{ showToast('Paste an arxiv/HF URL or ID'); return; }}
  fetch('/add-paper', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{url}})
  }}).then(r => r.json()).then(data => {{
    if (data.error) {{ showToast('Error: ' + data.error); return; }}
    if (data.status === 'already_exists') {{ showToast(data.arxiv_id + ' already in DB'); return; }}
    showToast('Added ' + data.arxiv_id + (data.has_meta ? ' (with metadata)' : ''));
    input.value = '';
  }}).catch(err => showToast('Error: ' + err.message));
}}

function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 1500);
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate HTML paper viewer")
    parser.add_argument("--json", required=True, type=Path, help="Path to all_scored.json")
    parser.add_argument("--out", required=True, type=Path, help="Output HTML file")
    parser.add_argument("--min-score", type=int, default=0, help="Min score to include")
    args = parser.parse_args()

    papers = json.loads(args.json.read_text())
    html_content = generate_html(papers, args.min_score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_content)
    print(f"Wrote {args.out} ({len([p for p in papers if p.get('score', 0) >= args.min_score])} papers)")


if __name__ == "__main__":
    main()
