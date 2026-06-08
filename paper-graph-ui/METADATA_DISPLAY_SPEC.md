# Spec: surface ALL paper metadata in paper-graph-ui

## TL;DR for the agent

paper-graph-ui shows only a slice of the metadata that exists per paper. The parent
pipeline just enriched Neon with **citation data** (`cited_by_count`, `fwci`), **abstracts**,
and **DOIs** across ~13k papers, and many already-present fields (`primary_category`,
`tag_categories_v2`, `similar_paper`, `score_reason`, `published`) are baked but never rendered.
**Goal: thread every useful metadata field from Neon → the baked JSON → the paper detail UI.**

**Trust / don't trust:**
- **DO** edit the data scripts (`scripts/pull-neon-metadata.py`, `scripts/build-paper-graph.mjs`),
  the data layer (`src/lib/papers.js`), and app-local UI (`src/app/p/[category]/[slug]/page.jsx`,
  new components under `src/app/_components/`).
- **DO NOT** touch the Neon-verbatim dirs (`src/components/`, `src/styles/` except `app.css`,
  `src/hooks/`, `src/utils/`, etc.) — see AGENTS.md "Neon verbatim".
- **DO NOT bake the full-text `markdown` column into any JSON.** It averages ~68k chars across
  ~12k papers (~hundreds of MB). It would blow the 1 MB `graph.generated.json` and the 250 MB
  Vercel function limit (see CLAUDE.md). Paper *bodies* already render from
  `src/content/papers/<cat>/<slug>.md`. The OCR `markdown` is OUT OF SCOPE for this task.
- This is Next.js **16** (App Router, React 19, async `params`). Client components need
  `'use client'`. See AGENTS.md.

## Background: the data pipeline (no prior context assumed)

```
Neon "nextjs-ui_paper"
  │  scripts/pull-neon-metadata.py   (SELECT FIELDS WHERE id = ANY(...))  ← needs $DATABASE_URL
  ▼
src/lib/neon-metadata.generated.json
  │  scripts/build-paper-graph.mjs   (joins .md files + maps Neon meta -> node objects)
  ▼
src/lib/graph.generated.json         (~1 MB; one node per paper, ~25 keys today)
  │  src/lib/papers.js               (getPaper(category, slug) -> the node)
  ▼
src/app/p/[category]/[slug]/page.jsx (PaperMetaLine renders the chips)
```

Refresh sequence (must run in order; `pnpm build` alone won't pick up Neon changes):
```bash
uv run python scripts/pull-neon-metadata.py   # writes src/lib/neon-metadata.generated.json
node scripts/build-paper-graph.mjs            # writes src/lib/graph.generated.json
pnpm build                                    # next pre-renders every detail page
```

## Metadata inventory — the gap

Current node keys in `graph.generated.json`: arxivComment, arxivId, authors, category,
companyOnly, github, githubStars, id, isArxiv, month, organization, primaryCategory,
readTimeMin, relativePath, score, scoreReason, similarPaper, slug, tagCategories,
tagCategoryV2, title, topics, upvotes, wordCount, year.

| Neon column | pulled? (pull-neon-metadata.py) | baked? (build-paper-graph.mjs node) | displayed? | action |
|---|---|---|---|---|
| `cited_by_count` | ❌ no | ❌ no | ❌ no | **add** to pull + bake (`citedByCount`) + display |
| `fwci` | ❌ no | ❌ no | ❌ no | **add** to pull + bake (`fwci`) + display |
| `doi` | ❌ no | ❌ no | ❌ no | **add** to pull + bake (`doi`) + display (link to doi.org) |
| `abstract` | ✅ yes | ❌ no | ❌ no | **bake** (`abstract`) + display (collapsible) |
| `published` | ✅ yes | ❌ no | ❌ no | **bake** (`published`) + display (date) |
| `primary_category` | ✅ yes | ✅ `primaryCategory` | ❌ no | **display** |
| `tag_categories_v2` | ✅ yes | ✅ `tagCategories` | partial (routing only) | **display** as chips |
| `similar_paper` | ✅ yes | ✅ `similarPaper` | ❌ no | **display** |
| `score_reason` | ✅ yes | ✅ `scoreReason` | tooltip only | **display** (expandable) |
| `markdown` | ❌ (KEEP OUT) | ❌ (KEEP OUT) | ❌ | **out of scope — do NOT add** |

## Plan (ordered; each step has a deliverable)

**Step 1 — pull the missing columns.** In `scripts/pull-neon-metadata.py`, add to the
`FIELDS` list: `cited_by_count`, `fwci`, `doi`. (`abstract`, `published` already present.)
- Deliverable: re-running `pull-neon-metadata.py` writes a `neon-metadata.generated.json`
  whose entries carry `cited_by_count`/`fwci`/`doi` for OpenAlex-enriched papers.

**Step 2 — bake the fields into the node.** In `scripts/build-paper-graph.mjs`, in the node
object (the `// enriched from Neon` block, ~line 333), add:
`citedByCount: meta?.cited_by_count ?? null`, `fwci: meta?.fwci ?? null`,
`doi: meta?.doi ?? null`, `published: meta?.published ?? null`,
`abstract: meta?.abstract ?? null`.
- Deliverable: `node scripts/build-paper-graph.mjs` then
  `node -e "const g=require('./src/lib/graph.generated.json'); ..."` shows the new keys on
  nodes that have them.

**Step 3 — confirm passthrough.** Verify `src/lib/papers.js` `getPaper()` returns the node
wholesale (so new keys reach the page). If it picks fields explicitly, add the new ones.
- Deliverable: the detail page's `paper` object has `citedByCount`/`fwci`/`doi`/`published`/`abstract`.

**Step 4 — render in the detail page.** In `src/app/p/[category]/[slug]/page.jsx`:
- Extend `PaperMetaLine` (~line 222): add chips for **citations** (`📊 {citedByCount.toLocaleString()} cites`),
  **fwci** (`fwci {fwci.toFixed(1)}` when present), **published date**, **primaryCategory**,
  and a **DOI link** (`https://doi.org/{doi}`). Guard every field with a null check (most papers
  lack most fields).
- Add a metadata section/panel below the title rendering: **abstract** (collapsible, only if no
  markdown body or as an "Abstract" disclosure), **tagCategories** as chips, **similarPaper**,
  and **score_reason** as an expandable note (currently tooltip-only).
- Keep it consistent with the existing chip styling (font-mono, `text-gray-new-60`, border-left
  dividers). Put any new component under `src/app/_components/`, not `src/components/`.
- Deliverable: a paper with citation data (e.g. a high-cited Google/Meta paper) shows citations,
  fwci, doi, published, primary category, tags, and abstract in the UI.

**Step 5 — validate.** `node scripts/compute-closure.mjs` (import graph resolvable),
`pnpm build` succeeds, `pnpm dev` and visually confirm a high-cited paper page renders all new
fields and a sparse paper (no citations/abstract) renders cleanly with no empty chips.
- Deliverable: clean build + closure; screenshot/ό description of an enriched paper page.

## Definition of done
- All five missing fields (`citedByCount`, `fwci`, `doi`, `published`, `abstract`) flow Neon →
  pull → bake → page, and render with null-safety.
- The four pulled-but-unshown fields (`primaryCategory`, `tagCategories`, `similarPaper`,
  `score_reason`) are now visible.
- `markdown` full-text is NOT baked anywhere.
- `compute-closure.mjs` clean, `pnpm build` green, no empty/`null` chips on sparse papers.

## References
- Data scripts: `scripts/pull-neon-metadata.py` (FIELDS list ~line 28), `scripts/build-paper-graph.mjs` (node map ~line 333).
- Data layer: `src/lib/papers.js` (`getPaper` ~line 49).
- UI: `src/app/p/[category]/[slug]/page.jsx` (`PaperMetaLine` ~line 222, chips at ~228-264).
- Conventions: `AGENTS.md` (Neon-verbatim boundary, Next 16, compute-closure), `CLAUDE.md`
  (scale, 250 MB Vercel limit, refresh flow).
- Parent Neon schema (the new columns): `../src/paper_pipeline/core/neon_db.py` SCHEMA_COLUMNS.

---

# PHASE 2 — surface summary-less company/interested papers in the TEXT views (NOT the graph)

**Context:** the parent pipeline added ~18,300 papers (OpenAlex company-affiliated + `interested`)
that have rich metadata (title, authors, org, citations, fwci, doi, published, **abstract**) but
**no `docs/.md` summary**. Right now nodes come ONLY from `src/content/papers/**/*.md`, so these
papers are invisible. Goal: make them browsable in the **text/list/timeline/category/detail**
surfaces, hidden behind a toggle — and do **NOT** add them to the `/graph` SVG view (it would
blow up and is effectively unused).

**Scope of the set (do this filter in `scripts/pull-neon-metadata.py`):**
`is_only_important_because_of_company = true OR COALESCE(interested,0)=1`, restricted to rows
WITHOUT a summary. **~12,750 papers.** The OpenAlex 2000-2024 company backfill has ALREADY been
marked `is_only_important_because_of_company = true` in Neon, so this single clean filter captures
company + interested + the company backfill. NOT the generic HF feed.

**Decisions (from the user):** hide by default; bake the **full abstract**.

## Plan

1. **`scripts/pull-neon-metadata.py`** — today it pulls metadata only for arxiv_ids that already have
   a `.md`. Extend it to ALSO pull the scope set above (full `abstract`, plus the Phase-1 fields
   cited_by_count/fwci/doi). Output the union into `neon-metadata.generated.json`, each row tagged so
   the builder can tell which have summaries.

2. **`scripts/build-paper-graph.mjs`** — after the existing `.md` scan, emit a **summary-less node**
   for every scope-set paper that has no `.md`. Each carries: id, arxivId, title, authors,
   organization, citedByCount, fwci, doi, published, year, **abstract (full)**, a category (derive
   from `primary_category`/`tag_categories_v2`, else `uncategorized`), and **`hasSummary: false`**.
   Mark the existing `.md`-backed nodes `hasSummary: true`.

3. **`/graph` view stays lean** — wherever the graph SVG builds its node list
   (`src/app/graph/**`, `roadmap-canvas.jsx`), filter to `hasSummary !== false`. The summary-less
   papers must NOT appear in the graph (prevents blowup). Likewise keep them out of any
   dependency/influence-tree edges.

4. **Text/list views + toggle** — add a toggle "Show papers without summaries" (default **OFF**;
   localStorage, mirror the existing `merit-only-toggle.jsx` / `pg.paper-filter.v2` pattern under
   `src/app/_components/`). When OFF, list/timeline/category routes exclude `hasSummary===false`.
   When ON, they appear inline with their metadata (title, org, citations, published) and a clear
   "no summary yet" marker.

5. **Detail page `/p/[category]/[slug]`** — `generateStaticParams` now includes summary-less papers.
   For a `hasSummary===false` paper, render the Phase-1 `PaperMetaLine` (citations/fwci/doi/etc.) +
   the **full abstract**, and SKIP the markdown-body render (there is none). Show a small "Summary
   not generated yet" note.

## Watch items (flag, don't silently break)
- **Build scale:** this ~9×'s the page count (~2.2k → ~20k static pages) and adds ~18 MB to
  `graph.generated.json`. Confirm `pnpm build` still completes and `next.config.ts`
  `outputFileTracingExcludes` keeps the serverless function under 250 MB (abstracts live in the
  JSON, not in paper bodies — verify the JSON isn't bloating the function bundle; if it is, consider
  a separate `abstracts.generated.json` loaded only by the detail route).
- If build time is unacceptable, fall back to: list summary-less papers inline in list views WITHOUT
  individual detail routes (no `generateStaticParams` for them) — but the user asked for full
  abstract on detail, so prefer generating the pages unless the build genuinely breaks.

## Definition of done (Phase 2)
- Toggle OFF (default): UI looks like today. Toggle ON: ~18k company/interested no-summary papers
  appear in timeline/category/list views with metadata, deep-link to a detail page showing full
  abstract + citations, marked "no summary yet".
- `/graph` SVG unchanged (no summary-less nodes). `pnpm build` green. `compute-closure.mjs` clean.
