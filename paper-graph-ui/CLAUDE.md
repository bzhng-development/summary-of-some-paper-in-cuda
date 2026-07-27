@README.md

# What changed since AGENTS.md / README.md were written

Those two are accurate on stack + architecture but **stale on scale and on which features actually shipped**. Treat the list below as the current truth.

## Scale

`src/content/papers/` holds **2,208 markdown files (~329 MB)**, not ~500. The build pre-renders every detail page via `generateStaticParams`.

## Vercel 250 MB function-size workaround

`next.config.ts` sets `outputFileTracingExcludes: { "*": ["src/content/papers/**/*"] }`. The runtime serverless function never re-reads paper bodies — they're baked into static HTML at build time — but Next's tracer would otherwise bundle all 329 MB. **Do not remove this exclusion** unless you also stop using `generateStaticParams` for `/p/[category]/[slug]`.

## Multi-tag taxonomy (a paper can live in multiple `/c/[category]` indexes)

`src/lib/neon-metadata.generated.json` carries `tag_categories_v2: string[]` per paper, populated by the V4-Pro multi-tagger in the parent repo. `scripts/build-paper-graph.ts` reads it as `tagCategories` and emits it into `graph.generated.json`; `/c/[category]` filters surface a paper whenever the category is in that array (falls back to the single-tag `category` field when `tag_categories_v2` is empty).

## All / Merit / Company filter

`src/app/_components/merit-only-toggle.jsx` is a 3-way toggle persisted in `localStorage` key **`pg.paper-filter.v2`** (values: `all` / `merit` / `company`). Pair: `src/styles/app.css` carries `.pg-company-only [data-company-only="false"] { display: none !important; }` rules so the filter applies via CSS, not React re-renders.

## Graph wiring — scaling-laws domain + cross-domain bridges

A recent commit (`8b28067 graph: wire scaling-laws into CATEGORY_META + DOMAIN_BRIDGES`) added **scaling-laws** as a first-class domain in `CATEGORY_META` and registered new entries in `DOMAIN_BRIDGES` for cross-domain edges on `/graph`. When adding another domain, mirror that pattern: register in `CATEGORY_META` (color, label, sort), then add bridge entries so the SVG has edges to render. The bridges affect `/topic/[topic]` thread suggestions too — not just the graph view.

## Components that the README's "Brainstormed but unbuilt" list claims are pending — but are actually shipped

`completion-ring.jsx`, `domain-progress.jsx`, `mark-read.jsx`, `queue-button.jsx`, `queue-pill.jsx`, `progress-strip.jsx`, `read-dot.jsx`, `recent-reads.jsx`, `record-visit.jsx`, `resume-banner.jsx`, `scroll-tracker.jsx`, `streak-counter.jsx`, `surprise-me.jsx`, `whats-next.jsx`. The remaining items in that list (swipe gestures, inline highlights, margin notes, radial back-nav, offline PWA, math-overflow indicator) are still genuinely unbuilt.

## Refreshing data — current flow

```bash
# from this dir, paper-graph-ui/
uv run python scripts/pull-neon-metadata.py     # needs $DATABASE_URL; writes src/lib/neon-metadata.generated.json
tsx scripts/build-paper-graph.ts              # writes src/lib/graph.generated.json (multi-tag aware)
pnpm build                                       # next build pre-renders everything
vercel --prod --yes                              # deploy
```

`pnpm build` alone won't pick up Neon changes — you must re-run the two scripts first so the `*.generated.json` files are fresh on disk before Next reads them.

## Search ownership

The compact `paper-index.generated.json` plus `paper-filter.js` own only
in-browser list filtering. Cross-corpus ranked search is implemented once in
the sibling binutils pipeline's `pipeline.company_blogs.search` module, which
adapts this app's `graph.generated.json` alongside the company-blog exports.

From the binutils `classification/pipeline/` directory:

```bash
uv run company-content-search search "expert parallelism" --limit 20
```

Do not reintroduce a server-only `searchPapers` helper in `src/lib/papers.js`;
it was unused and duplicated the browser filter with fewer fields. If the
generated graph changes, preserve the fields listed in the parent
`CLAUDE.md`'s unified-search contract or update the adapter and tests together.

## What's stored client-side (current localStorage keys)

- `pg.recent.v1` — last 16 papers opened
- `pg.read.v1` — set of paper IDs marked read
- `pg.streak.v1` — daily reading streak counter
- `pg.paper-filter.v2` — all/merit/company filter (NEW since README)
- `pg.queue.v1` — reading queue / Up Next pill (NEW since README)
