# Paper Graph

A mobile-first reading interface for ~500 LLM/systems/RL paper summaries.
Forked from `neon-clone-mini` (Neon design system + Next.js 16).

Live: https://paper-graph-ui.vercel.app

## Stack

- Next.js 16 (App Router, React 19, Turbopack, React Compiler on)
- Neon design system (components + Tailwind tokens, verbatim from
  `neondatabase/website`)
- Markdown via `react-markdown` + KaTeX + GFM (no MDX — paper bodies have
  raw `<1%`-style snippets that MDX rejects)
- pnpm
- No runtime database — all data baked at build time into
  `src/lib/graph.generated.json`

## Routes

- `/` — domain cards, topic chips, year bars, recently-opened, surprise-me
- `/timeline` — every paper by year, anchor nav, year deep-links
- `/c/[category]` — papers in one domain, grouped by year, related domains
- `/p/[category]/[slug]` — paper detail: markdown body, dependency tree,
  influence tree, prev/next, mark-read
- `/topic/[topic]` — cross-domain topic thread (e.g. "Speculative decoding")
- `/graph` — pan/zoom SVG of domains sized by paper count, with bridges

## Data pipeline

```
src/content/papers/<category>/<slug>.md          ← copied from ../docs/
                          │
                          ▼
scripts/build-paper-graph.ts
   reads every .md, parses arxiv ID + H1 title,
   classifies topics by title keyword,
   joins src/lib/neon-metadata.generated.json,
   emits src/lib/graph.generated.json (~1 MB)
                          │
                          ▼
src/lib/papers.js
   server-only data layer: listPapers, getPaper,
   buildDependencyTree, buildInfluenceTree,
   yearTimeline, listCategories
                          │
                          ▼
            (all routes consume from here)
```

## Refreshing metadata from Neon

The Vercel build runs in CI without DB access, so it consumes the cached
`src/lib/neon-metadata.generated.json` shipped in the repo. To refresh:

```bash
uv run python scripts/pull-neon-metadata.py   # needs $DATABASE_URL
git add src/lib/neon-metadata.generated.json
pnpm build                                    # regenerate graph.generated.json
vercel deploy --prod
```

## Local dev

```bash
pnpm install
pnpm dev                                       # → http://localhost:3000
```

## What's stored client-side (localStorage)

- `pg.recent.v1` — last 16 papers opened
- `pg.read.v1` — set of paper IDs marked read
- `pg.streak.v1` — daily reading streak counter

No accounts, no sync — entirely device-local.

## Brainstormed but unbuilt

A sub-agent reviewed the design and proposed 15 mobile-commute features.
Built: surprise-me, streak counter, mark-read, recently-opened, read-time
(stored as `wordCount`/`readTimeMin` on every paper). Pending:

1. Reading queue + "Up Next" pill
2. Auto-advance after end-of-paper
3. Swipe-left/right for prev/next paper
4. Session-resume banner with scroll-ratio
5. Domain completion rings on category cards
6. Inline highlights (text-select → save range to localStorage)
7. Margin notes anchored to headings
8. Radial back-navigation on graph long-press
9. Topic-thread progress bar
10. Offline PWA shell
11. Math-block horizontal-scroll indicator
12. "What's Next" digest card (score by topic overlap)

All are S/M difficulty and localStorage-only — no backend required.
