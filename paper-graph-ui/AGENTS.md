# Notes for AI agents

## Next.js version

This project uses Next.js 16 (App Router, Turbopack, React 19). APIs differ from older
versions that may dominate training data — `params` is an async Promise, route segments
ship as async server components by default, and client components must declare
`'use client'` at the top of the file. When in doubt read the relevant guide in
`node_modules/next/dist/docs/` before editing route files.

## Project shape

### App-local code (edit freely)

- `src/app/layout.jsx`, `src/app/page.jsx`, `src/app/c/[course]/[chapter]/**`
  — routes.
- `src/app/_components/` — project-local UI (e.g. `site-header.jsx`). Put new
  app-specific components here, not under `src/components/`.
- `src/lib/psets.js` — filesystem reader; all reads go through
  `React.cache`-wrapped helpers and a `safeJoin` that blocks path traversal.
- `src/lib/headings.js` — markdown → TOC items (matches `AnchorHeading`'s slug
  algorithm).
- `src/content/psets/<course>/<chapter>/<pset>.md` — content.
- `src/styles/app.css` — app-local CSS overrides.
- `src/config/docs-icons-config.js` — 4-line stub for Neon's build-time
  tech-icon map. Empty by default. Populate with entries like
  `{ foo: { lightIconPath: '/path.svg', darkIconPath: null } }` if you want
  `TechCards` / `PromptCards` to render icons.

### Neon verbatim (do NOT modify)

These are bit-identical copies from the public `neondatabase/website` repo.
Any local modification becomes an upgrade conflict. Add overrides to
`src/styles/app.css` or wrap components in an app-local file instead of
patching.

- `src/components/` — UI primitives + MDX tags
- `src/styles/` — Neon's full Tailwind v4 stylesheet stack (except `app.css`)
- `src/hooks/`, `src/utils/`, `src/contexts/`, `src/constants/`
- `src/icons/`, `src/fonts/`
- `src/lib/shiki.js`, `src/lib/rehype-code-props.js`
- `tailwind.config.js`, `postcss.config.js`, `empty.js`

## Component wiring

All Neon UI primitives available as MDX tags are registered in
`src/app/c/[course]/[chapter]/[pset]/page.jsx` → `mdxComponents`. To add a
new one:

1. Add an import at the top of the pset reader.
2. Add it to the `mdxComponents` map.
3. Run `tsx scripts/compute-closure.ts` to verify the transitive import
   graph stays resolvable (every referenced file must exist on disk).
4. Restart the dev server; if it errors on `'use client'` missing or a
   missing transitive dep (e.g. `config/foo`), either add a stub or skip the
   component. Two known-broken components in Neon's tree are `BlinkingText`
   and `TypingText` (missing `'use client'` marker).

## Tooling

- pnpm (see `pnpm-workspace.yaml`).
- Tailwind v4 via Neon's `tailwind.config.js`.
- MDX via `next-mdx-remote/rsc`.
- Shiki for code highlighting (v1.x — Neon's `lib/shiki.js` uses the old
  `getHighlighter` API, not v2's `createHighlighter`).
- KaTeX for math (`remark-math` + `rehype-katex`, plus `styles/app.css` has
  a horizontal-overflow override for `.katex-display`).
- SVGR wired via `next.config.ts` for `*.inline.svg` files.

## Dead-code checker

`tsx scripts/compute-closure.ts` prints every file reachable from the app's
entry points, walking ES import / require / `@import` / `@config`. Useful to
run after adding or removing a component and before shipping.

Known quirks:
- Resolves aliases: `components/`, `hooks/`, `utils/`, `lib/`, `constants/`,
  `contexts/`, `icons/`, `images/`, `styles/`, `app/`, `fonts/`, `config/`.
  Add to the list in `scripts/compute-closure.ts` if you introduce new
  top-level dirs under `src/`.
- Follows `.jsx`/`.js`/`.ts`/`.tsx`/`.css` files; leaves other file types
  (svg, png, woff2, md) in the closure but doesn't descend into them.
