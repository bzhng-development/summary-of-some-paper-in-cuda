# papers-ui

SvelteKit frontend for the paper scorer. Browse the scored HF daily papers, filter by score/date/tag, mark interested, add papers by URL.

Replaces an old 58 MB monolithic HTML viewer.

## Stack

- SvelteKit 2 + Svelte 5 (runes) + TypeScript
- Vite 7
- Data comes from `paper-server` (FastAPI, `src/paper_pipeline/ingest/paper_server.py`) via `/api/papers`, `/interested`, `/interested-ids`, `/add-paper`
- Vite dev server proxies those paths to the backend (default `http://localhost:8787`, override with `PAPER_SERVER_URL`)

## Run (two processes)

```bash
# 1. Start the Python backend (from repo root)
uv run paper-server --port 8787

# 2. Start the Svelte dev server (this directory)
pnpm install   # first time only
pnpm dev
```

Open **http://localhost:5173**. Both processes must be running — the Svelte UI fetches all data from the FastAPI backend.

Point at a remote backend:

```bash
PAPER_SERVER_URL=http://other-host:8787 pnpm dev
```

## Build

```bash
pnpm build     # production bundle -> .svelte-kit/output
pnpm preview   # serve the built app locally
```

Deployment adapter is `@sveltejs/adapter-auto`; swap in `@sveltejs/adapter-static` (already in devDependencies) if you want a static export to drop behind any HTTP server.

## Typecheck

```bash
pnpm check
```

Runs `svelte-check` against the TS config. Expect 0 errors / 0 warnings.

## Layout

```
src/
├── routes/
│   ├── +layout.svelte   # page shell, fonts, Material Symbols
│   └── +page.svelte     # the whole paper viewer (filters + list)
├── lib/
│   └── types.ts         # `Paper` type mirroring all_scored.json rows
├── app.html             # HTML shell with Google Sans + Material Symbols links
└── app.d.ts
vite.config.ts           # dev-server proxy config (the important piece)
```

Single-file page on purpose — easier to edit, no premature componentization. Split when it actually hurts.
