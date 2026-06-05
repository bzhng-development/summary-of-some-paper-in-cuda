# Where the data lives

The large `local_data/` corpus (regen jsonls, `papers.db`/`papers_unified.db`,
backups — the paper-tagging pipeline's intelligence) is **not committed to git**.
It lives in the private Hugging Face Storage bucket **`vincentzed-hf/data`** under
the **`cuda/`** prefix, indexed by
[`data-bucket.manifest.ndjson`](./data-bucket.manifest.ndjson)
(one line per file: `path` → `bytes` → `sha256` → `hf://` URI).

## Restore on a fresh clone

```bash
hf auth login
uv run scripts/data_bucket.py status
uv run scripts/data_bucket.py pull                       # everything
uv run scripts/data_bucket.py pull --include local_data  # subset
```

## Re-upload after regenerating data

```bash
uv run scripts/data_bucket.py push --bucket vincentzed-hf/data --prefix cuda
```

Note: `local_data/` can also be rebuilt from committed sources via `sync_db.py`;
the bucket preserves the exact snapshots + backups so nothing is lost. Data stays
gitignored; only this manifest + `scripts/data_bucket.py` are committed.
