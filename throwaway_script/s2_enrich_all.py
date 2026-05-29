"""Enrich every arxiv paper in Neon with Semantic Scholar metadata.

Pulls the WHOLE response body for each paper via /paper/batch (up to 500 IDs
per request), writes the raw JSON to `local_data/s2_enrichment.jsonl` for
later reuse, and updates a handful of Neon columns (title, abstract,
authors, affiliations, published) where they were missing.

Why so many fields requested: user wants the data captured "for
convenience" so future workflows can re-read venues / TLDRs / citation
counts / fields-of-study / openAccess PDF URLs without re-hitting S2.

Scope is controlled by --year-min (default 2020). Re-run with --year-min 2015
once 2020+ lands.

Rate-limit: strictly 1.05s between requests (key allows 1 rps).

Usage:
    DATABASE_URL=... S2_API_KEY=s2k-... uv run python \\
        throwaway_script/s2_enrich_all.py --year-min 2020
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neon_db import NeonDB, TABLE


S2_BASE = "https://api.semanticscholar.org/graph/v1"
API_KEY = os.environ.get("S2_API_KEY")
if not API_KEY:
    raise SystemExit("S2_API_KEY env var required")
HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

BATCH_SIZE = 500  # /paper/batch hard limit
MIN_GAP_SEC = 1.05  # API key allows 1 rps — leave a tiny buffer
_last_request_at = 0.0


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_GAP_SEC:
        time.sleep(MIN_GAP_SEC - elapsed)
    _last_request_at = time.monotonic()


# Comprehensive field set — user wants "as much as possible" stored for
# downstream reuse. Skipped: citations/references (could be hundreds each per
# paper), embedding (768 floats), citationStyles (large formatted bibtex).
FIELDS = ",".join([
    "paperId",
    "corpusId",
    "externalIds",  # DOI, ARXIV, MAG, ACL, PMID, etc.
    "url",
    "title",
    "abstract",
    "venue",
    "publicationVenue",  # structured venue object
    "year",
    "publicationDate",
    "publicationTypes",
    "journal",
    "fieldsOfStudy",
    "s2FieldsOfStudy",
    "isOpenAccess",
    "openAccessPdf",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "authors.authorId",
    "authors.name",
    "authors.affiliations",
    "tldr",  # S2's auto-generated short summary
])


def post_batch(ids: list[str]) -> list[dict | None]:
    """POST /paper/batch with up to BATCH_SIZE IDs."""
    _throttle()
    for attempt in range(6):
        try:
            r = httpx.post(
                f"{S2_BASE}/paper/batch",
                params={"fields": FIELDS},
                json={"ids": ids},
                headers=HEADERS,
                timeout=60,
            )
        except Exception as e:
            logger.warning(f"  request error: {e}; retrying in 3s")
            time.sleep(3)
            continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"  429, sleeping {wait}s (attempt {attempt+1}/6)")
            time.sleep(wait)
            _throttle()
            continue
        if r.status_code in (502, 503, 504):
            time.sleep(5)
            continue
        if r.status_code == 400:
            logger.error(f"  400 bad request — body: {r.text[:300]}")
            return []
        logger.warning(f"  HTTP {r.status_code}: {r.text[:200]}")
        time.sleep(3)
    return []


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.is_file():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            aid = rec.get("_arxiv_id")
            if aid:
                seen.add(aid)
        except json.JSONDecodeError:
            continue
    return seen


def update_neon_from_s2(db: NeonDB, arxiv_id: str, body: dict) -> None:
    """Fill missing fields on the Neon row from the S2 response.

    Never OVERWRITES — only fills when the existing column is null/empty.
    User can decide later whether to bulk-overwrite from the raw JSONL.
    """
    if not body:
        return
    fields: dict = {}
    title = (body.get("title") or "").strip()
    if title:
        fields["title"] = title[:500]
    abstract = (body.get("abstract") or "").strip()
    if abstract:
        fields["abstract"] = abstract
    if body.get("publicationDate"):
        fields["published"] = body["publicationDate"]
    authors = body.get("authors") or []
    if authors:
        names = [a.get("name") for a in authors if a.get("name")]
        if names:
            fields["authors"] = json.dumps(names)
        affs_map = {}
        for a in authors:
            name = a.get("name")
            aff_list = a.get("affiliations") or []
            if name and aff_list:
                affs_map[name] = aff_list
        if affs_map:
            fields["affiliations"] = json.dumps(affs_map)
    if not fields:
        return
    db.save_paper(arxiv_id, **fields)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, default=Path("local_data/s2_enrichment.jsonl"))
    ap.add_argument("--year-min", type=int, default=2020,
                    help="Lowest arxiv year prefix to enrich (default 2020).")
    ap.add_argument("--year-max", type=int, default=2026)
    ap.add_argument("--dry-run", action="store_true",
                    help="Fetch + log to JSONL but DO NOT update Neon.")
    args = ap.parse_args()

    db = NeonDB()
    yy_prefixes = [f"{y - 2000:02d}" for y in range(args.year_min, args.year_max + 1)]
    in_clause = " OR ".join([f"id LIKE '{yy}%'" for yy in yy_prefixes])
    with db.get_conn() as c, c.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} WHERE ({in_clause}) AND id NOT LIKE 'ext%' ORDER BY id DESC")
        all_ids = [r[0] for r in cur.fetchall()]
    logger.info(f"arxiv IDs in scope ({args.year_min}-{args.year_max}): {len(all_ids)}")

    seen = load_seen(args.output)
    todo = [aid for aid in all_ids if aid not in seen]
    logger.info(f"already enriched: {len(seen)}; todo: {len(todo)}")

    if not todo:
        logger.info("nothing to do")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fh = args.output.open("a", encoding="utf-8")
    n_hit = n_miss = 0

    try:
        for i in range(0, len(todo), BATCH_SIZE):
            chunk = todo[i:i + BATCH_SIZE]
            s2_ids = [f"ARXIV:{aid}" for aid in chunk]
            t0 = time.perf_counter()
            results = post_batch(s2_ids)
            dt = time.perf_counter() - t0
            if not isinstance(results, list):
                logger.warning(f"  unexpected response shape, skipping chunk")
                continue
            assert len(results) == len(chunk), f"got {len(results)} results for {len(chunk)} ids"
            chunk_hits = 0
            for aid, body in zip(chunk, results):
                rec = {"_arxiv_id": aid, "_s2_response": body}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if body:
                    n_hit += 1
                    chunk_hits += 1
                    if not args.dry_run:
                        try:
                            update_neon_from_s2(db, aid, body)
                        except Exception as e:
                            logger.warning(f"  neon update {aid} failed: {e}")
                else:
                    n_miss += 1
            fh.flush()
            done = i + len(chunk)
            logger.info(
                f"batch {i // BATCH_SIZE + 1}/{(len(todo) + BATCH_SIZE - 1) // BATCH_SIZE}: "
                f"{chunk_hits}/{len(chunk)} hit, {dt:.1f}s — total {done}/{len(todo)}"
            )
    finally:
        fh.close()

    logger.info(f"DONE — {n_hit} S2 hits, {n_miss} not-in-S2")
    print(f"\noutput: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
