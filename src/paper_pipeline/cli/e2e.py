"""paper-e2e — one run, the whole company-paper pipeline, end to end.

Spawns every canon discovery source, lands the results in Neon, enriches them,
then (optionally) OCRs and summarizes — in one command:

    discover ─┬─ OpenAlex (API, --save-neon)
              ├─ Semantic Scholar (API, --save-neon)      [needs S2_API_KEY]
              ├─ playwright web-scrape (browser → JSONL)   [--no-web to skip]
              └─ arxiv affiliation search (Meta/LinkedIn → JSONL)
        absorb  (web + meta JSONL → Neon stubs)
        enrich  (arxiv metadata for stubs missing it)
        ocr     (full-text markdown)                        [--ocr, off by default]
        summarize (long-form 7-section)                     [needs local LLM server]

Each stage is spawned as its own subprocess (or run in-process for DB steps),
failures are isolated (one dead source doesn't abort the run), and a summary
table is printed at the end. Re-running is safe: every write is net-new / upsert.

    uv run paper-e2e --window 2025-2026
    uv run paper-e2e --window 2026 --no-web --ocr      # skip browser, enable OCR
    uv run paper-e2e --dry-run                          # print the plan only
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from paper_pipeline.core.neon_db import TABLE, NeonDB

console = Console()
app = typer.Typer(add_completion=False)

DISCOVERY = "paper_pipeline.discovery"
SCRATCH = Path("local_data")


class StageResult(BaseModel):
    """Outcome of one pipeline stage, collected for the final summary table."""

    name: str
    status: str  # "ok" | "skipped" | "failed"
    detail: str = ""
    added: int | None = None  # net rows added to Neon, when measurable
    seconds: float = 0.0


def _launcher(*extra_pkgs: str) -> list[str]:
    """Interpreter prefix: the venv python, or `uv run --with` for ad-hoc deps."""
    missing = [p for p in extra_pkgs if importlib.util.find_spec(p) is None]
    if not missing:
        return [sys.executable]
    return ["uv", "run", *(f"--with={p}" for p in missing), "python"]


def _total_rows(db: NeonDB) -> int:
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE}")
        return int(cur.fetchone()[0])


def _llm_server_up() -> bool:
    """Cheap reachability probe for the local OpenAI-compatible server."""
    import httpx

    from paper_pipeline.summarize.config import LOCAL_BASE_URL

    try:
        httpx.get(f"{LOCAL_BASE_URL}/models", timeout=2.0)
    except httpx.HTTPError:
        return False
    return True


def _run(cmd: Sequence[str], *, name: str) -> StageResult:
    """Spawn a stage, streaming its output, isolating failure to this stage."""
    console.print(Rule(f"[bold]{name}[/]  ·  {' '.join(cmd[:6])}…"))
    start = time.monotonic()
    try:
        proc = subprocess.run(cmd, check=False)
    except (OSError, ValueError) as exc:
        return StageResult(name=name, status="failed", detail=str(exc)[:120], seconds=time.monotonic() - start)
    secs = time.monotonic() - start
    if proc.returncode != 0:
        return StageResult(name=name, status="failed", detail=f"exit {proc.returncode}", seconds=secs)
    return StageResult(name=name, status="ok", seconds=secs)


def _absorb(db: NeonDB, path: Path, *, source: str) -> int:
    """Upsert net-new arxiv ids from a scraper JSONL into Neon as company stubs."""
    if not path.exists():
        return 0
    recs: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        aid = row.get("arxiv_id")
        if aid:
            recs.setdefault(aid, row)
    if not recs:
        return 0
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT id FROM {TABLE} WHERE id = ANY(%s)", (list(recs),))
        present = {r[0] for r in cur.fetchall()}
    fresh = {aid: row for aid, row in recs.items() if aid not in present}
    with db.batch() as batch:
        for aid, row in fresh.items():
            batch.save_paper(
                aid,
                url=f"https://arxiv.org/abs/{aid}",
                organization=row.get("org_label") or None,
                title=row.get("title") or None,
                score_source=source,
            )
    return len(fresh)


def _render_summary(results: list[StageResult]) -> None:
    table = Table(title="paper-e2e summary", title_style="bold", show_lines=False)
    table.add_column("stage")
    table.add_column("status")
    table.add_column("added", justify="right")
    table.add_column("time", justify="right")
    glyph = {"ok": "[green]ok[/]", "skipped": "[yellow]skipped[/]", "failed": "[red]failed[/]"}
    for r in results:
        detail = f" [dim]{r.detail}[/]" if r.detail else ""
        table.add_row(
            r.name,
            glyph.get(r.status, r.status) + detail,
            "—" if r.added is None else str(r.added),
            f"{r.seconds:.0f}s",
        )
    console.print(table)


@app.command()
def main(
    window: Annotated[
        str, typer.Option(help="Year window for API scrapers, e.g. '2026' or '2025-2026'.")
    ] = "2025-2026",
    web: Annotated[bool, typer.Option(help="Run the (slow) playwright browser scrape.")] = True,
    s2: Annotated[bool, typer.Option(help="Run the Semantic Scholar scrape (needs S2_API_KEY).")] = True,
    meta: Annotated[bool, typer.Option(help="Run arxiv affiliation search (Meta/LinkedIn).")] = True,
    enrich: Annotated[bool, typer.Option(help="Fill arxiv metadata for stubs missing it.")] = True,
    ocr: Annotated[bool, typer.Option(help="Run the full-text OCR stage (needs the GPU box).")] = False,
    summarize: Annotated[
        bool, typer.Option(help="Run the long-form summary backfill (needs local LLM server).")
    ] = True,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the stage plan and exit.")] = False,
) -> None:
    """Run the company-paper pipeline end to end."""
    s2_window = window
    oa_window = window

    # Resolve prerequisites up front so the plan reflects what will actually run.
    has_s2_key = bool(os.environ.get("S2_API_KEY"))
    server_up = _llm_server_up() if (summarize and not dry_run) else summarize

    plan: list[tuple[str, bool, str]] = [
        ("discover:openalex", True, f"OpenAlex API, years {oa_window}, --save-neon"),
        ("discover:s2", s2 and has_s2_key, "S2 API, --save-neon" if has_s2_key else "no S2_API_KEY"),
        ("discover:web", web, "playwright browser scrape (slow)"),
        ("discover:meta", meta, "arxiv affiliation search (Meta/LinkedIn)"),
        ("absorb", web or meta, "web + meta JSONL → Neon stubs"),
        ("enrich", enrich, "arxiv metadata for stubs missing it"),
        ("ocr", ocr, "full-text markdown (GPU box only)"),
        ("summarize", summarize and server_up, "local LLM server" if server_up else "no LLM server at :30000"),
    ]

    console.print(Rule("[bold cyan]paper-e2e plan[/]"))
    for name, on, note in plan:
        mark = "[green]RUN[/]" if on else "[dim]skip[/]"
        console.print(f"  {mark}  [bold]{name}[/]  [dim]{note}[/]")
    if dry_run:
        console.print("\n[yellow]dry run — nothing executed.[/]")
        raise typer.Exit(0)

    db = NeonDB()
    results: list[StageResult] = []

    def discover(name: str, module: str, args: list[str], *extra_pkgs: str) -> None:
        before = _total_rows(db)
        res = _run([*_launcher(*extra_pkgs), "-m", f"{DISCOVERY}.{module}", *args], name=name)
        if res.status == "ok":
            res.added = max(0, _total_rows(db) - before)
        results.append(res)

    # 1–2: API discovery sources write to Neon directly.
    discover("discover:openalex", "openalex_2026_audit", ["--years", oa_window, "--save-neon"], "pyalex")
    if s2 and has_s2_key:
        discover("discover:s2", "s2_company_scrape", ["--year-range", s2_window])
    elif s2:
        results.append(StageResult(name="discover:s2", status="skipped", detail="no S2_API_KEY"))

    # 3–4: browser + arxiv discovery write JSONL; absorbed below.
    web_jsonl = SCRATCH / f"e2e_web_{window}.jsonl"
    meta_jsonl = SCRATCH / f"e2e_meta_{window}.jsonl"
    if web:
        results.append(
            _run(
                [sys.executable, "-m", f"{DISCOVERY}.playwright_pub_scrape", "--output", str(web_jsonl)],
                name="discover:web",
            )
        )
    if meta:
        results.append(
            _run(
                [*_launcher("arxiv"), "-m", f"{DISCOVERY}.arxiv_org_search", "--output", str(meta_jsonl)],
                name="discover:meta",
            )
        )

    # 5: absorb the JSONL discovery into Neon.
    if web or meta:
        start = time.monotonic()
        added = 0
        if web:
            added += _absorb(db, web_jsonl, source="e2e_web")
        if meta:
            added += _absorb(db, meta_jsonl, source="e2e_meta")
        results.append(StageResult(name="absorb", status="ok", added=added, seconds=time.monotonic() - start))

    # 6: enrich stubs from the arxiv API (in-process — reuses the sync helper).
    if enrich:
        from paper_pipeline.cli.sync import enrich_from_arxiv

        start = time.monotonic()
        try:
            n = enrich_from_arxiv(db)
            results.append(StageResult(name="enrich", status="ok", added=n, seconds=time.monotonic() - start))
        except Exception as exc:
            results.append(
                StageResult(name="enrich", status="failed", detail=str(exc)[:120], seconds=time.monotonic() - start)
            )

    # 7: OCR — find what needs it; the OCR model itself runs on the GPU box.
    if ocr:
        results.append(_run([sys.executable, "-m", "paper_pipeline.ocr.find_missing"], name="ocr:find_missing"))
        console.print(
            "[yellow]OCR scan done. Run `glm_ocr_driver.py` on the GPU box to fill `markdown` (see ocr/RUNBOOK.md).[/]"
        )

    # 8: summarize the backlog (needs the local LLM server).
    if summarize and server_up:
        results.append(_run([sys.executable, "-m", "paper_pipeline.summarize.cli", "--backfill"], name="summarize"))
    elif summarize:
        results.append(StageResult(name="summarize", status="skipped", detail="no LLM server at :30000"))

    console.print(Rule("[bold cyan]done[/]"))
    _render_summary(results)
    if any(r.status == "failed" for r in results):
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
