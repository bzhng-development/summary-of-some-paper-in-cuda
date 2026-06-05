# /// script
# requires-python = ">=3.12"
# dependencies = ["typer", "rich", "pydantic"]
# ///
"""Sync a repo's gitignored data "intelligence" to a Hugging Face Storage bucket.

The repo commits only `data-bucket.manifest.ndjson` (path, bytes, sha256, hf:// URI);
the data itself stays gitignored and lives in the bucket. `pull` restores it on a
fresh clone — so nothing is lost to `.gitignore`, and nothing big bloats git.

    uv run scripts/data_bucket.py push --bucket vincentzed-hf/data --prefix binutils
    uv run scripts/data_bucket.py pull
    uv run scripts/data_bucket.py status

Requires the HF CLI signed in (`hf auth whoami`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, TypeAdapter
from rich.console import Console
from rich.progress import track
from rich.table import Table

console = Console()
err = Console(stderr=True)

MANIFEST_NAME = "data-bucket.manifest.ndjson"

# Directories/extensions that are software junk or regenerable build output — never bucketed.
_SKIP_DIR_PARTS = {
    ".venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    ".mypy_cache",
    ".git",
    ".pnpm-store",
    ".firefox-profile",
    ".scraper-profile",
    ".tmp-profile",
    "target",
    "build",
    "dist",
    "out",
    ".turbo",
    ".vercel",
    "coverage",
    # agent/AI-tool state dirs (config, not intelligence) — note: .review-bundles is KEPT
    ".claude",
    ".entire",
    ".agents",
    ".adal",
    ".augment",
    ".continue",
    ".cortex",
    ".crush",
    ".goose",
    ".iflow",
    ".junie",
    ".kilocode",
    ".kode",
    ".mcpjam",
    ".mux",
    ".neovate",
    ".openhands",
    ".pochi",
    ".qoder",
    ".qwen",
    ".roo",
    ".trae",
    ".windsurf",
    ".zencoder",
    ".codebuddy",
    ".commandcode",
    ".factory",
    ".kiro",
    ".pi",
    ".vibe",
    ".playwright-cli",
    ".idea",
    ".vscode",
    ".vscode-test",
    ".trunk",
    # embedded vendored clones (their own git history / not our intelligence)
    "ente",
    "markitdown",
    "bunnylol.rs",
    "google-maps-review-scraper",
    "cs166-s26",
    "prefect-repo",
    "temporal-repo",
    "temporal-samples",
    "inngest-repo",
    "abseil-src",
}
# Junk filenames (suffix check misses dotfiles like .DS_Store, whose Path.suffix == "").
_SKIP_NAMES = {".DS_Store", "Thumbs.db", ".localized"}
_SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pth",
    ".so",
    ".dylib",
    ".tsbuildinfo",
    ".o",
    ".rlib",
    ".rmeta",
    ".a",
    ".map",
    ".DS_Store",
}


class ManifestEntry(BaseModel):
    path: str  # repo-relative POSIX path
    bytes: int
    sha256: str
    uri: str  # hf://buckets/<namespace>/<bucket>/<prefix>/<path>


_ENTRIES = TypeAdapter(list[ManifestEntry])


def _run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)  # type: ignore[arg-type]


def _repo_root() -> Path:
    out = _run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(out)


def _is_junk(rel: str) -> bool:
    p = Path(rel)
    if set(p.parts) & _SKIP_DIR_PARTS:
        return True
    if p.name in _SKIP_NAMES:
        return True
    return p.suffix in _SKIP_SUFFIXES


def _data_files(root: Path) -> list[str]:
    """Gitignored files that are data, not software junk — sorted, repo-relative POSIX."""
    out = _run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"], cwd=root
    ).stdout.splitlines()
    return sorted(f for f in out if f and not _is_junk(f) and (root / f).is_file())


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _uri(bucket: str, prefix: str, rel: str) -> str:
    base = f"hf://buckets/{bucket}"
    return f"{base}/{prefix}/{rel}" if prefix else f"{base}/{rel}"


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def push(
    bucket: Annotated[str, typer.Option(help="HF bucket id, e.g. vincentzed-hf/data")],
    prefix: Annotated[
        str, typer.Option(help="Path prefix inside the bucket (repo namespace)")
    ] = "",
    dry_run: Annotated[
        bool, typer.Option(help="Plan only; don't upload or write manifest")
    ] = False,
) -> None:
    """Upload gitignored data to the bucket and (re)write the manifest."""
    root = _repo_root()
    files = _data_files(root)
    total = sum((root / f).stat().st_size for f in files)
    console.print(
        f"[bold]{len(files)}[/] data files, {total / 1e9:.2f} GB -> "
        f"hf://buckets/{bucket}/{prefix or '(root)'}"
    )
    if dry_run:
        for f in files[:20]:
            console.print(f"  {f}")
        if len(files) > 20:
            console.print(f"  … +{len(files) - 20} more")
        return

    # Stage into a temp tree preserving paths, then one batched `hf buckets sync`.
    with tempfile.TemporaryDirectory(prefix="data-bucket-") as tmp:
        stage = Path(tmp)
        for f in track(files, description="staging"):
            dst = stage / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / f, dst)
        dest = f"hf://buckets/{bucket}/{prefix}" if prefix else f"hf://buckets/{bucket}"
        console.print("[bold]uploading via hf buckets sync…[/]")
        subprocess.run(["hf", "buckets", "sync", str(stage), dest], check=True)

    entries = [
        ManifestEntry(
            path=f,
            bytes=(root / f).stat().st_size,
            sha256=_sha256(root / f),
            uri=_uri(bucket, prefix, f),
        )
        for f in track(files, description="hashing")
    ]
    manifest = root / MANIFEST_NAME
    with manifest.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.model_dump(), sort_keys=True) + "\n")
    console.print(f"[green]wrote {manifest.name} ({len(entries)} entries)[/]")


@app.command()
def manifest(
    bucket: Annotated[str, typer.Option(help="HF bucket id, e.g. vincentzed-hf/data")],
    prefix: Annotated[
        str, typer.Option(help="Path prefix inside the bucket (repo namespace)")
    ] = "",
) -> None:
    """(Re)write the manifest from local data files WITHOUT uploading (data already in bucket)."""
    root = _repo_root()
    files = _data_files(root)
    entries = [
        ManifestEntry(
            path=f,
            bytes=(root / f).stat().st_size,
            sha256=_sha256(root / f),
            uri=_uri(bucket, prefix, f),
        )
        for f in track(files, description="hashing")
    ]
    out = root / MANIFEST_NAME
    with out.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e.model_dump(), sort_keys=True) + "\n")
    console.print(f"[green]wrote {out.name} ({len(entries)} entries)[/]")


def _load_manifest(root: Path) -> list[ManifestEntry]:
    mf = root / MANIFEST_NAME
    if not mf.exists():
        err.print(f"[red]no {MANIFEST_NAME} found in {root}[/]")
        raise typer.Exit(1)
    return _ENTRIES.validate_python(
        [json.loads(ln) for ln in mf.read_text().splitlines() if ln.strip()]
    )


@app.command()
def pull(
    include: Annotated[str, typer.Option(help="Only pull paths containing this substring")] = "",
) -> None:
    """Download bucket data named in the manifest that is missing or changed locally."""
    root = _repo_root()
    entries = [e for e in _load_manifest(root) if include in e.path]
    todo = [
        e for e in entries if not (root / e.path).exists() or _sha256(root / e.path) != e.sha256
    ]
    console.print(f"{len(todo)} of {len(entries)} files to download")
    for e in track(todo, description="pulling"):
        dst = root / e.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["hf", "buckets", "cp", e.uri, str(dst)], check=True)
    console.print("[green]pull complete[/]")


@app.command()
def status() -> None:
    """Show present / missing / changed counts against the manifest."""
    root = _repo_root()
    entries = _load_manifest(root)
    present = missing = changed = 0
    for e in entries:
        p = root / e.path
        if not p.exists():
            missing += 1
        elif p.stat().st_size == e.bytes:
            present += 1
        else:
            changed += 1
    t = Table("state", "count")
    t.add_row("present", str(present))
    t.add_row("missing", str(missing))
    t.add_row("changed", str(changed))
    console.print(t)


if __name__ == "__main__":
    app()
