"""Assemble the four per-section JSONL files from offline_regen into one
per-paper JSONL ready for absorb_regen_output.py.

Reads:
  local_data/regen_pulldown/regen_output_s14.s1.jsonl
  ...                                        .s4.jsonl

Each per-section line is {arxiv_id, section, text}. We group by arxiv_id,
keep papers that have ALL FOUR sections, concatenate them in order, and
emit {arxiv_id, summary} per paper. title/category/url stay None — the
absorb script only updates fields that are explicitly set, so existing
Neon values are preserved.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-dir", type=Path,
                    default=Path("local_data/regen_pulldown"))
    ap.add_argument("--output", type=Path,
                    default=Path("local_data/regen_assembled_s14.jsonl"))
    ap.add_argument("--require-all", action="store_true", default=True,
                    help="Only emit papers that have all 4 sections (default).")
    args = ap.parse_args()

    sections: dict[str, dict[int, str]] = defaultdict(dict)
    for n in (1, 2, 3, 4):
        p = args.pull_dir / f"regen_output_s14.s{n}.jsonl"
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                aid = r["arxiv_id"]
                sections[aid][n] = r["text"]
        print(f"s{n}: read {sum(1 for v in sections.values() if n in v)} papers")

    n_full = 0
    n_partial = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for aid, secs in sections.items():
            if args.require_all and len(secs) < 4:
                n_partial += 1
                continue
            ordered = [secs[i] for i in sorted(secs)]
            summary = "\n\n".join(ordered)
            out.write(json.dumps({"arxiv_id": aid, "summary": summary},
                                 ensure_ascii=False) + "\n")
            n_full += 1

    print(f"assembled {n_full} papers ({n_partial} partial skipped)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
