#!/usr/bin/env python3
"""Write the published data files from a tagging run.

`scripts/lxx_tagging/tag_septuagint.py` writes three files to `outputs/`.
This script turns them into what the repository publishes under `data/`:

  data/tagged/NN-Book.tsv   one row per word of that book: the tag, how it
                            was chosen, how sure the choice is, and the other
                            readings considered (docs/FORMAT.md)
  data/untagged.tsv         every word left without a tag, with the reason
  data/report.json          the run's own report: coverage by tier and book,
                            methods, confidence bands, the name stage
  data/MANIFEST.json        size and SHA-256 of every input and output,
                            including outputs/LXX_tagged.json, which is too
                            large to commit and is attached to releases

The same run gives the same bytes: rows keep the tagger's order, nothing is
timestamped, and JSON is written with sorted keys.

    python3 scripts/export_data.py
    python3 scripts/export_data.py --check    # compare with what is committed
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The published column order: the tag first, then how it was reached. The
# tagger's `choice` column repeats method, confidence and margin and is left out.
LEADING = [
    "ref", "osis", "token", "start", "end", "surface",
    "strongs", "lemma", "morph", "confidence", "method", "margin",
    "tier", "reason", "key", "n_readings", "candidates", "runner_up", "set_aside", "part",
]
DROPPED = {"choice"}
UNTAGGED = ["ref", "osis", "token", "surface", "tier", "reason", "name_note"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_record(path: Path) -> dict:
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
            size += len(block)
    return {"bytes": size, "sha256": sha.hexdigest()}


def read_decisions(path: Path) -> tuple[list[str], list[list[str]]]:
    with io.open(path, encoding="utf-8", newline="\n") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in handle]
    for number, row in enumerate(rows, 2):
        if len(row) != len(header):
            raise SystemExit(f"{path.name}:{number} has {len(row)} fields, the header {len(header)}")
    return header, rows


def table(columns: list[str], rows: list[list[str]]) -> bytes:
    lines = ["\t".join(columns)] + ["\t".join(row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def build(outputs: Path, source: Path) -> dict[str, bytes]:
    """Published path (relative to data/) -> bytes."""
    header, rows = read_decisions(outputs / "lxx_tagging_decisions.tsv")
    missing = [name for name in LEADING + UNTAGGED if name not in header]
    if missing:
        raise SystemExit(f"the decisions file lacks columns: {', '.join(missing)}")
    columns = LEADING + [name for name in header if name not in LEADING and name not in DROPPED]
    index = {name: header.index(name) for name in header}
    pick = [index[name] for name in columns]

    files: dict[str, bytes] = {}
    books: dict[str, list[list[str]]] = {}
    order: list[str] = []
    for row in rows:
        number = int(row[index["ref"]]) // 1_000_000
        name = f"{number:02d}-{row[index['osis']].split('.')[0]}"
        if name not in books:
            books[name] = []
            order.append(name)
        books[name].append([row[i] for i in pick])
    for name in order:
        files[f"tagged/{name}.tsv"] = table(columns, books[name])

    gaps = [[row[index[name]] for name in UNTAGGED] for row in rows if not row[index["strongs"]]]
    files["untagged.tsv"] = table(UNTAGGED, gaps)

    report = json.loads((outputs / "lxx_tagging_report.json").read_text(encoding="utf-8"))
    files["report.json"] = (json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")

    manifest = {
        "inputs": {
            "data/source/LXX.json": file_record(source / "LXX.json"),
            "data/source/versemap_lxx_to_english.json": file_record(source / "versemap_lxx_to_english.json"),
        },
        "outputs": {
            "outputs/LXX_tagged.json": file_record(outputs / "LXX_tagged.json"),
            **{f"data/{name}": {"bytes": len(data), "sha256": digest(data)} for name, data in sorted(files.items())},
        },
        "words": len(rows),
        "tagged": sum(1 for row in rows if row[index["strongs"]]),
    }
    files["MANIFEST.json"] = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode("utf-8")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outputs", type=Path, default=REPOSITORY_ROOT / "outputs")
    parser.add_argument("--data", type=Path, default=REPOSITORY_ROOT / "data")
    parser.add_argument("--check", action="store_true", help="compare with the committed files; write nothing")
    options = parser.parse_args()

    files = build(options.outputs, options.data / "source")
    changed = 0
    for name, data in files.items():
        path = options.data / name
        if path.exists() and path.read_bytes() == data:
            continue
        changed += 1
        if options.check:
            print(f"differs  data/{name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    stale = [path for path in sorted((options.data / "tagged").glob("*.tsv"))
             if f"tagged/{path.name}" not in files] if (options.data / "tagged").exists() else []
    for path in stale:
        changed += 1
        if options.check:
            print(f"stale    data/tagged/{path.name}")
        else:
            path.unlink()
    print(f"{len(files)} files, {changed} {'differ' if options.check else 'written'}")
    return 1 if options.check and changed else 0


if __name__ == "__main__":
    sys.exit(main())
