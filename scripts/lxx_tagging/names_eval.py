#!/usr/bin/env python3
"""Validation of the name stage (`names.py`) on open data.

Reads a tagging run's outputs (LXX.json for the text, the decisions file for
every token's tier and tag) and reports:

  gold         For the Septuagint's occurrences of New Testament names whose
               TBESG entry gives their Hebrew number, the matcher run blind,
               as if they were tier d, with a model fitted without those
               names (one name in four held out, split by name, never by
               occurrence): how often it names a Hebrew number, and how often
               that is TBESG's (precision and coverage, by method and score
               band). Also the ceiling: how often TBESG's number is in the
               Hebrew verses searched at all.
  consistency  How many Greek name forms the run mapped to one Hebrew number,
               and the largest that map to several.
  sample       A stratified random sample of the run's name decisions, and of
               the candidates it left untagged, each beside the Greek verse
               and the Hebrew verse's names, for reading by hand
               (`--sample-out`).

The gold and consistency figures are the run's own (the report's `names`
section: `names.evaluate_gold` runs with every tagging run, and calibrates
its confidence); this prints them and draws the sample.

Usage:
    python3 scripts/lxx_tagging/names_eval.py --run outputs/lxx_tagging
    python3 scripts/lxx_tagging/names_eval.py --run /tmp/lxx --sample-out /tmp/lxx/name_sample.tsv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging import hebrew_names, names  # noqa: E402
from lxx_tagging.text import tokenize  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent


def verse_text(verse: dict) -> str:
    content = verse["content"]
    return content["text"] if isinstance(content, dict) else content


def read_decisions(path: Path) -> dict[tuple[int, int], dict]:
    csv.field_size_limit(1 << 30)
    rows = {}
    with io.open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            rows[(int(row["ref"]), int(row["token"]))] = row
    return rows


def band_of(score: float) -> str:
    return names.score_band(score)


def sample(module: dict, decisions: dict, hebrew, verse_map, size: int = 150, untagged: int = 50,
           seed: int = 0) -> tuple[list[dict], list[dict]]:
    """A stratified random sample of the run's name decisions (by method and
    score band, each stratum in proportion to its square root, so the rare
    methods are read too), and a random sample of tier-d candidates left
    untagged, each with its Greek verse (the word in brackets) and the names
    of the Hebrew verse it maps to."""
    texts = {verse["ref"]: verse_text(verse) for book in module["books"]
             for chapter in book["chapters"] for verse in chapter["verses"]}
    strata: dict[tuple[str, str], list] = defaultdict(list)
    left = []
    for (ref, index), row in sorted(decisions.items()):
        if row.get("method") in names.METHODS and row.get("name_score"):
            strata[(row["method"], band_of(float(row["name_score"])))].append((ref, index, row))
        elif row.get("tier") == "d" and not row.get("strongs"):
            left.append((ref, index, row))
    generator = random.Random(seed)
    weights = {key: len(rows) ** 0.5 for key, rows in strata.items()}
    total = sum(weights.values())
    chosen = []
    for key in sorted(strata):
        count = min(len(strata[key]), max(1, round(size * weights[key] / total)))
        chosen += [(key, item) for item in generator.sample(strata[key], count)]

    def context(ref: int, index: int) -> dict:
        tokens = tokenize(texts[ref])
        greek = " ".join(f"[{t.surface}]" if t.index == index else t.surface for t in tokens)
        target = verse_map[ref]
        listed = [f"{w.gloss} {w.key} {w.stem}" for w in hebrew.names(target)] if target else []
        return {"ref": ref, "greek": greek, "hebrewRef": target, "hebrewNames": listed}

    decided = [{**context(ref, index), "stratum": f"{key[0]} {key[1]}", "surface": row["surface"],
                "strongs": row["strongs"], "lemma": row["lemma"], "hebrew": row["name_hebrew"],
                "score": row["name_score"], "margin": row["margin"], "grammar": row["name_grammar"]}
               for key, (ref, index, row) in chosen]
    undecided = [{**context(ref, index), "surface": row["surface"], "note": row.get("name_note", "")}
                 for ref, index, row in generator.sample(left, min(untagged, len(left)))]
    return decided, undecided


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=REPOSITORY_ROOT / "outputs" / "lxx_tagging",
                        help="directory with a run's lxx_tagging_report.json and lxx_tagging_decisions.tsv")
    parser.add_argument("--input", type=Path, default=REPOSITORY_ROOT / "outputs" / "lxx_tagging" / "LXX_source.json")
    parser.add_argument("--sources", type=Path, default=hebrew_names.DEFAULT_SOURCES)
    parser.add_argument("--sample-out", type=Path, help="write the hand-reading sample here (JSON)")
    parser.add_argument("--size", type=int, default=150)
    parser.add_argument("--untagged", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args(argv)

    report = json.loads((options.run / "lxx_tagging_report.json").read_text(encoding="utf-8"))
    section = report.get("names")
    if not section:
        print("the run has no name stage (--name-keys off)")
        return 1
    gold = section["validation"] or {}
    print(f"tier d: {section['tierD']['before']['tokens']:,} tokens ({section['tierD']['before']['forms']:,} forms)"
          f" -> {section['tierD']['after']['tokens']:,} untagged ({section['tierD']['after']['forms']:,} forms)")
    print("tagged by method: " + ", ".join(f"{m} {n:,}" for m, n in section["tagged"]["byMethod"].items()))
    if gold:
        print(f"gold (TBESG names held out, {gold['heldOutNames']} names, {gold['tokens']:,} tokens): "
              f"coverage {gold['coveragePercent']}%, precision {gold['precisionPercent']}% "
              f"(ceiling {gold['ceilingPercent']}%); transliterated names: coverage "
              f"{gold['transliteratedNames']['coveragePercent']}%, precision "
              f"{gold['transliteratedNames']['precisionPercent']}%")
    consistency = section["consistency"]
    print(f"forms: {consistency['forms']:,}, one number {consistency['oneNumber']:,}, "
          f"several {consistency['severalNumbers']:,}")
    if options.sample_out:
        module = json.loads(options.input.read_text(encoding="utf-8"))
        decisions = read_decisions(options.run / "lxx_tagging_decisions.tsv")
        hebrew = hebrew_names.build(options.sources)
        verse_map = hebrew_names.versify_map([v["ref"] for b in module["books"] for c in b["chapters"]
                                              for v in c["verses"]])
        decided, undecided = sample(module, decisions, hebrew, verse_map, options.size, options.untagged,
                                    options.seed)
        options.sample_out.write_text(json.dumps({"decisions": decided, "untagged": undecided},
                                                 ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"sample: {len(decided)} decisions, {len(undecided)} untagged -> {options.sample_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
