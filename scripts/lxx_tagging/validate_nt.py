#!/usr/bin/env python3
"""Open cross-validation of the tagger and its chooser on STEPBible's tagged
New Testament.

Everything here is CC BY 4.0 data, so the per-word results may be inspected
freely. (It does not import, and must never import, `validate_ccat` or any
`ccat_*` module.)

WHAT IS MEASURED

The NA text of TAGNT is set in Brenton's capitals and tokenised as the LXX is
(`ntcorpus`). The books are split into `--folds` folds (`--holdout
stratified`, the default: every fold has Gospel, Acts and epistle material;
or `book`: whole books). For each fold:

  * its tokens get their candidates from a form table built from the other
    folds' rows only (every edition's), through the tagger's own providers,
    case filter and paradigm rules, so no form, lemma or tag of the held-out
    fold reaches the tables;
  * the chooser is trained on the other folds, each of whose tokens has its
    candidates from a table without its own fold *and* without the held-out
    one (nested: `--no-nested` uses the other folds' tables as the CLI's
    training does, which lets the held-out fold's rows into the training
    tokens' candidate counts, never into their labels);
  * every held-out token is chosen and compared with TAGNT.

Scores (overall and per fold): accuracy of the full reading (Strong's, lemma
and morphology; a name or loanword tagged without morphology counts as right
when its Strong's number and lemma are), of Strong's number and lemma, and of
each morphological field (part of speech, case, number, gender, person,
tense, voice, mood; over the tokens whose TAGNT code has the field, tokens
tagged without morphology left out), for:

  all        every held-out token (untagged ones count as wrong)
  ambiguous  tiers b, c and u: the tokens the chooser decides
  unseen     tokens whose form the training folds never have (no NT-table
             candidate): the Septuagint's commonest situation
  tier a-e,u each tier

against two baselines: the reading with most evidence for the form
(`TokenCandidates.readings()[0]`) and the first candidate.

Calibration: the confidence a token is given comes from a table fitted on
the other folds' held-out decisions (so it too is out of sample); the report
lists, per confidence band, the tokens, their mean confidence and the
observed accuracy.

The JSON report goes to `.cache/lxx_tagging/reports/nt_cross_validation.json`
(or `--report`); a summary is printed.

Usage:
    python3 scripts/lxx_tagging/validate_nt.py
    python3 scripts/lxx_tagging/validate_nt.py --folds 5 --holdout book --no-nested
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging import chooser as chooser_module  # noqa: E402
from lxx_tagging import lexicon as lexicon_module  # noqa: E402
from lxx_tagging import ntcorpus, ntforms  # noqa: E402
from lxx_tagging.pipeline import TaggerSpec  # noqa: E402
from lxx_tagging.chooser import Calibration, ModelChooser, parse_morph  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_REPORT = REPOSITORY_ROOT / ".cache" / "lxx_tagging" / "reports" / "nt_cross_validation.json"
FIELDS = ("pos", "case", "num", "gen", "person", "tense", "voice", "mood")
SUBSETS = ("all", "ambiguous", "unseen", "tier:a", "tier:b", "tier:c", "tier:d", "tier:e", "tier:u")
SYSTEMS = ("chooser", "frequent", "first")


def default_factory(sources: Path = ntforms.DEFAULT_SOURCES, analyser: bool = True) -> TaggerSpec:
    """The CLI's own tagger: the same providers, analyser included."""
    return TaggerSpec(tbesg_path=sources / "TBESG.txt", analyser=analyser)


def cross_validate(
    sources: Path = ntforms.DEFAULT_SOURCES,
    *,
    folds: int = chooser_module.FOLDS,
    holdout: str = "stratified",
    nested: bool = True,
    epochs: int = chooser_module.EPOCHS,
    beam: int = chooser_module.BEAM,
    factory=None,
    report_path: Path | None = DEFAULT_REPORT,
    log=print,
    records_path: Path | None = None,
) -> dict:
    """Cross-validate on TAGNT and return (and write) the aggregate report.
    See the module docstring."""
    started = time.time()
    factory = factory or default_factory(sources)
    lexicon = lexicon_module.build(sources / "TBESG.txt")
    data = chooser_module.training_data(sources, factory, folds, holdout, lexicon=lexicon, log=log)
    nested_sets = chooser_module.nested_sets(data, sources, factory, folds, holdout, log=log) if nested else {}
    fold_results = []
    for held in range(folds):
        log(f"fold {held + 1} of {folds}: {len(data.fold_verses(held)):,} verses held out")
        training = [f for f in range(folds) if f != held]
        if nested:
            sets = {f: nested_sets[(held, f)] for f in training}
        else:
            sets = {f: data.sets[f] for f in training}
        model = chooser_module.Model()
        model.text_counts = data.counts
        model.verb_objects = data.objects
        model.name_genders = chooser_module.name_genders(lexicon)
        examples = []
        for fold in training:
            examples += chooser_module.examples_from(model, data.fold_verses(fold), sets[fold])
        model = chooser_module.train(examples, model, epochs=epochs, beam=beam)
        fold_results.append(chooser_module.evaluate(
            model, ModelChooser(model, Calibration({}), beam=beam), data.fold_verses(held), data.sets[held], held))
    # Each fold's confidences from a calibration fitted on the other folds.
    for result in fold_results:
        others = [r for r in fold_results if r.fold != result.fold]
        calibration = Calibration.fit(chooser_module.calibration_records(others))
        for record in result.records:
            if record["chosen"] is not None:
                margin = record["margin"]
                record["confidence"] = calibration.confidence(
                    record["method"], record["tier"], record["family"], math.inf if margin is None else margin,
                    record["n_options"])
    records = [record for result in fold_results for record in result.records]
    if records_path is not None:
        import pickle
        records_path.write_bytes(pickle.dumps(
            [{**record, "osis": data.fold_verses(record["fold"])[record["verse"]].osis} for record in records]))
    report = summarise(records, folds)
    report["settings"] = {"analyser": getattr(factory, "analyser", None), "folds": folds, "holdout": holdout, "nested": nested, "epochs": epochs, "beam": beam,
                          "tokens": len(records), "seconds": round(time.time() - started)}
    report["folds"] = {
        str(result.fold): {
            "books": sorted({r["book"] for r in result.records}),
            **{k: v for k, v in summarise(result.records, 1, calibration=False).items() if k == "accuracy"},
        }
        for result in fold_results
    }
    final = Calibration.fit(chooser_module.calibration_records(fold_results))
    report["calibrationTable"] = final.groups
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")
    return report


def _fields(morph: str | None) -> dict[str, str] | None:
    if morph is None:
        return None
    attrs = parse_morph(morph)
    return {name: getattr(attrs, name) for name in FIELDS}


def summarise(records: list[dict], folds: int, calibration: bool = True) -> dict:
    totals: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        subsets = ["all", f"tier:{record['tier']}"]
        if record["tier"] in ("b", "c", "u"):
            subsets.append("ambiguous")
        if record["unseen"]:
            subsets.append("unseen")
        gold = record["gold"]
        gold_fields = _fields(gold[2])
        for system in SYSTEMS:
            chosen = record["chosen" if system == "chooser" else system]
            ok = chooser_module.correct(chosen, gold)
            word = chosen is not None and chosen[0] == gold[0] and _nfc(chosen[1]) == _nfc(gold[1])
            chosen_fields = _fields(chosen[2]) if chosen is not None else None
            for subset in subsets:
                counter = totals[f"{subset}|{system}"]
                counter["tokens"] += 1
                counter["tagged"] += chosen is not None
                counter["reading"] += ok
                counter["word"] += word
                if chosen is not None and chosen[2] is None:
                    counter["noMorphology"] += 1
                    continue
                for name in FIELDS:
                    if gold_fields and gold_fields[name]:
                        counter[f"n:{name}"] += 1
                        counter[f"ok:{name}"] += bool(chosen_fields) and chosen_fields[name] == gold_fields[name]
    accuracy = {}
    for subset in SUBSETS:
        accuracy[subset] = {}
        for system in SYSTEMS:
            counter = totals.get(f"{subset}|{system}")
            if not counter:
                continue
            n = counter["tokens"]
            entry = {
                "tokens": n,
                "tagged": round(counter["tagged"] / n, 4),
                "reading": round(counter["reading"] / n, 4),
                "strongsLemma": round(counter["word"] / n, 4),
                "taggedWithoutMorphology": counter["noMorphology"],
            }
            for name in FIELDS:
                if counter[f"n:{name}"]:
                    entry[name] = round(counter[f"ok:{name}"] / counter[f"n:{name}"], 4)
            accuracy[subset][system] = entry
    out = {"accuracy": accuracy}
    if calibration:
        bands: dict[str, Counter] = defaultdict(Counter)
        sums: dict[str, float] = defaultdict(float)
        for record in records:
            if record.get("confidence") is None:
                continue
            for group in ("all", f"method:{record['method']}", f"tier:{record['tier']}"):
                name = f"{group}|{chooser_module.band(record['confidence'])}"
                bands[name]["n"] += 1
                bands[name]["correct"] += chooser_module.correct(record["chosen"], record["gold"])
                sums[name] += record["confidence"]
        table: dict[str, dict] = defaultdict(dict)
        for name in sorted(bands):
            group, band = name.split("|")
            n = bands[name]["n"]
            table[group][band] = {"tokens": n, "meanConfidence": round(sums[name] / n, 4),
                                  "accuracy": round(bands[name]["correct"] / n, 4)}
        out["calibration"] = {group: {band: table[group][band] for band in chooser_module.BANDS if band in table[group]}
                              for group in sorted(table)}
        # Ambiguous tokens whose true reading was among the candidates: the ceiling.
        ambiguous = [r for r in records if r["tier"] in ("b", "c", "u")]
        if ambiguous:
            out["ambiguousOracle"] = round(sum(r["gold_available"] for r in ambiguous) / len(ambiguous), 4)
    return out


def _nfc(text: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", text)


def print_summary(report: dict) -> None:
    accuracy = report["accuracy"]
    print(f"\n{report['settings']['tokens']:,} held-out tokens, {report['settings']['folds']} folds "
          f"({report['settings']['holdout']}, nested={report['settings']['nested']}), "
          f"{report['settings']['seconds']}s")
    print(f"{'subset':10} {'system':9} {'tokens':>7} {'tagged':>7} {'reading':>8} {'word':>7} "
          + " ".join(f"{name:>6}" for name in FIELDS))
    for subset in SUBSETS:
        for system in SYSTEMS:
            entry = accuracy.get(subset, {}).get(system)
            if not entry:
                continue
            print(f"{subset:10} {system:9} {entry['tokens']:>7,} {entry['tagged']:>7.2%} {entry['reading']:>8.2%} "
                  f"{entry['strongsLemma']:>7.2%} "
                  + " ".join(f"{entry.get(name, float('nan')):>6.1%}" for name in FIELDS))
    if "ambiguousOracle" in report:
        print(f"ambiguous tokens with the true reading among the candidates: {report['ambiguousOracle']:.2%}")
    print("\ncalibration (confidence band: tokens, mean confidence, observed accuracy)")
    for group, bands in report["calibration"].items():
        cells = ", ".join(f"{band} {v['tokens']:,} {v['meanConfidence']:.3f}/{v['accuracy']:.3f}" for band, v in bands.items())
        print(f"  {group:16} {cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", type=Path, default=ntforms.DEFAULT_SOURCES,
                        help="directory with TAGNT_*.txt and TBESG.txt")
    parser.add_argument("--folds", type=int, default=chooser_module.FOLDS)
    parser.add_argument("--holdout", choices=("stratified", "book"), default="stratified")
    parser.add_argument("--no-nested", action="store_true",
                        help="train on the other folds' own candidate sets (faster; see the docstring)")
    parser.add_argument("--epochs", type=int, default=chooser_module.EPOCHS)
    parser.add_argument("--beam", type=int, default=chooser_module.BEAM)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-analyser", action="store_true", help="candidates without the morphological analyser")
    parser.add_argument("--records", type=Path, help="also write every held-out token's record (pickle), for error analysis")
    options = parser.parse_args(argv)
    report = cross_validate(options.sources, folds=options.folds, holdout=options.holdout,
                            nested=not options.no_nested, epochs=options.epochs, beam=options.beam,
                            factory=default_factory(options.sources, not options.no_analyser),
                            report_path=options.report, records_path=options.records)
    print_summary(report)
    print(f"\nreport: {options.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
