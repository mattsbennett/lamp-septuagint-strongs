#!/usr/bin/env python3
"""Tag Brenton's Greek Septuagint with Strong's numbers, lemmas and morphology.

Reads `outputs/lxx_tagging/LXX_source.json`, the untagged text as
`usfm_to_translation_schema.py` converts it (never changed), and writes
beside it:

  LXX_tagged.json            the same module with `strongs` annotations in
                             TAGNT's shape, features.strongs and
                             features.morphology set, and the sources of its
                             tags in meta.sourceTexts.annotations; every text
                             is the input's, byte for byte
  lxx_tagging_report.json    coverage by tier and book, reasons tokens were
                             left untagged, the commonest untagged forms, and
                             how often each rule applied
  lxx_tagging_decisions.tsv  one row per token, with its candidates, for
                             review and for the later stages

`--bundle-module outputs/LXX.json` also writes the tagged module there, where
the bundler finds the app's `LXX`. The untagged text and the run's other
files stay in `outputs/lxx_tagging/`, outside the bundler's reach: both
modules are called LXX, and only one may be bundled.

Every token with a candidate the case filter kept is tagged (tiers a, b, c
and u): by its one reading, or by the contextual chooser (`chooser.py`), which
is trained on TAGNT on first use and cached under .cache/lxx_tagging/chooser/.
`--chooser unique` tags tier a only, as before the chooser existed. The tiers
in the report and the decisions file describe the candidates, not the outcome;
the decisions file's `method`, `confidence` and `margin` say how each token was
chosen and how sure the choice is.

Then the name stage (`names.py`) tags the tier-d names the New Testament never
uses by the Hebrew name they render, from STEPBible's tagged Hebrew (TAHOT,
TBESH) in the same verse: `--name-keys hebrew` (the default) gives the Hebrew
Strong's number as the shipped Hebrew module keys it, with no morphology code;
`greek-extended` the Hebrew number + 10000 as a Greek key, with the code;
`off` leaves tier d untagged. Its methods are hebrew-verse, hebrew-order,
hebrew-nearby and hebrew-consistency, and the report's `names` section says
what it tagged, what it left and why, and how it measured on TBESG's own
name relations. The stage runs after the chooser, in code outside the
chooser's cache key, so changing it never retrains the chooser.

After the name stage, three passes outside the chooser's cache key:
`capitals.py` tags the common words Brenton capitalises that the name stage
found no Hebrew name for (Βοῤῥᾶν, Νότον, Λιβός, Οὐαί) from their lower-case
readings; `name_parse.py` reads the case of a name the chooser tagged from a
loose NT match from the Septuagint word itself; and the decisions file's
`name_relation` says whether each name the stage tagged transliterates the
Hebrew name or stands for it. A misprint of the digitised text is read as
the word intended only when `text.MISPRINT_TABLE` lists it; its row's
`reason` then says so. Confidences stay on the scale measured on held-out
New Testament text: nothing measured against a reference tagging of the
Septuagint is applied to them, or to anything else written here.

Output is deterministic: the same input gives the same bytes.

Usage:
    python3 scripts/lxx_tagging/tag_septuagint.py --bundle-module outputs/LXX.json
    python3 scripts/lxx_tagging/tag_septuagint.py --books Gen,Ps --output-dir /tmp/lxx
    python3 scripts/lxx_tagging/tag_septuagint.py --name-keys greek-extended --output-dir /tmp/lxx
"""

from __future__ import annotations

import argparse
import copy
import io
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging import capitals  # noqa: E402
from lxx_tagging import chooser as chooser_module  # noqa: E402
from lxx_tagging import hebrew_names  # noqa: E402
from lxx_tagging import lexicon as lexicon_module  # noqa: E402
from lxx_tagging import name_parse  # noqa: E402
from lxx_tagging import names as names_module  # noqa: E402
from lxx_tagging import ntforms  # noqa: E402
from lxx_tagging.candidates import (  # noqa: E402
    CaseProfile, Choice, Tagger, TokenCandidates, annotation_data, default_tagger, unique_reading_chooser,
)
from lxx_tagging.pipeline import TaggerSpec  # noqa: E402
from lxx_tagging.text import MISPRINT_TABLE, bare, normalize, tokenize  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_RUN = REPOSITORY_ROOT / "outputs" / "lxx_tagging"
TIERS = {
    "a": "one reading: tagged",
    "b": "one Strong's number and lemma, several parses",
    "c": "several Strong's numbers or lemmas",
    "d": "looks like a name, no acceptable candidate",
    "e": "no candidate",
    "u": "unconfirmed: tolerant match only, or names for a lower-case word",
}
DECISION_COLUMNS = (
    "ref", "osis", "token", "start", "end", "surface", "key", "tier", "reason",
    "strongs", "lemma", "morph", "choice", "n_readings", "candidates", "set_aside", "part",
    # Added with the chooser, at the end so readers by header name are unaffected.
    "method", "confidence", "margin", "runner_up",
    # Added with the name stage: the Hebrew word a name was matched to, the
    # match's score, the case/number/gender the Greek shows (a TAGNT-shaped
    # code, ? for unknown; never on an H key itself), where the key came
    # from (tahot, or tbesg for an NT name's Greek number), and for a
    # candidate left untagged, why.
    "name_hebrew", "name_score", "name_grammar", "name_key", "name_note", "name_replaces",
    # Whether a name the stage tagged transliterates the Hebrew name it is
    # tagged with or stands for it (another name printed there; `names.name_relations`).
    "name_relation",
)
UNTAGGED_FORMS = 300


def verses(module: dict, books: set[str] | None = None):
    """(book id, chapter, verse record) for each verse, in the module's order."""
    for book in module["books"]:
        if books and book["id"] not in books:
            continue
        for chapter in book["chapters"]:
            for verse in chapter["verses"]:
                yield book["id"], chapter["chapter"], verse


def verse_text(verse: dict) -> str:
    content = verse["content"]
    return content["text"] if isinstance(content, dict) else content


class Results(list):
    """Per-verse results, (book id, chapter, verse record, outcome), with the
    chooser's runner-up readings by result index and token index."""

    def __init__(self) -> None:
        super().__init__()
        self.runner_ups: dict[int, dict] = {}


def tag_module(module: dict, tagger: Tagger, books: set[str] | None = None, after=None):
    """Tag every verse. Returns (tagged module, per-verse results).

    `after(results)`, if given, runs once every verse is chosen and before
    the annotations are written: the name stage, which reads the chooser's
    tags around a name and may replace a token's choice."""
    tagged = copy.deepcopy(module)
    results = Results()
    for book_id, chapter, verse in verses(tagged, books):
        outcome = tagger.tag_tokens(tokenize(verse_text(verse)))
        runners = dict(getattr(tagger.chooser, "runner_up", {}) or {})
        if runners:
            results.runner_ups[len(results)] = runners
        results.append((book_id, chapter, verse, outcome))
    if after is not None:
        after(results)
    annotate(results)
    features = tagged["meta"].setdefault("features", {})
    features["strongs"] = True
    features["morphology"] = True
    return tagged, results


def annotate(results) -> None:
    """Write each verse's chosen readings as `strongs` annotations."""
    for _, _, verse, outcome in results:
        text = verse_text(verse)
        annotations = []
        for item, choice in outcome:
            if choice is None:
                continue
            token = item.token
            if text[token.start:token.end] != token.surface:
                raise AssertionError(f"{verse['ref']}: span {token.start}-{token.end} is not {token.surface!r}")
            annotations.append({
                "type": "strongs", "start": token.start, "end": token.end,
                "text": token.surface, "data": annotation_data(choice.candidate),
            })
        if annotations:
            if not isinstance(verse["content"], dict):
                verse["content"] = {"text": text}
            existing = verse["content"].get("annotations", [])
            verse["content"]["annotations"] = sorted(
                existing + annotations, key=lambda a: (a["start"], a["end"], a["type"]))


def apply_names(results, run, codes: set[str] | None) -> None:
    """Put the name stage's decisions into the results, as the chooser's
    choices are: an H key never carries a morphology code (the app reads a
    key's first letter as its language), and any code a name carries is one
    TAGNT uses, as `Tagger` requires of the chooser's."""
    for (number, position), decision in run.decisions.items():
        choice = names_module.choice_for(decision)
        morph = choice.candidate.morph
        if morph and choice.candidate.strongs.startswith("H"):
            raise ValueError(f"{choice.candidate.strongs} would carry the Greek code {morph}")
        if morph and codes is not None and morph not in codes:
            raise ValueError(f"{morph!r} is not a TAGNT morphology code")
        book_id, chapter, verse, outcome = results[number]
        item, _ = outcome[position]
        outcome[position] = (item, choice)
    results.names = run


def apply_name_parses(results, codes: set[str] | None) -> None:
    """Read the case of every name the chooser tagged from a loose NT match
    from the Septuagint word itself (`name_parse`), and say how in the
    decisions file's `name_note`."""
    done = name_parse.apply(results, codes)
    results.name_parses = done
    run = getattr(results, "names", None)
    if run is not None:
        for position, rule in done.items():
            note = run.notes.get(position, "")
            run.notes[position] = (note + "; " if note else "") + name_parse.NOTES[rule]


def check_texts_unchanged(original: dict, tagged: dict) -> None:
    before = [verse_text(verse) for _, _, verse in verses(original)]
    after = [verse_text(verse) for _, _, verse in verses(tagged)]
    if before != after:
        raise AssertionError("a verse's text changed while tagging")


def compact(candidates) -> str:
    return " ; ".join(candidate.compact() for candidate in candidates)


def name_columns(run, number: int, position: int) -> tuple[str, ...]:
    """The name stage's columns for one token (empty for the others)."""
    if run is None:
        return ("",) * 7
    note = run.notes.get((number, position), "")
    match = run.found.get((number, position))
    decision = run.decisions.get((number, position))
    relation = run.relations.get((number, position), "") if decision else ""
    if match is None:
        return ("", "", "", "", note, "", "")
    match = match[0]
    word = match.hebrew
    hebrew = f"{word.ref}#{word.index} {word.dstrongs} {word.stem} {word.gloss}"
    if match.spelling:
        hebrew += f" (the word {match.spelling})"
    grammar = decision.grammar if decision else ""
    key = decision.key_source if decision else ""
    replaces = decision.replaces if decision and decision.replaces else ""
    return (hebrew, f"{match.score:.2f}", grammar, key, note, replaces, relation)


def decision_rows(results):
    runner_ups = getattr(results, "runner_ups", {})
    run = getattr(results, "names", None)
    for number, (book_id, chapter, verse, outcome) in enumerate(results):
        runners = runner_ups.get(number, {})
        for position, (item, choice) in enumerate(outcome):
            token = item.token
            chosen = choice.candidate if choice else None
            runner = runners.get(token.index)
            named = run is not None and (number, position) in run.decisions
            if named:
                runner_text = run.decisions[(number, position)].runner_up
            else:
                runner_text = runner.compact() if runner else ""
            yield (
                str(verse["ref"]), f"{book_id}.{chapter}.{verse['v']}", str(token.index),
                str(token.start), str(token.end), token.surface, token.key, item.tier,
                item.reason + (f"; assumed misprint: {token.surface} read as {token.key}"
                               if "misprint" in token.rules else ""),
                chosen.strongs if chosen else "", chosen.lemma if chosen else "",
                (chosen.morph or "") if chosen else "",
                f"{choice.method}:{choice.confidence:g}:{choice.margin:g}" if choice else "",
                str(len(item.readings())), compact(item.candidates),
                " ; ".join(f"{c.compact()}|{reason}" for c, reason in item.set_aside),
                f"{token.part + 1}/{token.parts}" if token.joined else "",
                choice.method if choice else "",
                f"{choice.confidence:.4f}" if choice else "",
                f"{choice.margin:g}" if choice else "",
                runner_text,
            ) + name_columns(run, number, position)


def write_decisions(path: Path, results) -> None:
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(DECISION_COLUMNS) + "\n")
        for row in decision_rows(results):
            handle.write("\t".join(field.replace("\t", " ").replace("\n", " ") for field in row) + "\n")


def build_report(results, tagger: Tagger, source: Path) -> dict:
    tiers: Counter = Counter()
    by_book: dict[str, Counter] = defaultdict(Counter)
    reasons: Counter = Counter()
    chosen_sources: Counter = Counter()
    rules: Counter = Counter()
    without_morph = 0
    untagged: dict[str, Counter] = defaultdict(Counter)
    untagged_info: dict[str, dict] = {}
    tokens = 0
    tagged = 0
    tagged_by_tier: Counter = Counter()
    tagged_by_book: Counter = Counter()
    methods: Counter = Counter()
    bands: dict[str, Counter] = defaultdict(Counter)
    for book_id, _, _, outcome in results:
        for item, choice in outcome:
            tokens += 1
            tier = item.tier
            tiers[tier] += 1
            by_book[book_id][tier] += 1
            for rule in item.token.rules:
                rules[rule] += 1
            if choice:
                tagged += 1
                tagged_by_tier[tier] += 1
                tagged_by_book[book_id] += 1
                methods[choice.method] += 1
                bands["all"][chooser_module.band(choice.confidence)] += 1
                bands[f"method:{choice.method}"][chooser_module.band(choice.confidence)] += 1
                bands[f"tier:{tier}"][chooser_module.band(choice.confidence)] += 1
                chosen_sources[choice.candidate.source] += 1
                if not choice.candidate.morph:
                    without_morph += 1
                continue
            reasons[item.reason] += 1
            key = item.token.key
            untagged[key][item.token.surface] += 1
            info = untagged_info.setdefault(key, {"tiers": Counter(), "candidates": item.readings()[:3]})
            info["tiers"][item.tier] += 1

    def share(counter: Counter, total: int) -> dict:
        return {tier: {"tokens": counter[tier], "percent": round(100 * counter[tier] / total, 2) if total else 0.0}
                for tier in TIERS}

    ranked = sorted(untagged, key=lambda key: (-sum(untagged[key].values()), key))[:UNTAGGED_FORMS]
    try:
        shown = source.resolve().relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        shown = source.name    # the same input reports the same, wherever it is
    return {
        "source": shown,
        "providers": tagger.provider_names(),
        "tierDescriptions": TIERS,
        "tokens": tokens,
        "tagged": tagged,
        "taggedPercent": round(100 * tagged / tokens, 2) if tokens else 0.0,
        "taggedWithoutMorphology": without_morph,
        "tiers": share(tiers, tokens),
        "taggedByTier": {tier: tagged_by_tier[tier] for tier in TIERS},
        "taggedByMethod": dict(sorted(methods.items(), key=lambda item: (-item[1], item[0]))),
        "confidenceBands": {
            group: {name: counts[name] for name in chooser_module.BANDS}
            for group, counts in sorted(bands.items())
        },
        "chooser": getattr(tagger.chooser, "meta", None) or {"policy": "unique reading only"},
        "byBook": {
            book: {"tokens": sum(counts.values()), "tiers": {tier: counts[tier] for tier in TIERS},
                   "tagged": tagged_by_book[book],
                   "taggedPercent": round(100 * tagged_by_book[book] / sum(counts.values()), 2)}
            for book, counts in by_book.items()
        },
        "untaggedReasons": dict(sorted(reasons.items(), key=lambda item: (-item[1], item[0]))),
        "taggedBySource": dict(sorted(chosen_sources.items(), key=lambda item: (-item[1], item[0]))),
        "normalisationRules": dict(sorted(rules.items(), key=lambda item: (-item[1], item[0]))),
        "untaggedForms": [
            {
                "key": key,
                "tokens": sum(untagged[key].values()),
                "surfaces": [surface for surface, _ in sorted(untagged[key].items(), key=lambda i: (-i[1], i[0]))[:3]],
                "tiers": dict(sorted(untagged_info[key]["tiers"].items())),
                "readings": ["|".join(part or "" for part in reading) for reading in untagged_info[key]["candidates"]],
            }
            for key in ranked
        ],
    }


# What the module says of its own tags, where the apps show a text's sources
# (schemas/translation_schema.json, meta.sourceTexts.annotations). The apps
# print one sentence per entry, "Strong's numbers in {name} come from
# {source}, {method}.", and tell the entries apart by their source, so the
# two sources must differ and each method must read on from its source.
WORD_TAGS = {
    "part": "strongs",
    "source": "STEPBible's tagged Greek New Testament and Greek lexicon (TAGNT and TBESG), created by "
              "STEPBible.org based on work at Tyndale House, Cambridge, CC BY 4.0",
    "method": "matched to each word, with its dictionary form and parse, by Lamp Bible's own program from "
              "the word's form and its context; made by machine and not checked by hand, so some are wrong",
}
NAME_TAGS = {
    "part": "strongs",
    "source": "STEPBible's tagged Hebrew Old Testament and Hebrew lexicon (TAHOT and TBESH), CC BY 4.0",
    "method": "for names the New Testament does not use: each takes the number of the Hebrew name it "
              "stands for in the same verse, matched by how the Septuagint spells Hebrew names",
}
TAGGED_LICENSE = "Public Domain; tagging CC BY 4.0"
TAGGED_COPYRIGHT = ("Strong's numbers, dictionary forms and morphology from data created by STEPBible.org "
                    "based on work at Tyndale House, Cambridge.")
TAGGED_DESCRIPTION = (" Each word's Strong's number, dictionary form and parse were assigned by Lamp Bible's "
                      "own program and have not been checked by hand.")


def describe_tags(meta: dict, name_keys: str) -> None:
    """Say in the module's own metadata where its tags came from and that a
    program assigned them: the credit STEPBible's licence asks for, and the
    disclosure the project owes for labelling a text by its own analysis."""
    sources = [dict(WORD_TAGS)] + ([dict(NAME_TAGS)] if name_keys != "off" else [])
    meta["sourceTexts"] = {**(meta.get("sourceTexts") or {}), "annotations": sources}
    meta["license"] = TAGGED_LICENSE
    meta["copyright"] = TAGGED_COPYRIGHT
    if meta.get("description") and not meta["description"].endswith(TAGGED_DESCRIPTION):
        meta["description"] += TAGGED_DESCRIPTION


def write_json(path: Path, data: dict, like: Path | None = None) -> None:
    """JSON as the repository writes it: UTF-8, sorted keys, two-space indent,
    a final newline if the file it is modelled on has one."""
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    newline = True
    if like is not None:
        with open(like, "rb") as handle:
            handle.seek(-1, 2)
            newline = handle.read(1) == b"\n"
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text + ("\n" if newline else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_RUN / "LXX_source.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bundle-module", type=Path,
                        help="also write the tagged module here (outputs/LXX.json, for the bundle)")
    parser.add_argument("--sources", type=Path, default=ntforms.DEFAULT_SOURCES,
                        help="directory with TAGNT_*.txt and TBESG.txt")
    parser.add_argument("--books", help="book ids to tag, comma-separated (e.g. Gen,Ps); the case profile still reads every book")
    parser.add_argument("--no-paradigm", action="store_true",
                        help="do not add the forms every paradigm shares (for measuring their effect)")
    parser.add_argument("--chooser", choices=("model", "unique"), default="model",
                        help="model (default): choose every token with candidates in context; "
                             "unique: tag only tokens with one reading")
    parser.add_argument("--chooser-cache", type=Path, default=chooser_module.CACHE_DIR,
                        help="where the trained chooser is cached")
    parser.add_argument("--retrain", action="store_true", help="train the chooser even if a cached model fits")
    parser.add_argument("--verbose", action="store_true", help="report the chooser's training epoch by epoch")
    parser.add_argument("--no-analyser", action="store_true",
                        help="leave out the morphological analyser (forms TAGNT lacks stay untagged)")
    parser.add_argument("--name-keys", choices=names_module.KEY_MODES, default="hebrew",
                        help="hebrew (default): tier-d names take the Hebrew Strong's number of the name they "
                             "render, without morphology; greek-extended: that number + 10000 as a Greek key, "
                             "with morphology; off: tier d stays untagged")
    parser.add_argument("--hebrew-sources", type=Path,
                        help="directory with TAHOT_*.txt and TBESH.txt (default: --sources)")
    parser.add_argument("--verse-map", help="a JSON file mapping Septuagint references to English ones, or "
                        "'identity'; default: lamp-bible-core's `lamp-module versify lxx eng`, cached")
    parser.add_argument("--jobs", type=int, default=4,
                        help="worker processes for building the chooser's training folds, if it must be "
                             "trained (default 4)")
    options = parser.parse_args(argv)
    # The fold builder asks `default_jobs` how many workers to start.
    chooser_module.default_jobs = lambda: max(1, options.jobs)

    module = json.loads(options.input.read_text(encoding="utf-8"))
    forms = ntforms.build(ntforms.tagnt_paths(options.sources))
    lexicon = lexicon_module.build(options.sources / "TBESG.txt")
    profile = CaseProfile.from_texts(verse_text(verse) for _, _, verse in verses(module))
    paradigm = not options.no_paradigm

    # One spec builds the tagger for the LXX and for every training fold of
    # the chooser, so the chooser learns from candidates made as the LXX's
    # are (`pipeline.py`). The analyser reads forms TAGNT lacks; it generates
    # only the forms whose letters the text has.
    factory = TaggerSpec(tbesg_path=options.sources / "TBESG.txt", paradigm=paradigm,
                         analyser=not options.no_analyser)
    vocabulary = {bare(token.key) for _, _, verse in verses(module) for token in tokenize(verse_text(verse))}
    if options.chooser == "model":
        chooser = chooser_module.load_or_train(options.sources, factory, options.chooser_cache,
                                               retrain=options.retrain, verbose=options.verbose)
    else:
        chooser = unique_reading_chooser
    tagger = factory(forms, lexicon, profile, vocabulary=vocabulary)
    tagger.chooser = chooser

    if hasattr(tagger.chooser, "prepare"):
        # The chooser reads the whole text's candidates first (which words
        # are certain where: see `chooser.certain_lexemes`).
        tagger.chooser.prepare((tagger.candidates_for(token) for _, _, verse in verses(module)
                                for token in tokenize(verse_text(verse))), lexicon)
    books = set(options.books.split(",")) if options.books else None
    name_stage = None
    if options.name_keys != "off":
        hebrew_sources = options.hebrew_sources or options.sources
        refs = [verse["ref"] for _, _, verse in verses(module)]
        if options.verse_map == "identity":
            verse_map = hebrew_names.VerseMap.identity(refs)
        elif options.verse_map:
            verse_map = hebrew_names.VerseMap.from_json(json.loads(Path(options.verse_map).read_text(encoding="utf-8")))
        else:
            verse_map = hebrew_names.versify_map(refs)
        hebrew = hebrew_names.build(hebrew_sources)
        tbesh = hebrew_names.read_tbesh(hebrew_sources / "TBESH.txt")

        def factory(exclude):
            return names_module.build_stage(hebrew_sources, lexicon, verse_map, options.name_keys,
                                            tagger.codes, hebrew=hebrew, exclude=exclude, forms=forms, tbesh=tbesh)

        def name_stage(results):
            run = names_module.tag_names(names_module.views_from_results(results), factory)
            apply_names(results, run, tagger.codes)
            # Common words Brenton capitalises, which the name stage found no
            # Hebrew name for, as the common words they are (`capitals.py`).
            common = capitals.apply(results, tagger, run)
            results.capital_common = common
            for position, note in common.items():
                run.notes[position] = note

    def after(results):
        if name_stage is not None:
            name_stage(results)
        apply_name_parses(results, tagger.codes)
    tagged, results = tag_module(module, tagger, books, after)
    check_texts_unchanged(module, tagged)

    output = options.output_dir
    output.mkdir(parents=True, exist_ok=True)
    describe_tags(tagged["meta"], options.name_keys)
    write_json(output / "LXX_tagged.json", tagged, like=options.input)
    if options.bundle_module is not None:
        if options.books:
            raise SystemExit("--bundle-module needs the whole text: drop --books")
        options.bundle_module.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(output / "LXX_tagged.json", options.bundle_module)
    report = build_report(results, tagger, options.input)
    misprinted: Counter = Counter()
    for _, _, _, outcome in results:
        for item, _ in outcome:
            if "misprint" in item.token.rules:
                misprinted[normalize(item.token.surface)] += 1
    report["misprints"] = {
        "description": "printed forms read as another word because text.MISPRINT_TABLE lists them as "
                       "misprints of the digitised text; the text itself is unchanged",
        "table": [{"printed": printed, "intended": intended, "readIn": where,
                   "tokens": misprinted[normalize(printed)]} for printed, intended, where in MISPRINT_TABLE],
    }
    report["capitalisedCommonWords"] = {
        "description": "tier-d words the name stage found no Hebrew name for, tagged from their lower-case "
                       "readings (capitals.py)",
        "tokens": len(getattr(results, "capital_common", {})),
        "forms": dict(Counter(results[n][3][p][0].token.surface for n, p in
                              getattr(results, "capital_common", {})).most_common(30)),
    }
    report["tolerantNameParses"] = {
        "description": "names tagged from a loose match with an NT name: the rule that read their case "
                       "from the Septuagint word (name_parse.py), or unknown (the NT form's parse kept, "
                       "its confidence capped)",
        "byRule": dict(sorted(Counter(getattr(results, "name_parses", {}).values()).items())),
    }
    if getattr(results, "names", None) is not None:
        report["names"] = names_module.report(results.names)
    write_json(output / "lxx_tagging_report.json", report)
    write_decisions(output / "lxx_tagging_decisions.tsv", results)

    print(f"{report['tokens']:,} tokens; tagged {report['tagged']:,} ({report['taggedPercent']}%), "
          f"{report['taggedWithoutMorphology']:,} of them without morphology")
    for tier, description in TIERS.items():
        print(f"  {tier}  {report['tiers'][tier]['tokens']:>8,}  {report['tiers'][tier]['percent']:>6}%  "
              f"tagged {report['taggedByTier'][tier]:>8,}  {description}")
    if "all" in report["confidenceBands"]:
        print("  confidence: " + ", ".join(f"{name} {count:,}" for name, count in report["confidenceBands"]["all"].items()))
    if "names" in report:
        names = report["names"]
        print(f"  names ({names['keyMode']} keys): tier d {names['tierD']['before']['tokens']:,} -> "
              f"{names['tierD']['after']['tokens']:,} untagged; " + ", ".join(
                  f"{method} {count:,}" for method, count in names["tagged"]["byMethod"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
