#!/usr/bin/env python3
"""Open evaluation of the morphological analyser (`analyser.py`).

Two checks, both on CC BY data only:

  nt      Leave forms out on the New Testament. TAGNT's distinct forms are
          split into folds (by a hash of the form); for each fold the
          analyser is built with that fold's rows withheld from everything it
          learns (stems, declensions, voice letters, suffixes), then asked to
          read each held-out form. Reported: recall (the true reading among
          the candidates), the number of candidates, and how often the top
          candidate is the true reading, by part of speech and, for verbs,
          by tense and mood; by token and by form. Withholding a whole fold
          of forms is stricter than leaving one form out at a time.
  lxx     Brenton's Septuagint tagged with and without the analyser: tiers
          before and after, the share of tier-e tokens and forms that get a
          candidate, the candidate-set sizes, the forms still without one
          (with a guess at why and at the missing lemma), and a random
          sample of analysed tier-e tokens in their verse for reading.

Usage:
    python3 scripts/lxx_tagging/analyser_eval.py nt [--folds 5] [--only-fold 0]
    python3 scripts/lxx_tagging/analyser_eval.py lxx [--sample 150]
    python3 scripts/lxx_tagging/analyser_eval.py all --output-dir DIR

Writes `analyser_nt_eval.json` (and `analyser_nt_misses.tsv`, every held-out
reading the analyser missed), `analyser_lxx_eval.json` and
`analyser_lxx_sample.tsv` to --output-dir (default .cache/lxx_tagging/analyser_eval/).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging import analyser as analyser_module  # noqa: E402
from lxx_tagging import lexicon as lexicon_module  # noqa: E402
from lxx_tagging import ntforms  # noqa: E402
from lxx_tagging.candidates import default_tagger, CaseProfile  # noqa: E402
from lxx_tagging.text import bare, tokenize  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent


def _pos(morph: str) -> str:
    return {"V": "verb", "N": "noun", "A": "adjective"}.get(morph.split("-")[0], "other")


def _verb_group(morph: str) -> str:
    match = re.match(r"V-2?([PIFARL])[A-Z]([ISOMNP])", morph)
    if not match:
        return "?"
    return {"P": "present", "I": "imperfect", "F": "future", "A": "aorist", "R": "perfect",
            "L": "pluperfect"}[match[1]] + " " + {"I": "indicative", "S": "subjunctive", "O": "optative",
                                                  "M": "imperative", "N": "infinitive", "P": "participle"}[match[2]]


class Tally:
    def __init__(self):
        self.tokens = Counter()
        self.forms = Counter()
        self.sizes: list[int] = []

    def add(self, found: bool, top: bool, size: int, count: int) -> None:
        self.tokens["n"] += count
        self.forms["n"] += 1
        self.tokens["recall"] += count * found
        self.forms["recall"] += found
        self.tokens["top1"] += count * top
        self.forms["top1"] += top
        self.tokens["answered"] += count * (size > 0)
        self.forms["answered"] += size > 0
        self.sizes.append(size)

    def report(self) -> dict:
        def share(counter, name):
            return round(counter[name] / counter["n"], 4) if counter["n"] else 0.0
        answered = [s for s in self.sizes if s]
        return {
            "forms": self.forms["n"], "tokens": self.tokens["n"],
            "recallTokens": share(self.tokens, "recall"), "recallForms": share(self.forms, "recall"),
            "top1Tokens": share(self.tokens, "top1"), "top1Forms": share(self.forms, "top1"),
            "answeredForms": share(self.forms, "answered"),
            "meanCandidates": round(statistics.mean(answered), 2) if answered else 0.0,
            "medianCandidates": statistics.median(answered) if answered else 0,
        }


def nt_leave_out(forms, lexicon, folds: int = 5, only: int | None = None, misses_path: Path | None = None) -> dict:
    """See the module docstring."""
    by_fold: dict[int, list[str]] = defaultdict(list)
    for key in forms.keys():
        by_fold[zlib.crc32(key.encode("utf-8")) % folds].append(key)
    overall, by_pos, by_group = Tally(), defaultdict(Tally), defaultdict(Tally)
    out_of_scope = Counter()
    misses: list[tuple] = []
    started = time.time()
    hints = analyser_module.genitive_hints(lexicon_module.DEFAULT_TBESG)
    for fold in range(folds):
        if only is not None and fold != only:
            continue
        keys = by_fold[fold]
        analyser = analyser_module.Analyser(lexicon, forms, vocabulary={bare(k) for k in keys},
                                            exclude_keys=keys, hints=hints)
        in_scope = {lexeme.strongs: lexeme.pos for lexeme in analyser.lexemes}
        for key in keys:
            candidates = None
            for reading, count in forms.readings(key).items():
                morph = reading.morph
                pos = _pos(morph)
                if pos == "other" or " + " in morph or re.search(r"-(P|L|PG|LG)$", morph):
                    out_of_scope["other part of speech or name"] += count
                    continue
                if reading.strongs not in in_scope:
                    out_of_scope["lemma not a TBESG verb, noun or adjective"] += count
                    continue
                if candidates is None:
                    candidates = analyser.candidates(key)
                readings = [(c.strongs, c.morph) for c in candidates]
                found = (reading.strongs, morph) in readings
                top = bool(readings) and readings[0] == (reading.strongs, morph)
                size = len({(c.strongs, c.lemma, c.morph) for c in candidates})
                overall.add(found, top, size, count)
                by_pos[pos].add(found, top, size, count)
                if pos == "verb":
                    by_group[_verb_group(morph)].add(found, top, size, count)
                if not found:
                    misses.append((key, reading.strongs, reading.lemma, morph, count,
                                   " ; ".join(c.compact() for c in candidates[:6])))
        print(f"  fold {fold}: {len(keys)} forms, {time.time() - started:.0f}s", file=sys.stderr)
    if misses_path:
        with open(misses_path, "w", encoding="utf-8") as handle:
            handle.write("key\tstrongs\tlemma\tmorph\tcount\tcandidates\n")
            for row in sorted(misses, key=lambda r: (-r[4], r[0])):
                handle.write("\t".join(str(x) for x in row) + "\n")
    return {
        "method": f"{folds}-fold leave-forms-out" + (f" (fold {only} only)" if only is not None else ""),
        "overall": overall.report(),
        "byPartOfSpeech": {pos: tally.report() for pos, tally in sorted(by_pos.items())},
        "verbsByTenseMood": {group: tally.report() for group, tally in sorted(by_group.items())},
        "outOfScopeTokens": dict(out_of_scope),
    }


# ---------------------------------------------------------------------------

def _verses(module):
    for book in module["books"]:
        for chapter in book["chapters"]:
            for verse in chapter["verses"]:
                content = verse["content"]
                yield book["id"], chapter["chapter"], verse, content["text"] if isinstance(content, dict) else content


def guess_lemma(key: str, lexicon) -> str:
    """A guess at the dictionary form of a form no lexeme generates: the
    letters with a common ending replaced, if TBESG has the result."""
    letters = bare(key)
    for ending, replacements in (
            ("ουσιν", ("ω",)), ("ουσι", ("ω",)), ("ομεν", ("ω",)), ("ετε", ("ω",)), ("ειν", ("ω",)),
            ("σεν", ("ω",)), ("σαν", ("ω",)), ("θη", ("ω",)), ("ων", ("οσ", "η", "ον", "α")),
            ("οισ", ("οσ", "ον")), ("ου", ("οσ", "ον")), ("ουσ", ("οσ",)), ("ῳ", ("οσ", "ον")),
            ("ησ", ("η",)), ("ην", ("η",)), ("ασ", ("α",)), ("αν", ("α",)), ("αι", ("α", "η")),
            ("οσ", ("οσ",)), ("ον", ("οσ", "ον")), ("α", ("ον",))):
        if letters.endswith(ending) and len(letters) > len(ending) + 2:
            for replacement in replacements:
                found = lexicon.by_letters(letters[:-len(ending)] + replacement)
                if found:
                    return found[0].lemma
    return ""


_GREEK = "αβγδεζηθικλμνξοπρστυφχψω"
_MISSING_GUESSES = (
    ("ουσιν", "ω"), ("ουσι", "ω"), ("ομεν", "ω"), ("ετε", "ω"), ("ειν", "ω"), ("σεν", "ω"), ("σε", "ω"),
    ("σαν", "ω"), ("θησαν", "ω"), ("θη", "ω"), ("ονται", "ομαι"), ("εται", "ομαι"), ("οισ", "οσ"),
    ("ου", "οσ"), ("ουσ", "οσ"), ("ῳ", "οσ"), ("ον", "οσ"), ("οι", "οσ"), ("ων", "οσ"), ("ησ", "η"),
    ("ην", "η"), ("ῃ", "η"), ("αι", "η"), ("ασ", "α"), ("αν", "α"), ("ατα", "α"), ("ατοσ", "α"))


def explain_unresolved(key: str, lexicon, known_letters: set[str]) -> dict:
    """Why a form got no candidate, with a guess: a suspected misprint (one
    letter from a form TAGNT or the analyser knows: γἱοί), a phrase printed
    as one word (ἐξεναντίας), a TBESG lemma whose form was not derived, or
    a lemma TBESG probably lacks (with a guess at its dictionary form)."""
    letters = bare(key)
    if lexicon.by_letters(letters):
        entry = lexicon.by_letters(letters)[0]
        return {"reason": f"in TBESG as {entry.pos_class}, not generated", "guess": entry.lemma}
    for index in range(1, len(letters)):
        head, tail = letters[:index], letters[index:]
        if len(head) >= 2 and len(tail) >= 2 and head in known_letters and tail in known_letters:
            return {"reason": "phrase printed as one word", "guess": f"{head} + {tail}"}
    neighbours = set()
    for index in range(len(letters) + 1):
        for letter in _GREEK:
            neighbours.add(letters[:index] + letter + letters[index:])
            if index < len(letters):
                neighbours.add(letters[:index] + letter + letters[index + 1:])
        if index < len(letters):
            neighbours.add(letters[:index] + letters[index + 1:])
    neighbours.discard(letters)
    typo = sorted(n for n in neighbours if n in known_letters)
    guess = guess_lemma(key, lexicon)
    if guess:
        return {"reason": "lemma in TBESG, form not derived", "guess": guess}
    if typo and len(letters) > 3:
        return {"reason": "suspected misprint", "guess": typo[0]}
    for ending, replacement in _MISSING_GUESSES:
        if letters.endswith(ending) and len(letters) > len(ending) + 2:
            return {"reason": "lemma probably missing from TBESG", "guess": letters[:-len(ending)] + replacement}
    return {"reason": "lemma probably missing from TBESG", "guess": ""}


def lxx_evaluation(forms, lexicon, sample_size: int = 150, sample_path: Path | None = None,
                   seed: int = 0) -> dict:
    module = json.loads((REPOSITORY_ROOT / "outputs" / "lxx_tagging" / "LXX_source.json").read_text(encoding="utf-8"))
    verses = list(_verses(module))
    profile = CaseProfile.from_texts(text for *_, text in verses)
    vocabulary = {bare(token.key) for *_, text in verses for token in tokenize(text)}
    started = time.time()
    analyser = analyser_module.Analyser(lexicon, forms, vocabulary=vocabulary,
                                        tbesg_path=lexicon_module.DEFAULT_TBESG)
    print(f"  analyser built in {time.time() - started:.0f}s", file=sys.stderr)
    before = default_tagger(forms, lexicon, profile)
    after = default_tagger(forms, lexicon, profile)
    after.register("analyser", analyser, before="nt-tolerant")
    tiers_before, tiers_after = Counter(), Counter()
    e_tokens, e_answered = 0, 0
    e_types, e_types_answered = set(), set()
    sizes = Counter()
    unresolved = Counter()
    unresolved_info: dict[str, dict] = {}
    analysed_e: list[tuple] = []
    sources = Counter()
    large: dict[str, str] = {}
    for book, chapter, verse, text in verses:
        tokens = tokenize(text)
        for token in tokens:
            first = before.candidates_for(token)
            second = after.candidates_for(token)
            tiers_before[first.tier] += 1
            tiers_after[second.tier] += 1
            if first.tier not in ("e", "u"):
                continue
            e_tokens += 1
            e_types.add(token.key)
            ours = [c for c in second.candidates if c.source == analyser_module.SOURCE]
            if ours:
                e_answered += 1
                e_types_answered.add(token.key)
                size = len({c.reading for c in ours})
                sizes[size if size < 10 else "10+"] += 1
                if size >= 10:
                    large[token.key] = " ; ".join(f"{c.lemma}|{c.morph or ''}|{c.weight:g}" for c in ours[:12])
                analysed_e.append((f"{book}.{chapter}.{verse['v']}", token.surface, token.key, first.tier,
                                   " ; ".join(f"{c.strongs}|{c.lemma}|{c.morph or ''}|{c.weight:g}" for c in ours[:8]),
                                   text))
                sources[ours[0].notes.split(":")[0]] += 1
            elif first.tier == "e":
                unresolved[token.key] += 1
                if token.key not in unresolved_info:
                    unresolved_info[token.key] = {"surface": token.surface, "ref": f"{book}.{chapter}.{verse['v']}"}
    known_letters = {bare(key) for key in forms.keys()} | set(analyser.index)
    for key in unresolved:
        unresolved_info[key].update(explain_unresolved(key, lexicon, known_letters))
    reasons = Counter()
    for key, count in unresolved.items():
        reasons[unresolved_info[key]["reason"]] += count
    total = sum(tiers_before.values())
    if sample_path:
        random.Random(seed).shuffle(analysed_e)
        with open(sample_path, "w", encoding="utf-8") as handle:
            handle.write("ref\tsurface\tkey\ttier_before\tcandidates\tverse\n")
            for row in analysed_e[:sample_size]:
                handle.write("\t".join(field.replace("\t", " ") for field in row) + "\n")

    def tiers(counter):
        return {tier: {"tokens": counter[tier], "percent": round(100 * counter[tier] / total, 2)}
                for tier in "abcdeu"}

    return {
        "tokens": total,
        "tiersBefore": tiers(tiers_before),
        "tiersAfter": tiers(tiers_after),
        "tierEU": {"tokens": e_tokens, "tokensWithCandidate": e_answered,
                   "tokenShare": round(e_answered / e_tokens, 4) if e_tokens else 0,
                   "forms": len(e_types), "formsWithCandidate": len(e_types_answered),
                   "formShare": round(len(e_types_answered) / len(e_types), 4) if e_types else 0},
        "candidateSetSizes": {str(k): v for k, v in sorted(sizes.items(), key=lambda kv: str(kv[0]))},
        "largeSetExamples": dict(sorted(large.items())[:40]),
        "unresolvedTokens": sum(unresolved.values()),
        "unresolvedForms": len(unresolved),
        "unresolvedReasons": dict(reasons.most_common()),
        "topUnresolved": [{"key": key, "tokens": count, **unresolved_info[key]}
                          for key, count in unresolved.most_common(100)],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("nt", "lxx", "all"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int)
    parser.add_argument("--sample", type=int, default=150)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / ".cache" / "lxx_tagging" / "analyser_eval")
    options = parser.parse_args(argv)
    forms = ntforms.load()
    lexicon = lexicon_module.load()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    if options.what in ("nt", "all"):
        report = nt_leave_out(forms, lexicon, options.folds, options.only_fold,
                              options.output_dir / "analyser_nt_misses.tsv")
        (options.output_dir / "analyser_nt_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report["overall"], ensure_ascii=False))
        for pos, numbers in report["byPartOfSpeech"].items():
            print(pos, json.dumps(numbers, ensure_ascii=False))
    if options.what in ("lxx", "all"):
        report = lxx_evaluation(forms, lexicon, options.sample, options.output_dir / "analyser_lxx_sample.tsv")
        (options.output_dir / "analyser_lxx_eval.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("tiersBefore", "tiersAfter", "tierEU", "unresolvedReasons")},
                         ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
