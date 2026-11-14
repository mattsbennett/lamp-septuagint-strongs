"""Common words Brenton capitalises, tagged as the common words they are.

WHY

Brenton capitalises the points of the compass and a few other common words
where they stand like names (πρὸς Βοῤῥᾶν, ἀπὸ Λιβός, κατὰ Νότον; Ἄγγελος,
Οὐαί). A capital in the middle of a sentence makes the case filter set their
common readings aside (`candidates.py`), so they reach the name stage as
tier d; and having no Hebrew name to match, they stayed untagged.

WHAT

After the name stage, a tier-d word it left untagged is looked up again as if
printed in lower case, through the same providers and case filter
(`Tagger.candidates_for`), when:

  * the name stage found no Hebrew name for it: its note says there is no
    matching Hebrew name, or none in the verse, and not that a Hebrew name of
    the verse may be its counterpart, nor that it is ambiguous between names,
    nor that the name it would match was taken by another Greek word
    (`BLOCKING_NOTES`);
  * the lower-case lookup gives readings that are not guesses: TAGNT's own
    form, or the analyser's form with its marks as printed (weight 1, not an
    accent or breathing mismatch), or TBESG's entry for the word itself when
    the verse's Hebrew has no name at all; never a tolerant match;
  * Brenton prints the same word (Strong's number) in lower case somewhere
    else, so he uses it as a common word: Ῥαφίς (Rezeph, and Rabsaris) is not
    ῥαφίς "needle", which he never prints; Σίκιμα (Shechem) is not TBESG's
    σίκιμα "a portion", Ἡρώων πόλις not ἥρως;
  * the reading is not a verb, unless it is a participle the article just
    before it agrees with (ὁ Ὤν): a capitalised verb in the middle of a
    sentence is likelier a name spelt like one (Ὤν, the city On; Κύρου,
    Cyrus, not an imperative of κυρόω).

The verse is then chosen again by the chooser with those tokens' new
candidates, and only those tokens take the new choice; every other token
keeps its own. The decisions file's `reason` says "capitalised common word"
and `name_note` how it was found. A real name is not turned into a common
word, because it is never offered unless the Hebrew verse lacks a name the
matcher could pair it with.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging.candidates import TokenCandidates, classify  # noqa: E402

REASON = "capitalised common word"
NOTE = "a common word Brenton capitalises: tagged from its lower-case readings (no Hebrew name for it)"
ALLOWED_NOTES = ("no matching Hebrew name", "no Hebrew name in the verse")
BLOCKING_NOTES = ("counterpart", "ambiguous between Hebrew names", "taken by another",
                  "matched several Hebrew numbers")


def _family(source: str) -> str:
    return source.split("+")[0].split(":")[0]


def acceptable(candidate, verse_has_names: bool) -> bool:
    """A lower-case reading that is not a guess (see the module docstring)."""
    if not candidate.taggable or candidate.kind != "common":
        return False
    family = _family(candidate.source)
    if family in ("nt-form", "nt-caps", "nt-compound"):
        return True
    if family == "analyser":
        return candidate.weight >= 1.0
    if family == "tbesg":
        return not verse_has_names
    return False


def _agrees(article: str | None, morph: str | None) -> bool:
    """Whether an article's code and a participle's agree in case, number and gender."""
    if not article or not morph or not article.startswith("T-"):
        return False
    return morph.split("-")[-1] == article.split("-")[1]


def lower_case_candidates(tagger, item: TokenCandidates, note: str, common_numbers: set[str] | None = None,
                          previous_morph: str | None = None) -> TokenCandidates | None:
    """The token's readings as a lower-case word, if it qualifies."""
    if item.tier != "d":
        return None
    if not note.startswith(ALLOWED_NOTES) or any(block in note for block in BLOCKING_NOTES):
        return None
    verse_has_names = not note.startswith("no Hebrew name in the verse")
    lowered = dataclasses.replace(item.token, case="lower")
    found = tagger.candidates_for(lowered)
    kept = [c for c in found.candidates if acceptable(c, verse_has_names)
            and (common_numbers is None or c.strongs in common_numbers)
            and (not (c.morph or "").startswith("V-") or _agrees(previous_morph, c.morph))]
    if not kept:
        return None
    tier, _ = classify(lowered, kept, [], "common")
    return TokenCandidates(item.token, kept, list(item.set_aside), tier, REASON)


def apply(results, tagger, run) -> dict[tuple[int, int], str]:
    """Tag the qualifying tier-d words of `tag_septuagint`'s results in place,
    choosing each such verse again. Returns {(verse number, position): note}."""
    if run is None or tagger.chooser is None:
        return {}
    done: dict[tuple[int, int], str] = {}
    common_numbers = {choice.candidate.strongs for _, _, _, outcome in results for item, choice in outcome
                      if choice is not None and item.token.case == "lower"}
    for number, (_, _, _, outcome) in enumerate(results):
        changed: dict[int, TokenCandidates] = {}
        for position, (item, choice) in enumerate(outcome):
            if choice is not None or item.tier != "d":
                continue
            before = outcome[position - 1][1] if position > 0 else None
            replacement = lower_case_candidates(tagger, item, run.notes.get((number, position), ""), common_numbers,
                                                before.candidate.morph if before else None)
            if replacement is not None:
                changed[position] = replacement
        if not changed:
            continue
        verse = [changed.get(position, item) for position, (item, _) in enumerate(outcome)]
        choices = tagger.chooser(verse)
        runners = getattr(tagger.chooser, "runner_up", {}) or {}
        for position, replacement in changed.items():
            choice = choices[position]
            if choice is None:
                continue
            if tagger.codes is not None and choice.candidate.morph and choice.candidate.morph not in tagger.codes:
                continue
            outcome[position] = (replacement, choice)
            done[(number, position)] = NOTE
            runner = runners.get(replacement.token.index)
            if runner is not None and hasattr(results, "runner_ups"):
                results.runner_ups.setdefault(number, {})[replacement.token.index] = runner
    return done
