"""The parse of a name the chooser tagged from a loose match with an NT name.

WHY

A capitalised word TAGNT lacks can still be found among TAGNT's forms with the
same letters and other marks (`nt-tolerant`, tier u): Brenton's Ἄχαζ is the
NT's Ἀχάζ, his Ἠλιού the NT's Ἡλίου. The chooser picks one of those NT forms
(method `tolerant` or `tolerant-mismatch`), and with it that form's parse. The
Strong's number and lemma are usually right; the parse is the NT form's, not
the Septuagint word's: Ἠλιού is indeclinable in the Septuagint but Ἡλίου is
the NT's genitive of Ἡλίας, and an indeclinable Ἄχαζ is accusative or
nominative only as the NT happens to use it.

WHAT

`reparse` reads the case of such a name from its own form and its context,
by rules of Greek grammar, in this order (`RULES`):

  subscript    a final ᾳ ῃ ῳ is a dative singular (Φαραῷ)
  article      the article just before it, as the chooser tagged it, gives
               case, number and gender (τῷ Ἰσαάκ, τοῦ Μαδιάμ)
  preposition  a preposition just before it that takes one case (ἐν, εἰς,
               ἐκ, ἀπό, ἐνώπιον ...: `names.PREPOSITION_CASES`)
  ending       a form whose letters differ from its NT lemma's, and whose
               case ending read back to a nominative gives that lemma, accent
               and all, has that ending's case (Ἡσαΐαν, accusative of
               Ἡσαΐας); Ἠλιού is not Ἡλίου, the genitive of Ἡλίας, since the
               accent has moved: it is undeclined
  kinship      after υἱός, θυγάτηρ, ἀδελφός, πατήρ, γυνή, οἶκος, χείρ and the
               like, a name is the genitive of the person (υἱὸς Νάθαν)
  apposition   after a title the chooser tagged in a case (ὁ βασιλεὺς Ἄχαζ,
               ὁ προφήτης Ἠλιού) the name takes the title's case

Number and gender are the NT form's (a person's name is singular); the
suffix (-P, -L, -T ...) is kept. A rule's code is used only if TAGNT uses it.

When no rule applies the case is unknown: the chooser's parse is kept, but its
confidence is capped (unless it is a dative the printed form rules out, a long
-α -η -ω with no iota subscript: then the name is tagged without a parse, as
TBESG's names are), at `UNKNOWN_CONFIDENCE` (the share of such parses the
New Testament check finds right), so the decisions file does not claim a
borrowed parse is likely. The rule that set a parse is written to the
decisions file's `name_note`.

The rules were measured on STEPBible's tagged New Testament (indeclinable
names there carry a case, so the rules can be scored on the cross-validation's
own choices of the word before), and each rule's confidence is that
measurement (`RULE_CONFIDENCE`).
"""

from __future__ import annotations

import sys
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging.candidates import Candidate, Choice  # noqa: E402
from lxx_tagging.lexicon import split_variants  # noqa: E402
from lxx_tagging.names import PREPOSITION_CASES, nominative, paradigm_readings  # noqa: E402
from lxx_tagging.text import bare, normalize  # noqa: E402

METHODS = ("tolerant", "tolerant-mismatch")
RULES = ("subscript", "article", "preposition", "ending", "kinship", "apposition")
NAME_SUFFIXES = ("P", "L", "PG", "LG", "T", "PI", "LI")
KINSHIP = frozenset(normalize(w) for w in (
    "υἱός", "θυγάτηρ", "δοῦλος", "παῖς", "οἶκος", "χείρ", "τέκνον", "σπέρμα", "πρόσωπον", "θεός", "λαός",
    "γῆ", "πόλις", "ὄρος", "ἔρημος", "βασιλεία", "φυλή", "δῆμος", "ἡμέρα", "ὀφθαλμός", "στόμα"))
# Left out, because a name after them is as often in apposition as a
# genitive (the NT check): ὄνομα (ὀνόματι Ἰωσήφ), ἀνήρ, πατήρ, ἀδελφός.
TITLES = frozenset(normalize(w) for w in (
    "βασιλεύς", "προφήτης", "ἱερεύς", "ἀρχιερεύς", "ἄρχων", "βασίλισσα", "γραμματεύς", "ἡγούμενος"))

_PREPOSITIONS = {normalize(k): v for k, v in PREPOSITION_CASES.items() if v}

# Measured on the NT's names, each rule's share right smoothed (one error
# and one success added), with the 5-fold cross-validation's own choices for
# the word before: on indeclinable names, article 1,217 of 1,241,
# preposition 144 of 144, subscript 57 of 57, kinship 105 of 113, apposition
# 6 of 7; on declined names, ending 533 of 541. Where no rule applied (1,311
# indeclinable names) the chooser's own parse was right 596 times.
RULE_CONFIDENCE = {"subscript": 0.98, "article": 0.98, "preposition": 0.99, "ending": 0.98, "kinship": 0.92,
                   "apposition": 0.78}
UNKNOWN_CONFIDENCE = 0.45


@dataclass(frozen=True)
class Neighbour:
    """What a rule reads of the token before: its key, the chosen lemma's
    key and the chosen code."""

    key: str
    lemma: str = ""
    morph: str = ""


def _letters(text: str) -> str:
    return bare(normalize(text)).replace("ς", "σ")


def _cng(code: str | None) -> tuple[str, str, str] | None:
    if not code:
        return None
    parts = code.split("-")
    if len(parts) > 1 and len(parts[1]) == 3 and parts[1][0] in "NGDAV":
        return parts[1][0], parts[1][1], parts[1][2]
    return None


def is_name_code(code: str | None) -> bool:
    if not code or not code.startswith("N-"):
        return False
    parts = code.split("-")
    return len(parts) > 2 and parts[2] in NAME_SUFFIXES


def no_dative(surface: str) -> bool:
    """A form ending in a long α, η or ω without an iota subscript is no
    first- or second-declension dative singular (Μωυσῆ is not the NT's
    Μωϋσῇ), and a name has no other dative in those letters."""
    last = unicodedata.normalize("NFD", surface)
    letters = [c for c in last if not unicodedata.combining(c)]
    return bool(letters) and letters[-1].lower() in "αηω" and "\u0345" not in last[-3:]


def name_case(surface: str, key: str, lemma: str, parses: Iterable[str], previous: Neighbour | None,
              person: bool = True) -> tuple[str | None, str | None, str | None, str]:
    """(case, number, gender, rule) for a name from its form and the token
    before it; case None and rule "" when no rule decides."""
    plain = unicodedata.normalize("NFC", surface)
    if plain[-1:] in ("ᾳ", "ῃ", "ῳ") or (plain[-1:] and unicodedata.normalize("NFD", plain[-1])[-1:] == "ͅ"):
        return "D", "S", None, "subscript"
    if previous is not None:
        article = _cng(previous.morph) if previous.morph.startswith("T-") else None
        if article:
            return article[0], article[1], article[2], "article"
        fixed = _PREPOSITIONS.get(previous.key)
        if fixed and previous.morph.startswith("PREP"):
            return fixed, None, None, "preposition"
    variants = split_variants(lemma) or [lemma]
    if _letters(key) not in {_letters(v) for v in variants}:
        # An inflected form of the NT lemma: its ending, read back to the
        # lemma itself with the accent where the lemma has it (Ἡσαΐαν ->
        # Ἡσαΐας). Ἠλιού is not Ἡλίου, the genitive of Ἡλίας: its accent
        # has moved, so it is the Septuagint's own undeclined spelling.
        wanted = {normalize(v) for v in variants}
        cases = set()
        for paradigm, _, case, number, ending in paradigm_readings(surface):
            made = nominative(surface, ending, paradigm)
            if made and normalize(made) in wanted and case != "?":
                cases.add((case, number))
        if no_dative(surface):
            cases = {c for c in cases if c[0] != "D"}
        if len(cases) == 1:
            case, number = cases.pop()
            return case, number, None, "ending"
    if previous is not None:
        if previous.lemma in KINSHIP and previous.morph.startswith("N-"):
            return "G", "S", None, "kinship"
        title = _cng(previous.morph)
        if previous.lemma in TITLES and title and person:
            # A person's name only: βασιλεὺς Ἰσραήλ is the king of Israel.
            return title[0], title[1], None, "apposition"
    return None, None, None, ""


def reparse(candidate: Candidate, surface: str, key: str, parses: Sequence[str], previous: Neighbour | None,
            codes: set[str] | None) -> tuple[Candidate, str]:
    """The candidate with its parse read from the Septuagint word (see the
    module docstring), and the rule used ("" for none)."""
    if not is_name_code(candidate.morph):
        return candidate, ""
    case, number, gender, rule = name_case(surface, key, candidate.lemma, parses, previous,
                                           person=candidate.morph.endswith("-P"))
    if case is None:
        return candidate, ""
    head, cng, *rest = candidate.morph.split("-")
    number = number or cng[1]
    gender = gender or cng[2]
    code = "-".join([head, case + number + gender, *rest])
    if codes is not None and code not in codes:
        # The NT form's gender with the rule's case, if the article's gender
        # makes a code TAGNT lacks.
        code = "-".join([head, case + number + cng[2], *rest])
        if codes is not None and code not in codes:
            return candidate, ""
    if code == candidate.morph:
        return candidate, rule
    return replace(candidate, morph=code, source=candidate.source + "+name-parse",
                   notes=(candidate.notes + "; " if candidate.notes else "") + f"case from the {rule}"), rule


def apply(results, codes: set[str] | None, lemma_key=None) -> dict:
    """Reparse every tolerant name choice of `tag_septuagint`'s results in
    place. Returns {(verse number, position): rule or "unknown"}."""
    from lxx_tagging.chooser import lemma_key as default_lemma_key
    lemma_key = lemma_key or default_lemma_key
    done: dict[tuple[int, int], str] = {}
    for number, (_, _, _, outcome) in enumerate(results):
        for position, (item, choice) in enumerate(outcome):
            if choice is None or choice.method not in METHODS or not is_name_code(choice.candidate.morph):
                continue
            previous = None
            if position > 0:
                before_item, before = outcome[position - 1]
                previous = Neighbour(before_item.token.key,
                                     lemma_key(before.candidate.lemma) if before else "",
                                     (before.candidate.morph or "") if before else "")
            parses = [c.morph for c in item.candidates
                      if c.morph and c.strongs == choice.candidate.strongs]
            candidate, rule = reparse(choice.candidate, item.token.surface, item.token.key, parses, previous, codes)
            if rule:
                confidence = RULE_CONFIDENCE[rule]
            else:
                confidence = min(choice.confidence, UNKNOWN_CONFIDENCE)
                if _cng(candidate.morph) and _cng(candidate.morph)[0] == "D" and no_dative(item.token.surface):
                    # The NT form's dative, which the printed form rules out:
                    # no parse rather than a wrong one.
                    candidate = replace(candidate, morph=None, source=candidate.source + "+name-parse",
                                        notes=(candidate.notes + "; " if candidate.notes else "")
                                        + "the NT form's dative does not fit; case unknown")
                    rule = "contradicted"
            outcome[position] = (item, Choice(candidate, confidence, choice.margin, choice.method))
            done[(number, position)] = rule or "unknown"
    return done


NOTES = {
    "subscript": "parse: dative from the iota subscript",
    "article": "parse: case, number and gender from the article before it",
    "preposition": "parse: case from the preposition before it",
    "ending": "parse: case from its ending",
    "kinship": "parse: genitive after a word of kinship or possession",
    "apposition": "parse: the case of the title before it",
    "unknown": "parse: the NT form's; the case is not known, so its confidence is capped",
    "contradicted": "parse: none; the NT form's dative does not fit the printed form and the case is not known",
}
