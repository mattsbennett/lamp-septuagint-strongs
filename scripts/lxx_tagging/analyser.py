"""A morphological analyser for word forms the Greek New Testament never uses.

The tagger's first provider finds Brenton's word among TAGNT's forms. Most of
the rest are inflected forms of words the lexicon (TBESG) has: a dative the
New Testament happens not to use, a future of a verb it has only in the
present, an LXX-only compound. This provider reads such a form by generating
the paradigms of the lexicon's words and looking the form up in them.

HOW A FORM IS READ

  * Every TBESG verb, noun, adjective and adjective-like pronoun becomes
    one or more lexemes (`build_lexemes`, `_classes`, which also allows for
    TBESG's slips: ἐφοράω filed as a noun, λοβός as a name, εἰκοσαετής as
    indeclinable). A word TBESG files as an adjective that does not decline
    (ἐνταῦθα, ἑκκαίδεκα) is its own only form, coded ADV or A-NUI.
  * A verb lemma is split into preverbs and a base (κατα-λαμβάνω,
    ἐπι-συν-άγω; `morph_phon.split_preverbs`): every split whose base is a
    verb the lexicon or the irregular tables know, and the lemma whole too
    if that base is short (ἀπ-ορέω is not ἀπο-ρέω).
  * Each lexeme gets hypotheses for its stems and inflection class, from
    four sources, strongest first (`EVIDENCE_WEIGHTS`):
      nt         the lexeme's own TAGNT forms (`NTEvidence`): tense stems cut
                 from its attested forms by the ending tables
                 (`cut_verb_stems`), the declension its attested forms fit,
                 its TAGNT voice letters and "2" convention, the suffixes
                 (-T by number, -C) its codes carry
      nt-base    the same from TAGNT forms of other compounds of the same
                 base (προσκαταλαμβάνω inherits λαμβάνω's ἔλαβον, εἴληφα,
                 ἐλήμφθην), where the stems resemble the base
      irregular  the stated principal parts of irregular verbs and the
                 written-out athematic and irregular paradigms
                 (`morph_irregular`)
      rule       the regular derivations from the present or the
                 nominative (`regular_verb_stems`, `morph_nominal`)
    A hypothesis TAGNT's forms of the word contradict is demoted
    (`CONTRADICTED`); rules for a tense the evidence covers are weakened.
  * The paradigms (`morph_verbs`, `morph_nominal`) generate every form with
    its TAGNT code, each weighted by its hypothesis and by a mild preference
    for words TAGNT uses often (`_prior`); the forms are indexed by their
    letters. With a `vocabulary` only forms with those letters are kept.
  * A token is looked up by its letters and compared mark by mark: the
    accent where the rules place it, the breathing, the iota subscript
    (`_compare`). A form whose subscript differs counts as matching, at
    `SUBSCRIPT_MISMATCH` (Brenton's ἀποθνῄσκω, σῴζω); one whose accent or
    breathing differs is kept only if nothing matches better.

CANDIDATES

`Analyser.__call__(key, token)` is a provider (see `candidates.py`): every
reading whose lemma is a TBESG entry and whose generated form is the token's,
as `Candidate(source="analyser")`, the weights relative to the token's best
reading (1.0; ACCENT_MISMATCH or BREATHING_MISMATCH when only such forms
matched), readings below `MIN_SHARE` of the best dropped. `notes`
names the rule path: the hypothesis's source and system or class (e.g.
`nt-base:aorp2`, `rule:-ις -εως`, `irregular:stated forms`), any mismatch
tolerated, and the TBESG entry. A parse whose code TAGNT never uses is
offered with TAGNT's code for the same parse under its other voice letter
where there is one (D for M, O for P ...), else with morphology None and the
parse in `notes`, at `UNCODED` weight; a vocative TAGNT has no code for is
left to the nominative's code. An elided word (ἀντ’, καθ’) is read as its
full forms.

Words whose lemma is not in TBESG get no candidate: no number is invented.
`analyser_eval.py` reports them, with a guess at the missing lemma.

The analyser never consults TAGNT's form table at analysis time: TAGNT
reaches it only through the lexeme-level evidence above, which
`exclude_keys` can withhold (the leave-forms-out evaluation does).
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging import morph_irregular as irregular  # noqa: E402
from lxx_tagging import morph_nominal as nominal  # noqa: E402
from lxx_tagging import morph_phon as ph  # noqa: E402
from lxx_tagging import morph_verbs as verbs  # noqa: E402
from lxx_tagging.candidates import Candidate  # noqa: E402
from lxx_tagging.lexicon import LexEntry, Lexicon  # noqa: E402
from lxx_tagging.text import Token, bare, normalize  # noqa: E402

SOURCE = "analyser"
MIN_SHARE = 0.05          # readings below this share of the best are dropped
ACCENT_MISMATCH = 0.3     # weight of a form whose accent differs, when nothing matches exactly
SUBSCRIPT_MISMATCH = 0.8  # Brenton's ἀποθνῄσκω / σῴζω against a generated form without (or with) ι
BREATHING_MISMATCH = 0.2  # ἁμάξας against TBESG's ἄμαξα: kept only when nothing matches better
UNCODED = 0.5             # a parse TAGNT has no code for (offered with morphology None)
EVIDENCE_WEIGHTS = {"nt": 1.0, "nt-base": 0.85, "table": 0.95, "irregular": 0.9, "rule": 0.6}
CONTRADICTED = 0.15
# A verb TAGNT has at least this many times, never in the active, is a
# deponent whatever its headword; its active forms are weighted by
# ACTIVE_OF_DEPONENT.
DEPONENT_EVIDENCE = 3
ACTIVE_OF_DEPONENT = 0.1       # a hypothesis the word's TAGNT forms do not support, when another does


# ---------------------------------------------------------------------------
# Lexemes

@dataclass
class Lexeme:
    """A TBESG word the analyser generates: its number, the lemma it is
    tagged with (TAGNT's spelling where TAGNT has the number) and class."""

    index: int
    strongs: str
    lemma: str
    key: str              # the lemma variant this lexeme was built from, normalised
    pos: str              # "verb", "noun", "adjective"
    gender: str | None = None
    genitive: str | None = None
    weight: float = 1.0
    note: str = ""
    grammar: str = ""     # TBESG's grammar code ("V", "N-F", "K", "F-2")


_GENITIVE = re.compile(r"^\s*<b>[^<]+</b>\s*,\s*(-[^\s,<\[;]+)")


def genitive_hints(path: Path) -> dict[str, str]:
    """TBESG definition lines that state a noun's genitive (Abbott-Smith's
    `<b>ἀγάπη</b>, -ης, ἡ`), by extended Strong's number."""
    hints: dict[str, str] = {}
    if not path.exists():
        return hints
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 8 or not re.match(r"G\d+$", columns[0].strip()):
                continue
            match = _GENITIVE.match(columns[7])
            if match:
                hints.setdefault(columns[1].partition("=")[0].strip() or columns[0].strip(), match.group(1)[1:])
    return hints


def _guess_class(entry: LexEntry) -> list[tuple[str, float]]:
    """The class of an entry with no part of speech, from its lemma and gloss."""
    key = normalize(entry.variants[0] if entry.variants else entry.lemma)
    letters = bare(key)
    gloss = entry.gloss.lower()
    position = ph.accent_position(key)
    if letters.endswith("ω") and position and position == (ph.nucleus_count(key) - 1, ph.ACUTE):
        return [("noun", 0.6)]          # φειδώ, πειθώ: oxytone -ώ nouns, not verbs
    if letters.endswith(("ω", "ομαι", "μι")) or gloss.startswith(("to ", "be ")):
        return [("verb", 1.0)] if letters.endswith(("ω", "ομαι", "μι", "μαι")) else []
    if letters.endswith(("οσ",)):
        return [("noun", 0.6), ("adjective", 0.6)]
    if letters.endswith(("ησ", "ων")):
        return [("noun", 0.5), ("adjective", 0.6)]
    if letters.endswith(("η", "α", "ον", "μα", "ισ", "ευσ", "ξ", "ψ", "ηρ", "ωρ", "υσ")):
        if _NUMERAL.search(letters) and re.search(r"\d|one|two|three|four|five|six|seven|eight|nine|ten|twenty|"
                                                  r"thirty|forty|fifty|hundred|thousand", gloss):
            return [("invariable", 1.0)]
        return [("noun", 0.8)]
    if _NUMERAL.search(letters) or re.search(r"^(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
                                             r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)", gloss):
        return [("invariable", 1.0)]
    return []


# Endings the nominal paradigms decline (`morph_nominal`).
_DECLINABLE = ("οσ", "ον", "η", "α", "ησ", "ασ", "ισ", "υσ", "ευσ", "ων", "ην", "ηρ", "ωρ", "αρ", "ξ", "ψ", "μα",
               "ουσ", "ουν", "οι", "αι", "ωσ")
_ADJECTIVE_ENDINGS = ("οσ", "οι", "ησ", "υσ", "ων", "ην", "ουσ", "ξ", "ψ", "λασ")
_NUMERAL = re.compile(r"(κοντα|δεκα|κοσι|πεντε|εξ|επτα|οκτω|εννεα|δυο|τρεισ|τεσσαρεσ|εισ)$")
_PRONOUN_HEADS = ("K", "D", "R", "F", "S", "C", "Q", "X", "I")


def _classes(entry: LexEntry) -> list[tuple[str, float]]:
    """The classes an entry is generated as, weighted. TBESG's part of
    speech is followed, with its slips allowed for: a verb-shaped lemma
    tagged as a noun (ἐφοράω G:N-F) is a verb too; a lower-case "name"
    with a Greek ending (λοβός N:N, κοιλάς N:N) a noun too; an
    "indeclinable" adjective with a declining ending (εἰκοσαετής A-NUI) an
    adjective."""
    key = normalize(entry.variants[0] if entry.variants else entry.lemma)
    letters = bare(key)
    if entry.pos_class == "verb":
        return [("verb", 1.0)]
    stated = irregular.irregular_nominal(key)
    if stated and entry.pos_class in ("noun", "adjective", "unknown"):
        return [("adjective" if stated[0] == "A" else "noun", 1.0)]    # πᾶς, εἷς, οὐδείς; ἀνήρ, γυνή
    if entry.pos_class in ("noun", "adjective"):
        found = [(entry.pos_class, 1.0)]
        if letters.endswith(("αω", "εω", "οω", "ζω", "ευω", "ομαι", "νυμι", "ημι", "ωμι", "σσω", "πτω", "σκω")):
            found.append(("verb", 1.0))
        if entry.indeclinable and not letters.endswith(_ADJECTIVE_ENDINGS if entry.pos_class == "adjective"
                                                         else _DECLINABLE):
            return [("invariable", 1.0)]
        if entry.pos_class == "adjective" and not letters.endswith(_ADJECTIVE_ENDINGS):
            return [("invariable", 1.0)]
        return found
    if entry.pos_class == "unknown":
        return _guess_class(entry)
    if entry.pos_class == "transliteration" and letters.endswith(_DECLINABLE) and len(letters) > 3:
        return [("noun", 0.5)]
    if entry.pos_class == "pronoun" and entry.grammar.split("-")[0] in _PRONOUN_HEADS:
        return [("pronoun", 1.0)]
    return []


def build_lexemes(lexicon: Lexicon, forms=None, hints: dict[str, str] | None = None) -> list[Lexeme]:
    """The verbs, nouns, adjectives and declining pronouns of TBESG as
    lexemes, one per spelling of the lemma (and both spellings of a
    λημψ-/ληψ- word). Names, transliterations, invariables and the personal
    pronouns are left to the other providers."""
    lexemes: list[Lexeme] = []
    seen: set[tuple[str, str, str]] = set()
    hints = hints or {}
    for entry in lexicon.entries:
        classes = _classes(entry)
        if not classes:
            continue
        lemma = forms.lemma_for(entry.strongs) if forms is not None else None
        variants = list(entry.variants)
        for variant in entry.variants:
            nfd = ph.nfd(variant)
            # The Attic λήψομαι / Koine λήμψομαι spellings, both ways.
            if "λημψ" in bare(variant) or "λημφ" in bare(variant):
                variants.append(ph.nfc(re.sub(r"λη([\u0301\u0342]?)μ([ψφ])", r"λη\1\2", nfd)))
            elif "ληψ" in bare(variant) or "ληφ" in bare(variant):
                variants.append(ph.nfc(re.sub(r"λη([\u0301\u0342]?)([ψφ])", r"λη\1μ\2", nfd)))
        for variant in dict.fromkeys(variants):
            key = normalize(variant)
            if not bare(key) or len(bare(key)) < 2:
                continue
            for pos, weight in classes:
                if (entry.strongs, key, pos) in seen:
                    continue
                seen.add((entry.strongs, key, pos))
                lexemes.append(Lexeme(len(lexemes), entry.strongs, lemma or entry.lemma, key, pos,
                                      entry.gender, hints.get(entry.extended), weight,
                                      f"TBESG {entry.extended} {entry.pos or '(no part of speech)'}",
                                      entry.grammar))
    return lexemes


# ---------------------------------------------------------------------------
# Regular verb stems

def present_stem(lemma_key: str) -> tuple[str, str | None, bool, str] | None:
    """(present stem, contract vowel, deponent, kind) of a verb lemma (plain
    or key): "λύω" -> ("λυ", None, False, "thematic"); "ποιέω" -> ("ποιε",
    "ε", ...); "ἔρχομαι" -> ("ἐρχ", None, True, ...); "δίδωμι" -> ("διδω",
    None, False, "mi"); an aorist lemma (εἶπον, φαγεῖν) -> kind "aorist"."""
    plain = ph.plain(lemma_key)
    letters = ph.letters(plain)
    parts = ph.clusters(plain)

    def cut(count: int) -> str:
        return ph.compose(parts[:len(parts) - count])

    if letters.endswith("ομαι") and len(letters) > 4:
        stem, deponent = cut(4), True
    elif letters.endswith("ουμαι") and len(letters) > 5:
        stem, deponent = cut(5) + "ε", True
    elif letters.endswith("μαι") and len(letters) > 4:
        return cut(3), None, True, "mi"
    elif letters.endswith("μι") and len(letters) > 3:
        return cut(2), None, False, "mi"
    elif letters.endswith("ω") and len(letters) > 1:
        stem, deponent = cut(1), False
    elif letters.endswith("ειν") and len(letters) > 4:
        return cut(3), None, False, "aorist"
    elif letters.endswith("ον") and len(letters) > 3:
        return cut(2), None, False, "aorist"
    else:
        return None
    stem_letters = ph.letters(stem)
    contract = None
    if stem_letters[-1:] in ("α", "ε", "ο") and len(stem_letters) > 1:
        contract = stem_letters[-1]
        if contract == "ε" and len(stem_letters) <= 3 and sum(1 for c in stem_letters if c in ph.VOWELS) == 1:
            contract = "ε1"     # πλέω, πνέω, δέω, χέω: only ε+ε, ε+ει contract
        if stem_letters in ("ζα", "χρα", "πεινα", "διψα", "κνα", "σμα"):
            contract = "η"
    return stem, contract, deponent, "thematic"


def _lengthen(vowel: str, before_rho: bool) -> str:
    if vowel == "α":
        return "ᾱ" if before_rho else "η"
    return {"ε": "η", "ο": "ω"}.get(vowel, vowel)


def regular_verb_stems(base_key: str) -> list[verbs.Stem]:
    """The tense stems the rules derive from a base's present (see the
    module docstring of `morph_verbs` for the systems), weighted: the usual
    outcome 1.0, real alternatives lower (a -ζω verb's ξ, an -έω verb's
    short ε)."""
    found = present_stem(base_key)
    if not found:
        return []
    stem, contract, deponent, kind = found
    stems: list[verbs.Stem] = []

    def add(system, text, weight=1.0, contract=None, second=False, note=""):
        stems.append(verbs.Stem(system, text, (), weight, "rule", contract, second, note))

    if kind == "mi":
        letters = ph.letters(stem)
        if letters.endswith("νυ"):
            stems.append(verbs.Stem("table", stem, (), 1.0, "rule", forms=irregular.nymi_forms(stem),
                                    note="-νυμι present"))
            add("pres", stem, 0.5, None, note="thematic -νύω")
            root = stem[:-2]
            root_letters = ph.letters(root)
            if root_letters.endswith("ν") and root_letters[-2:-1] in ph.VOWELS:
                # ζώννυμι, σβέννυμι, κορέννυμι, στρώννυμι: a vowel root with σ (ἔζωσα, ἔσβεσα, κεκόρεσμαι).
                vowel_root = root[:-1]
                add("fut", vowel_root + "σ")
                add("aor1", vowel_root + "σ")
                add("aorp", vowel_root + "σθ")
                add("aorp", vowel_root + "θ", 0.5)
                for text, w in ph.reduplicate(vowel_root + "σ"):
                    add("perfm", text, w)
                for text, w in ph.reduplicate(vowel_root):
                    add("perfm", text, 0.5 * w)
                    add("perf", text + "κ", 0.5 * w)
            else:
                stems.extend(_consonant_or_vowel_stems(root, add_present=False))
        return stems
    if kind == "aorist":
        augmented = [(stem, 1.0)]
        unaugmented = _deaugment(stem) or [(stem, 1.0)]
        for text, weight in unaugmented:
            stems.append(verbs.Stem("aor2", text, tuple(augmented), weight, "rule", None, True, "aorist lemma"))
        return stems
    add("pres", stem, 1.0, contract)
    if contract == "η":
        add("pres", stem, 0.6, "α", note="Koine -ᾶν beside -ῆν")
    stems.extend(_consonant_or_vowel_stems(stem, contract=contract, add_present=False))
    return stems


def _consonant_or_vowel_stems(stem: str, contract: str | None = None, add_present: bool = True) -> list[verbs.Stem]:
    stems: list[verbs.Stem] = []

    def add(system, text, weight=1.0, contract=None, second=False, note=""):
        stems.append(verbs.Stem(system, text, (), weight, "rule", contract, second, note))

    letters = ph.letters(stem)
    if not letters:
        return stems
    last = letters[-1]
    redup = ph.reduplicate

    def perfects(perf_stem: str | None, middle_stem: str | None, weight=1.0, second=False):
        for text, w in (redup(perf_stem) if perf_stem else ()):
            add("perf", text, weight * w, second=second)
        for text, w in (redup(middle_stem) if middle_stem else ()):
            add("perfm", text, weight * w)

    if contract in ("α", "ε", "ο", "η", "ε1"):
        head = stem[:-1]
        vowel = "α" if contract in ("α", "η") else ("ε" if contract.startswith("ε") else "ο")
        long_vowel = _lengthen(vowel, letters[-2:-1] in ("ε", "ι", "ρ"))
        options = [(long_vowel, 1.0), (vowel, 0.3)]
        if vowel == "α" and long_vowel != "η":
            options.append(("η", 0.4))         # ξυρήσω, ἐξύρησα beside ἰάσομαι
        for vowel_used, weight in options:
            if vowel_used == vowel and contract == "ο":
                continue
            add("fut", head + vowel_used + "σ", weight)
            add("aor1", head + vowel_used + "σ", weight)
            add("aorp", head + vowel_used + "θ", weight)
            perfects(head + vowel_used + "κ", head + vowel_used, weight)
        if contract.startswith("ε") or contract == "α":
            add("aorp", head + vowel + "σθ", 0.2)
        return stems
    if last == "ζ":
        head = stem[:-1]
        if letters.endswith("ιζ"):
            add("fut", head + "σ", 1.0)
            add("fut", head + "ε", 0.6, "ε", note="Attic future -ιῶ")
            add("aor1", head + "σ", 1.0)
            add("aorp", head + "σθ", 1.0)
            perfects(head + "κ", head + "σ")      # ἥγνικα? (ἡγνίκασι), ἥγνισμαι
        else:
            for consonant, weight in (("σ", 1.0), ("ξ", 0.4)):
                add("fut", head + consonant, weight)
                add("aor1", head + consonant, weight)
            add("aorp", head + "σθ", 1.0)
            add("aorp", head + "χθ", 0.3)
            add("aorp2", head + "γ", 0.3, second=True)
            perfects(head + "κ", head + "σ")
            perfects(None, head + "γ", 0.3)
        return stems
    if letters.endswith(("σσ", "ττ")):
        head = stem[:-2]
        for consonant, weight in (("ξ", 1.0), ("σ", 0.3)):
            add("fut", head + consonant, weight)
            add("aor1", head + consonant, weight)
        add("aorp", head + "χθ", 1.0)
        add("aorp2", head + "γ", 0.5, second=True)
        add("aorp", head + "σθ", 0.3)
        perfects(head + "χ", head + "γ")
        return stems
    if letters.endswith("πτ"):
        head = stem[:-2]
        add("fut", head + "ψ")
        add("aor1", head + "ψ")
        add("aorp", head + "φθ", 1.0)
        add("aorp2", head + "β", 0.4, second=True)
        add("aorp2", head + "φ", 0.4, second=True)
        perfects(head + "φ", head + "π")
        return stems
    if letters.endswith("σκ"):
        head = stem[:-2]
        for addition, weight in (("σ", 0.7), ("ησ", 0.5)):
            add("fut", head + addition, weight)
            add("aor1", head + addition, weight)
        add("aorp", head + "θ", 0.4)
        add("aorp", head + "σθ", 0.4)
        add("aorp", head + "ηθ", 0.4)
        perfects(head + "κ", head, 0.4)
        return stems
    if last in ph.LABIALS:
        head = stem[:-1]
        add("fut", head + "ψ")
        add("aor1", head + "ψ")
        add("aorp", head + "φθ")
        add("aorp2", head + "φ", 0.4, second=True)
        perfects(head + "φ", stem, 1.0, second=True)
        return stems
    if last in ph.VELARS:
        head = stem[:-1]
        add("fut", head + "ξ")
        add("aor1", head + "ξ")
        add("aorp", head + "χθ")
        add("aorp2", head + "γ", 0.4, second=True)
        perfects(head + "χ", stem, 1.0, second=True)
        return stems
    if last in ph.DENTALS:
        head = stem[:-1]
        add("fut", head + "σ")
        add("aor1", head + "σ")
        add("aorp", head + "σθ")
        perfects(head + "κ", head + "σ")
        return stems
    if last in ph.LIQUIDS:
        _liquid_stems(stem, add, perfects)
        return stems
    if last in ph.VOWELS:
        add("fut", stem + "σ")
        add("aor1", stem + "σ")
        add("aorp", stem + "θ", 1.0)
        add("aorp", stem + "σθ", 0.4)
        perfects(stem + "κ", stem)
        return stems
    return stems


def _liquid_stems(stem: str, add, perfects) -> list:
    """Liquid and nasal stems: contracted future, aorist without σ with the
    stem vowel lengthened (ἔκρινα, ἤγγειλα, ἔμεινα, ἐσήμανα/ἐσήμηνα)."""
    letters = ph.letters(stem)
    stems_before = []

    def lengthened(short_stem: str) -> list[tuple[str, float]]:
        bare_short = ph.letters(short_stem)
        vowel = bare_short[-2:-1]
        head, tail = short_stem[:-2], short_stem[-1:]
        if vowel == "ε":
            return [(head + "ει" + tail, 1.0)]
        if vowel == "α":
            return [(head + "η" + tail, 0.8), (head + "ᾱ" + tail, 0.8)]
        if vowel in ("ι", "υ"):
            return [(head + {"ι": "ῑ", "υ": "ῡ"}[vowel] + tail, 1.0)]
        return [(short_stem, 0.8)]

    if letters.endswith("λλ"):
        short = stem[:-1]
        add("fut", short + "ε", 1.0, "ε")
        for text, weight in lengthened(short):
            add("aor1", text, weight)
        vowel = ph.letters(short)[-2:-1]
        add("aorp2", short[:-2] + "α" + short[-1:], 0.6 if vowel == "ε" else 0.3, second=True)
        add("aorp2", short, 0.5, second=True)
        add("aorp", short + "θ", 0.3)
        perfects(short + "κ", short, 0.8)
        return stems_before
    # -αίνω, -είνω, -ίνω, -ύνω, -αίρω, -είρω, -ίρω, -ύρω: the present's ι is a suffix.
    if letters[-3:-1] in ("αι", "ει") and letters[-1] in "νρ":
        vowel = letters[-3]
        short = stem[:-2] + stem[-1:]        # σημαιν > σημαν, κτειν > κτεν
        add("fut", short + "ε", 1.0, "ε")
        if vowel == "α":
            for text, weight in ((stem[:-3] + "η" + stem[-1:], 0.8), (stem[:-3] + "ᾱ" + stem[-1:], 0.8)):
                add("aor1", text, weight)
            add("aorp", short + "θ", 1.0)
            add("aorp2", short, 0.4, second=True)
        else:
            add("aor1", stem, 1.0)
            add("aorp", stem[:-3] + "α" + "θ", 0.5)
            add("aorp2", stem[:-3] + "α" + stem[-1:], 0.5, second=True)
            add("aorp", short + "θ", 0.3)
        perfects(None, short, 0.4)
        return stems_before
    if letters[-1] in "νρ" and letters[-2:-1] in ("ι", "υ"):
        add("fut", stem + "ε", 1.0, "ε")
        for text, weight in lengthened(stem):
            add("aor1", text, weight)
        if letters[-1] == "ν":
            add("aorp", stem[:-1] + "θ", 0.7)      # ἐκρίθην
            add("aorp", stem + "θ", 0.5)           # ἐπληθύνθην
            perfects(stem[:-1] + "κ", stem[:-1], 0.6)
        else:
            add("aorp", stem + "θ", 0.6)
        return stems_before
    # Other liquids and nasals (μένω, νέμω, φέρω-like): contracted future and aorist in -ειν-/σ.
    add("fut", stem + "ε", 1.0, "ε")
    add("fut", stem + "ησ", 0.4)
    for text, weight in lengthened(stem):
        add("aor1", text, weight * 0.8)
    add("aor1", stem + "ησ", 0.3)
    add("aorp", stem + "θ", 0.5)
    add("aorp", stem + "ηθ", 0.4)
    perfects(stem + "ηκ", stem + "η", 0.3)
    return stems_before


def _deaugment(stem: str) -> list[tuple[str, float]]:
    """Unaugmented spellings an augmented stem may come from (ἠγειρ- from
    ἐγειρ- or ἀγειρ-; ἐλαβ- from λαβ-), weighted."""
    letters = ph.letters(stem)
    parts = ph.clusters(stem)
    breathing = ph.initial_breathing(stem)
    found: list[tuple[str, float]] = []
    if letters.startswith("ερρ"):
        found.append(("ῥ" + ph.compose(parts[3:]), 1.0))
    if letters[:1] == "ε" and len(letters) > 1 and letters[1] in ph.CONSONANTS and not letters.startswith("ερρ"):
        found.append((ph.set_breathing(ph.compose(parts[1:]), None), 1.0))
    group, rest, _ = ph._initial_vowels(stem)
    reverse = {"η": ("ε", "α"), "ω": ("ο",), "ῃ": ("αι", "ει"), "ῳ": ("οι",), "ηυ": ("ευ", "αυ"),
               "ει": ("ε", "ι", "ει"), "ι": ("ι",), "υ": ("υ",), "ευ": ("ευ",), "αυ": ("αυ",), "οι": ("οι",),
               "ου": ("ου",), "αι": ("αι",)}
    if group:
        for number, original in enumerate(reverse.get(group, ())):
            found.append((ph.set_breathing(original + rest, breathing or ph.SMOOTH), 1.0 if number == 0 else 0.6))
    return found


# ---------------------------------------------------------------------------
# Evidence from TAGNT

_VERB_CODE = re.compile(r"^V-(?P<second>2?)(?P<tense>[PIFARL])(?P<voice>[AMPDONE])(?P<mood>[ISOMNP])"
                        r"(?:-(?P<rest>[0-9A-Z]+))?$")
_NOMINAL_CODE = re.compile(r"^(?P<head>[NA])-(?P<case>[NGDAV])(?P<number>[SP])(?P<gender>[MFN])(?:-(?P<suffix>[A-Z]+))?$")

# A verb form's endings by (tense, mood), to cut its stem: (system, endings, augmented).
_SECONDARY_ACTIVE = [e for _, e, _, _ in verbs.ACT_SECONDARY]
_SECONDARY_MIDDLE = [e for _, e, _, _ in verbs.MID_SECONDARY]


def _endings(table) -> list[str]:
    return sorted({ph.letters(e) for _, e, _, _ in table}, key=len, reverse=True)


_PTC = {name: _endings(table) for name, table in (
    ("ont", verbs.PTC_ONT), ("ant", verbs.PTC_ANT), ("ent", verbs.PTC_ENT), ("ot", verbs.PTC_OT),
    ("men", verbs.PTC_MEN), ("amen", verbs.PTC_AMEN), ("pmen", verbs.PTC_PMEN))}
_CONTRACT_FUTURE = ["ω", "εισ", "ει", "ουμεν", "ειτε", "ουσι", "ουσιν", "ουμαι", "ῃ", "ειται", "ουμεθα",
                    "εισθε", "ουνται", "ειν", "εισθαι", "ων", "ουντοσ", "ουσα", "ουμενοσ", "ουμενου",
                    "ουμενοι", "ουμενων", "ουμενα", "ουμενη"]


def _cut_letters(plain: str, count: int) -> str:
    parts = ph.clusters(plain)
    return ph.compose(parts[:len(parts) - count]) if count else plain


def _strip(plain: str, endings: list[str]) -> list[tuple[str, str]]:
    """(stem, ending) for each ending (letters) the plain form ends with, longest first."""
    letters = ph.letters(plain)
    found = []
    for ending in endings:
        if letters.endswith(ending) and len(letters) > len(ending) + 1:
            found.append((_cut_letters(plain, len(ending)), ending))
    return found


def cut_verb_stems(plain: str, code: str) -> list[tuple[str, str, bool]]:
    """(system, stem, augmented) readings of a base form (no preverb) with a
    TAGNT code: the stem the generator would need to make this form."""
    match = _VERB_CODE.match(code)
    if not match:
        return []
    tense, voice, mood = match["tense"], match["voice"], match["mood"]
    past = mood == "I" and tense in "IAL"
    letters = ph.letters(plain)
    found: list[tuple[str, str, bool]] = []

    def take(system, endings, augmented=past, require=None):
        for stem, ending in _strip(plain, endings):
            if require and not ph.letters(stem).endswith(require):
                continue
            found.append((system, stem, augmented))
            return True
        return False

    if tense == "P":
        return found
    if tense == "I":
        take("impf", [e for e in _endings(verbs.ACT_SECONDARY) + _endings(verbs.MID_SECONDARY)], True)
        return found
    if tense == "F":
        if voice in "PO" or (letters.endswith(("θησομαι", "θησεται", "θησονται", "θησῃ"))):
            for system, require in (("aorp", "θ"), ("aorp2", None)):
                if mood == "P":
                    ok = take(system, ["ησ" + e for e in _PTC["men"]], False, require)
                elif mood == "N":
                    ok = take(system, ["ησεσθαι"], False, require)
                else:
                    ok = take(system, ["ησ" + e for e in _endings(verbs.MID_PRIMARY) + _endings(verbs.MID_OPT)],
                              False, require)
                if ok:
                    break
            return found
        endings = (_endings(verbs.ACT_PRIMARY) + _endings(verbs.MID_PRIMARY) + _endings(verbs.ACT_OPT)
                   + _endings(verbs.MID_OPT) + ["ειν", "εσθαι"] + _PTC["ont"] + _PTC["men"] + ["εσαι"])
        endings = sorted(set(endings), key=len, reverse=True)
        for stem, ending in _strip(plain, endings):
            if ph.letters(stem)[-1:] in ("σ", "ξ", "ψ"):
                found.append(("fut", stem, False))
                break
        else:
            for stem, ending in _strip(plain, sorted(_CONTRACT_FUTURE, key=len, reverse=True)):
                found.append(("futc", stem, False))
                break
            else:
                for stem, ending in _strip(plain, endings):
                    found.append(("fut", stem, False))
                    break
        return found
    if tense == "A":
        passive_endings = {
            "I": _endings(verbs.AP_IND), "S": _endings(verbs.AP_SUBJ), "O": _endings(verbs.AP_OPT),
            "M": ["ητι", "ηθι", "ητω", "ητε", "ητωσαν"], "N": ["ηναι"], "P": _PTC["ent"]}[mood]
        if voice in "PO" or (voice in "DN" and any(letters.endswith(e) for e in passive_endings)
                             and any(ph.letters(s).endswith("θ") for s, _ in _strip(plain, passive_endings))):
            for stem, ending in _strip(plain, passive_endings):
                found.append(("aorp" if ph.letters(stem).endswith("θ") else "aorp2", stem, past))
                return found
            return found
        first = {"I": _endings(verbs.A1_ACT_IND) + _endings(verbs.A1_MID_IND), "S": _endings(verbs.ACT_SUBJ)
                 + _endings(verbs.MID_SUBJ), "O": _endings(verbs.A1_ACT_OPT) + _endings(verbs.A1_MID_OPT),
                 "M": _endings(verbs.A1_ACT_IMP) + _endings(verbs.A1_MID_IMP), "N": ["αι", "ασθαι"],
                 "P": _PTC["ant"] + _PTC["amen"]}[mood]
        second = {"I": _endings(verbs.ACT_SECONDARY) + _endings(verbs.MID_SECONDARY),
                  "S": _endings(verbs.ACT_SUBJ) + _endings(verbs.MID_SUBJ),
                  "O": _endings(verbs.ACT_OPT) + _endings(verbs.MID_OPT),
                  "M": ["ε", "ετω", "ετε", "ετωσαν", "ου", "εσθω", "εσθε"], "N": ["ειν", "εσθαι"],
                  "P": _PTC["ont"] + _PTC["men"]}[mood]
        first = sorted(set(first), key=len, reverse=True)
        second = sorted(set(second), key=len, reverse=True)
        if mood == "S":
            # The subjunctive does not show which aorist: a σ/ξ/ψ stem is first.
            for stem, _ in _strip(plain, second):
                system = "aor1" if ph.letters(stem)[-1:] in ("σ", "ξ", "ψ") else "aor2"
                found.append((system, stem, False))
                return found
            return found
        one = _strip(plain, first)
        two = _strip(plain, second)
        if match["second"] and two:
            found.append(("aor2", two[0][0], past))
        elif one and (not two or len(one[0][1]) >= len(two[0][1])):
            found.append(("aor1", one[0][0], past))
        elif two:
            found.append(("aor2", two[0][0], past))
        return found
    if tense in "RL":
        if voice == "A":
            if tense == "L":
                take("perf", _endings(verbs.PLPF_ACT), False)
                return found
            endings = {"I": _endings(verbs.PF_ACT_IND), "S": _endings(verbs.PF_ACT_SUBJ), "M": ["ε", "ετε"],
                       "N": ["εναι"], "P": _PTC["ot"], "O": []}[mood]
            take("perf", endings, False)
            return found
        endings = {"I": _endings(verbs.PLPF_MID if tense == "L" else verbs.PF_MID_IND), "M": ["σο", "σθω", "σθε"],
                   "N": ["σθαι", "θαι"], "P": _PTC["pmen"], "S": [], "O": []}[mood]
        for stem, ending in _strip(plain, endings):
            stem_letters = ph.letters(stem)
            # Undo the assimilation: γέγραμ-μαι, γέγραπ-ται, γέγραψ-αι, γεγράφ-θαι > a labial stem.
            if stem_letters.endswith("μ") and ending.startswith("μ"):
                stem = stem[:-1] + "π"
            elif stem_letters.endswith("ψ") and ending in ("αι",):
                stem = stem[:-1] + "π"
            elif stem_letters.endswith("ξ") and ending in ("αι",):
                stem = stem[:-1] + "κ"
            elif stem_letters[-1:] in ("φ", "χ") and ending.startswith("θ"):
                stem = stem[:-1] + {"φ": "π", "χ": "κ"}[stem_letters[-1]]
            found.append(("perfm", stem, False))
            break
        return found
    return found


def _preverb_strip(plain: str, chain: tuple) -> tuple[str, str | None] | None:
    """A form of a compound without its preverbs: (base form, breathing the
    preverb's spelling shows), or None if the form does not start with them."""
    rest = plain
    breathing = None
    for preverb in chain:
        letters = ph.letters(rest)
        options = []
        for spelling in {preverb.consonant, preverb.vowel, preverb.rough, *preverb.assimilated("β"),
                         *preverb.assimilated("γ"), *preverb.assimilated("λ"), *preverb.assimilated("ρ"),
                         *preverb.assimilated("στ"), *preverb.assimilated("σα"), "πρ" if preverb.name == "πρό" else ""}:
            if spelling and letters.startswith(ph.letters(spelling)):
                options.append(spelling)
        if not options:
            return None
        options.sort(key=lambda s: -len(ph.letters(s)))
        chosen = None
        for spelling in options:
            after = ph.letters(rest)[len(ph.letters(spelling)):]
            if not after:
                continue
            if spelling == preverb.consonant and after[0] in ph.VOWELS and preverb.consonant != preverb.vowel:
                continue
            chosen = spelling
            break
        if chosen is None:
            return None
        if chosen == preverb.rough and preverb.rough != preverb.vowel:
            breathing = ph.ROUGH
        elif chosen == preverb.vowel and preverb.rough != preverb.vowel:
            breathing = ph.SMOOTH
        else:
            breathing = None
        rest = _drop(rest, len(ph.letters(chosen)))
        if ph.letters(rest).startswith("ρρ") and ph.letters(chosen)[-1:] in ph.VOWELS:
            rest = _drop(rest, 1)
    if rest and ph.letters(rest)[0] in ph.VOWELS:
        rest = ph.set_breathing(rest, breathing or ph.SMOOTH)
    return rest, breathing


def _drop(word: str, count: int) -> str:
    return ph.compose(ph.clusters(word)[count:])


@dataclass
class VerbEvidence:
    stems: Counter = field(default_factory=Counter)        # (system, stem, augmented) -> count
    voices: dict = field(default_factory=lambda: defaultdict(Counter))   # slot -> letter counts
    second: dict = field(default_factory=lambda: defaultdict(Counter))   # (system, mood) -> {"2": n, "": n}
    forms: set = field(default_factory=set)                              # letters of the base forms read
    active: int = 0
    total: int = 0


def _resembling(evidence: "VerbEvidence", base_stem: str) -> "VerbEvidence":
    """The evidence's stems that share their first letters with the base's
    present stem, allowing for an augment or reduplication."""
    head = base_stem[:3]
    kept = Counter()
    for (system, stem, augmented), count in evidence.stems.items():
        letters = ph.letters(stem)
        if head and any(letters[i:i + len(head)] == head for i in range(0, 3)):
            kept[(system, stem, augmented)] = count
    filtered = VerbEvidence(kept, evidence.voices, evidence.second, evidence.forms, evidence.active, evidence.total)
    return filtered


def _slot(system: str, tense: str, plain: str, code_voice: str) -> str:
    if system in ("aorp", "aorp2"):
        return "fp" if tense == "F" else "ap"
    if tense == "F":
        return "act" if code_voice == "A" else "fm"
    if tense == "A":
        return "act" if code_voice == "A" else "am"
    return "act" if code_voice == "A" else "mp"


class NTEvidence:
    """What TAGNT's forms say about each lexeme, with some forms withheld.

    `exclude_keys` are form keys whose rows are not used (the
    leave-forms-out evaluation passes the forms it tests)."""

    def __init__(self, forms, exclude_keys: Iterable[str] = ()):
        excluded = set(exclude_keys)
        self.by_strongs: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for key in forms.keys():
            if key in excluded:
                continue
            for reading, count in forms.readings(key).items():
                if " + " in reading.morph:
                    continue
                self.by_strongs[reading.strongs].append((key, reading.morph, count))

    def rows(self, strongs: str) -> list[tuple[str, str, int]]:
        return self.by_strongs.get(strongs, [])


# ---------------------------------------------------------------------------
# The analyser

@dataclass(frozen=True)
class Entry:
    """One generated form in the index."""

    key: str
    lexeme: int
    code: str | None
    weight: float
    note: str


class Analyser:
    """The provider. Build once (`Analyser(lexicon, forms)`); call per token."""

    name = SOURCE

    def __init__(self, lexicon: Lexicon, forms=None, *, vocabulary: Iterable[str] | None = None,
                 exclude_keys: Iterable[str] = (), codes="tagnt",
                 hints: dict[str, str] | None = None, tbesg_path: Path | None = None):
        """`forms` is TAGNT's form table (evidence, and the code inventory);
        `vocabulary` the letters (`text.bare`) of the forms that will be
        asked about, which saves generating the rest; `exclude_keys` form
        keys whose TAGNT rows are withheld from the evidence; `codes` the
        morphology codes a candidate may carry ("tagnt": TAGNT's own; None:
        any); `hints`/`tbesg_path` TBESG's stated genitives."""
        self.lexicon = lexicon
        self.forms = forms
        if codes == "tagnt":
            codes = forms.morph_codes if forms is not None else None
        self.codes = codes
        self.vocabulary = set(vocabulary) if vocabulary is not None else None
        if self.vocabulary is not None:
            # A spelling bridge (`bridged_keys`) looks up other letters.
            self.vocabulary |= {bare(v) for letters in list(self.vocabulary) for v, _ in bridged_keys(letters)}
        if hints is None and tbesg_path is not None:
            hints = genitive_hints(tbesg_path)
        self.lexemes = build_lexemes(lexicon, forms, hints)
        self.evidence = NTEvidence(forms, exclude_keys) if forms is not None else None
        self.index: dict[str, list[Entry]] = defaultdict(list)
        self._priors: dict[str, float] = {}
        self._known_bases = self._bases()
        self._base_evidence: dict[str, VerbEvidence] = defaultdict(VerbEvidence)
        self._lexeme_evidence: dict[tuple, VerbEvidence] = {}
        self._verb_plans: dict[int, list] = {}
        self._collect_verb_evidence()
        generator = verbs.Generator(self.vocabulary)
        for lexeme in self.lexemes:
            if lexeme.pos == "verb":
                self._index_verb(lexeme, generator)
            elif lexeme.pos == "noun":
                self._index_noun(lexeme)
            elif lexeme.pos == "adjective":
                self._index_adjective(lexeme)
            elif lexeme.pos == "invariable":
                self._index_invariable(lexeme)
            elif lexeme.pos == "pronoun":
                self._index_pronoun(lexeme)
        self.index = dict(self.index)

    # -- verbs ---------------------------------------------------------------

    def _bases(self) -> dict[str, list[str]]:
        """Letters of every known simplex verb -> its keys (TBESG verbs,
        irregular tables). εἰμί "be" and εἶμι "go" share their letters."""
        known: dict[str, list[str]] = defaultdict(list)
        for lexeme in self.lexemes:
            if lexeme.pos == "verb" and lexeme.key not in known[ph.letters(ph.plain(lexeme.key))]:
                known[ph.letters(ph.plain(lexeme.key))].append(lexeme.key)
        for name in list(irregular.PRINCIPAL_PARTS) + list(irregular.TABLES):
            key = normalize(name)
            if key not in known[ph.letters(ph.plain(name))]:
                known[ph.letters(ph.plain(name))].append(key)
        return dict(known)

    def _analyses(self, lexeme: Lexeme) -> list[tuple[tuple, str, float]]:
        """(preverb chain, base key, weight) readings of a verb lemma: every
        split whose base is a known verb (καθ-αίρω beside κατα-αἴρω: the
        forms decide), else the lemma whole and its likeliest split."""
        plain = ph.plain(lexeme.key)
        splits = ph.split_preverbs(plain)
        found: dict[tuple, tuple] = {}
        for chain, base in splits:
            if not chain:
                continue
            for base_key in self._known_bases.get(ph.letters(base), ()):
                found.setdefault((tuple(p.name for p in chain), base_key), (chain, base_key, 1.0))
        if found:
            plans = sorted(found.values(), key=lambda plan: (-len(plan[0]), plan[1]))[:4]
            if min(len(ph.letters(ph.plain(base))) for _, base, _ in plans) <= 4:
                # ἀπ-ορέω is not ἀπο-ρέω: a short base may be a coincidence.
                plans.append(((), plain, 0.5))
            return plans
        plausible = [(chain, base) for chain, base in splits if chain and len(ph.letters(base)) >= 4]
        if plausible and not ph.letters(plain).startswith(("ευ", "δυσ")):
            plausible.sort(key=lambda item: (-len(item[0]), -len(ph.letters(item[1]))))
            return [((), plain, 0.6)] + [(chain, base, 0.6) for chain, base in plausible[:2]]
        return [((), plain, 1.0)]

    def _base_letters(self, base_key: str) -> str:
        found = present_stem(base_key)
        return ph.letters(found[0]) if found else ph.letters(base_key)

    @staticmethod
    def _plan_id(lexeme: Lexeme, chain: tuple, base_key: str) -> tuple:
        return (lexeme.index, tuple(p.name for p in chain), base_key)

    def _collect_verb_evidence(self) -> None:
        if self.evidence is None:
            return
        for lexeme in self.lexemes:
            if lexeme.pos != "verb":
                continue
            rows = self.evidence.rows(lexeme.strongs)
            if not rows:
                continue
            plans = self._analyses(lexeme)
            self._verb_plans[lexeme.index] = plans
            for chain, base_key, _ in plans:
                evidence = self._read_rows(rows, chain, base_key)
                self._lexeme_evidence[self._plan_id(lexeme, chain, base_key)] = evidence
                pooled = self._base_evidence[self._base_letters(base_key)]
                pooled.stems.update(evidence.stems)
                pooled.forms |= evidence.forms
                for slot, counter in evidence.voices.items():
                    pooled.voices[slot].update(counter)
                for slot, counter in evidence.second.items():
                    pooled.second[slot].update(counter)
                pooled.active += evidence.active
                pooled.total += evidence.total

    def _read_rows(self, rows, chain: tuple, base_key: str) -> "VerbEvidence":
        """A verb's TAGNT rows as evidence about its base: stems, voice letters, conventions."""
        evidence = VerbEvidence()
        base_breathing = ph.initial_breathing(ph.plain(base_key))
        for key, code, count in rows:
            match = _VERB_CODE.match(code)
            if not match:
                continue
            evidence.total += count
            plain = ph.plain(key)
            stripped = _preverb_strip(plain, chain) if chain else (plain, None)
            if stripped is None:
                continue
            base_form, shown = stripped
            if chain and ph.starts_with_vowel(base_form) and shown is None and base_breathing:
                base_form = ph.set_breathing(base_form, base_breathing)
            evidence.forms.add((ph.letters(base_form), match["second"] + match["tense"], match["mood"]))
            cut = cut_verb_stems(base_form, code)
            if ph.letters(base_key).endswith("μι"):
                # An athematic aorist (ἔδομεν, δῶ, θῶ, στῶ) is no thematic
                # stem (δ-: ἔδεσθε); the irregular tables write it out.
                cut = [entry for entry in cut if entry[0] != "aor2"]
            for system, stem, augmented in cut:
                evidence.stems[(system, stem, augmented)] += count
                slot = _slot(system, match["tense"], base_form, match["voice"])
                evidence.voices[slot][match["voice"]] += count
                generator_system = {"futc": "fut", "impf": "pres"}.get(system, system)
                evidence.second[(generator_system, match["mood"])][match["second"]] += count
                evidence.second[(generator_system, "")][match["second"]] += count
            if not cut:
                tense = match["tense"]
                slot = "act" if match["voice"] == "A" else ("mp" if tense in "PIRL" else ("fm" if tense == "F" else "am"))
                evidence.voices[slot][match["voice"]] += count
            if match["voice"] == "A":
                evidence.active += count
        return evidence

    def _stems_for(self, lexeme: Lexeme, chain: tuple, base_key: str) -> list[verbs.Stem]:
        """Every stem hypothesis of a verb, merged and weighted by evidence."""
        hypotheses: list[verbs.Stem] = []
        own = self._lexeme_evidence.get(self._plan_id(lexeme, chain, base_key))
        pooled = self._base_evidence.get(self._base_letters(base_key))
        base_stem = self._base_letters(base_key)
        suppletive = bool(self._irregular_name(base_key))
        for evidence, source in ((own, "nt"), (pooled, "nt-base")):
            if not evidence:
                continue
            if source == "nt-base" and not suppletive:
                # Another word's forms filed under this base by TAGNT's lemma
                # (κατεκληρονόμησεν under κατακληροδοτέω) are not its stems.
                evidence = _resembling(evidence, base_stem)
            augmented_by_system: dict[str, list] = defaultdict(list)
            unaugmented_by_system: dict[str, set] = defaultdict(set)
            for (system, stem, augmented), count in evidence.stems.items():
                if augmented:
                    augmented_by_system[system].append((stem, count))
                else:
                    unaugmented_by_system[system].add(ph.letters(stem))
            for (system, stem, augmented), count in evidence.stems.items():
                if system == "impf":
                    continue
                second = system in ("aor2", "aorp2") or (
                    system == "perf" and evidence.second[("perf", "")]["2"] > evidence.second[("perf", "")][""])
                if system == "futc":
                    hypotheses.append(verbs.Stem("fut", stem + "ε", (), EVIDENCE_WEIGHTS[source], source, "ε"))
                    continue
                if augmented:
                    attested = unaugmented_by_system.get(system)
                    for unaugmented, weight in _deaugment(stem):
                        if attested and ph.letters(unaugmented) not in attested:
                            # The unaugmented stem is attested (ἰδεῖν beside
                            # εἶδον): no other guess at it (not ἐδ-).
                            continue
                        hypotheses.append(verbs.Stem(system, unaugmented, ((stem, 1.0),),
                                                     EVIDENCE_WEIGHTS[source] * weight * 0.9, source, None, second))
                    continue
                hypotheses.append(verbs.Stem(system, stem, tuple((s, 1.0) for s, _ in augmented_by_system[system]),
                                             EVIDENCE_WEIGHTS[source], source, None, second))
        name = self._irregular_name(base_key)
        parts = irregular.principal_parts(name, EVIDENCE_WEIGHTS["irregular"])
        table = irregular.table_stem(name)
        if own and table:
            # A written-out paradigm the word's own TAGNT forms contradict
            # (forms of a tense and mood it has, none of them the table's)
            # belongs to another word with these letters (εἴδω "see", εἴδω "know").
            table_forms = defaultdict(set)
            for entry in table.forms:
                table_forms[(entry[0].lstrip("2"), entry[2])].add(ph.letters(entry[4]))
            shared = [group for group in {(t.lstrip("2"), m) for _, t, m in own.forms} if group in table_forms]
            if shared and not any(letters in table_forms[(t.lstrip("2"), m)] for letters, t, m in own.forms
                                  if (t.lstrip("2"), m) in table_forms):
                table = replace(table, weight=table.weight * CONTRADICTED)
        hypotheses += parts
        if table:
            hypotheses.append(table)
        rule = [replace(stem, weight=stem.weight * EVIDENCE_WEIGHTS["rule"]) for stem in regular_verb_stems(base_key)]
        # Systems the evidence or the tables cover make the rules' guesses for them weak.
        groups = {"pres": "pres", "fut": "fut", "aor1": "aor", "aor2": "aor", "aorp": "aorp", "aorp2": "aorp",
                  "perf": "perf", "perfm": "perfm"}
        covered = {groups[stem.system] for stem in hypotheses if stem.system in groups}
        if table:
            for tense, slot, *_ in table.forms:
                tense = tense.lstrip("2")
                covered.add({"P": "pres", "I": "pres", "F": "fut", "R": "perf" if slot == "act" else "perfm",
                             "L": "perf" if slot == "act" else "perfm"}.get(tense, "aorp" if slot == "ap" else "aor"))
        for stem in rule:
            group = groups.get(stem.system)
            if group in covered:
                if group == "pres" and table and "νύω" not in stem.note:
                    continue
                if group != "pres":
                    stem = replace(stem, weight=stem.weight * 0.3)
            hypotheses.append(stem)
        # Imperfects: the attested and stated augmented present stems.
        impf = []
        for evidence in (own, pooled):
            if evidence:
                impf += [stem for (system, stem, augmented), _ in evidence.stems.items() if system == "impf"]
        impf += [a for stem in irregular.principal_parts(name) if stem.system == "impf" for a, _ in stem.augmented]
        merged: dict[tuple, verbs.Stem] = {}
        for stem in hypotheses:
            if stem.system == "impf":
                continue
            if stem.system == "pres" and impf:
                stem = replace(stem, augmented=tuple(dict.fromkeys(
                    [(ph.plain(a), 1.0) for a in impf] + list(ph.augment(stem.stem)))))
            if stem.system == "table":
                merged[("table", stem.stem)] = stem
                continue
            identity = (stem.system, ph.letters(stem.stem), stem.contract)
            if identity in merged:
                old = merged[identity]
                augmented = tuple(dict.fromkeys(old.augmented + stem.augmented))
                keep = stem if stem.weight > old.weight else old
                merged[identity] = replace(keep, augmented=augmented, second=old.second or stem.second)
            else:
                merged[identity] = stem
        return list(merged.values())

    def _irregular_name(self, base_key: str) -> str:
        letters = ph.letters(ph.plain(base_key))
        for name in list(irregular.PRINCIPAL_PARTS) + list(irregular.TABLES):
            if ph.letters(ph.plain(name)) == letters and (normalize(name) == base_key or True):
                if normalize(name) == base_key:
                    return name
        for name in list(irregular.PRINCIPAL_PARTS) + list(irregular.TABLES):
            if ph.letters(ph.plain(name)) == letters:
                return name
        return ""

    def _voices(self, lexeme: Lexeme, chain: tuple, base_key: str) -> tuple[dict, dict]:
        own = self._lexeme_evidence.get(self._plan_id(lexeme, chain, base_key))
        pooled = self._base_evidence.get(self._base_letters(base_key))
        found = present_stem(lexeme.key)
        deponent = bool(found and found[2])
        evidence = own if own and own.total else pooled
        if evidence and evidence.total and not deponent:
            middle = sum(evidence.voices["mp"].values())
            if not evidence.active and middle and evidence.voices["mp"]["N"] + evidence.voices["mp"]["D"] >= middle / 2:
                deponent = True
        defaults = verbs.DEPONENT_VOICES if deponent else verbs.DEFAULT_VOICES
        voices = {}
        for slot, default in defaults.items():
            learned = Counter()
            for source, factor in ((own, 1.0), (pooled, 0.5)):
                if source:
                    for letter, count in source.voices[slot].items():
                        learned[letter] += count * factor
            total = sum(learned.values())
            if total:
                shares = {letter: count / total for letter, count in learned.items() if count / total >= 0.05}
                for letter, weight in default:
                    shares.setdefault(letter, 0.15 * weight)
                voices[slot] = tuple(sorted(shares.items(), key=lambda item: -item[1]))
            else:
                voices[slot] = default
        if deponent and lexeme.key.endswith("μαι") and not (own and own.active) and not (pooled and pooled.active):
            voices["act"] = ()
        elif deponent and evidence and evidence.total >= DEPONENT_EVIDENCE and not evidence.active:
            # TAGNT lemmatises some deponents under an active headword
            # (πορεύω, φοβέω): if all its forms of the word are middle or
            # passive, an active form (πορεύσῃ as an aorist subjunctive
            # active) is unlikely, though not impossible.
            voices["act"] = tuple((letter, weight * ACTIVE_OF_DEPONENT) for letter, weight in voices["act"])
        second = {}
        for source in (pooled, own):
            if source:
                for slot, counter in source.second.items():
                    total = sum(counter.values())
                    if total:
                        second[slot] = counter["2"] / total
        return voices, second

    def _index_verb(self, lexeme: Lexeme, generator: verbs.Generator) -> None:
        if bare(lexeme.key).endswith(("ναι", "σθαι")) and not self._irregular_name(lexeme.key):
            # An infinitive as the lemma (εἶναι G1511): the lemma is that form.
            self._add(lexeme.key, lexeme, "V-PAN", 0.8, "rule:infinitive lemma")
            return
        plans = self._verb_plans.get(lexeme.index) or self._analyses(lexeme)
        for chain, base_key, plan_weight in plans:
            stems = self._stems_for(lexeme, chain, base_key)
            voices, second = self._voices(lexeme, chain, base_key)
            by_system = defaultdict(list)
            for stem in stems:
                by_system[stem.system].append(stem)
            verb = verbs.VerbLexeme(lexeme.strongs, lexeme.lemma, chain, dict(by_system), voices, second,
                                    plan_weight * lexeme.weight,
                                    "+".join(p.name for p in chain) + ("+" if chain else "") + base_key)
            for form in generator.forms(verb):
                self._add(form.key, lexeme, form.code, form.weight, form.note)

    # -- nominals ------------------------------------------------------------

    def _nominal_rows(self, lexeme: Lexeme) -> list[tuple[str, str, int]]:
        if self.evidence is None:
            return []
        return [row for row in self.evidence.rows(lexeme.strongs) if _NOMINAL_CODE.match(row[1])]

    def _support(self, hypotheses: list[tuple[list, float]], rows) -> list[float]:
        """For each hypothesis (a list of Declensions), how many TAGNT rows it generates."""
        support = []
        wanted = Counter()
        for key, code, count in rows:
            match = _NOMINAL_CODE.match(code)
            wanted[(key, match["case"] + match["number"], match["gender"])] += count
        for declensions, _ in hypotheses:
            generated = set()
            for decl in declensions:
                for form in nominal.decline(decl):
                    generated.add((form.key, form.case_number, form.gender))
                    if form.case_number[0] == "N":
                        generated.add((form.key, "V" + form.case_number[1], form.gender))
            support.append(sum(count for item, count in wanted.items() if item in generated))
        return support

    def _suffixes(self, lexeme: Lexeme, rows) -> dict[str, dict[str, float]]:
        """The code suffixes ("" or "T", a title) TAGNT gives the word's
        forms, by number: θεός, κύριος are -T in the singular, and "gods",
        "lords" in the plural are not."""
        counts: dict[str, Counter] = {"S": Counter(), "P": Counter()}
        for _, code, count in rows:
            match = _NOMINAL_CODE.match(code)
            counts[match["number"]][match["suffix"] or ""] += count
        shares = {}
        for number, counter in counts.items():
            total = sum(counter.values())
            shares[number] = ({suffix: n / total for suffix, n in counter.items()
                               if suffix in ("", "T") and n / total >= 0.05} if total else {"": 1.0})
            if not shares[number]:
                shares[number] = {"": 1.0}
        return shares

    def _learned_stems(self, lexeme: Lexeme, rows, gender: str) -> list[nominal.Declension]:
        """Third-declension stems cut from TAGNT forms no hypothesis generates (γυναικ-, ἀνδρ-)."""
        found = Counter()
        for key, code, count in rows:
            match = _NOMINAL_CODE.match(code)
            if match["case"] + match["number"] in ("GS", "DS", "AS", "NP", "GP", "AP"):
                plain = ph.plain(key)
                for ending in {"GS": ("οσ", "ουσ"), "DS": ("ι",), "AS": ("α",), "NP": ("εσ",), "GP": ("ων",),
                               "AP": ("ασ",)}[match["case"] + match["number"]]:
                    if ph.letters(plain).endswith(ending) and len(ph.letters(plain)) > len(ending) + 1:
                        found[_cut_letters(plain, len(ph.letters(ending)))] += count
                        break
        plain = ph.plain(lexeme.key)
        out = []
        for stem, count in found.most_common(2):
            out.append(nominal.third_declension(plain, lexeme.key, stem, gender, EVIDENCE_WEIGHTS["nt"],
                                                f"stem {stem}- from TAGNT"))
        return out

    def _index_noun(self, lexeme: Lexeme) -> None:
        rows = self._nominal_rows(lexeme)
        genders = [lexeme.gender] if lexeme.gender else [None]
        seen_genders = {(_NOMINAL_CODE.match(code)["gender"]) for _, code, _ in rows
                        if _NOMINAL_CODE.match(code)["head"] == "N"}
        if seen_genders:
            genders = sorted(seen_genders)
        if lexeme.gender is None and "M/F" in lexeme.note:
            genders = ["M", "F"] if not seen_genders else genders
        stated = irregular.irregular_nominal(lexeme.key)
        if stated and stated[0] == "A":
            stated = None
        hypotheses: list[tuple[list, float]] = []
        for gender in genders:
            for decl in nominal.noun_hypotheses(lexeme.key, gender, lexeme.genitive):
                hypotheses.append(([decl], decl.weight * (0.3 if stated else 1.0)))
        noun_rows = [row for row in rows if row[1].startswith("N-")]
        if noun_rows and not stated:
            oblique = [row for row in noun_rows if row[0] != lexeme.key]
            if oblique and not any(self._support(hypotheses, oblique)):
                for gender in (genders if genders != [None] else ["M"]):
                    for decl in self._learned_stems(lexeme, oblique, gender or "M"):
                        hypotheses.append(([decl], decl.weight))
            support = self._support(hypotheses, noun_rows)
            if not any(support):
                for gender in (genders if genders != [None] else ["M"]):
                    for decl in self._learned_stems(lexeme, noun_rows, gender or "M"):
                        hypotheses.append(([decl], decl.weight))
                support = self._support(hypotheses, noun_rows)
            best = max(support) if support else 0
            if best:
                hypotheses = [(decls, weight * (1.0 if s == best else (0.5 if s else CONTRADICTED)))
                              for (decls, weight), s in zip(hypotheses, support)]
        suffixes = self._suffixes(lexeme, noun_rows)

        def emit(key, slot, gender, weight, note):
            for suffix, share in suffixes[slot[1]].items():
                code = f"N-{slot}{gender}" + (f"-{suffix}" if suffix else "")
                self._add(key, lexeme, code, weight * share, note)
                if slot == "NS":
                    # The nominative serves as the vocative (κύριος, ἅγιος).
                    self._add(key, lexeme, f"N-VS{gender}" + (f"-{suffix}" if suffix else ""), 0.2 * weight * share,
                              note + "; nominative for vocative")

        if stated:
            gender, forms = stated
            for slot, key in forms:
                emit(key, slot, gender if gender in ("M", "F", "N") else (lexeme.gender or "M"),
                     EVIDENCE_WEIGHTS["table"], "irregular:stated forms")
        for decls, weight in hypotheses:
            for decl in decls:
                for form in nominal.decline(decl):
                    emit(form.key, form.case_number, form.gender,
                         form.weight / max(decl.weight, 1e-9) * weight, f"{decl.source}:{decl.note}")

    def _index_adjective(self, lexeme: Lexeme) -> None:
        rows = [row for row in self._nominal_rows(lexeme) if row[1].startswith("A-")]
        two = 0.6 if _compound_adjective(lexeme.key) else 0.15
        stated = irregular.irregular_nominal(lexeme.key)
        if stated and stated[0] != "A":
            stated = None
        hypotheses = [(list(decls), max(d.weight for d in decls) * (0.3 if stated else 1.0))
                      for decls in nominal.adjective_hypotheses(lexeme.key, two)]
        if rows and not stated:
            plain_rows = [row for row in rows if (_NOMINAL_CODE.match(row[1])["suffix"] or "") not in ("C", "S")]
            support = self._support(hypotheses, plain_rows)
            best = max(support) if support else 0
            if best:
                hypotheses = [(decls, weight * (1.0 if s == best else (0.5 if s else CONTRADICTED)))
                              for (decls, weight), s in zip(hypotheses, support)]
        letters = bare(lexeme.key)
        degree = ""
        if (re.search(r"(ιων|σσων|ττων|τερον|τεροσ)$", letters) and letters not in NOT_COMPARATIVES) \
                or letters in COMPARATIVES:
            degree = "C"          # a comparative lemma (κρείσσων, χείρων, κομψότερον)
        if rows:
            # TAGNT's own use decides: πρεσβύτερος -C, ἐλάχιστος plain.
            counts = Counter(_NOMINAL_CODE.match(code)["suffix"] or "" for _, code, _ in rows)
            degree = "C" if counts["C"] > counts[""] else ("S" if counts["S"] > counts[""] else "")

        def emit(key, slot, weight, note, suffix=degree):
            code = f"A-{slot}" + (f"-{suffix}" if suffix else "")
            self._add(key, lexeme, code, weight, note)
            if slot[:2] == "NS" and slot[2] in "MF":
                self._add(key, lexeme, f"A-VS{slot[2]}" + (f"-{suffix}" if suffix else ""), 0.2 * weight,
                          note + "; nominative for vocative")

        if stated:
            for slot, key in stated[1]:
                emit(key, slot, EVIDENCE_WEIGHTS["table"], "irregular:stated forms")
        for decls, weight in hypotheses:
            for decl in decls:
                for form in nominal.decline(decl):
                    emit(form.key, form.case_number + form.gender, form.weight / max(decl.weight, 1e-9) * weight,
                         f"{decl.source}:{decl.note}")
        if degree:
            return
        compared = [(normalize(form), grade, 0.9, "irregular comparison")
                    for name, items in irregular.IRREGULAR_COMPARISON.items() if normalize(name) == lexeme.key
                    for form, grade in items]
        compared += [(ph.finish(ph.accentuate(nominative, "recessive")[0]), grade, weight * 0.8, "rule:comparison")
                     for nominative, grade, weight in nominal.comparison_stems(lexeme.key)]
        for nominative, grade, weight, note in compared:
            for decls in nominal.adjective_hypotheses(nominative, 0):
                for decl in decls:
                    for form in nominal.decline(decl):
                        emit(form.key, form.case_number + form.gender, form.weight * weight, note, grade)

    def _index_invariable(self, lexeme: Lexeme) -> None:
        """A word TBESG files as an adjective that does not decline
        (ἐνταῦθα, κάτωθεν, ἑκκαίδεκα, παράπαν): its lemma is its form,
        coded as TAGNT codes its kind (A-NUI a numeral, ADV an adverb)."""
        letters = bare(lexeme.key)
        if re.search(r"(κοντα|δεκα|κοσιοι|πεντε|εξ|επτα|οκτω|εννεα|δυο|τρεισ|τεσσαρεσ)$", letters) or "NUI" in lexeme.grammar:
            code = "A-NUI"
        elif re.search(r"(θεν|θα|θι|ωσ|δον|ακισ|ου|ει|ι|η|α|ον|ην|ιν|εσ|ισ|ξ)$", letters):
            code = "ADV"
        else:
            code = None
        self._add(lexeme.key, lexeme, code, 0.8, "rule:undeclined" + ("" if code else " (code unknown)"))

    def _index_pronoun(self, lexeme: Lexeme) -> None:
        """Pronouns that decline like adjectives (ὅσος, ἐκεῖνος, τοιοῦτος,
        σεαυτοῦ, ἡμέτερος), with TAGNT's pronoun codes: K-, D-, R-, Q-, X-,
        I-, C- {case}{number}{gender}; F-{person}..., S-{person}{number}...
        Neuters in -ο where the pronoun has them (ἐκεῖνο, αὐτό, τοῦτο)."""
        head = lexeme.grammar.split("-")[0]
        person = lexeme.grammar.split("-")[1] if "-" in lexeme.grammar else ""
        letters = bare(lexeme.key)
        plain = ph.plain(lexeme.key)
        forms: list[tuple[str, str]] = []
        if letters.endswith("ουτοσ") and len(letters) > 5:
            prefix = plain[:-len("ουτοσ")] if plain.endswith("ουτοσ") else ph.compose(ph.clusters(plain)[:-5])
            for slot, ending in (("NSM", "οῦτοσ"), ("GSM", "ούτου"), ("DSM", "ούτῳ"), ("ASM", "οῦτον"), ("NPM", "οῦτοι"),
                                 ("GPM", "ούτων"), ("DPM", "ούτοισ"), ("APM", "ούτουσ"), ("NSF", "αύτη"),
                                 ("GSF", "αύτησ"), ("DSF", "αύτῃ"), ("ASF", "αύτην"), ("NPF", "αῦται"),
                                 ("GPF", "ούτων"), ("DPF", "αύταισ"), ("APF", "αύτασ"), ("NSN", "οῦτο"),
                                 ("NSN", "οῦτον"), ("ASN", "οῦτο"), ("ASN", "οῦτον"), ("GSN", "ούτου"),
                                 ("DSN", "ούτῳ"), ("NPN", "αῦτα"), ("APN", "αῦτα"), ("GPN", "ούτων"),
                                 ("DPN", "ούτοισ")):
                forms.append((slot, ph.finish(prefix + ending)))
        else:
            if letters.endswith(("ου",)) and head == "F":
                lemma_key = ph.finish(ph.accentuate(_cut_letters(plain, 2) + "οσ", "syllable", None, 0, -1, ph.ACUTE)[0])
            else:
                lemma_key = lexeme.key
            for decls in nominal.adjective_hypotheses(lemma_key, 0):
                for decl in decls:
                    for form in nominal.decline(decl):
                        forms.append((form.case_number + form.gender, form.key))
            if head in ("D", "F", "R") or letters.endswith("οσ") and head == "P":
                # ἐκεῖνο, αὐτό: the neuter in -ο.
                forms = [(slot, key[:-1] if slot[:2] in ("NS", "AS", "VS") and slot[2] == "N" and key.endswith("ν")
                          else key) for slot, key in forms]
        for slot, key in forms:
            if head == "F":
                if slot[:2] == "NS":
                    continue
                code = f"F-{person}{slot}"
            elif head == "S":
                code = f"S-{person}{'P' if bare(lexeme.key).startswith(('ημ', 'υμ')) else 'S'}{slot}"
            else:
                code = f"{head}-{slot}"
            self._add(key, lexeme, code, 0.8, "rule:pronoun declined as an adjective")

    # -- the index -----------------------------------------------------------

    def _add(self, key: str, lexeme: Lexeme, code: str, weight: float, note: str) -> None:
        letters = bare(key)
        if self.vocabulary is not None and letters not in self.vocabulary:
            return
        self.index.setdefault(letters, []).append(Entry(key, lexeme.index, code, weight * self._prior(lexeme), note))

    def _prior(self, lexeme: Lexeme) -> float:
        """A mild preference for words TAGNT uses often, between words that
        share a form (θεῶν: θεός before θεά): 0.6 for a word TAGNT lacks, up
        to 1.0 for one it has a thousand times."""
        cached = self._priors.get(lexeme.strongs)
        if cached is None:
            count = sum(n for _, _, n in self.evidence.rows(lexeme.strongs)) if self.evidence else 0
            cached = 0.6 + 0.4 * min(1.0, math.log10(1 + count) / 3)
            self._priors[lexeme.strongs] = cached
        return cached

    # -- lookup ----------------------------------------------------------------

    def readings(self, key: str, caps: bool = False) -> list[tuple[Lexeme, str | None, float, str]]:
        """(lexeme, code, weight, note) for every generated form matching `key`."""
        letters = bare(key)
        entries = self.index.get(letters, ())
        exact, near = [], []
        for entry in entries:
            if caps or entry.key == key:
                exact.append((entry, 1.0, ""))
                continue
            factor, why = _compare(entry.key, key)
            if factor >= SUBSCRIPT_MISMATCH:
                exact.append((entry, factor, why))
            elif factor > 0:
                near.append((entry, factor, why))
        if len(letters) <= 2:
            near = []          # ὦ is not ὤ: too short for a mark to be a slip
        chosen = exact or near
        self._last_match = max((factor for _, factor, _ in chosen), default=0.0)
        return [(self.lexemes[entry.lexeme], entry.code, entry.weight * factor,
                 entry.note + (f"; {why}" if why else "")) for entry, factor, why in chosen]

    def candidates(self, key: str, caps: bool = False) -> list[Candidate]:
        best: dict[tuple, Candidate] = {}
        for lexeme, code, weight, note in self.readings(key, caps):
            morph, extra = self._code(code)
            if morph is None and code is not None:
                if _is_vocative(code):
                    continue        # the nominative's form, which TAGNT codes as that
                weight *= UNCODED
            # A lemma TBESG capitalises (Κυρία) is a name for the case filter.
            kind = "name" if lexeme.lemma[:1].isupper() else "common"
            candidate = Candidate(lexeme.strongs, lexeme.lemma, morph, SOURCE, weight, kind, True,
                                  f"{note}{extra}; {lexeme.note}")
            identity = candidate.reading
            if identity not in best or best[identity].weight < weight:
                best[identity] = candidate
        if not best and key.endswith("’"):
            return self._elided(key)
        if not best:
            return []
        top = max(c.weight for c in best.values())
        # The best reading weighs 1.0, or less when its form only matched
        # with its accent or breathing differing (then ACCENT_MISMATCH ...).
        scale = 1.0 if self._last_match >= SUBSCRIPT_MISMATCH else self._last_match
        kept = [replace(c, weight=round(scale * c.weight / top, 4)) for c in best.values()
                if c.weight >= MIN_SHARE * top]
        return sorted(kept, key=lambda c: (-c.weight, c.strongs, c.lemma, c.morph or ""))

    def _code(self, code: str | None) -> tuple[str | None, str]:
        """A parse as a code TAGNT uses: the code itself, or TAGNT's code for
        the same parse under its other voice letter (D for M, O for P ...),
        or a superlative's or comparative's as the positive's, or None."""
        if code is None or self.codes is None or code in self.codes:
            return code, ""
        for other in _nearest_codes(code):
            if other in self.codes:
                return other, f"; parse {code} coded as {other} (TAGNT lacks {code})"
        return None, f"; parse {code} (TAGNT lacks the code)"

    def _elided(self, key: str) -> list[Candidate]:
        """An elided word (ἀντ’, καθ’, οὐδ’): its full forms, the elided vowel
        restored (and an aspirate before a rough breathing undone)."""
        stem = key[:-1]
        stems = {stem}
        if bare(stem)[-1:] in ("φ", "θ", "χ"):
            stems.add(stem[:-1] + {"φ": "π", "θ": "τ", "χ": "κ"}[bare(stem)[-1]])
        found: dict[tuple, Candidate] = {}
        for base in stems:
            for vowel in ("α", "ε", "ι", "ο"):
                for spelling in {base + vowel, *(ph.finish(a) for a in ph.accentuate(ph.plain(base) + vowel,
                                                                                         "syllable", None, 0, -1))}:
                    full = normalize(spelling)
                    for entry in self.lexicon.by_key(full):
                        if entry.pos_class in ("preposition", "conjunction", "particle", "adverb"):
                            morph = entry.grammar if self.codes is None or entry.grammar in self.codes else None
                            lemma = (self.forms.lemma_for(entry.strongs) if self.forms is not None else None) or entry.lemma
                            candidate = Candidate(entry.strongs, lemma, morph, SOURCE, 1.0, "common", True,
                                                  f"elided {full}; TBESG {entry.extended}")
                            found[candidate.reading] = candidate
                    for candidate in self.candidates(full):
                        found.setdefault(candidate.reading, replace(candidate, notes=f"elided {full}; {candidate.notes}"))
        return sorted(found.values(), key=lambda c: (-c.weight, c.strongs, c.lemma, c.morph or ""))

    def __call__(self, key: str, token: Token) -> list[Candidate]:
        found = self.candidates(key, caps=token.case == "caps")
        if found or token.case == "caps":
            return found
        # Spellings that vary between editions and the lexicon, tried only
        # when the form itself matches nothing (`bridged_keys`).
        merged: dict[tuple, Candidate] = {}
        for variant, rule in bridged_keys(key):
            for candidate in self.candidates(variant):
                bridged = replace(candidate, weight=round(candidate.weight * BRIDGED, 4),
                                  notes=f"{candidate.notes}; spelt {rule} ({variant})")
                if candidate.reading not in merged or merged[candidate.reading].weight < bridged.weight:
                    merged[candidate.reading] = bridged
        return sorted(merged.values(), key=lambda c: (-c.weight, c.strongs, c.lemma, c.morph or ""))


# A reading found through a spelling bridge, relative to a direct match.
BRIDGED = 0.8
_BRIDGES = (("ει", "ι", "ει/ι"), ("ι", "ει", "ι/ει"), ("ρευ", "ραυ", "ερευν-/εραυν-"), ("ραυ", "ρευ", "εραυν-/ερευν-"))


def bridged_keys(key: str) -> list[tuple[str, str]]:
    """Other spellings of a form, one change at a time, for the spellings
    the editions and the lexicon disagree on: ει for ι and back (σάπφειρος,
    σάπφιρος; ἀπώλεια, Brenton's ἀπωλίᾳ; δαβείρ, δαβίρ; -εία/-ία), and
    ἐραυν-/ἐρευν-. A change keeps the marks: the accent of εῖ goes to ῖ.
    Works on keys and on bare letters alike."""
    import unicodedata
    clusters: list[list[str]] = []
    for char in unicodedata.normalize("NFD", key):
        if unicodedata.combining(char) and clusters:
            clusters[-1].append(char)
        else:
            clusters.append([char])
    letters = "".join(c[0] for c in clusters)
    out: list[tuple[str, str]] = []
    for old, new, rule in _BRIDGES:
        start = letters.find(old)
        while start >= 0:
            if old == "ι" and start > 0 and letters[start - 1] in "αεου":
                start = letters.find(old, start + 1)    # a diphthong's ι is not a spelling of ει
                continue
            span = clusters[start:start + len(old)]
            marks = [m for c in span for m in c[1:] if m != "\u0308"]
            replaced = [[letter] for letter in new]
            replaced[-1] += marks if old != "ρευ" and old != "ραυ" else []
            if old in ("ρευ", "ραυ"):
                replaced = [[n, *c[1:]] for n, c in zip(new, span)]
            variant = clusters[:start] + replaced + clusters[start + len(old):]
            text = unicodedata.normalize("NFC", "".join("".join(c) for c in variant))
            if text != key and (text, rule) not in out:
                out.append((text, rule))
            start = letters.find(old, start + 1)
    return out


def _is_vocative(code: str) -> bool:
    return bool(re.search(r"-V[SP][MFN]?(?:-|$)", code))


_VOICE_ALTERNATIVES = {"D": "MN", "N": "MPD", "O": "PN", "E": "PM", "M": "DN", "P": "ON"}


def _nearest_codes(code: str) -> list[str]:
    """Codes TAGNT may use for the same parse, nearest first."""
    found = []
    match = _VERB_CODE.match(code)
    if match:
        tense = match["tense"]
        rest = f"-{match['rest']}" if match["rest"] else ""
        for voice in _VOICE_ALTERNATIVES.get(match["voice"], ""):
            found.append(f"V-{match['second']}{tense}{voice}{match['mood']}{rest}")
        return [c for c in dict.fromkeys(found) if c != code]
    if re.search(r"-[SC]$", code):
        found.append(code[:-2])
    return found


def _compare(generated: str, key: str) -> tuple[float, str]:
    """How well a generated form with the same letters matches a key: 1.0
    the same; SUBSCRIPT_MISMATCH an iota subscript differs; ACCENT_MISMATCH
    the accent differs; BREATHING_MISMATCH the breathing (with or without
    the accent) differs."""
    a, b = ph.clusters(generated), ph.clusters(key)
    if len(a) != len(b):
        return 0.0, ""
    factor, why = 1.0, []
    for x, y in zip(a, b):
        mx, my = set(x[1:]), set(y[1:])
        if mx == my:
            continue
        if (mx & ph.BREATHINGS) != (my & ph.BREATHINGS):
            # Editions differ on some words' breathing (ἅμαξα, ἄμαξα).
            factor = min(factor, BREATHING_MISMATCH)
            why.append("breathing differs")
            if (mx - ph.BREATHINGS) == (my - ph.BREATHINGS):
                continue
        if (mx ^ my) <= {ph.SUBSCRIPT}:
            factor = min(factor, SUBSCRIPT_MISMATCH)
            why.append("iota subscript differs")
            continue
        factor = min(factor, ACCENT_MISMATCH)
        why.append("accent differs")
    return factor, ", ".join(dict.fromkeys(why))


# -τερος words that are not comparatives, and comparatives the pattern misses.
NOT_COMPARATIVES = frozenset(bare(normalize(w)) for w in (
    "ἕτερος", "δεύτερος", "ἀμφότερος", "πότερος", "ἑκάτερος", "ὕστερος", "ἀριστερός", "δεξιτερός",
    "ἡμέτερος", "ὑμέτερος", "πρότερος", "ἀμφότεροι", "ἕτερον", "δεύτερον", "πρότερον", "ὕστερον"))
COMPARATIVES = frozenset(bare(normalize(w)) for w in (
    "χείρων", "μείζων", "πλείων", "πλέων", "ἐλάσσων", "ἐλάττων", "ἥσσων", "ἥττων", "ἀμείνων", "βελτίων",
    "κρείσσων", "κρείττων", "καλλίων", "ἡδίων", "θάσσων"))


def _compound_adjective(key: str) -> bool:
    """Whether an -ος adjective looks compound, so may have two terminations
    (ἀλλόφυλος, ἄδικος, εὔκαιρος, πολυέλεος)."""
    letters = bare(key)
    if letters.startswith(("α", "δυσ", "ευ", "πολυ", "παν", "ομο", "αλλο", "μονο", "φιλο", "ισο")):
        return True
    return any(chain for chain, _ in ph.split_preverbs(ph.plain(key)))


def default_analyser(forms=None, lexicon=None, vocabulary: Iterable[str] | None = None) -> Analyser:
    """The analyser over the repository's TAGNT and TBESG."""
    from lxx_tagging import lexicon as lexicon_module, ntforms
    if forms is None:
        forms = ntforms.load()
    if lexicon is None:
        lexicon = lexicon_module.load()
    return Analyser(lexicon, forms, vocabulary=vocabulary, tbesg_path=lexicon_module.DEFAULT_TBESG)
