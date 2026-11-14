"""Candidate readings for a token, from an ordered list of providers, and the
policy that turns them into a decision.

PROVIDERS

A provider is any callable `(key, token) -> list[Candidate]`: the token's
normalised key (`Token.key`) and the token itself, for whatever else it wants
to know (case, position, the joined word it is part of). Providers are
consulted in order. A "fallback" provider is consulted only while no earlier
provider has offered a candidate that survived the case filter; an "always"
provider is consulted every time and its candidates are added to the others'.
The default order:

  nt-forms       TAGNT's readings of the exact form (or, for a word printed in
                 capitals, of every form with those letters; for part of a
                 joined word TAGNT also joins, of that part), then of licensed
                 alternates (movable nu, οὕτω/οὕτως, λήψεται/λήμψεται ...),
                 then, for a capitalised word, of a name whose dictionary
                 form is the word with the other initial breathing (Ἁβραάμ/
                 Ἀβραάμ), without its case; see
                 `NTFormProvider._name_spelt_otherwise`.
  tbesg          TBESG entries whose lemma is the form, for words whose lemma
                 is their only form: names, transliterated Hebrew words,
                 indeclinables, and adverbs, prepositions, conjunctions,
                 particles and interjections. A lemma TBESG prints with no
                 breathing (αιλάμ, εξεναντίας) matches the word with either;
                 an entry TBESG gives no part of speech (εἰκοσιοκτώ, κανοῦν)
                 is the word its lemma prints, tagged without morphology; the
                 Hebrew letters heading the acrostics that Brenton spells
                 otherwise than TBESG (ΑΛΕΦ, TBESG's Αλφ) are listed
                 (`HEBREW_LETTER_SPELLINGS`).
  analyser       the morphological analyser (`analyser.py`): every reading of
                 the form in the generated paradigms of TBESG's verbs, nouns,
                 adjectives and pronouns, for forms TAGNT lacks. Registered by
                 the CLI, or `default_tagger(analyser=...)`, or
                 `tagger.register("analyser", fn, before="nt-tolerant")`.
  nt-tolerant    TAGNT's readings of forms with the same letters whatever their
                 accents and breathings. Never tagged (`taggable=False`): a hint
                 for review and for later stages.
  tbesg-tolerant TBESG names matched the same way. Never tagged.

CASE FILTER

Brenton capitalises names and the titles of God; TAGNT marks names -P, -L,
-PG, -LG and titles -T (`Candidate.kind`). What a token's capital says is used
to set aside candidates of the wrong kind:

  * lower case anywhere: a common word. Names are set aside, and titles too if
    a common reading remains (κυρίου μου is "my master", not the LORD).
  * a capital mid-sentence (`Token.name_like`): a name or title. Common words
    are set aside: Λίβανον is Lebanon, not λίβανος "frankincense".
  * a capital that starts a sentence or follows a comma or ano teleia, or a
    word in capitals, says nothing by itself, so Brenton's own use of the same
    form mid-sentence decides (`CaseProfile`): lower case there 95% of the
    time or more, a common word; capitalised 95% or more, a name or title;
    in between, either, so a single common reading is not tagged (Λίβανον,
    Αἰλάμ, Ὤν); never seen there, no filter.

PARADIGM RULES

TAGNT attests a form only in the parses the New Testament happens to use, so
ὄνομα seen only as an accusative would otherwise be tagged accusative
everywhere. `paradigm_codes` adds the parses every Greek paradigm gives the
same form (neuter nominative = accusative = vocative, and the like), as
candidates with weight 0 and source "...+paradigm", when TAGNT uses the code.

CODES

Every morphology code a candidate carries is a TAGNT code verbatim (or None).
`Tagger` refuses a chosen candidate whose code TAGNT does not use, so a later
provider or chooser cannot put a code into the module the decoder has never
seen.

Set-aside candidates are kept on the result with the reason, so the
decisions file shows them.

TIERS (`TokenCandidates.tier`)

  a  one reading (Strong's, lemma, morphology): tagged. A name or loanword
     whose case the form does not show has morphology None and is tagged with
     its number and lemma only.
  b  one Strong's number and lemma, several parses.
  c  several Strong's numbers or lemmas.
  d  looks like a name and has no acceptable candidate: none at all, or only
     common-word readings the capital rules out (kept in `set_aside`).
  e  no candidate.
  u  candidates, but unconfirmed: found only by a tolerant lookup, or only
     names for a word Brenton prints in lower case. Not tagged.

CHOOSERS

A chooser takes one verse's `TokenCandidates`, in order, and returns one
`Choice` or None per token. The default, `unique_reading_chooser`, chooses
tier a's reading and nothing else. A disambiguator plugs in with
`Tagger(chooser=...)` or `tagger.chooser = ...`.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Protocol, Sequence

from lxx_tagging.lexicon import INVARIABLE_CLASSES, Lexicon, split_variants
from lxx_tagging.ntforms import NTForms, Reading, reading_kind
from lxx_tagging.text import (
    ROUGH, SMOOTH, Token, alternates, bare, caps_pattern, licensed, normalize, other_breathing, tokenize,
)


@dataclass(frozen=True)
class Candidate:
    """One possible reading of a token, and where it came from.

    `morph` is a TAGNT code verbatim, or None where the source cannot give one
    (a name whose case its form does not show). `weight` is the evidence
    count (TAGNT occurrences) or a provider's own score; `kind` is "common",
    "name" or "title"; `taggable` False marks a hint that is never tagged
    by itself.
    """

    strongs: str
    lemma: str
    morph: str | None
    source: str
    weight: float = 0.0
    kind: str = "common"
    taggable: bool = True
    notes: str = ""

    @property
    def reading(self) -> tuple[str, str, str | None]:
        return (self.strongs, self.lemma, self.morph)

    def compact(self) -> str:
        """`G976|βίβλος|N-NSF|nt-form|12` for the decisions file."""
        weight = f"{self.weight:g}"
        return f"{self.strongs}|{self.lemma}|{self.morph or ''}|{self.source}|{weight}"


Provider = Callable[[str, Token], list[Candidate]]


@dataclass
class ProviderEntry:
    name: str
    provider: Provider
    mode: str = "fallback"   # "fallback" | "always"


@dataclass
class TokenCandidates:
    """A token, its candidates after the case filter, and the decision tier."""

    token: Token
    candidates: list[Candidate]
    set_aside: list[tuple[Candidate, str]] = field(default_factory=list)
    tier: str = "e"
    reason: str = ""

    def readings(self) -> list[tuple[str, str, str | None]]:
        """The distinct readings, most evidence first, then alphabetically."""
        weights: Counter = Counter()
        for candidate in self.candidates:
            weights[candidate.reading] += candidate.weight
        return sorted(weights, key=lambda reading: (-weights[reading], tuple(x or "" for x in reading)))


@dataclass(frozen=True)
class Choice:
    """A chooser's decision for one token: a candidate, how sure, and by what margin."""

    candidate: Candidate
    confidence: float = 1.0
    margin: float = 1.0
    method: str = "unique"


class Chooser(Protocol):
    def __call__(self, verse: Sequence[TokenCandidates]) -> list[Choice | None]: ...


def unique_reading_chooser(verse: Sequence[TokenCandidates]) -> list[Choice | None]:
    """Choose a token's reading only when there is exactly one (tier a)."""
    choices: list[Choice | None] = []
    for item in verse:
        if item.tier != "a":
            choices.append(None)
            continue
        best = max((c for c in item.candidates if c.taggable), key=lambda c: (c.weight, c.source))
        choices.append(Choice(best, 1.0, 1.0, "unique"))
    return choices


# ---------------------------------------------------------------------------
# Brenton's own use of capitals

class CaseProfile:
    """How often Brenton prints each key in lower case and with a capital in
    mid-sentence, where the capital means a name or title."""

    def __init__(self) -> None:
        self.lower: Counter[str] = Counter()
        self.capital: Counter[str] = Counter()

    def add(self, token: Token) -> None:
        if not token.case_informative:
            return
        (self.capital if token.case == "capital" else self.lower)[token.key] += 1

    @classmethod
    def from_tokens(cls, tokens: Iterable[Token]) -> "CaseProfile":
        profile = cls()
        for token in tokens:
            profile.add(token)
        return profile

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CaseProfile":
        return cls.from_tokens(token for text in texts for token in tokenize(text))

    # A form Brenton prints mid-sentence with a capital at most this share of
    # the time (or at least 1 minus it) leans one way: the odd capital Καὶ
    # (once in 7,889) is a slip, the 2.6% of lower-case κύριος are "master".
    MINORITY = 0.05

    def leaning(self, key: str) -> str:
        """"lower", "capital", "mixed" or "unknown"."""
        lower, capital = self.lower[key], self.capital[key]
        total = lower + capital
        if not total:
            return "unknown"
        share = capital / total
        if share <= self.MINORITY:
            return "lower"
        if share >= 1 - self.MINORITY:
            return "capital"
        return "mixed"


def expectation(token: Token, profile: CaseProfile | None) -> str | None:
    """What the token's case says it is: "name", "common", "either" or None."""
    if token.case == "lower":
        return "common"
    if token.name_like:
        return "name"
    leaning = profile.leaning(token.key) if profile else "unknown"
    return {"lower": "common", "capital": "name", "mixed": "either"}.get(leaning)


def case_filter(token: Token, candidates: list[Candidate], expected: str | None
                ) -> tuple[list[Candidate], list[tuple[Candidate, str]]]:
    """Candidates kept and set aside (with the reason) by what the case says."""
    if expected == "name":
        kept = [c for c in candidates if c.kind in ("name", "title")]
        aside = [(c, "capitalised mid-sentence: a name or title") for c in candidates if c not in kept]
        return kept, aside
    if expected == "common":
        common = [c for c in candidates if c.kind == "common"]
        titles = [c for c in candidates if c.kind == "title"]
        kept = common or titles
        aside = [(c, "lower case: not a name" if c.kind == "name" else "lower case: not a title")
                 for c in candidates if c not in kept]
        return kept, aside
    return list(candidates), []


def classify(token: Token, kept: list[Candidate], aside: list[tuple[Candidate, str]],
             expected: str | None) -> tuple[str, str]:
    """The tier and reason for a token's surviving candidates. See the module docstring."""
    capital = token.case != "lower"
    if not kept:
        if aside and expected == "name":
            return "d", "name-like: only common readings"
        if aside:
            return "u", "lower case: only name readings"
        if expected == "name" or (capital and expected == "either"):
            return "d", "name-like: no candidate"
        return "e", "no candidate"
    if not any(c.taggable for c in kept):
        return "u", "tolerant match only"
    readings = {c.reading for c in kept}
    if len(readings) == 1:
        only = kept[0]
        if capital and expected == "either" and only.kind == "common":
            return "d", "capital may be a name: only a common reading"
        return "a", "one reading"
    if len({(s, l) for s, l, _ in readings}) == 1:
        return "b", "several parses"
    return "c", "several words"


# ---------------------------------------------------------------------------
# Paradigm rules: forms that are the same word in every Greek paradigm

_NOMINAL = re.compile(r"^(?P<head>[NATPRCDKIXQF])-(?P<person>\d?)(?P<case>[NGDAV])(?P<number>[SP])(?P<gender>[MFN]?)(?P<rest>(?:-.*)?)$")
_PARTICIPLE = re.compile(r"^V-(?P<tvm>[^-]+)-(?P<case>[NGDAV])(?P<number>[SP])(?P<gender>[MFN])$")
_PRESENT_2P = re.compile(r"^V-P(?P<voice>[A-Z])(?P<mood>[IM])-2P$")


def _accented_ending(key: str) -> bool:
    """Whether the accent falls on the last vowel of the form."""
    nfd = unicodedata.normalize("NFD", key)
    vowels = [index for index, char in enumerate(nfd) if char in "αεηιουω"]
    if not vowels:
        return False
    tail = nfd[vowels[-1]:]
    # A diphthong's accent stands on its second vowel; -ας has none.
    return any(mark in tail for mark in ("\u0301", "\u0300", "\u0342"))


# Verbs whose aorist subjunctive is not the future's form: the κ-aorists
# (δώσω/δῶ, θήσω/θῶ, ἥσω/ὧ) and their compounds.
_KAPPA_AORISTS = ("διδωμι", "τιθημι", "ιημι")


def paradigm_codes(morph: str, key: str = "", lemma: str = "") -> list[str]:
    """Other codes every Greek word has the same form for, given one code and
    (for the rules that depend on the ending) the form's key.

    Only identities that hold in every paradigm are used:
      * neuter nominative, accusative and vocative are one form;
      * plural nominative and vocative are one form;
      * masculine and neuter genitive and dative are one form in adjectives,
        pronouns, participles and the article (nouns have one gender);
      * a feminine genitive singular in -ας not accented on the ending is a
        first-declension -α word, whose accusative plural is the same
        (ἀνομίας, ἡμέρας; but χαρᾶς and χαράς differ);
      * a plural in -εις is nominative and accusative alike (πόλεις,
        ἱερεῖς, ἀληθεῖς, τρεῖς);
      * present indicative and imperative second plural are one form;
      * an active first singular in -ω is present indicative and subjunctive
        alike (λύω, ποιῶ);
      * a first singular in -σω (-ξω, -ψω) is future indicative and aorist
        subjunctive alike (ποιήσω), so either attested gives both. (Where a
        verb's future and aorist stems differ, as in a second aorist, the
        extra parse is only an ambiguity, never a tag; the κ-aorists of
        δίδωμι, τίθημι and ἵημι, whose subjunctives are δῶ, θῶ, ὧ, are
        left out by name.)
    So a reading TAGNT happens to attest for one of them does not become a
    tag that rules out the others.
    """
    found: list[str] = []
    match = _NOMINAL.match(morph)
    participle = _PARTICIPLE.match(morph) if not match else None
    letters = bare(key) if key else ""
    if match or participle:
        parts = (match or participle).groupdict()
        case, number, gender = parts["case"], parts["number"], parts["gender"]

        def code(c: str, n: str, g: str) -> str:
            if match:
                return f"{parts['head']}-{parts['person']}{c}{n}{g}{parts['rest']}"
            return f"V-{parts['tvm']}-{c}{n}{g}"

        if gender == "N" and case in "NAV":
            found += [code(c, number, gender) for c in "NAV" if c != case]
        if number == "P" and case in "NV":
            found.append(code("V" if case == "N" else "N", number, gender))
        declines_by_gender = participle or (match and parts["head"] != "N")
        if declines_by_gender and gender in ("M", "N") and case in "GD":
            found.append(code(case, number, "N" if gender == "M" else "M"))
        if (case, number, gender) == ("G", "S", "F") and letters.endswith("ασ") and not _accented_ending(key):
            found.append(code("A", "P", "F"))
        if number == "P" and case in "NA" and letters.endswith("εισ"):
            found.append(code("A" if case == "N" else "N", "P", gender))
    present = _PRESENT_2P.match(morph)
    if present:
        other = "M" if present["mood"] == "I" else "I"
        found.append(f"V-P{present['voice']}{other}-2P")
    if letters.endswith("ω"):
        if morph == "V-PAI-1S":
            found.append("V-PAS-1S")
        elif morph == "V-PAS-1S":
            found.append("V-PAI-1S")
    if letters.endswith(("σω", "ξω", "ψω")) and not bare(normalize(lemma)).endswith(_KAPPA_AORISTS):
        if morph == "V-FAI-1S":
            found.append("V-AAS-1S")
        elif morph == "V-AAS-1S":
            found.append("V-FAI-1S")
    return [c for c in dict.fromkeys(found) if c != morph]


# ---------------------------------------------------------------------------
# Providers

# Suffixes of loanwords TAGNT tags by their sentence, like names.
_UNDECLINED_SUFFIXES = ("-HEB", "-ARAM", "-LI")


class NTFormProvider:
    """Readings from TAGNT's form table. See the module docstring."""

    name = "nt-forms"

    def __init__(self, forms: NTForms, paradigm: bool = True):
        self.forms = forms
        self.paradigm = paradigm
        self._key = ""

    def __call__(self, key: str, token: Token) -> list[Candidate]:
        self._key = key
        if token.joined:
            parts = self._compound_part(token)
            if parts:
                return parts
        if token.case == "caps":
            found = self.forms.caps_readings(caps_pattern(key, token.first_marks_known))
            if found:
                return self._candidates(found, "nt-caps")
            other = other_breathing(key) if token.first_marks_known else None
            names = Counter({reading: count for reading, count in
                             self.forms.caps_readings(caps_pattern(other, True)).items()
                             if reading_kind(reading.morph) == "name"}) if other else Counter()
            return self._names_without_case(names, "nt-caps:name-breathing")
        found = self.forms.readings(key)
        if found:
            return self._candidates(found, "nt-form")
        if "apostrophe-breathing" in token.rules:
            rough = unicodedata.normalize("NFC", unicodedata.normalize("NFD", key).replace(SMOOTH, ROUGH, 1))
            found = self.forms.readings(rough)
            if found:
                return self._candidates(found, "nt-form:apostrophe-breathing")
        candidates: list[Candidate] = []
        for alternate in alternates(key):
            readings = Counter({reading: count for reading, count in self.forms.readings(alternate.key).items()
                                if licensed(alternate.rule, alternate.key, reading.lemma, reading.morph)})
            candidates += self._candidates(readings, f"nt-form:{alternate.rule}")
        if not candidates and token.case != "lower":
            candidates = self._name_spelt_otherwise(key)
        return candidates

    def _name_spelt_otherwise(self, key: str) -> list[Candidate]:
        """A name whose dictionary form is this word with the other initial
        breathing (Ἁβραάμ/Ἀβραάμ, Ἡσαῦ/Ἠσαῦ, Ὠσηέ/Ὡσηέ): Hebrew names'
        breathings vary by edition. Only the dictionary form is matched, never
        an inflected form, which can be another name's: Brenton's Ἄνναν is
        Hannah, and the NT's Ἅνναν is Annas. The case is not taken from the
        match. Accents are not relaxed: Brenton's Ἀχαϊά (Ahlai) is not the
        NT's Ἀχαΐα (Achaia), nor Ἠλιού (Elihu) WH's Ἠλίου (Elijah's)."""
        other = other_breathing(key)
        names: Counter[Reading] = Counter()
        for reading, count in (self.forms.readings(other).items() if other else ()):
            if reading_kind(reading.morph) == "name" and any(
                    normalize(variant) == other for variant in split_variants(reading.lemma)):
                names[reading] += count
        return self._names_without_case(names, "nt-form:name-breathing") if names else []

    def _names_without_case(self, names: Counter[Reading], source: str) -> list[Candidate]:
        """Names found under another spelling, without the case, which a
        differently spelt form does not vouch for."""
        candidates = [
            replace(candidate, morph=None, notes=f"name spelt otherwise ({source}); case not taken from it")
            for candidate in self._candidates(names, source, paradigm=False)
        ]
        return list({c.reading + (c.kind,): c for c in candidates}.values())

    def _compound_part(self, token: Token) -> list[Candidate]:
        """For a part of a joined word TAGNT also prints joined (ἱνατί), the
        NT's reading of that part."""
        found: Counter[Reading] = Counter()
        for parts, count in self.forms.parts_of(token.word_key).items():
            if len(parts) == token.parts:
                found[parts[token.part]] += count
        return self._candidates(found, "nt-compound") if found else []

    def _candidates(self, readings: Counter[Reading], source: str, paradigm: bool | None = None) -> list[Candidate]:
        merged: Counter = Counter()
        notes: dict = {}
        for reading, count in readings.items():
            morph: str | None = reading.morph
            kind = reading_kind(morph)
            loanword = kind == "title" or morph.endswith(_UNDECLINED_SUFFIXES)
            if ((kind == "name" and not self.forms.case_shown_by_form(reading.strongs))
                    or (loanword and not self.forms.case_shown_by_form(reading.strongs, strict=False))):
                morph = None
                notes[(reading.strongs, reading.lemma, None, kind)] = "case not shown by the form"
            merged[(reading.strongs, reading.lemma, morph, kind)] += count
        candidates = [
            Candidate(strongs, lemma, morph, source, float(count), kind, True,
                      notes.get((strongs, lemma, morph, kind), ""))
            for (strongs, lemma, morph, kind), count in merged.items()
        ]
        if self.paradigm if paradigm is None else paradigm:
            attested = {(c.strongs, c.lemma, c.morph) for c in candidates}
            for candidate in list(candidates):
                if not candidate.morph:
                    continue
                for code in paradigm_codes(candidate.morph, self._key, candidate.lemma):
                    if code in self.forms.morph_codes and (candidate.strongs, candidate.lemma, code) not in attested:
                        attested.add((candidate.strongs, candidate.lemma, code))
                        candidates.append(Candidate(candidate.strongs, candidate.lemma, code,
                                                    f"{source}+paradigm", 0.0, candidate.kind, True,
                                                    f"same form as {candidate.morph}"))
        return sorted(candidates, key=lambda c: (-c.weight, c.strongs, c.lemma, c.morph or ""))


# The Hebrew letters heading the acrostics (Ps 118, Lamentations), as Brenton
# spells them in capitals where TBESG spells the letter's entry otherwise
# (ΑΛΕΦ, TBESG's Αλφ; ΡΗΧΣ, Ρης): letters -> TBESG's number for the letter.
# The other letters (ΒΗΘ, ΜΗΜ, ΘΑΥ ...) Brenton spells as TBESG does.
HEBREW_LETTER_SPELLINGS = {
    "αλεφ": "G21477", "γιμελ": "G21479", "δαλεθ": "G21480", "ζαιν": "G21483", "ιωδ": "G21486",
    "λαμεδ": "G21488", "σαμεχ": "G21491", "τσαδη": "G21494", "ρηχσ": "G21496", "χσεν": "G21497",
}


def _without_breathing(key: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", key) if c not in ("\u0313", "\u0314")))


class LexiconProvider:
    """TBESG entries whose lemma is the form itself. See the module docstring."""

    name = "tbesg"

    def __init__(self, lexicon: Lexicon, forms: NTForms, tolerant: bool = False):
        self.lexicon = lexicon
        self.forms = forms
        self.tolerant = tolerant
        if tolerant:
            self.name = "tbesg-tolerant"

    def __call__(self, key: str, token: Token) -> list[Candidate]:
        if token.case == "caps":
            # Letters are all a word in capitals has: every accentuation of
            # them is looked up, as for TAGNT's forms. The tolerant provider
            # has nothing to add.
            if self.tolerant:
                return []
            entries = self.lexicon.by_letters(bare(key))
            if bare(key) in HEBREW_LETTER_SPELLINGS:
                entries = tuple(entries) + tuple(self.lexicon.by_strongs(HEBREW_LETTER_SPELLINGS[bare(key)]))
        elif self.tolerant:
            exact = set(self.lexicon.by_key(key))
            entries = tuple(e for e in self.lexicon.by_letters(bare(key)) if e not in exact)
        else:
            entries = self.lexicon.by_key(key)
            if not entries and bare(key).endswith("θε"):
                # The movable nu of -θε(ν) adverbs (κατόπισθε, κατόπισθεν),
                # as `text.alternates` allows for TAGNT's forms.
                entries = tuple(e for e in self.lexicon.by_key(key + "ν")
                                if e.indeclinable or e.pos_class in INVARIABLE_CLASSES)
            if not entries and token.case != "lower" and other_breathing(key):
                # As for TAGNT's names; TBESG's lemma is the dictionary form.
                entries = tuple(e for e in self.lexicon.by_key(other_breathing(key)) if e.is_name)
            if not entries:
                # TBESG prints some Septuagint lemmas without a breathing
                # (αιλάμ "porch", εξεναντίας "opposite"): a lemma with no
                # breathing at all matches the word with either.
                bare_key = _without_breathing(key)
                entries = tuple(e for e in self.lexicon.by_letters(bare(key))
                                if any(_without_breathing(normalize(v)) == bare_key
                                       and normalize(v) == _without_breathing(normalize(v)) for v in e.variants))
        candidates = []
        for entry in entries:
            morph: str | None
            if entry.is_name:
                kind, morph = "name", None
            elif self.tolerant:
                continue
            elif entry.indeclinable or entry.pos_class in INVARIABLE_CLASSES:
                kind = "common"
                morph = entry.grammar if entry.grammar in self.forms.morph_codes else None
            elif entry.pos_class == "unknown" and any(
                    _without_breathing(normalize(v)) == _without_breathing(key) for v in entry.variants):
                # TBESG gives no part of speech (εἰκοσιοκτώ, κανοῦν, εἰρ): the
                # word as its lemma prints it, tagged without morphology.
                kind, morph = "common", None
            else:
                continue    # a declining word's lemma is one form of many: an analyser's job
            lemma = self._lemma(entry)
            candidates.append(Candidate(
                entry.strongs, lemma, morph, self.name, 1.0, kind,
                taggable=not self.tolerant,
                notes=f"TBESG {entry.extended} {entry.pos or '(no part of speech)'}".strip(),
            ))
        unique = {c.reading: c for c in candidates}
        return sorted(unique.values(), key=lambda c: (c.strongs, c.lemma, c.morph or ""))


    def _lemma(self, entry) -> str:
        """The lemma as the NT spells it, when TAGNT's lemma for the number is
        this entry's word; otherwise TBESG's. (G1 is both Ἀλφα and ἆ.)"""
        nt_lemma = self.forms.lemma_for(entry.strongs)
        if nt_lemma:
            spelt = {normalize(variant) for variant in split_variants(nt_lemma)}
            if any(normalize(variant) in spelt for variant in entry.variants):
                return nt_lemma
        return entry.lemma


class NTTolerantProvider:
    """TAGNT readings of forms with the same letters, accents and breathings
    ignored. A hint only: never tagged."""

    name = "nt-tolerant"

    def __init__(self, forms: NTForms):
        self.forms = forms

    def __call__(self, key: str, token: Token) -> list[Candidate]:
        readings = self.forms.bare_readings(bare(key))
        merged: Counter = Counter()
        for reading, count in readings.items():
            merged[(reading.strongs, reading.lemma, reading.morph, reading_kind(reading.morph))] += count
        return sorted(
            (Candidate(s, l, m, self.name, float(n), k, taggable=False) for (s, l, m, k), n in merged.items()),
            key=lambda c: (-c.weight, c.strongs, c.lemma, c.morph or ""),
        )


# ---------------------------------------------------------------------------
# The aggregator

def candidates_for(token: Token, providers: Sequence[ProviderEntry],
                   profile: CaseProfile | None = None) -> TokenCandidates:
    """Consult `providers` in order for `token`, filter by case, and classify."""
    expected = expectation(token, profile)
    kept: list[Candidate] = []
    aside: list[tuple[Candidate, str]] = []
    # A word in capitals is looked up by its letters alone, so every source
    # that knows those letters is asked: ΝΟΥΝ heading an acrostic is the
    # letter nun as well as νοῦν "mind", and the two must meet to be ambiguous.
    exhaustive = token.case == "caps"
    for entry in providers:
        # The analyser stays a fallback even then: a form it generates from
        # the letters ΕΝ (ἕν "one") is no rival to the reading TAGNT attests
        # 2,829 times (ἐν), and the chooser never learns to weigh the two,
        # since outside capitals the analyser meets no TAGNT reading.
        if entry.mode != "always" and kept and (not exhaustive or entry.name == "analyser"):
            continue
        offered = entry.provider(token.key, token)
        if not offered:
            continue
        accepted, rejected = case_filter(token, offered, expected)
        kept += accepted
        aside += rejected
    if exhaustive and any(c.taggable for c in kept):
        kept = [c for c in kept if c.taggable]
    tier, reason = classify(token, kept, aside, expected)
    return TokenCandidates(token, kept, aside, tier, reason)


class Tagger:
    """Providers, a case profile and a chooser, applied verse by verse."""

    def __init__(self, providers: list[ProviderEntry] | None = None,
                 profile: CaseProfile | None = None, chooser: Chooser | None = None,
                 codes: set[str] | frozenset[str] | None = None):
        self.providers: list[ProviderEntry] = list(providers or [])
        self.profile = profile
        self.chooser: Chooser = chooser or unique_reading_chooser
        # The morphology codes a choice may carry; None allows any.
        self.codes = codes
        self._cache: dict = {}

    def register(self, name: str, provider: Provider, mode: str = "fallback",
                 before: str | None = None, after: str | None = None) -> None:
        """Add a provider, at the end or before/after a named one."""
        if mode not in ("fallback", "always"):
            raise ValueError(f"mode must be 'fallback' or 'always', not {mode!r}")
        entry = ProviderEntry(name, provider, mode)
        names = [existing.name for existing in self.providers]
        if before is not None:
            self.providers.insert(names.index(before), entry)
        elif after is not None:
            self.providers.insert(names.index(after) + 1, entry)
        else:
            self.providers.append(entry)
        self._cache.clear()

    def provider_names(self) -> list[str]:
        return [entry.name for entry in self.providers]

    def candidates_for(self, token: Token) -> TokenCandidates:
        # Everything a provider or the filter reads from a token, so tokens
        # alike in these get the same result without asking again.
        signature = (token.key, token.word_key, token.part, token.parts, token.case, token.position,
                     token.first_marks_known, "apostrophe-breathing" in token.rules)
        cached = self._cache.get(signature)
        if cached is None:
            cached = candidates_for(token, self.providers, self.profile)
            self._cache[signature] = cached
        return TokenCandidates(token, cached.candidates, cached.set_aside, cached.tier, cached.reason)

    def tag_tokens(self, tokens: Sequence[Token]) -> list[tuple[TokenCandidates, Choice | None]]:
        verse = [self.candidates_for(token) for token in tokens]
        choices = self.chooser(verse)
        if len(choices) != len(verse):
            raise ValueError(f"chooser returned {len(choices)} choices for {len(verse)} tokens")
        if self.codes is not None:
            for choice in choices:
                if choice and choice.candidate.morph and choice.candidate.morph not in self.codes:
                    raise ValueError(f"{choice.candidate.morph!r} is not a TAGNT morphology code")
        return list(zip(verse, choices))

    def tag_text(self, text: str) -> list[tuple[TokenCandidates, Choice | None]]:
        return self.tag_tokens(tokenize(text))


def default_providers(forms: NTForms, lexicon: Lexicon, paradigm: bool = True) -> list[ProviderEntry]:
    return [
        ProviderEntry("nt-forms", NTFormProvider(forms, paradigm)),
        ProviderEntry("tbesg", LexiconProvider(lexicon, forms)),
        ProviderEntry("nt-tolerant", NTTolerantProvider(forms)),
        ProviderEntry("tbesg-tolerant", LexiconProvider(lexicon, forms, tolerant=True)),
    ]


def default_tagger(forms: NTForms | None = None, lexicon: Lexicon | None = None,
                   profile: CaseProfile | None = None, chooser: Chooser | None = None,
                   paradigm: bool = True, analyser: Provider | None = None) -> Tagger:
    """The standard providers over the repository's TAGNT and TBESG, and the
    morphological analyser (`analyser.Analyser`) before the tolerant lookups
    when one is given (building one takes a while: see `analyser.py`)."""
    if forms is None:
        from lxx_tagging import ntforms
        forms = ntforms.load()
    if lexicon is None:
        from lxx_tagging import lexicon as lexicon_module
        lexicon = lexicon_module.load()
    tagger = Tagger(default_providers(forms, lexicon, paradigm), profile, chooser, forms.morph_codes)
    if analyser is not None:
        tagger.register("analyser", analyser, before="nt-tolerant")
    return tagger


def annotation_data(candidate: Candidate) -> dict:
    """An annotation's `data`, in TAGNT's shape: strongs, morphology, lemma."""
    data = {"strongs": candidate.strongs}
    if candidate.morph:
        data["morphology"] = candidate.morph
    data["lemma"] = candidate.lemma
    return data
