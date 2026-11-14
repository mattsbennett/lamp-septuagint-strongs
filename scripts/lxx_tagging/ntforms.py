"""The form table of STEPBible's tagged Greek New Testament (TAGNT).

Every row of the raw TAGNT files is evidence for what a Greek form can be,
including words read only by the Textus Receptus, the Byzantine text or other
editions, which the shipped NT module leaves out of its running text. Each row
gives a form and one reading, (Strong's, lemma, morphology); the table counts
readings per normalised form.

The row is read with `stepbible_to_translation_schema`'s own functions, so a
reading carries exactly the key, lemma and code `outputs/TAGNT.json` would give
the same word: `G0976` becomes `G976`, `G2424G` becomes `G2424`, the lemma is
the dictionary-form column before its `=gloss`, and the morphology is TAGNT's
code verbatim. A word TAGNT tags as two (κἀγώ is `G1473=P-1NS + G2532=CONJ`)
keeps the shipped shape: the first key, and the rest of the field as its code.
Its parts are also kept separately (`parts_of`), which is how a joined word
Brenton prints as one (ἱνατί) gets a reading for each part.
"""

from __future__ import annotations

import functools
import io
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stepbible_to_translation_schema import (  # noqa: E402
    clean_greek, lemma_from, parse_header, split_strongs_grammar, strongs_keys,
)
from lxx_tagging.text import bare, caps_pattern, normalize_with_rules  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_SOURCES = REPOSITORY_ROOT / "sources" / "stepbible"
TAGNT_FILES = ("TAGNT_Mat-Jhn.txt", "TAGNT_Act-Rev.txt")

# TAGNT's surface column can carry the verse's punctuation and its own marks:
# ¶ for a paragraph, ¬, and [[ ]] around the long ending of Mark and the
# adulteress passage.
_TRIM = ".,;·:!?¶¬[]();· "

# Suffixes of noun and adjective codes that mark proper names and titles.
NAME_SUFFIXES = frozenset(("P", "L", "PG", "LG"))
TITLE_SUFFIXES = frozenset(("T",))


@dataclass(frozen=True, order=True)
class Reading:
    """One way TAGNT tags a form: what an annotation's `data` would carry."""

    strongs: str
    lemma: str
    morph: str


def reading_kind(morph: str | None) -> str:
    """"name", "title" or "common", from the code's suffix: TAGNT marks a
    person, place or gentilic -P, -L, -PG, -LG and a title (θεός, κύριος,
    Χριστός used as titles) -T."""
    if not morph:
        return "common"
    head = morph.split(" + ")[0]
    fields = head.split("-")
    if fields[0] in ("N", "A") and len(fields) > 2:
        if fields[2] in NAME_SUFFIXES:
            return "name"
        if fields[2] in TITLE_SUFFIXES:
            return "title"
    return "common"


def case_of(morph: str) -> str | None:
    """The case letter of a noun or adjective code (`N-GSM-P` -> `G`), else None."""
    fields = morph.split(" + ")[0].split("-")
    if fields[0] in ("N", "A") and len(fields) > 1 and re.fullmatch(r"[NGDAV][SP][MFN]?", fields[1]):
        return fields[1][0]
    return None


@dataclass(frozen=True)
class Row:
    """One TAGNT word row, parsed."""

    reference: str
    word_type: str
    surface: str
    reading: Reading
    # A compound row's parts, each a Reading; empty for an ordinary row.
    parts: tuple[Reading, ...] = ()


def surface_of(column: str) -> str:
    """`Ἀβραάμ. (Abraam)` -> `Ἀβραάμ`: NFC, transliteration and punctuation removed."""
    return unicodedata.normalize("NFC", clean_greek(unicodedata.normalize("NFC", column))).strip(_TRIM)


def _compound_parts(strongs_field: str, lemma_field: str) -> tuple[Reading, ...]:
    """`G2443=CONJ + G5101=I-NSN` with `ἵνα=in order that + τίς=which?` ->
    the two parts' readings. Empty if the two columns do not line up."""
    tags = [tag.strip() for tag in strongs_field.split(" + ")]
    lemmas = [lemma.strip() for lemma in lemma_field.split(" + ")]
    if len(tags) < 2 or len(tags) != len(lemmas):
        return ()
    parts = []
    for tag, lemma in zip(tags, lemmas):
        keys, code = split_strongs_grammar(tag)
        normalised = strongs_keys(keys)
        word = lemma_from("", lemma)
        if not normalised or not code or not word:
            return ()
        parts.append(Reading(",".join(normalised), word, code.strip()))
    return tuple(parts)


def read_rows(path: Path) -> Iterator[Row]:
    """Every word row of one TAGNT file, whatever edition reads it."""
    with io.open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 6:
                continue
            header = parse_header(columns[0])
            if not header:
                continue
            surface = surface_of(columns[1])
            keys, code = split_strongs_grammar(columns[3])
            normalised = strongs_keys(keys)
            lemma = lemma_from("", columns[4])
            if not surface or not normalised or not code.strip() or not lemma:
                continue
            reading = Reading(",".join(normalised), lemma, code.strip())
            yield Row(columns[0].strip(), header[4], surface, reading,
                      _compound_parts(columns[3], columns[4]))


class NTForms:
    """Normalised form -> counted readings, with the indexes lookups need.

    `readings(key)` is the exact table. `caps_readings(pattern)` serves words
    printed in capitals (see `text.caps_pattern`), and `bare_readings(letters)`
    the accent- and breathing-insensitive lookup, which is only ever a hint.
    """

    def __init__(self, rows: Iterable[Row], skip_rules: frozenset[str] = frozenset()):
        self.skip_rules = skip_rules
        self._readings: dict[str, Counter[Reading]] = defaultdict(Counter)
        self._surfaces: dict[str, Counter[str]] = defaultdict(Counter)
        self._parts: dict[str, Counter[tuple[Reading, ...]]] = defaultdict(Counter)
        self._lemmas: dict[str, Counter[str]] = defaultdict(Counter)
        self.morph_codes: set[str] = set()
        self.rows = 0
        for row in rows:
            key = self.key(row.surface)
            self._readings[key][row.reading] += 1
            self._surfaces[key][row.surface] += 1
            self.morph_codes.add(row.reading.morph)
            if row.parts:
                self._parts[key][row.parts] += 1
                for part in row.parts:
                    self.morph_codes.add(part.morph)
            else:
                self._lemmas[row.reading.strongs][row.reading.lemma] += 1
            self.rows += 1
        self._caps: dict[str, Counter[Reading]] = defaultdict(Counter)
        self._bare: dict[str, Counter[Reading]] = defaultdict(Counter)
        for key, readings in self._readings.items():
            for pattern in {caps_pattern(key, True), caps_pattern(key, False)}:
                self._caps[pattern].update(readings)
            self._bare[bare(key)].update(readings)

    def key(self, word: str) -> str:
        return normalize_with_rules(word, self.skip_rules)[0]

    def __contains__(self, key: str) -> bool:
        return key in self._readings

    def __len__(self) -> int:
        return len(self._readings)

    def keys(self) -> Iterable[str]:
        return self._readings.keys()

    def readings(self, key: str) -> Counter[Reading]:
        """The readings of a normalised form, counted. Empty if unseen."""
        return Counter(self._readings.get(key, {}))

    def caps_readings(self, pattern: str) -> Counter[Reading]:
        """Readings of every form a word printed in capitals could be."""
        return Counter(self._caps.get(pattern, {}))

    def bare_readings(self, letters: str) -> Counter[Reading]:
        """Readings of every form with these letters, whatever its accents and breathings."""
        return Counter(self._bare.get(letters, {}))

    def forms_with_letters(self, letters: str) -> list[str]:
        """The keys whose letters (see `text.bare`) are these, sorted."""
        if not hasattr(self, "_keys_by_letters"):
            index: dict[str, list[str]] = defaultdict(list)
            for key in sorted(self._readings):
                index[bare(key)].append(key)
            self._keys_by_letters = dict(index)
        return list(self._keys_by_letters.get(letters, ()))

    def surfaces(self, key: str) -> Counter[str]:
        """How TAGNT prints the forms behind a key."""
        return Counter(self._surfaces.get(key, {}))

    def parts_of(self, key: str) -> Counter[tuple[Reading, ...]]:
        """For a form TAGNT tags as several words with a lemma for each
        (ἱνατί, οὐκέτι), each part's reading. κἀγώ has one lemma, so none."""
        return Counter(self._parts.get(key, {}))

    def keys_of(self, strongs: str) -> list[str]:
        """The keys of every form TAGNT tags with this Strong's number."""
        if not hasattr(self, "_keys_by_strongs"):
            index: dict[str, set[str]] = defaultdict(set)
            for key, readings in self._readings.items():
                for reading in readings:
                    index[reading.strongs].add(key)
            self._keys_by_strongs = {strongs: sorted(keys) for strongs, keys in index.items()}
        return list(self._keys_by_strongs.get(strongs, ()))

    def case_shown_by_form(self, strongs: str, strict: bool = True) -> bool:
        """Whether this word's case can be read off its form.

        A Hebrew name or loanword (Ἀβραάμ, Φαραώ, μάννα) does not decline:
        TAGNT's case for it comes from the sentence, so a form seen once as a
        genitive says nothing about the next occurrence. A word whose forms
        all have the same letters (Ἀβραάμ and Tyndale's Ἁβραάμ are one
        spelling) does not decline. With `strict`, for names, a word also
        counts as undeclined if TAGNT tags any one of its forms with two
        different cases (Δαυίδ, in three spellings, is genitive, dative and
        accusative alike); a title like θεός is not held to that, because a
        declining noun's forms can serve two cases (θεός for the vocative).
        """
        keys = self.keys_of(strongs)
        if len({bare(key) for key in keys}) < 2:
            return False
        if not strict:
            return True
        for key in keys:
            cases = {case_of(reading.morph) for reading in self._readings[key] if reading.strongs == strongs}
            if len(cases) > 1:
                return False
        return True

    def lemma_for(self, strongs: str) -> str | None:
        """The lemma TAGNT most often gives a Strong's number, so a reading from
        another source spells its lemma as the NT does. None if TAGNT lacks it."""
        lemmas = self._lemmas.get(strongs)
        if not lemmas:
            return None
        return sorted(lemmas.items(), key=lambda item: (-item[1], item[0]))[0][0]


def tagnt_paths(sources: Path = DEFAULT_SOURCES) -> list[Path]:
    paths = [sources / name for name in TAGNT_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TAGNT files not found: {', '.join(missing)}")
    return paths


def build(paths: Iterable[Path], skip_rules: frozenset[str] = frozenset()) -> NTForms:
    """The form table of the given TAGNT files."""
    return NTForms((row for path in paths for row in read_rows(path)), skip_rules)


@functools.lru_cache(maxsize=2)
def load(sources: Path = DEFAULT_SOURCES) -> NTForms:
    """The form table of the repository's TAGNT files, built once per process."""
    return build(tagnt_paths(sources))


def is_tagnt_code(code: str, forms: NTForms) -> bool:
    """Whether a morphology code is one TAGNT itself uses, so the shipped NT
    module carries it too and core's decoder treats the two alike. (Core reads
    every such code but the compound `CONJ + G1437=COND` of κἄν, κἀκεῖ and
    κἀκεῖθεν, which it returns empty for in the NT as well.)"""
    return code in forms.morph_codes
