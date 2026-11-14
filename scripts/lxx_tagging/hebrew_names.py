"""The names of STEPBible's tagged Hebrew Old Testament (TAHOT), verse by verse,
and the Septuagint's verses mapped onto the Hebrew's.

This is the Hebrew side of the name stage (`names.py`). It reads TAHOT
(CC BY 4.0) with `stepbible_to_translation_schema`'s own functions, so a word
is in the verse exactly when `outputs/TAHOT.json` has it (Leningrad, Qere and
restored words; not the words TAHOT reconstructs from the Septuagint) and its
key is the one the shipped Hebrew module gives it: `H0040G` is `H40`, the
disambiguating letter dropped.

WHAT COUNTS AS A NAME

TAHOT's grammar column gives each morpheme a code; the word proper is the
morpheme in braces in the dStrongs column (`H9009/{H2983G}` with `HTd/Ngmsa`:
the article, then a gentilic). A word is a name when that morpheme is

  Np   a proper noun: Npm a man, Npf a woman, Npl a place, Npt a title (a god,
       a month, and the divine names יהוה, אֲדֹנָי, שַׁדַּי)
  Ng   a gentilic (הַכְּנַעֲנִי "the Canaanite", לֵוִי "Levite")

and its number is the gentilic's own when TAHOT tags a gentilic (H3669
Canaanite, not H3667 Canaan): the Greek word that renders it gets the number
of the Hebrew word it renders. Every other word is kept too, marked common,
so the matcher can tell a transliterated common noun (Brenton's capitalised
Θεραφίν) from a name.

A name TAHOT spreads over two words joined by `+` (בֵּית־ + שֶׁמֶשׁ, both
`{H1053G}`) is one name here, with both words' letters.

THE VERSE MAP

Brenton's verses are numbered as the Septuagint numbers them (`lxx`) and
TAHOT's as Lamp's English Bibles do (`eng`; TAHOT's header gives the English
reference with the Hebrew in brackets, and the shipped module keeps the
English). lamp-bible-core's `lamp-module versify lxx eng` maps one onto the
other; the whole Septuagint is mapped in one call and cached under
`.cache/lxx_tagging/names/`, keyed by the references and the binary's hash.
Where it is imperfect (Jeremiah's order, Exodus 35-40, 3 Kingdoms'
rearrangements, the additions to Esther and Daniel, which have no Hebrew) the
name stage searches neighbouring verses itself (`names.py`).
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import re
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sword_to_translation_schema import BOOK_INFO, normalize_strongs  # noqa: E402
from stepbible_to_translation_schema import (  # noqa: E402
    clean_hebrew, hebrew_is_text, osis, parse_header,
)

REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_SOURCES = REPOSITORY_ROOT / "sources" / "stepbible"
TAHOT_FILES = ("TAHOT_Gen-Deu.txt", "TAHOT_Jos-Est.txt", "TAHOT_Job-Sng.txt", "TAHOT_Isa-Mal.txt")
CACHE_DIR = REPOSITORY_ROOT / ".cache" / "lxx_tagging" / "names"

# The kinds of name, by TAHOT's code for the word proper.
NAME_KINDS = {"Npm": "person", "Npf": "person", "Npl": "place", "Npt": "title", "Np": "person"}


@dataclass(frozen=True)
class HebrewWord:
    """One word of a TAHOT verse (or one name spread over joined words)."""

    ref: int            # the English reference, BBCCCVVV, as TAHOT.json numbers it
    index: int          # TAHOT's word number in the verse (the first, for a joined name)
    surface: str        # the word as printed, prefixes and all, without TAHOT's slashes
    stem: str           # the word proper, without prefixes and suffixes
    dstrongs: str       # TAHOT's disambiguated number of the word proper: "H0040G"
    grammar: str        # TAHOT's code for the word proper: "Npm", "Ngmsa", "Ncmsa"
    kind: str           # "person", "place", "title", "gentilic" or "common"
    lemma: str          # the dictionary form from TAHOT's expanded tags
    gloss: str          # TAHOT's name for it: "Abimelech"

    @property
    def key(self) -> str:
        """The key the shipped Hebrew module gives the word: `H0040G` -> `H40`."""
        return normalize_strongs(self.dstrongs)

    @property
    def is_name(self) -> bool:
        return self.kind != "common"


def stem_of(dstrongs_field: str, grammar_field: str) -> tuple[str, str] | None:
    """(the word proper's dStrongs, its grammar code) from TAHOT's two columns:
    `H9009/{H2983G}` with `HTd/Ngmsa` -> (`H2983G`, `Ngmsa`). The language
    letter (H or A) heads the code of the first morpheme only."""
    numbers = dstrongs_field.split("\\")[0].split("/")
    codes = grammar_field.split("/")
    for position, number in enumerate(numbers):
        found = re.search(r"\{(H\d+[A-Za-z]?)\}", number)
        if not found:
            continue
        code = codes[position] if position < len(codes) else ""
        if position == 0 and code[:1] in ("H", "A"):
            code = code[1:]
        return found.group(1), code
    return None


def kind_of(code: str) -> str:
    if code.startswith("Ng"):
        return "gentilic"
    if code.startswith("Np"):
        return NAME_KINDS.get(code[:3], "person")
    return "common"


_EXPANDED = re.compile(r"\{(H\d+[A-Za-z]?)=([^=}]*)=([^}]*)\}")


def _expanded(field: str, dstrongs: str) -> tuple[str, str]:
    """The lemma and gloss of the word proper from TAHOT's expanded tags,
    `{H0040G=אֲבִימֶ֫לֶךְ=Abimelech»Abimelech@Gen.20.2-}`."""
    for number, lemma, gloss in _EXPANDED.findall(field):
        if number == dstrongs:
            gloss = gloss.split("»")[0].strip().lstrip(":").strip()
            return lemma.strip(), gloss
    return "", ""


def read_words(paths: Iterable[Path]) -> Iterable[HebrewWord]:
    """Every word of TAHOT's reading text, in order (see the module docstring)."""
    for path in paths:
        pending: HebrewWord | None = None
        with io.open(path, encoding="utf-8-sig") as handle:
            for line in handle:
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 6:
                    continue
                header = parse_header(columns[0])
                if not header or not hebrew_is_text(header[4]):
                    continue
                code, chapter, verse, number, _ = header
                text, _ = clean_hebrew(columns[1])
                if not text:
                    continue
                found = stem_of(columns[4], columns[5])
                if not found:
                    continue
                dstrongs, grammar = found
                parts = columns[1].split("\\")[0].split("/")
                numbers = columns[4].split("\\")[0].split("/")
                stem = next((clean_hebrew(part)[0] for part, n in zip(parts, numbers) if "{" in n), text)
                lemma, gloss = _expanded(columns[11] if len(columns) > 11 else "", dstrongs)
                ref = BOOK_INFO[osis(code)][1] * 1_000_000 + chapter * 1_000 + verse
                word = HebrewWord(ref, number, text, stem, dstrongs, grammar, kind_of(grammar),
                                  lemma or stem, gloss)
                joined = "{" + dstrongs + "}+" in columns[4]
                if pending is not None:
                    if pending.ref == ref and pending.dstrongs == dstrongs:
                        word = HebrewWord(ref, pending.index, f"{pending.surface} {text}",
                                          f"{pending.stem} {stem}", dstrongs, pending.grammar, pending.kind,
                                          pending.lemma, pending.gloss)
                    else:
                        yield pending
                    pending = None
                if joined:
                    pending = word
                    continue
                yield word
        if pending is not None:
            yield pending


def tahot_paths(sources: Path = DEFAULT_SOURCES) -> list[Path]:
    paths = [sources / name for name in TAHOT_FILES]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"TAHOT files not found: {', '.join(missing)}")
    return paths


class HebrewVerses:
    """TAHOT's words by English verse reference."""

    def __init__(self, words: Iterable[HebrewWord]):
        self._verses: dict[int, list[HebrewWord]] = defaultdict(list)
        for word in words:
            self._verses[word.ref].append(word)
        self._verses = dict(self._verses)
        self._last: dict[int, int] = defaultdict(int)
        for ref in self._verses:
            chapter = ref // 1000
            self._last[chapter] = max(self._last[chapter], ref % 1000)

    def words(self, ref: int) -> list[HebrewWord]:
        return list(self._verses.get(ref, ()))

    def names(self, ref: int) -> list[HebrewWord]:
        return [word for word in self._verses.get(ref, ()) if word.is_name]

    def __contains__(self, ref: int) -> bool:
        return ref in self._verses

    def last_verse(self, chapter: int) -> int:
        """The last verse number of a chapter (BBCCC), 0 if TAHOT lacks it."""
        return self._last.get(chapter, 0)

    def refs(self) -> list[int]:
        return sorted(self._verses)


def build(sources: Path = DEFAULT_SOURCES) -> HebrewVerses:
    return HebrewVerses(read_words(tahot_paths(sources)))


# ---------------------------------------------------------------------------
# TBESH

@dataclass(frozen=True)
class HebrewEntry:
    """One TBESH row."""

    dstrongs: str       # "H0040G"
    lemma: str          # pointed, first of several spellings
    transliteration: str
    pos: str            # "N:N-M-P", "H:N-M"
    gloss: str
    relation: str       # "", "a Name of", "a Spelling of"
    related: str

    @property
    def key(self) -> str:
        return normalize_strongs(self.dstrongs)


def read_tbesh(path: Path) -> dict[str, HebrewEntry]:
    """TBESH by disambiguated number (`H0040G`), and by plain number (`H0040`)
    for its first row."""
    entries: dict[str, HebrewEntry] = {}
    with io.open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 7 or not re.match(r"H\d+[A-Za-z]?$", columns[0].strip()):
                continue
            extended, _, relation = columns[1].partition("=")
            extended = extended.strip() or columns[0].strip()
            entry = HebrewEntry(extended, columns[3].split(",")[0].strip(), columns[4].strip(),
                                columns[5].strip(), columns[6].strip(), relation.strip(), columns[2].strip())
            entries.setdefault(extended, entry)
            entries.setdefault(columns[0].strip(), entry)
    return entries


# ---------------------------------------------------------------------------
# Hebrew sounds, for comparing a Greek spelling with a Hebrew one

SHEVA, HATAF_SEGOL, HATAF_PATAH, HATAF_QAMATS = "\u05b0", "\u05b1", "\u05b2", "\u05b3"
HIRIQ, TSERE, SEGOL, PATAH, QAMATS, HOLAM, HOLAM_VAV, QUBUTS = (
    "\u05b4", "\u05b5", "\u05b6", "\u05b7", "\u05b8", "\u05b9", "\u05ba", "\u05bb")
DAGESH, SHIN_DOT, SIN_DOT, QAMATS_QATAN = "\u05bc", "\u05c1", "\u05c2", "\u05c7"
VOWEL_POINTS = {
    SHEVA: "ə", HATAF_SEGOL: "e", HATAF_PATAH: "a", HATAF_QAMATS: "o", HIRIQ: "i", TSERE: "e", SEGOL: "e",
    PATAH: "a", QAMATS: "a", HOLAM: "o", HOLAM_VAV: "o", QUBUTS: "u", QAMATS_QATAN: "o",
}
CONSONANTS = {
    "א": "ʔ", "ב": "b", "ג": "g", "ד": "d", "ה": "h", "ו": "w", "ז": "z", "ח": "ḥ", "ט": "ṭ", "י": "y",
    "כ": "k", "ך": "k", "ל": "l", "מ": "m", "ם": "m", "נ": "n", "ן": "n", "ס": "s", "ע": "ʕ", "פ": "p",
    "ף": "p", "צ": "ṣ", "ץ": "ṣ", "ק": "q", "ר": "r", "ש": "š", "ת": "t",
}
HEBREW_VOWELS = frozenset("aeiouə")
# Which vowel a letter pointed with two is read with (only the first is
# read). A letter's marks are a set, whose order changes from run to run
# with Python's string hashing, so the choice is made by this fixed rank:
# a full vowel before a sheva, and a hiriq last, since beside another vowel
# it is the vowel of an unwritten yod (Jerusalem's Qere יְרוּשָׁלִַם, -layim:
# the lamed's own vowel is the patah).
VOWEL_RANK = "aeouiə"


def _clusters(word: str) -> list[tuple[str, set[str]]]:
    """(letter, its points) for each Hebrew letter; accents and other marks dropped."""
    clusters: list[tuple[str, set[str]]] = []
    for char in unicodedata.normalize("NFD", word):
        if char in CONSONANTS:
            clusters.append((char, set()))
        elif clusters and (char in VOWEL_POINTS or char in (DAGESH, SHIN_DOT, SIN_DOT)):
            clusters[-1][1].add(char)
    return clusters


@functools.lru_cache(maxsize=1 << 16)
def sounds(word: str) -> tuple[str, ...]:
    """A pointed Hebrew word as a sequence of sounds: consonants (ʔ b g d h w z
    ḥ ṭ y k l m n s ʕ p ṣ q r š ś t) and vowels (a e i o u, ə for sheva).

    Vowel letters are read as vowels, not consonants: וֹ after a letter without
    a vowel is o, וּ there is u; י without a vowel after i or e lengthens it;
    a final ה without a vowel or mappiq is silent. A furtive patah (נֹחַ) is
    read before its letter. A final sheva is silent. Spaces and maqaf, which
    join the words of one name, are ignored.
    """
    clusters = _clusters(word)
    out: list[str] = []
    for position, (letter, marks) in enumerate(clusters):
        vowels = sorted((VOWEL_POINTS[m] for m in marks if m in VOWEL_POINTS), key=VOWEL_RANK.index)
        previous_vowel = out[-1] if out and out[-1] in HEBREW_VOWELS else None
        last = position == len(clusters) - 1
        if letter == "ו" and not [v for v in vowels if v != "o"]:
            if "o" in vowels and previous_vowel is None and out:
                out.append("o")
                continue
            if DAGESH in marks and not vowels and (previous_vowel is None or not out):
                out.append("u")
                continue
        if letter == "י" and not vowels and previous_vowel in ("i", "e"):
            continue
        if letter == "ה" and last and not vowels and DAGESH not in marks:
            continue
        consonant = CONSONANTS[letter]
        if letter == "ש" and SIN_DOT in marks:
            consonant = "ś"
        if last and vowels == ["a"] and letter in "חעה" and previous_vowel is not None:
            out.extend(("a", consonant))      # furtive patah
            continue
        out.append(consonant)
        for vowel in vowels[:1]:
            if vowel == "ə" and last:
                continue
            out.append(vowel)
    return tuple(out)


def spellings(lemma: str) -> list[str]:
    """The spellings a lemma lists (`נְבוּכַדְנֶאצַּר, נְבוּכַדְרֶאצַּר`)."""
    return [part.strip() for part in lemma.split(",") if part.strip()] or [lemma]


def parts(lemma: str) -> list[str]:
    """The words of a name of several (קָדֵשׁ בַּרְנֵעַ, בֵּית־שֶׁמֶשׁ), each of
    which the Septuagint may print as a word of its own (Κάδης Βαρνή); empty
    for a name of one word."""
    words = [w for w in re.split(r"[\s\u05be]+", spellings(lemma)[0]) if w]
    return words if len(words) > 1 else []


@functools.lru_cache(maxsize=1 << 16)
def sound_variants(word: str, gentilic: bool = False) -> tuple[tuple[tuple[str, ...], str], ...]:
    """The Hebrew name's sounds, for each spelling its lemma lists, and the
    shorter forms the Septuagint's renderings follow, each named: the
    theophoric -יָהוּ read as -יָה (עֲזַרְיָהוּ, Ἀζαρίας), and a gentilic
    without its -î (כְּנַעֲנִי, Χανανα-ῖος)."""
    found: list[tuple[tuple[str, ...], str]] = []
    for spelling in spellings(word):
        full = sounds(spelling)
        found.append((full, ""))
        if full[-4:] == ("y", "a", "h", "u"):
            found.append((full[:-2], "yahu"))
        if gentilic and len(full) > 2 and full[-1] == "i":
            found.append((full[:-1], "gentilic"))
        if gentilic and len(full) > 3 and full[-2:] == ("i", "m"):
            found.append((full[:-2], "gentilic"))
    return tuple(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# The verse map

DEFAULT_LAMP_MODULE = REPOSITORY_ROOT.parent / "lamp-bible-core" / ".build" / "release" / "lamp-module"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class VerseMap:
    """Septuagint reference -> the English reference it corresponds to, or None."""

    def __init__(self, mapping: dict[int, int | None]):
        self.mapping = dict(mapping)

    def __getitem__(self, ref: int) -> int | None:
        return self.mapping.get(ref)

    def to_json(self) -> dict:
        return {str(ref): target for ref, target in sorted(self.mapping.items())}

    @classmethod
    def from_json(cls, data: dict) -> "VerseMap":
        return cls({int(ref): (int(target) if target is not None else None) for ref, target in data.items()})

    @classmethod
    def identity(cls, refs: Iterable[int]) -> "VerseMap":
        """Every verse onto the verse with the same number: for tests and for
        text already in the English numbering."""
        return cls({ref: ref for ref in refs})


def versify(refs: Sequence[int], source: str, target: str, binary: Path = DEFAULT_LAMP_MODULE) -> dict:
    """Each BBCCCVVV reference in `source`'s numbering as `target` numbers it
    (None where it has none), from lamp-bible-core's built `lamp-module`. The
    protocol is `core_module_builder.versify`'s; the binary is run as built,
    never rebuilt. Where the target numbers two verses for one ("a,b", core
    8b505b1 onwards), the first is taken, as core's `convert` did."""
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found: build lamp-bible-core's lamp-module, or pass --verse-map")
    refs = sorted(set(refs))
    result = subprocess.run([str(binary), "versify", source, target],
                            input="".join(f"{ref}\n" for ref in refs), capture_output=True, text=True, check=True)
    lines = result.stdout.splitlines()
    if len(lines) != len(refs):
        raise RuntimeError(f"lamp-module versify answered {len(lines)} of {len(refs)} references")
    mapped: dict[int, int | None] = {}
    for ref, line in zip(refs, lines):
        verse = line.partition(" ")[0].split(",")[0]
        mapped[ref] = None if verse in ("-", "") or ref % 1_000 == 0 else int(verse)
    return mapped


def versify_map(refs: Sequence[int], cache_dir: Path | None = CACHE_DIR,
                binary: Path = DEFAULT_LAMP_MODULE) -> VerseMap:
    """The Septuagint's verses mapped onto the English numbering with
    `lamp-module versify lxx eng`, in one call, cached (see the module docstring)."""
    refs = sorted(set(refs))
    digest = hashlib.sha256()
    digest.update(_file_digest(binary).encode("ascii") if binary.exists() else b"missing")
    digest.update(",".join(map(str, refs)).encode("ascii"))
    key = digest.hexdigest()[:20]
    path = cache_dir / f"versemap-{key}.json" if cache_dir is not None else None
    if path is not None and path.exists():
        return VerseMap.from_json(json.loads(path.read_text(encoding="utf-8")))
    found = VerseMap(versify(refs, "lxx", "eng", binary))
    if path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(found.to_json(), sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    return found
