"""STEPBible's Greek lexicon (TBESG), indexed for the tagger.

TBESG is keyed by extended Strong's numbers and covers the New Testament, the
Septuagint (G6000 and above, 5,326 of its 11,035 rows) and the Apocrypha. Its
row is `G6000 \t G6000 = \t G6000 \t ἀγγέλλω \t angellō \t G:V \t to report \t <definition>`:
the number, the number with any disambiguation letter and its relation to
another entry ("a Spelling of", "the Greek of"), that other entry, the lemma
as printed (which can list variants: `Δαυείδ, Δαυίδ, Δαβίδ`), the
transliteration, the part of speech, the gloss and the definition.

The part-of-speech column reads `language:grammar`: `G:V` a verb, `G:N-F` a
feminine noun, `N:N-M-P` a man's name, `N:N--L` a place, `G:A-NUI` an
indeclinable numeral. 1,263 rows leave it blank.

Strong's numbers are normalised as the shipped modules write them (`G0011` ->
`G11`, `G6202G` -> `G6202`), with `normalize_strongs` from the SWORD
converter, which `stepbible_to_translation_schema` uses too.
"""

from __future__ import annotations

import functools
import io
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sword_to_translation_schema import normalize_strongs  # noqa: E402
from lxx_tagging.text import bare, normalize  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent
DEFAULT_TBESG = REPOSITORY_ROOT / "sources" / "stepbible" / "TBESG.txt"

# The grammar code's head, as a broad class.
POS_CLASSES = {
    "V": "verb", "N": "noun", "A": "adjective", "ADV": "adverb", "PREP": "preposition",
    "CONJ": "conjunction", "PRT": "particle", "COND": "conjunction", "INJ": "interjection",
    "T": "article", "D": "pronoun", "P": "pronoun", "R": "pronoun", "C": "pronoun",
    "X": "pronoun", "Q": "pronoun", "I": "pronoun", "K": "pronoun", "S": "pronoun", "F": "pronoun",
}
# Classes whose words do not inflect, so a lemma is also the only form.
INVARIABLE_CLASSES = frozenset(("adverb", "preposition", "conjunction", "particle", "interjection"))


@dataclass(frozen=True)
class LexEntry:
    """One TBESG row."""

    strongs: str          # normalised: "G11", "G6202"
    extended: str         # as the row numbers it: "G0011", "G6202G"
    lemma: str            # the lemma column as printed, variants and all
    variants: tuple[str, ...]  # each spelling the lemma column gives
    transliteration: str
    pos: str              # the part-of-speech column verbatim: "N:N-M-P", "G:V", ""
    pos_class: str        # "verb", "noun", ..., "proper", "transliteration", "unknown"
    grammar: str          # the code after "G:" or "N:" ("N-M-P", "V", "ADV-I"), first of several
    gender: str | None    # "M", "F", "N" where the code gives one
    name_kind: str | None  # for names: "P" person, "L" place, "T" title, "LG"/"PG" gentilic
    gloss: str
    relation: str         # "", "the Greek of", "a Spelling of", "a Name of", ...
    related: str          # the entry the relation points to: "H0175", "G2501K"

    @property
    def is_name(self) -> bool:
        return self.pos_class == "proper"

    @property
    def indeclinable(self) -> bool:
        """The code says the word does not decline (`-PRI`, `-NUI`, `-LI`), or it
        is a transliterated Hebrew word (`N:N` with a lower-case lemma)."""
        return (bool(re.search(r"(?:^|-)(?:PRI|NUI|LI)\b", self.grammar))
                or self.pos_class == "transliteration")


@dataclass(frozen=True)
class LemmaMatch:
    """The entries a lemma lookup found, and how strict the match was.

    `tolerance` is "exact" (the normalised key matched), or "accents" (only
    the letters did: accents, breathings, diaeresis and subscript ignored).
    """

    entries: tuple[LexEntry, ...]
    tolerance: str

    @property
    def exact(self) -> bool:
        return self.tolerance == "exact"


def split_variants(lemma: str) -> tuple[str, ...]:
    """`Δαυείδ, Δαυίδ, Δαβίδ` -> three spellings; `Ἑλληνιστί=Ἑλληνικός` -> two;
    `α, Ἀλφα` -> two."""
    parts = [part.strip() for part in re.split(r"[,=;/]", lemma)]
    return tuple(dict.fromkeys(part for part in parts if part))


def parse_pos(code: str) -> tuple[str, str, str | None, str | None]:
    """A part-of-speech column as (class, grammar, gender, name kind).

    Of several codes (`G:A / G:ADV`), the first is read.
    """
    first = code.split("/")[0].strip()
    language, _, grammar = first.partition(":")
    grammar = grammar.strip()
    if not grammar:
        return "unknown", "", None, None
    fields = grammar.split("-")
    gender = fields[1] if len(fields) > 1 and fields[1] in ("M", "F", "N") else None
    if language == "N":
        kind = "-".join(fields[2:]) or None
        return "proper", grammar, gender, kind
    head = fields[0].split("+")[0].strip()
    return POS_CLASSES.get(head, "other"), grammar, gender, None


def read_entries(path: Path) -> list[LexEntry]:
    entries = []
    with io.open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 7 or not re.match(r"G\d+$", columns[0].strip()):
                continue
            extended, _, relation = columns[1].partition("=")
            pos_class, grammar, gender, kind = parse_pos(columns[5])
            lemma = columns[3].strip()
            # `N:N` rows with a lower-case lemma (αιλάμ "porch", ἄκαν "thorn")
            # are Hebrew common words transliterated, not names.
            if pos_class == "proper" and not lemma[:1].isupper():
                pos_class = "transliteration"
            entries.append(LexEntry(
                strongs=normalize_strongs(columns[0].strip()),
                extended=extended.strip() or columns[0].strip(),
                lemma=lemma,
                variants=split_variants(lemma),
                transliteration=columns[4].strip(),
                pos=columns[5].strip(),
                pos_class=pos_class,
                grammar=grammar,
                gender=gender,
                name_kind=kind,
                gloss=columns[6].strip(),
                relation=relation.strip(),
                related=columns[2].strip(),
            ))
    return entries


class Lexicon:
    """TBESG indexed by Strong's number, by each variant's normalised key, and
    by each variant's letters alone."""

    def __init__(self, entries: list[LexEntry]):
        self.entries = entries
        self._by_strongs: dict[str, list[LexEntry]] = defaultdict(list)
        self._by_key: dict[str, list[LexEntry]] = defaultdict(list)
        self._by_letters: dict[str, list[LexEntry]] = defaultdict(list)
        for entry in entries:
            self._by_strongs[entry.strongs].append(entry)
            for variant in entry.variants:
                # A letter's own sign (the "ο" of "ο, Ουαυ") is not a word,
                # and would otherwise answer for a bare capital Ο.
                if len(variant) == 1 and variant.isalpha() and variant == bare(variant):
                    continue
                key = normalize(variant)
                if entry not in self._by_key[key]:
                    self._by_key[key].append(entry)
                letters = bare(key)
                if entry not in self._by_letters[letters]:
                    self._by_letters[letters].append(entry)

    def by_strongs(self, strongs: str) -> tuple[LexEntry, ...]:
        """The rows of a Strong's number (normalised or extended: `G0011`, `G11`, `G6202G`)."""
        return tuple(self._by_strongs.get(normalize_strongs(strongs), ()))

    def by_key(self, key: str) -> tuple[LexEntry, ...]:
        """The entries one of whose variants normalises to `key` exactly."""
        return tuple(self._by_key.get(key, ()))

    def by_letters(self, letters: str) -> tuple[LexEntry, ...]:
        """The entries one of whose variants has these letters (see `text.bare`)."""
        return tuple(self._by_letters.get(letters, ()))

    def lookup(self, lemma: str, tolerant: bool = True) -> LemmaMatch | None:
        """The entries whose lemma (any variant) is `lemma`.

        `lemma` may be printed or already a key. An exact key match is
        preferred; failing that, with `tolerant`, a match of letters alone,
        reported as such so a caller can decide whether to trust it.
        """
        key = normalize(lemma)
        found = self._by_key.get(key)
        if found:
            return LemmaMatch(tuple(found), "exact")
        if tolerant:
            found = self._by_letters.get(bare(key))
            if found:
                return LemmaMatch(tuple(found), "accents")
        return None

    def of_class(self, pos_class: str) -> list[LexEntry]:
        """Every entry of a class ("verb", "noun", "adjective", "proper", ...), in file order."""
        return [entry for entry in self.entries if entry.pos_class == pos_class]

    def names(self) -> list[LexEntry]:
        return self.of_class("proper")

    def lemma_keys(self, pos_class: str | None = None) -> dict[str, list[LexEntry]]:
        """Normalised lemma key -> entries, optionally of one class. For an
        analyser matching stems against the lexicon."""
        if pos_class is None:
            return {key: list(found) for key, found in self._by_key.items()}
        return {
            key: [entry for entry in found if entry.pos_class == pos_class]
            for key, found in self._by_key.items()
            if any(entry.pos_class == pos_class for entry in found)
        }

    def classes(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for entry in self.entries:
            counts[entry.pos_class] += 1
        return dict(sorted(counts.items()))


def build(path: Path = DEFAULT_TBESG) -> Lexicon:
    return Lexicon(read_entries(path))


@functools.lru_cache(maxsize=2)
def load(path: Path = DEFAULT_TBESG) -> Lexicon:
    """The repository's TBESG, read once per process."""
    return build(path)
