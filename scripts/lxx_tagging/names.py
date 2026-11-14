"""Proper names the New Testament never uses, identified by the Hebrew name
they render.

WHAT IT DOES

After the chooser, Brenton's tier-d words (capitalised, no acceptable Greek
reading: Μωάβ, Ἰωάβ, Γαλαάδ, Ἀβιμέλεχ) are compared with the names in the
Hebrew of the same verse, from STEPBible's tagged Hebrew Old Testament
(`hebrew_names.py`), by how the Septuagint transliterates Hebrew
(`translit.py`). A Greek name that matches one Hebrew name clearly is tagged
with that name's Hebrew Strong's number, in the key convention of the
shipped Hebrew module (`H40` for Ἀβιμέλεχ, TAHOT's `H0040G`). STEPBible's
Greek lexicon has no numbers for these names; this is the owner's narrow,
recorded exception to the rule against labelling one text from another, and
it is limited to proper names and gentilics.

Names the New Testament does use keep the Greek numbers TAGNT and TBESG give
them. A tier-d word that is one of those names in a form the NT lacks
(Λευῖται, Δαμασκοῦ, Βαβυλῶνα) takes the NT's Greek number, not the Hebrew
one, when TBESG says that Greek entry renders the Hebrew name matched and
the two spellings have the same stem (`tbesg_name`).

CANDIDATES

  * tier d: looks like a name, no acceptable candidate;
  * tier e, capitalised: no candidate at all, and so never name-like to the
    case filter, a name at the start of a sentence or after a comma (Gen
    2:11 Φισῶν), under the same thresholds; not one whose ending is a
    verb's (`VERB_ENDINGS`: Ἐνεπάγησαν is an unread verb);
  * capitalised words the chooser tagged from a tolerant match with a name
    (Brenton's Ἠλιού matched the NT's Ἡλίας): see `tag_names` for when the
    Hebrew replaces such a tag.

Tagged names with a TBESG relation to a Hebrew number are anchors: each
takes one occurrence of the Hebrew name it renders, which is not offered to
another Greek word of the verse (`free`). The divine names the Septuagint
translates (יהוה, אֲדֹנָי) are never offered (`TRANSLATED_NAMES`).

MATCHING (`match_verse`)

Each distinct Greek form of the verse (spellings that share a stem with a
case ending, Βαναίας and Βαναίου, as one) is scored against each Hebrew
name of the verse the Septuagint's verse maps to (`hebrew_names.VerseMap`),
and against each word of a name of several (Κάδης Βαρνή for קָדֵשׁ בַּרְנֵעַ);
the pairs are assigned one to one, maximising the total score
(Kuhn-Munkres), so a list of names lines up as a whole. A pair is accepted
when its score is at least `Settings.threshold` and its margin (how much the
best assignment loses if this pair is forbidden, the others free to change)
at least `Settings.margin`, and no common Hebrew word of the verse matches the
Greek form better (a transliterated common noun is not a name). Then:

  * a list's order (`by_order`): a Greek name left alone between two matched
    names, with one Hebrew name left between theirs, renders it if it scores
    at least `Settings.order_floor`;
  * neighbouring Hebrew verses (`Settings.window` either side,
    `Settings.offset_cost` per verse of distance), then the chapter and the
    chapters either side (`chapter`), at stricter thresholds, for the books
    whose order the verse map does not follow (Exodus 35-40, Numbers 26,
    3 Kingdoms, Jeremiah's oracles, Daniel's additions). Not for a form one
    of the verse's own unmatched names could render (score >= 0), nor one
    that may be a reading of a common Hebrew word of the verse.

THE CONSISTENCY PASS (`consistency`)

A Greek form whose verse-level matches agree on one Hebrew number
(`Settings.consistency_share` of them, at least `Settings.consistency_min`
confident) takes that number where it is left untagged, in verses whose
Hebrew lacks it (Septuagint pluses, verse-map gaps), unless the verse has a
name or common word it may render. A form that ever confidently matched
another number is refused. Its confidence is lower.

METHODS (the decisions file's `method`)

  hebrew-verse        matched in the Hebrew verse the map gives
  hebrew-order        placed by a list's order in that verse
  hebrew-nearby       matched in a neighbouring verse, or the chapter
  hebrew-consistency  the number the form takes everywhere else
  hebrew-association  the Hebrew name the verses of an untagged Greek form
                      nearly always have unaccounted for, for names no
                      transliteration finds (Αἰθιοπία for Cush): see
                      `associate`

TRANSLITERATES OR STANDS FOR (`name_relations`, the decisions file's
`name_relation`)

A tagged name either transliterates the Hebrew name it is tagged with, or
stands for it: the Septuagint prints another name at that position (2 Chr
24:20 Ἀζαρίαν for Zechariah). The tag is the same either way; the column
says which, by the rule `name_relations` gives.

The confidence of each comes from the validation on TBESG's own name
relations (`evaluate_gold`, `calibration`), run with every tagging run.

LEMMAS AND GRAMMAR

A name's lemma is its dictionary form: the nominative, where the Greek
clearly declines it (`lemma_groups`), otherwise the form as printed; one
lemma for every occurrence of the name. The case, number and gender the form and its article or
preposition show are written to the decisions file (`name_grammar`, a
TAGNT-shaped code with ? for what is unknown). An H-keyed word carries no
morphology code: the app reads a key's first letter as its language, and a
Greek code on an H key would be decoded as Hebrew.

KEY MODES (`--name-keys`)

  hebrew          (default) the Hebrew number, no morphology
  greek-extended  STEPBible's convention for a Greek word that renders a
                  Hebrew name the lexicon lacks, the Hebrew number + 10000
                  as a Greek key (H40 -> G10040), with the morphology code
                  when the case, number and gender are known and TAGNT uses
                  the code
  off             tier d stays untagged
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from sword_to_translation_schema import normalize_strongs  # noqa: E402
from lxx_tagging import hebrew_names, translit  # noqa: E402
from lxx_tagging.candidates import Candidate, Choice  # noqa: E402
from lxx_tagging.hebrew_names import HebrewVerses, HebrewWord, VerseMap, sound_variants  # noqa: E402
from lxx_tagging.lexicon import Lexicon  # noqa: E402
from lxx_tagging.text import Token, bare, normalize, normalize_with_rules  # noqa: E402

METHODS = ("hebrew-verse", "hebrew-order", "hebrew-nearby", "hebrew-consistency", "hebrew-association")
KEY_MODES = ("hebrew", "greek-extended", "off")
SOURCE = "tahot-name"
# TBESG relations that say a Greek name renders a Hebrew one.
RELATIONS = ("the Greek of", "a Spelling of", "a Name of", "a Group member of")
# Prepositions that take one case only.
PREPOSITION_CASES = {
    "ἐν": "D", "σύν": "D", "εἰς": "A", "ἐκ": "G", "ἐξ": "G", "ἀπό": "G", "ἀπ’": "G", "ἐφ’": None,
    "ἀντί": "G", "πρό": "G", "ἄνευ": "G", "ἕως": "G", "ἐνώπιον": "G", "ἔναντι": "G", "ἐναντίον": "G",
}


@dataclass(frozen=True)
class Settings:
    threshold: float = 3.0        # least score of a verse-level match (nats)
    margin: float = 3.0           # least assignment margin
    confident: float = 6.0        # a match this sure (and `margin`) is confident: it teaches the
                                  # model and decides the consistency pass
    order_floor: float = 0.0      # least score of a match a list's order places
    nearby_threshold: float = 9.0
    nearby_margin: float = 4.0
    window: int = 2               # neighbouring Hebrew verses searched either side
    offset_cost: float = 1.0      # nats per verse of distance
    chapter_threshold: float = 12.0  # the same chapter (and the chapters either side), last
    chapter_margin: float = 6.0
    consistency_min: int = 2      # confident matches a form needs before the consistency pass
    consistency_share: float = 0.9  # of its verse-level matches that must agree
    common_margin: float = 0.0    # a common word scoring this much more than the name blocks it
    iterations: int = 8           # EM iterations per fit
    rounds: int = 2               # refits on the stage's own confident matches


# Endings of Greek verb forms no Hebrew name in Greek letters ends in: a
# capitalised tier-e word with one is an unread verb (Ἐνεπάγησαν, Ἐδέετο,
# Ἴδοισαν, Ψηλαφήσαισαν), not a name, and is not offered to the matcher.
VERB_ENDINGS = ("σαν", "θη", "θην", "ετο", "οντο", "ουσι", "ουσιν", "ασι", "ασιν", "σθαι", "μεθα", "σατο",
                "κασι", "οιεν", "οιην", "τω", "τωσαν", "ναι", "σθε", "ειν", "ηκα", "ηκε", "ησε", "ησεν", "ξε",
                "ψε", "θησαν", "σθη", "ομεν", "ετε", "ουσα", "οντεσ", "οτεσ")


def verb_like(key: str) -> bool:
    letters = bare(key).replace("ς", "σ")
    return letters.endswith(VERB_ENDINGS)


# ---------------------------------------------------------------------------
# The stage's view of a verse

@dataclass
class TokenView:
    """What the stage reads of one token: the token, its tier and reason,
    whether a common reading was set aside, and what the chooser chose."""

    token: Token
    tier: str
    reason: str = ""
    set_aside_common: bool = False
    strongs: str = ""
    lemma: str = ""
    morph: str = ""
    method: str = ""


@dataclass
class VerseView:
    ref: int                      # Septuagint reference, BBCCCVVV
    tokens: list[TokenView]


@dataclass
class NameDecision:
    """One token's name, and how it was found."""

    strongs: str
    lemma: str
    morph: str | None
    method: str
    score: float
    margin: float
    confidence: float
    hebrew: HebrewWord
    grammar: str = ""
    replaces: str = ""            # a tolerant tag this replaces
    key_source: str = "tahot"     # "tahot", or "tbesg" for an NT name's Greek number
    runner_up: str = ""


@dataclass
class Match:
    hebrew: HebrewWord
    score: float
    margin: float
    method: str
    runner_up: HebrewWord | None = None
    spelling: str = ""            # the word of a Hebrew name of several the Greek matched


# ---------------------------------------------------------------------------
# Greek forms

def letters_of(key: str) -> str:
    return translit.letters_of(key)


def display_form(surface: str) -> str:
    """A form as a lemma prints it: the key's normalisations, with the
    capital and final sigma kept (Ἀβιμέλεχ, Μαρδοχαῖος)."""
    return normalize_with_rules(surface, frozenset(("case", "final-sigma")))[0]


# Endings by what they say: (case, number) for each Greek case ending and
# the paradigm's nominative singular. Only endings whose case is certain are
# listed; a bare -α can be several.
PARADIGMS: dict[str, tuple[str, dict[str, tuple[str, str]]]] = {
    # name -> (nominative ending, {ending: (case, number)}), in the order a
    # tie between them is settled
    "os": ("ος", {"ος": ("N", "S"), "ου": ("G", "S"), "ῳ": ("D", "S"), "ον": ("A", "S"), "οι": ("N", "P"),
                  "ων": ("G", "P"), "οις": ("D", "P"), "ους": ("A", "P")}),
    # -α is Ἀζαρία's genitive in the Septuagint as well as its vocative: no case.
    "as": ("ας", {"ας": ("N", "S"), "ου": ("G", "S"), "ᾳ": ("D", "S"), "αν": ("A", "S"), "α": ("?", "S")}),
    "es": ("ης", {"ης": ("N", "S"), "ου": ("G", "S"), "ῃ": ("D", "S"), "ην": ("A", "S"), "αι": ("N", "P"),
                  "ων": ("G", "P"), "αις": ("D", "P"), "ας": ("A", "P")}),
    "is": ("ις", {"ις": ("N", "S"), "ιδος": ("G", "S"), "ιδι": ("D", "S"), "ιδα": ("A", "S"), "ιν": ("A", "S")}),
    "a": ("α", {"α": ("N", "S"), "ας": ("G", "S"), "ᾳ": ("D", "S"), "αν": ("A", "S")}),
    "e": ("η", {"η": ("N", "S"), "ης": ("G", "S"), "ῃ": ("D", "S"), "ην": ("A", "S")}),
    "ta": ("α", {"α": ("N", "P"), "ων": ("G", "P"), "οις": ("D", "P")}),     # Σίκιμα, Σικίμων
}
# The gender each paradigm gives a name (a place takes its paradigm's too).
PARADIGM_GENDERS = {"os": "M", "as": "M", "es": "M", "is": "F", "a": "F", "e": "F", "ta": "N"}
# A paradigm's lemma: its nominative singular, or for a plural place name
# (Σίκιμα) its nominative plural.
LEMMA_CASES = {paradigm: ("N", "P") if paradigm == "ta" else ("N", "S") for paradigm in PARADIGMS}


def _strip_marks(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c) or c == "ͅ")


def _plain(text: str) -> str:
    return unicodedata.normalize("NFC", "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c) or c == "ͅ"))


# The endings whose form keeps the nominative's accent where it is: a long
# final syllable can pull an accent forward (Ἀσσύριος, Ἀσσυρίων), so only
# endings as long as the nominative's are read back to it.
SAME_ACCENT = {"os": ("ον", "οι"), "as": ("ου", "ᾳ", "αν"), "es": ("ου", "ῃ", "ην"), "a": ("αν",),
               "e": ("ην",)}


def nominative(surface: str, ending: str, paradigm: str) -> str | None:
    """The nominative singular of a declined form: the form as printed with
    its case ending replaced by the paradigm's (Βαναίου -> Βαναίας). None if
    the ending may have moved the accent (`SAME_ACCENT`), or carried it."""
    if ending not in SAME_ACCENT.get(paradigm, ()):
        return None
    form = display_form(surface)
    nfd = unicodedata.normalize("NFD", form)
    count = len([c for c in unicodedata.normalize("NFD", ending) if not unicodedata.combining(c)])
    cut = len(nfd)
    while count and cut > 0:
        cut -= 1
        if not unicodedata.combining(nfd[cut]):
            count -= 1
    stem = nfd[:cut]
    if not any(c in stem for c in ("\u0301", "\u0300", "\u0342")):
        return None
    return unicodedata.normalize("NFC", stem + PARADIGMS[paradigm][0])


# ---------------------------------------------------------------------------
# Gold pairs and the model

def tbesg_hebrew(lexicon: Lexicon) -> dict[str, set[str]]:
    """Greek key (`G3019`) -> the Hebrew keys (`H3881`) TBESG says its name
    entries render ("the Greek of", "a Spelling of", "a Name of", "a Group
    member of" a Hebrew number)."""
    found: dict[str, set[str]] = defaultdict(set)
    for entry in lexicon.entries:
        if entry.is_name and entry.relation in RELATIONS and entry.related.startswith("H"):
            found[entry.strongs].add(normalize_strongs(entry.related.split()[0]))
    return dict(found)


def kind_from_pos(pos: str) -> str:
    """TBESH's `N:N-M-P`, `N:N--L`, `N:N--LG` as a kind of name."""
    if pos.endswith("G"):
        return "gentilic"
    if pos.endswith("-L"):
        return "place"
    if pos.endswith("-T"):
        return "title"
    return "person"


@dataclass(frozen=True)
class GoldPair:
    greek_strongs: str     # "G2408"
    hebrew_key: str        # "H3414"
    letters: str           # the Greek lemma's letters
    kind: str
    hebrew_lemma: str

    def training(self) -> tuple:
        return (self.letters, self.kind, sound_variants(self.hebrew_lemma, self.kind == "gentilic"))


def gold_pairs(lexicon: Lexicon, tbesh: dict) -> list[GoldPair]:
    """Each NT name whose TBESG entry gives its Hebrew number, as (Greek
    spelling, Hebrew spelling) pairs: one per spelling of the Greek lemma."""
    pairs = []
    for entry in lexicon.entries:
        if not (entry.is_name and entry.relation in RELATIONS and entry.related.startswith("H")):
            continue
        hebrew = tbesh.get(entry.related.split()[0])
        if hebrew is None:
            continue
        kind = kind_from_pos(hebrew.pos)
        for variant in entry.variants:
            pairs.append(GoldPair(entry.strongs, normalize_strongs(entry.related.split()[0]),
                                  letters_of(normalize(variant)), kind, hebrew.lemma))
    # Sorted by every field: two TBESH rows of one key can give one Greek
    # spelling two Hebrew lemmas (G7 αβια for אֲבִיָּה and אֲבִיָּהוּ), and the
    # order the model is fitted in must not be left to the set's.
    return sorted(set(pairs), key=lambda p: (p.greek_strongs, p.hebrew_key, p.letters, p.kind, p.hebrew_lemma))


def held_out(strongs: str, share: int = 4) -> bool:
    """Whether a Greek name is in the validation split: one in `share`,
    by name (its number), never by occurrence."""
    number = int(re.sub(r"\D", "", strongs) or 0)
    return (number * 2654435761) % 1000 < 1000 // share


def transliterations(pairs: Sequence[GoldPair], model: translit.Model, least: float = 0.0) -> list[GoldPair]:
    """The pairs that are transliterations at all (the conventions' model
    scores them above `least`): Αἴγυπτος for מִצְרַיִם is a translation,
    and would teach the model nonsense."""
    return [p for p in pairs if model.score(p.letters, p.kind, p.training()[2]) > least]


def train(pairs: Sequence[GoldPair], background: dict, stop: float, extra: Iterable[tuple] = (),
          iterations: int = 8) -> translit.Model:
    """The transducer fitted to the transliterated gold pairs and `extra`
    (letters, kind, variants) pairs."""
    prior = translit.initial(background, stop)
    training = [p.training() for p in transliterations(pairs, prior)] + list(extra)
    return translit.fit(training, background, stop, iterations=iterations)


# ---------------------------------------------------------------------------
# One-to-one assignment

def assign(weights: list[list[float]]) -> list[int | None]:
    """Maximum-weight one-to-one assignment of rows to columns, a row free to
    stay unassigned (weight 0). Returns each row's column or None."""
    rows = len(weights)
    if not rows:
        return []
    columns = len(weights[0])
    size = columns + rows       # one "unassigned" column per row
    # Kuhn-Munkres on costs (minimise), rows <= size.
    big = 1e9
    cost = [[(-weights[r][c] if weights[r][c] > 0 else big) if c < columns else 0.0 for c in range(size)]
            for r in range(rows)]
    u = [0.0] * (rows + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, rows + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j] = current
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result: list[int | None] = [None] * rows
    for j in range(1, size + 1):
        if p[j] and j - 1 < columns and weights[p[j] - 1][j - 1] > 0:
            result[p[j] - 1] = j - 1
    return result


def total(weights: list[list[float]], chosen: list[int | None]) -> float:
    return sum(weights[r][c] for r, c in enumerate(chosen) if c is not None)


def assign_with_margins(weights: list[list[float]]) -> list[tuple[int | None, float, int | None]]:
    """Each row's column, its margin (what the best assignment loses when that
    pair is forbidden), and the column the row takes then (its runner-up)."""
    chosen = assign(weights)
    best = total(weights, chosen)
    out = []
    for row, column in enumerate(chosen):
        if column is None:
            out.append((None, 0.0, None))
            continue
        forbidden = [list(r) for r in weights]
        forbidden[row][column] = 0.0
        other = assign(forbidden)
        out.append((column, best - total(forbidden, other), other[row]))
    return out


# ---------------------------------------------------------------------------
# The stage

def _morph_case(code: str) -> tuple[str, str, str] | None:
    """(case, number, gender) of an article or adjective code (`T-GSM`)."""
    match = re.match(r"^T-([NGDAV])([SP])([MFN])$", code or "")
    return match.groups() if match else None


@dataclass
class GreekUnit:
    """One Greek spelling of a verse's candidates (every token with those letters)."""

    letters: str
    positions: list[int]                      # token indexes in the verse
    surfaces: list[str]
    kind: str                                 # "candidate", "recheck" or "gold"
    group: int = 0                            # spellings of one name share a group


@dataclass(frozen=True)
class Column:
    """A Hebrew name a Greek spelling may be assigned to: the whole name, or
    one word of a name of several, which the Greek may print apart."""

    word: HebrewWord
    cost: float
    spelling: str                             # the lemma, or the word of it
    part: int = -1


# Divine names the Septuagint translates (κύριος, θεός) and never
# transliterates: a capitalised Greek word is never their rendering.
TRANSLATED_NAMES = frozenset(("H3068", "H3069", "H136"))


class NameStage:
    """Matches Greek names to Hebrew ones. See the module docstring."""

    def __init__(self, hebrew: HebrewVerses, verse_map: VerseMap, lexicon: Lexicon, model: translit.Model,
                 settings: Settings = Settings(), key_mode: str = "hebrew", codes: Iterable[str] | None = None,
                 tbesh: dict | None = None, exclude: Iterable[str] = ()):
        if key_mode not in KEY_MODES:
            raise ValueError(f"key mode must be one of {KEY_MODES}, not {key_mode!r}")
        self.hebrew = hebrew
        self.verse_map = verse_map
        self.lexicon = lexicon
        self.model = model
        self.settings = settings
        self.key_mode = key_mode
        self.codes = set(codes) if codes is not None else None
        self.tbesh = tbesh or {}
        self.forms = None
        # Greek names whose TBESG relation the stage may not use (the gold
        # names held out for validation).
        self.exclude = set(exclude)
        self.greek_to_hebrew = tbesg_hebrew(lexicon)
        self.hebrew_to_greek: dict[str, list] = defaultdict(list)
        for entry in lexicon.entries:
            if (entry.is_name and entry.relation in RELATIONS and entry.related.startswith("H")
                    and entry.strongs not in self.exclude):
                self.hebrew_to_greek[normalize_strongs(entry.related.split()[0])].append(entry)
        self._stems: dict[str, set[str]] = {}
        self._names: dict[str, bool] = {}
        self._local: dict[str, str] = {}

    def nt_lemma(self, entry) -> str:
        """An NT name's lemma as the module spells it for that number
        elsewhere: TAGNT's, when the stage was given the form table."""
        found = self.forms.lemma_for(entry.strongs) if self.forms is not None else None
        return found or entry.lemma

    # -- which tokens --------------------------------------------------------
    def is_name_entry(self, strongs: str) -> bool:
        found = self._names.get(strongs)
        if found is None:
            found = self._names[strongs] = any(entry.is_name for entry in self.lexicon.by_strongs(strongs))
        return found

    def role(self, view: TokenView, gold: set[str] | None = None) -> str | None:
        """"candidate", "recheck", "gold", "anchor" or None for a token."""
        token = view.token
        if token.case == "lower" or token.joined:
            return None
        if gold is not None and view.strongs in gold:
            return "gold"
        if view.tier == "d" and not view.strongs:
            return "candidate"
        if view.tier == "e" and not view.strongs and not verb_like(token.key):
            # A capitalised word with no reading at all: a name at the start
            # of a sentence or after a comma (Gen 2:11 Φισῶν), which the
            # case filter cannot call name-like. Offered under the same
            # thresholds, unless its ending is a verb's (Ἐνεπάγησαν).
            return "candidate"
        if view.method in ("tolerant", "tolerant-mismatch") and self.is_name_entry(view.strongs):
            return "recheck"
        if view.strongs in self.greek_to_hebrew:
            return "anchor"
        return None

    # -- Hebrew side ----------------------------------------------------------
    def target(self, ref: int) -> int | None:
        """The English verse the Septuagint verse maps to, if TAHOT has it."""
        mapped = self.verse_map[ref]
        if mapped is not None and mapped in self.hebrew:
            return mapped
        return None

    def nearby(self, ref: int) -> list[tuple[int, int]]:
        """(Hebrew verse, distance) around the mapped verse (or, unmapped, the
        verse with the same number), within the window, same chapter."""
        mapped = self.verse_map[ref]
        centre = mapped if mapped is not None and mapped in self.hebrew else ref
        chapter, verse = divmod(centre, 1000)
        last = self.hebrew.last_verse(chapter)
        found = []
        if centre != mapped and centre in self.hebrew:
            found.append((centre, 1))
        for offset in range(1, self.settings.window + 1):
            for number in (verse - offset, verse + offset):
                candidate = chapter * 1000 + number
                if 0 <= number <= last and candidate in self.hebrew:
                    found.append((candidate, offset + (1 if centre != mapped else 0)))
        return found

    def names_of(self, ref: int) -> list[HebrewWord]:
        return [w for w in self.hebrew.names(ref) if w.key not in TRANSLATED_NAMES]

    def unified(self, word: HebrewWord) -> set[str]:
        """The word's key and the one TBESH unifies it with (its uStrong
        column: לְוִיִּם, H3881 "Levite", is a Name of H3878 Levi)."""
        keys = {word.key}
        entry = self.tbesh.get(word.dstrongs)
        if entry is not None and entry.related.startswith("H"):
            keys.add(normalize_strongs(entry.related.split()[0]))
        return keys

    def tbesg_entry(self, letters: str, word: HebrewWord):
        """The NT name TBESG says renders this Hebrew name, when the Greek
        form is a form of its lemma (see `tbesg_name`), else None."""
        return tbesg_name(self, letters, self.unified(word))

    def _variants(self, word: HebrewWord, spelling: str | None):
        return sound_variants(spelling or word.lemma, word.kind == "gentilic")

    def score(self, letters: str, word: HebrewWord, spelling: str | None = None) -> float:
        """The transliteration score of a Greek spelling as this Hebrew word
        (or one word of it), or `TBESG_SCORE` when TBESG's relation says so."""
        kind = word.kind if word.is_name else "person"
        value = self.model.score(letters, kind, self._variants(word, spelling))
        if word.is_name and spelling is None and value < TBESG_SCORE and self.tbesg_entry(letters, word) is not None:
            value = TBESG_SCORE
        return value

    def split(self, letters: str, word: HebrewWord, spelling: str | None = None) -> translit.Split | None:
        kind = word.kind if word.is_name else "person"
        return self.model.explain(letters, kind, self._variants(word, spelling))[1]

    # -- matching -----------------------------------------------------------
    def units(self, verse: VerseView, roles: dict[int, str]) -> list[GreekUnit]:
        """The verse's candidate spellings, those that share a stem of four
        letters or more (Βαναίας, Βαναίου; Λευῖται, Λευιτῶν) in one group."""
        units: dict[str, GreekUnit] = {}
        for position, role in sorted(roles.items()):
            if role not in ("candidate", "recheck", "gold"):
                continue
            token = verse.tokens[position].token
            letters = letters_of(token.key)
            if len(letters) < 2:
                continue
            unit = units.setdefault(letters, GreekUnit(letters, [], [], role))
            unit.positions.append(position)
            unit.surfaces.append(token.surface)
            if role == "candidate":
                unit.kind = "candidate"
        found = list(units.values())
        stems = [{stem for stem in _stem_letters(unit.letters) if len(stem) >= 4} for unit in found]
        for index, unit in enumerate(found):
            unit.group = index
            for other in range(index):
                if stems[index] & stems[other]:
                    unit.group = found[other].group
                    break
        return found

    def anchored(self, verse: VerseView, roles: dict[int, str]) -> Counter:
        """The Hebrew keys the verse's tagged NT names render (TBESG's
        relations), each as many times as a Greek word renders it."""
        keys: Counter = Counter()
        for position, role in roles.items():
            if role == "anchor":
                keys.update(self.greek_to_hebrew.get(verse.tokens[position].strongs, set()))
        return keys

    def free(self, words: list[HebrewWord], anchored: Counter) -> list[HebrewWord]:
        """The Hebrew names no anchor renders. Each anchor takes one
        occurrence of the name with its own key (Jer 51:29 has Babylon twice,
        and Brenton's 28:29 one Βαβυλῶνος the NT's table knows and one
        Βαβυλῶνα it does not); failing that, one TBESH unifies with it
        (Λευΐτης, related to H3878 Levi, takes לְוִיִּם, H3881). Not both:
        TBESH unifies Ephrath with Bethlehem and Sarai with Sarah, and Gen 48:7
        has Βηθλεέμ and Ἐφραθά, Gen 17:15 Σάρρα and Σάρα."""
        present = {w.key for w in words}
        left = Counter(anchored)
        kept = []
        for word in words:
            if left[word.key] > 0:
                left[word.key] -= 1
                continue
            unified = [k for k in sorted(self.unified(word) - {word.key}) if k not in present and left[k] > 0]
            if unified:
                left[unified[0]] -= 1
                continue
            kept.append(word)
        return kept

    def columns(self, pool: Iterable[tuple[HebrewWord, float]]) -> list[Column]:
        """Each distinct Hebrew name at its least distant occurrence, and each
        word of a name of several. Names are distinct by the key the module
        ships (TAHOT's H1732 and H1732G are one name to the reader), so two
        of TAHOT's disambiguations never make a choice ambiguous."""
        best: dict[str, tuple[HebrewWord, float]] = {}
        for word, cost in pool:
            if word.key not in best or cost < best[word.key][1]:
                best[word.key] = (word, cost)
        columns = []
        for word, cost in best.values():
            columns.append(Column(word, cost, word.lemma))
            for index, part in enumerate(hebrew_names.parts(word.lemma)):
                columns.append(Column(word, cost, part, index))
        return columns

    def match_verse(self, verse: VerseView, roles: dict[int, str]) -> tuple[dict[str, Match], dict[str, str]]:
        """The verse-level and nearby matches of the verse's Greek forms, by
        letters, and why each unmatched form is unmatched."""
        units = self.units(verse, roles)
        self._local = {}
        if not units:
            return {}, {}
        anchored = self.anchored(verse, roles)
        matches: dict[str, Match] = {}
        why: dict[str, str] = {}
        target = self.target(verse.ref)
        settings = self.settings
        if target is not None:
            names = self.free(self.names_of(target), anchored)
            commons = [w for w in self.hebrew.words(target) if not w.is_name and w.grammar.startswith("Nc")]
            found = self._assign(units, self.columns((w, 0.0) for w in names), settings.threshold,
                                 settings.margin, "hebrew-verse", commons)
            for letters, (match, reason) in found.items():
                if match:
                    matches[letters] = match
                else:
                    why[letters] = reason
        else:
            for unit in units:
                why[unit.letters] = "no Hebrew verse"
        if target is not None:
            for letters, match in self.by_order(verse, roles, units, matches, why, target, anchored).items():
                matches[letters] = match
                why.pop(letters, None)
        # Beyond the verse only for a form none of the verse's own names left
        # over could render: the Septuagint's Ναβάτ in 1 Kgs 15:31 is the
        # verse's Nadab, however like the Nebat of 15:1 it is spelt.
        left = []
        if target is not None:
            used = {m.hebrew.key for m in matches.values()}
            left = [w for w in self.free(self.names_of(target), anchored) if w.key not in used]
        commons = [w for w in self.hebrew.words(target) if not w.is_name] if target is not None else []
        for unit in units:
            if unit.letters in matches:
                continue
            if any(self.score(unit.letters, w) >= settings.order_floor for w in left):
                why.setdefault(unit.letters, "no matching Hebrew name")
                self._local[unit.letters] = "a Hebrew name of the verse may be its counterpart"
            elif any(self.score(unit.letters, w) >= settings.threshold for w in commons):
                # Μοσόχ in Isa 66:19 reads מֹשְׁכֵי "drawing" as Meshech:
                # the Greek names what the Hebrew does not.
                why.setdefault(unit.letters, "no matching Hebrew name")
                self._local[unit.letters] = "it may read a common Hebrew word of the verse as a name"
        for search, threshold, least in (("nearby", settings.nearby_threshold, settings.nearby_margin),
                                         ("chapter", settings.chapter_threshold, settings.chapter_margin)):
            rest = [unit for unit in units if unit.letters not in matches and not self._local.get(unit.letters)
                    and why.get(unit.letters) not in ("ambiguous between Hebrew names",
                                                      "transliterates a common Hebrew word")]
            if not rest:
                break
            taken = {m.hebrew.key for m in matches.values()}
            pool: list[tuple[HebrewWord, float]] = []
            refs = self.nearby(verse.ref) if search == "nearby" else self.chapter(verse.ref)
            for ref, distance in refs:
                for word in self.free(self.names_of(ref), anchored):
                    if word.key not in taken:
                        pool.append((word, distance * settings.offset_cost if search == "nearby" else 0.0))
            if pool:
                found = self._assign(rest, self.columns(pool), threshold, least, "hebrew-nearby", [])
                for letters, (match, reason) in found.items():
                    if match:
                        matches[letters] = match
                        why.pop(letters, None)
                    elif letters not in why:
                        why[letters] = reason
        return matches, why

    def chapter(self, ref: int) -> list[tuple[int, int]]:
        """Every Hebrew verse of the mapped verse's chapter and the chapters
        either side, but those `nearby` searched: the last, strictest search,
        for the books whose order the verse map does not follow (Exodus
        35-40, Numbers 26, 3 Kingdoms)."""
        mapped = self.verse_map[ref]
        centre = mapped if mapped is not None and mapped in self.hebrew else ref
        near = {r for r, _ in self.nearby(ref)} | {centre}
        found = []
        for chapter in (centre // 1000 - 1, centre // 1000, centre // 1000 + 1):
            for number in range(0, self.hebrew.last_verse(chapter) + 1):
                candidate = chapter * 1000 + number
                if candidate in self.hebrew and candidate not in near:
                    found.append((candidate, 0))
        return found

    def by_order(self, verse: VerseView, roles: dict[int, str], units: list[GreekUnit], matches: dict[str, Match],
                 why: dict[str, str], target: int, anchored: set[str]) -> dict[str, Match]:
        """Names a list's order places. The Septuagint keeps the Hebrew's
        lists in order, so a Greek name left between two matched names (or
        after the last, or before the first), with exactly one Hebrew name
        left between their counterparts, renders that name, if it scores at
        least `Settings.order_floor` against it: Ἐρθουλά, between nothing
        and Βουλά (Bethul), is Eltolad (Josh 19:4). The matched pairs must be
        in the same order in both languages."""
        hebrew = self.names_of(target)
        pins: list[tuple[int, int]] = []
        for letters, match in matches.items():
            unit = next(u for u in units if u.letters == letters)
            pins.append((min(unit.positions), match.hebrew.index))
        for position, role in roles.items():
            if role != "anchor":
                continue
            keys = self.greek_to_hebrew.get(verse.tokens[position].strongs, set())
            found = [w.index for w in hebrew if w.key in keys]
            if len(found) == 1:
                pins.append((position, found[0]))
        pins.sort()
        if not pins or any(b[1] <= a[1] for a, b in zip(pins, pins[1:])):
            return {}
        # Hebrew names by occurrence: a second Jehoahaz is a name left over
        # even when the first is matched (2 Kgs 13:25's Ἰωάχαζ ... Ἰωάχας).
        pinned = {h for _, h in pins}
        free = sorted(((w.index, w) for w in self.free(hebrew, anchored) if w.index not in pinned),
                      key=lambda item: item[0])
        open_units = sorted(((min(u.positions), u) for u in units
                             if u.letters not in matches and why.get(u.letters) == "no matching Hebrew name"),
                            key=lambda item: item[0])
        placed: dict[str, Match] = {}
        bounds = [(-1, -1)] + pins + [(10 ** 6, 10 ** 6)]
        for (g_low, h_low), (g_high, h_high) in zip(bounds, bounds[1:]):
            between = [u for g, u in open_units if g_low < g < g_high]
            names = [w for h, w in free if h_low < h < h_high]
            if len(between) != 1 or len(names) != 1:
                continue
            if (g_low, h_low) == (-1, -1) and (g_high, h_high) == (10 ** 6, 10 ** 6):
                continue
            unit, word = between[0], names[0]
            score = self.score(unit.letters, word)
            if score < self.settings.order_floor:
                continue
            others = [self.score(unit.letters, w) for _, w in free if w.key != word.key]
            others += [self.score(unit.letters, m.hebrew) for m in matches.values() if m.hebrew.key != word.key]
            margin = score - max([0.0] + others)
            if margin <= 0:
                continue
            placed[unit.letters] = Match(word, score, margin, "hebrew-order")
        return placed

    def _assign(self, units: list[GreekUnit], columns: list[Column], threshold: float, least_margin: float,
                method: str, commons: list[HebrewWord]) -> dict[str, tuple[Match | None, str]]:
        out: dict[str, tuple[Match | None, str]] = {}
        if not columns:
            for unit in units:
                out[unit.letters] = (None, "no Hebrew name in the verse")
            return out
        scores = {unit.letters: [self.score(unit.letters, c.word, None if c.part < 0 else c.spelling) - c.cost
                                 for c in columns] for unit in units}
        # A group's row is its best spelling's; a name's several words may go
        # to several groups, but the whole name and its words to one only.
        groups: dict[int, list[GreekUnit]] = defaultdict(list)
        for unit in units:
            groups[unit.group].append(unit)
        order = sorted(groups)
        weights = [[max(max(scores[u.letters][c] for u in groups[g]), 0.0) for c in range(len(columns))]
                   for g in order]
        # The whole of a name and any of its words exclude each other; the
        # assignment cannot say so, so a group that takes a word of a name
        # whose whole another group took is dropped below.
        assigned = assign_with_margins(weights)
        whole_taken = {columns[c].word.key for c, _, _ in assigned if c is not None and columns[c].part < 0}
        for row, group in enumerate(order):
            column, margin, runner = assigned[row]
            for unit in groups[group]:
                if column is None:
                    best_any = max(max(scores[unit.letters]), 0.0)
                    out[unit.letters] = (None, "Hebrew name taken by another Greek name"
                                         if best_any >= threshold else "no matching Hebrew name")
                    continue
                chosen = columns[column]
                if chosen.part >= 0 and chosen.word.key in whole_taken:
                    out[unit.letters] = (None, "Hebrew name taken by another Greek name")
                    continue
                score = scores[unit.letters][column]
                if score < threshold:
                    out[unit.letters] = (None, "no matching Hebrew name")
                    continue
                if margin < least_margin:
                    out[unit.letters] = (None, "ambiguous between Hebrew names")
                    continue
                if commons:
                    common = max((self.score(unit.letters, w) for w in commons), default=-math.inf)
                    if common >= score + self.settings.common_margin:
                        out[unit.letters] = (None, "transliterates a common Hebrew word")
                        continue
                runner_word = columns[runner].word if runner is not None else None
                out[unit.letters] = (Match(chosen.word, score, margin, method, runner_word,
                                           "" if chosen.part < 0 else chosen.spelling), "")
        return out

    # -- the whole text ------------------------------------------------------
    def run(self, verses: Sequence[VerseView], gold: set[str] | None = None):
        """Every verse's matches, then the consistency pass. Returns
        ({(verse index, token index): (Match, GreekUnit)} of verse-level and
        nearby matches, {(verse index, token index): reason} for candidates
        left untagged, and the consistency pass's additions, keyed alike)."""
        found: dict[tuple[int, int], tuple[Match, GreekUnit]] = {}
        reasons: dict[tuple[int, int], str] = {}
        verse_units: dict[int, list[tuple[GreekUnit, str]]] = {}
        for number, verse in enumerate(verses):
            roles = {i: r for i, view in enumerate(verse.tokens) if (r := self.role(view, gold))}
            if not any(r in ("candidate", "recheck", "gold") for r in roles.values()):
                continue
            matches, why = self.match_verse(verse, roles)
            local = dict(self._local)
            units = self.units(verse, roles)
            verse_units[number] = []
            for unit in units:
                match = matches.get(unit.letters)
                for position in unit.positions:
                    if match:
                        found[(number, position)] = (match, unit)
                    else:
                        reasons[(number, position)] = why.get(unit.letters, "no matching Hebrew name")
                        if local.get(unit.letters):
                            reasons[(number, position)] += "; " + local[unit.letters]
                verse_units[number].append((unit, "local" if local.get(unit.letters) else why.get(unit.letters, "")))
        consistent = self.consistency(verses, found, reasons, verse_units)
        return found, reasons, consistent

    def confident(self, match: Match) -> bool:
        """A verse-level match of a whole name, sure enough to speak for the form."""
        return (match.method == "hebrew-verse" and not match.spelling and match.score >= self.settings.confident
                and match.margin >= self.settings.margin)

    def consistency(self, verses, found, reasons, verse_units) -> dict[tuple[int, int], tuple[Match, GreekUnit]]:
        """The consistency pass (see the module docstring). A form's number is
        the one its verse-level matches agree on (`Settings.consistency_share`
        of them, `Settings.consistency_min` confident), provided no confident
        match gave it another. A weak match of the form to another number,
        which that number's absence from its confident matches contradicts,
        is withdrawn first: Ἐλισαιέ, 60 times Elisha, is not "Philistine" in
        a verse whose Hebrew lacks Elisha."""
        settings = self.settings
        numbers: dict[str, Counter] = defaultdict(Counter)
        confident: dict[str, Counter] = defaultdict(Counter)
        words: dict[tuple[str, str], HebrewWord] = {}
        scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (verse, position), (match, unit) in found.items():
            if match.spelling:
                continue        # a word of a longer name (Μωάβ in Pahath-moab) says nothing of the form
            numbers[unit.letters][match.hebrew.key] += 1
            if self.confident(match):
                confident[unit.letters][match.hebrew.key] += 1
                words.setdefault((unit.letters, match.hebrew.key), match.hebrew)
                scores[(unit.letters, match.hebrew.key)].append(match.score)

        def number_of(letters: str) -> str | None:
            sure = confident.get(letters)
            if not sure or len(sure) != 1:
                return None
            key, count = next(iter(sure.items()))
            if count < settings.consistency_min or numbers[letters][key] < settings.consistency_share * sum(
                    numbers[letters].values()):
                return None
            return key

        for (verse, position), (match, unit) in list(found.items()):
            key = number_of(unit.letters)
            if key is not None and match.hebrew.key != key and not match.spelling and not self.confident(match) \
                    and match.hebrew.key not in confident[unit.letters]:
                del found[(verse, position)]
                reasons[(verse, position)] = "a weak match the form's other matches contradict"
        added: dict[tuple[int, int], tuple[Match, GreekUnit]] = {}
        for verse_number, units in verse_units.items():
            for unit, why in units:
                if any((verse_number, p) in found for p in unit.positions):
                    continue
                if why in ("ambiguous between Hebrew names", "transliterates a common Hebrew word", "local"):
                    continue
                key = number_of(unit.letters)
                if key is None:
                    if len(confident.get(unit.letters, ())) > 1:
                        for position in unit.positions:
                            if (verse_number, position) in reasons:
                                reasons[(verse_number, position)] += "; the form matched several Hebrew numbers"
                    continue
                word = words[(unit.letters, key)]
                typical = sorted(scores[(unit.letters, key)])[len(scores[(unit.letters, key)]) // 2]
                match = Match(word, typical, 0.0, "hebrew-consistency")
                for position in unit.positions:
                    added[(verse_number, position)] = (match, unit)
                    reasons.pop((verse_number, position), None)
        return added


# ---------------------------------------------------------------------------
# Keys, lemmas and grammar

def greek_extended(key: str) -> str:
    """STEPBible's Greek key for a Hebrew name: H40 -> G10040."""
    return f"G{int(key[1:]) + 10000}"


def name_suffix(word: HebrewWord, tbesh: dict) -> str:
    """TAGNT's suffix for the kind of name: P, L, PG, LG or T."""
    if word.kind == "gentilic":
        entry = tbesh.get(word.dstrongs) or tbesh.get(re.sub(r"[A-Za-z]$", "", word.dstrongs))
        return "LG" if entry is not None and entry.pos.endswith("LG") else "PG"
    return {"person": "P", "place": "L", "title": "T"}.get(word.kind, "P")


def grammar_of(view: TokenView, previous: TokenView | None, word: HebrewWord, shown: tuple[str, str] | None,
               paradigm: str | None, tbesh: dict) -> tuple[str, bool]:
    """A TAGNT-shaped code for the name (`N-GSM-P`, with ? for what is not
    known), and whether every part of it is known. The case and number come
    from a declined name's ending, else from an article just before it (the
    chooser's tag for it) or a preposition that takes one case; the gender
    from the article, the paradigm (-ος, -ας, -ης are masculine) or the
    Hebrew (a man's or a woman's name)."""
    case = number = gender = None
    if shown:
        case, number = shown
        case = None if case == "?" else case
        gender = PARADIGM_GENDERS.get(paradigm)
    if previous is not None:
        article = _morph_case(previous.morph)
        if article:
            a_case, a_number, a_gender = article
            if case is None or (case, number) == (a_case, a_number):
                case, number, gender = a_case, a_number, a_gender
        elif case is None and PREPOSITION_CASES.get(previous.token.key):
            case = PREPOSITION_CASES[previous.token.key]
    if gender is None:
        if word.grammar.startswith("Npm"):
            gender = "M"
        elif word.grammar.startswith("Npf"):
            gender = "F"
    if number is None and word.kind in ("person", "title"):
        number = "S"
    code = f"N-{case or '?'}{number or '?'}{gender or '?'}-{name_suffix(word, tbesh)}"
    return code, None not in (case, number, gender)


def _stem_letters(letters: str) -> set[str]:
    """A Greek spelling's letters with and without each case ending it may
    carry (not a gentilic suffix: Γαλααδῖτις is not a form of Γαλαάδ, nor
    Λευΐτης of Λευΐ)."""
    found = {letters}
    for split, _ in translit.decompositions(letters):
        if not split.gentilic:
            found.add("".join(split.stem))
    return found


# The score of a Greek form TBESG's own relation identifies: a lexicon fact
# (that entry renders this Hebrew name), confirmed by the name's being in
# the Hebrew verse; as sure as a long transliteration.
TBESG_SCORE = 20.0


def tbesg_name(stage: NameStage, letters: str, keys: Iterable[str]):
    """The NT's Greek entry for a Hebrew name (any of `keys`), if TBESG
    relates one to it and the Greek form is a form of its lemma, the same
    stem of three letters or more with or without a case ending: Λευῖται is
    Λευίτης (G3019, a Group member of H3878, which TBESH unifies with
    H3881), Βαβυλῶνα is Βαβυλών (G897, the Greek of H894)."""
    stems = {stem for stem in _stem_letters(letters) if len(stem) >= 3}
    best, best_length = None, 0
    for key in sorted(keys):
        for entry in stage.hebrew_to_greek.get(key, ()):
            for variant in entry.variants:
                variant_letters = letters_of(normalize(variant))
                cached = stage._stems.get(variant_letters)
                if cached is None:
                    cached = stage._stems[variant_letters] = _stem_letters(variant_letters)
                shared = max((len(stem) for stem in cached & stems), default=0)
                # The longest shared stem wins: Λευῖται shares λευιτ with
                # Λευΐτης and nothing with Λευΐ.
                if shared > best_length:
                    best, best_length = entry, shared
    return best


# Endings no Hebrew name ends in: a form with one of them declines.
GREEK_ONLY_ENDINGS = ("ος", "ον", "ῳ", "οι", "οις", "ους", "ᾳ", "ῃ", "ιδος", "ιδι")


def paradigm_readings(surface: str) -> list[tuple[str, str, str, str, str]]:
    """Every way the printed form can be a stem and a case ending:
    (paradigm, stem letters, case, number, ending), the longest ending of
    each paradigm. Read from the Greek alone."""
    plain = _plain(surface.lower())
    found = []
    for paradigm, (_, endings) in PARADIGMS.items():
        for ending, (case, number) in sorted(endings.items(), key=lambda item: -len(item[0])):
            if plain.endswith(ending) and len(bare(plain)) - len(bare(ending)) >= 2:
                stem = bare(plain[: len(plain) - len(ending)]).replace("ς", "σ")
                found.append((paradigm, stem, case, number, ending))
                break
    return found


def lemma_groups(stage: NameStage, everything, verses) -> dict[tuple[int, int], tuple]:
    """Each matched token's lemma, the case and number its ending shows, and
    the paradigm (None for an undeclined name).

    Forms matched to one Hebrew number are one declining name when they
    share a stem in one paradigm and carry two different case endings
    (Βαναίας, Βαναίου; Λεία, Λείας), or an ending no Hebrew name ends in
    (-ος, -ον, -ῳ ...: Μαρδοχαῖος). Its lemma is the nominative singular
    (a plural place's nominative plural, Σίκιμα), as Brenton prints it if he
    does, else built from a printed form whose ending keeps the accent where
    the nominative has it (`SAME_ACCENT`), else the commonest printed form.
    Any other form is undeclined, and its lemma is the commonest printing of
    its letters (words in capitals, which lose their accents, only when there
    is no other), so every occurrence of a name has the same lemma."""
    surfaces: dict[str, dict[str, list[tuple[tuple[int, int], str]]]] = defaultdict(lambda: defaultdict(list))
    genders: dict[str, str | None] = {}
    for (v, p), (match, unit) in everything.items():
        surfaces[match.hebrew.key][unit.letters].append(((v, p), verses[v].tokens[p].token.surface))
        grammar = match.hebrew.grammar
        genders.setdefault(match.hebrew.key, "M" if grammar.startswith(("Npm", "Ngm")) else
                           "F" if grammar.startswith(("Npf", "Ngf")) else None)
    out: dict[tuple[int, int], tuple] = {}
    paradigm_order = list(PARADIGMS)

    def printed(items) -> str:
        texts = [surface for _, surface in items]
        counts = Counter(display_form(t) for t in texts if not t.isupper()) or Counter(display_form(t) for t in texts)
        return sorted(counts.items(), key=lambda i: (-i[1], i[0]))[0][0]

    for key, forms in surfaces.items():
        groups: dict[tuple[str, str], dict[str, tuple[str, str, str]]] = defaultdict(dict)
        for letters, items in forms.items():
            for paradigm, stem, case, number, ending in paradigm_readings(items[0][1]):
                groups[(paradigm, stem)][letters] = (case, number, ending)

        def declined(group) -> bool:
            members = groups[group]
            return (len({(c, n) for c, n, _ in members.values() if c != "?"}) > 1
                    or any(e in GREEK_ONLY_ENDINGS for _, _, e in members.values()))

        chosen: dict[str, tuple[str, str]] = {}
        gender = genders.get(key)
        for letters in sorted(forms):
            # A man's name declines as a masculine, a woman's as a feminine
            # (Σαμαίας, not *Σαμαία, for Shemaiah; Ζαβδίας is not the genitive
            # of a *Ζαβδία).
            options = [g for g in groups if letters in groups[g] and declined(g)
                       and (gender is None or PARADIGM_GENDERS[g[0]] in (gender, "N"))]
            if options:
                chosen[letters] = sorted(options, key=lambda g: (
                    -len({(c, n) for c, n, _ in groups[g].values()}),
                    -sum(len(forms[f]) for f in groups[g]), paradigm_order.index(g[0]), g[1]))[0]
        lemma_of: dict[tuple[str, str], str] = {}
        for group in set(chosen.values()):
            members = [f for f in forms if chosen.get(f) == group]
            paradigm = group[0]
            nominatives: Counter = Counter()
            built: Counter = Counter()
            for letters in members:
                case, number, ending = groups[group][letters]
                for _, surface in forms[letters]:
                    if surface.isupper():
                        continue
                    if (case, number) == LEMMA_CASES[paradigm]:
                        nominatives[display_form(surface)] += 1
                    else:
                        made = nominative(surface, ending, paradigm)
                        if made:
                            built[made] += 1
            found = nominatives or built
            lemma_of[group] = (sorted(found.items(), key=lambda i: (-i[1], i[0]))[0][0] if found
                               else printed([item for f in members for item in forms[f]]))
        for letters, items in forms.items():
            group = chosen.get(letters)
            if group is None:
                lemma = printed(items)
                for position, _ in items:
                    out[position] = (lemma, None, None)
                continue
            case, number, _ = groups[group][letters]
            for position, _ in items:
                out[position] = (lemma_of[group], (case, number), group[0])
    return out


def decide(stage: NameStage, verses: Sequence[VerseView], everything, confidence) -> dict:
    """The NameDecision for every matched token: key, lemma (one per name),
    grammar and, in greek-extended mode or for an NT name, morphology."""
    lemmas = lemma_groups(stage, everything, verses)
    decisions = {}
    for (v, p), (match, unit) in everything.items():
        verse = verses[v]
        view = verse.tokens[p]
        previous = verse.tokens[p - 1] if p > 0 else None
        lemma, shown, paradigm = lemmas[(v, p)]
        grammar, known = grammar_of(view, previous, match.hebrew, shown, paradigm, stage.tbesh)
        key = match.hebrew.key
        key_source = "tahot"
        morph = None
        entry = stage.tbesg_entry(unit.letters, match.hebrew) if not match.spelling else None
        if entry is not None:
            strongs = entry.strongs
            lemma = stage.nt_lemma(entry)
            key_source = "tbesg"
            if known and (stage.codes is None or grammar in stage.codes):
                morph = grammar
        elif stage.key_mode == "greek-extended":
            strongs = greek_extended(key)
            if known and (stage.codes is None or grammar in stage.codes):
                morph = grammar
        else:
            strongs = key
        decisions[(v, p)] = NameDecision(
            strongs, lemma, morph, match.method, match.score, match.margin,
            confidence(match), match.hebrew, grammar,
            replaces=f"{view.strongs}|{view.lemma}" if view.strongs else "", key_source=key_source,
            runner_up=match.runner_up.key if match.runner_up else "")
    return decisions


def confirms(stage: NameStage, view: TokenView, match: Match) -> bool:
    """Whether a Hebrew match confirms a tolerant name tag: TBESG relates the
    tag's entry to the matched name (or the name TBESH unifies it with)."""
    return bool(stage.greek_to_hebrew.get(view.strongs, set()) & stage.unified(match.hebrew))


def candidate_for(decision: NameDecision) -> Candidate:
    return Candidate(decision.strongs, decision.lemma, decision.morph, SOURCE, decision.score, "name", True,
                     f"{decision.method}: {decision.hebrew.dstrongs} {decision.hebrew.stem} "
                     f"({decision.hebrew.ref})")


def choice_for(decision: NameDecision) -> Choice:
    return Choice(candidate_for(decision), decision.confidence, decision.margin, decision.method)


# ---------------------------------------------------------------------------
# Building it from the sources

def build_stage(sources: Path, lexicon: Lexicon, verse_map: VerseMap, key_mode: str = "hebrew",
                codes: Iterable[str] | None = None, settings: Settings = Settings(),
                hebrew: HebrewVerses | None = None, exclude: Iterable[str] = (), forms=None,
                tbesh: dict | None = None) -> NameStage:
    """The stage over the sources directory's TAHOT and TBESH, with a model
    fitted to TBESG's gold names, less those in `exclude` (Greek numbers),
    whose TBESG relations the stage will not use either."""
    exclude = set(exclude)
    tbesh = tbesh if tbesh is not None else hebrew_names.read_tbesh(sources / "TBESH.txt")
    hebrew = hebrew or hebrew_names.build(sources)
    every = gold_pairs(lexicon, tbesh)
    pairs = [p for p in every if p.greek_strongs not in exclude]
    background, stop = translit.null_model(p.letters for p in every)
    model = train(pairs, background, stop, (), settings.iterations)
    stage = NameStage(hebrew, verse_map, lexicon, model, settings, key_mode, codes, tbesh, exclude)
    stage.gold, stage.background, stage.stop = pairs, background, stop
    stage.forms = forms
    return stage


def self_train(stage: NameStage, verses: Sequence[VerseView], gold: set[str] | None = None,
               exclude_keys: Iterable[str] = ()) -> int:
    """Refit the model to the gold pairs and the stage's own confident
    matches of tier-d names (`Settings.rounds` times): a Greek spelling and
    the Hebrew name it matched in its verse, each distinct pair once.
    Matches of Hebrew numbers in `exclude_keys` (the held-out gold names')
    are not learnt from. Returns the number of pairs learnt from."""
    exclude_keys = set(exclude_keys)
    learnt = 0
    for _ in range(stage.settings.rounds):
        found, _, _ = stage.run(verses, gold)
        extra: dict[tuple, tuple] = {}
        for (_, _), (match, unit) in found.items():
            if unit.kind != "candidate" or match.method != "hebrew-verse":
                continue
            if stage.unified(match.hebrew) & exclude_keys:
                continue
            if not stage.confident(match):
                continue
            if stage.tbesg_entry(unit.letters, match.hebrew) is not None:
                continue
            word = match.hebrew
            pair = (unit.letters, word.kind, sound_variants(match.spelling or word.lemma, word.kind == "gentilic"))
            extra[pair] = pair
        stage.model = train(stage.gold, stage.background, stage.stop, sorted(extra), stage.settings.iterations)
        learnt = len(extra)
    return learnt


# ---------------------------------------------------------------------------
# Validation on TBESG's gold names, and the confidence it calibrates

BANDS = (0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 20.0)


def score_band(score: float) -> str:
    lower = max(b for b in BANDS if score >= b) if score >= 0 else 0.0
    upper = next((b for b in BANDS if b > lower), None)
    return f"{lower:g}-{upper:g}" if upper is not None else f">={lower:g}"


@dataclass
class GoldResult:
    tokens: int = 0
    ceiling: int = 0                       # gold's Hebrew number in the verses searched
    matched: Counter = field(default_factory=Counter)      # method -> tokens
    correct: Counter = field(default_factory=Counter)
    by_band: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(Counter)))   # method -> band -> matched/correct
    names: int = 0
    errors: Counter = field(default_factory=Counter)        # (greek, matched gloss, gold) -> tokens
    forms_correct: Counter = field(default_factory=Counter)  # method -> distinct (form, number) correct
    forms_matched: Counter = field(default_factory=Counter)
    # The same for the names whose Greek transliterates the Hebrew at all
    # (not Αἴγυπτος for מִצְרַיִם, which no transliteration finds).
    transliterated: Counter = field(default_factory=Counter)  # tokens, matched, correct

    def precision(self, method: str | None = None) -> float:
        matched = sum(self.matched.values()) if method is None else self.matched[method]
        correct = sum(self.correct.values()) if method is None else self.correct[method]
        return correct / matched if matched else 0.0

    def to_json(self) -> dict:
        matched, correct = sum(self.matched.values()), sum(self.correct.values())
        return {
            "heldOutNames": self.names,
            "tokens": self.tokens,
            "ceiling": self.ceiling,
            "ceilingPercent": round(100 * self.ceiling / self.tokens, 2) if self.tokens else 0.0,
            "matched": matched,
            "correct": correct,
            "coveragePercent": round(100 * matched / self.tokens, 2) if self.tokens else 0.0,
            "precisionPercent": round(100 * correct / matched, 2) if matched else 0.0,
            "byMethod": {method: {"matched": self.matched[method], "correct": self.correct[method],
                                  "precisionPercent": round(100 * self.precision(method), 2),
                                  "distinctForms": self.forms_matched[method],
                                  "distinctFormsCorrect": self.forms_correct[method]}
                         for method in METHODS if self.matched[method]},
            "byScoreBand": {method: {band: dict(counts) for band, counts in sorted(bands.items())}
                            for method, bands in sorted(self.by_band.items())},
            "transliteratedNames": {
                "tokens": self.transliterated["tokens"], "matched": self.transliterated["matched"],
                "correct": self.transliterated["correct"],
                "coveragePercent": round(100 * self.transliterated["matched"] / self.transliterated["tokens"], 2)
                if self.transliterated["tokens"] else 0.0,
                "precisionPercent": round(100 * self.transliterated["correct"] / self.transliterated["matched"], 2)
                if self.transliterated["matched"] else 0.0,
            },
            "commonestErrors": [{"greek": g, "matched": m, "tbesg": t, "tokens": n}
                                for (g, m, t), n in self.errors.most_common(25)],
        }


def evaluate_gold(stage_factory, verses: Sequence[VerseView], greek_to_hebrew: dict[str, set[str]]
                  ) -> tuple[GoldResult, NameStage]:
    """Run the matcher blind on the held-out gold names' occurrences (see
    `held_out`), with a stage that has not seen them: `stage_factory(exclude)`
    builds a stage whose model and TBESG relations leave them out, and its
    self-training learns nothing of their Hebrew numbers."""
    gold = {g for g in greek_to_hebrew if held_out(g)}
    stage = stage_factory(gold)
    exclude_keys = set().union(*(greek_to_hebrew[g] for g in gold)) if gold else set()
    self_train(stage, verses, gold, exclude_keys)
    found, _, consistent = stage.run(verses, gold)
    everything = {**found, **consistent}
    everything.update(associate(stage, verses, everything, gold))
    result = GoldResult(names=len(gold))
    prior = translit.initial(stage.background, stage.stop)
    readable = {p.greek_strongs for p in transliterations(
        [p for p in gold_pairs(stage.lexicon, stage.tbesh) if p.greek_strongs in gold], prior)}
    matched_forms: dict[str, set] = defaultdict(set)
    correct_forms: dict[str, set] = defaultdict(set)
    for number, verse in enumerate(verses):
        searched = None
        for position, view in enumerate(verse.tokens):
            if stage.role(view, gold) != "gold":
                continue
            expected = greek_to_hebrew[view.strongs]
            result.tokens += 1
            plain = view.strongs in readable
            result.transliterated["tokens"] += plain
            if searched is None:
                refs = ([stage.target(verse.ref)] if stage.target(verse.ref) else []) + [
                    r for r, _ in stage.nearby(verse.ref)]
                searched = set().union(*(stage.unified(w) for r in refs for w in stage.hebrew.names(r))) \
                    if refs else set()
            if expected & searched:
                result.ceiling += 1
            hit = everything.get((number, position))
            if hit is None:
                continue
            match, unit = hit
            ok = bool(stage.unified(match.hebrew) & expected)
            result.matched[match.method] += 1
            result.transliterated["matched"] += plain
            result.transliterated["correct"] += plain and ok
            band = score_band(match.score)
            result.by_band[match.method][band]["matched"] += 1
            matched_forms[match.method].add((unit.letters, match.hebrew.key))
            if ok:
                result.correct[match.method] += 1
                result.by_band[match.method][band]["correct"] += 1
                correct_forms[match.method].add((unit.letters, match.hebrew.key))
            else:
                result.errors[(view.token.surface, f"{match.hebrew.key} {match.hebrew.gloss}",
                               ",".join(sorted(expected)))] += 1
    for method in METHODS:
        result.forms_matched[method] = len(matched_forms[method])
        result.forms_correct[method] = len(correct_forms[method])
    return result, stage


def calibration(result: GoldResult) -> dict[str, list[tuple[float, float]]]:
    """Confidence by method and score band from the gold validation: the
    precision of each band, smoothed (one error and one success added) and
    never falling as the score rises; a method too rare to measure takes the
    one below it, less a twentieth."""
    table: dict[str, list[tuple[float, float]]] = {}
    fallback = None
    for method in METHODS:
        bands = result.by_band.get(method, {})
        rows = []
        best = 0.0
        for lower in BANDS:
            counts = bands.get(score_band(lower), Counter())
            if counts["matched"] < 20:
                continue
            value = (counts["correct"] + 1) / (counts["matched"] + 2)
            best = max(best, value)
            rows.append((lower, round(best, 4)))
        if not rows:
            base = fallback[0][1] if fallback else 0.5
            rows = [(0.0, round(base - 0.05, 4))]
        table[method] = rows
        fallback = rows
    return table


def confidence_from(table: dict[str, list[tuple[float, float]]]):
    def confidence(match: Match) -> float:
        rows = table.get(match.method) or table["hebrew-verse"]
        value = rows[0][1]
        for least, measured in rows:
            if match.score >= least:
                value = measured
        return value
    return confidence


# Before any validation has run (a stage built for a test): the precision
# the held-out gold names measured, by method (see the report's
# names.validation for the current figures).
DEFAULT_CALIBRATION: dict[str, list[tuple[float, float]]] = {
    "hebrew-verse": [(0.0, 0.9), (9.0, 0.95), (12.0, 0.98)],
    "hebrew-order": [(0.0, 0.85)],
    "hebrew-nearby": [(0.0, 0.9), (12.0, 0.95)],
    "hebrew-consistency": [(0.0, 0.85)],
    "hebrew-association": [(0.0, 0.8)],
}


# ---------------------------------------------------------------------------
# Association across verses: names no transliteration finds

# A Greek form is associated with a Hebrew name when, over the verses it
# stands in untagged, that name is in the Hebrew and no other Greek word of
# the verse accounts for it in at least this share of them, in at least
# `ASSOCIATION_MIN` verses, and no other Hebrew name comes near
# (`ASSOCIATION_RIVAL`).
ASSOCIATION_SHARE = 0.9
ASSOCIATION_MIN = 3
ASSOCIATION_RIVAL = 0.5


def associate(stage: NameStage, verses: Sequence[VerseView], everything: dict, gold: set[str] | None = None
              ) -> dict[tuple[int, int], tuple[Match, GreekUnit]]:
    """Names the transliteration model cannot find: Greek-origin or
    translated names (Αἰθιοπία for Cush, Συρία for Aram, Κῦρος for Koresh,
    Μεσοποταμία) and spellings too far from the Hebrew. For each Greek form
    left untagged, the Hebrew names of the verses it stands in that no other
    Greek word of the verse accounts for (not matched, not an anchor's) are
    counted; a form whose verses nearly always have one such name, and no
    rival, takes it wherever its verse has it unaccounted for (method
    `hebrew-association`). Only the verse the map gives is read."""
    occurrences: dict[str, list[tuple[int, list[int], dict[str, HebrewWord]]]] = defaultdict(list)
    used_by_verse: dict[int, set[str]] = defaultdict(set)
    for (v, _), (m, _) in everything.items():
        used_by_verse[v] |= stage.unified(m.hebrew)
    for number, verse in enumerate(verses):
        roles = {i: r for i, view in enumerate(verse.tokens) if (r := stage.role(view, gold))}
        open_positions = [i for i, r in roles.items() if r in ("candidate", "gold") and (number, i) not in everything]
        if not open_positions:
            continue
        target = stage.target(verse.ref)
        if target is None:
            continue
        used = used_by_verse.get(number, set())
        left: dict[str, HebrewWord] = {}
        for word in stage.free(stage.names_of(target), stage.anchored(verse, roles)):
            if not stage.unified(word) & used:
                left.setdefault(word.key, word)
        by_letters: dict[str, list[int]] = defaultdict(list)
        for position in open_positions:
            letters = letters_of(verse.tokens[position].token.key)
            if len(letters) >= 3:
                by_letters[letters].append(position)
        for letters, positions in by_letters.items():
            occurrences[letters].append((number, positions, left))
    added: dict[tuple[int, int], tuple[Match, GreekUnit]] = {}
    for letters, places in occurrences.items():
        if len(places) < ASSOCIATION_MIN:
            continue
        counts: Counter = Counter()
        for _, _, left in places:
            counts.update(left.keys())
        if not counts:
            continue
        ranked = counts.most_common()
        key, count = ranked[0]
        rival = ranked[1][1] if len(ranked) > 1 else 0
        if count < ASSOCIATION_MIN or count < ASSOCIATION_SHARE * len(places) or rival > ASSOCIATION_RIVAL * len(places):
            continue
        for number, positions, left in places:
            word = left.get(key)
            if word is None:
                continue
            surfaces = [verses[number].tokens[p].token.surface for p in positions]
            unit = GreekUnit(letters, list(positions), surfaces, "candidate")
            match = Match(word, stage.score(letters, word), count / len(places), "hebrew-association")
            for position in positions:
                added[(number, position)] = (match, unit)
    return added


# ---------------------------------------------------------------------------
# The whole stage

@dataclass
class NameRun:
    """What the stage decided for a text."""

    decisions: dict                 # (verse, token) -> NameDecision to apply
    notes: dict                     # (verse, token) -> why a candidate is untagged, or a recheck's outcome
    roles: dict                     # (verse, token) -> "candidate" or "recheck"
    found: dict                     # (verse, token) -> (Match, GreekUnit), every match
    gold: GoldResult | None
    calibration: dict
    learnt: int
    stage: NameStage
    verses: Sequence[VerseView] = ()
    relations: dict = field(default_factory=dict)   # (verse, token) -> RELATIONS_KINDS value


def tag_names(verses: Sequence[VerseView], factory, validate: bool = True) -> NameRun:
    """Run the stage over a text: validate on the held-out gold names (which
    calibrates the confidence), fit the model to every gold name and the
    text's own confident matches, match, and decide. `factory(exclude)` builds
    a stage (see `build_stage`).

    Tier-d candidates take their decision. A capitalised word the chooser
    tagged from a tolerant name match keeps that tag unless the Hebrew
    clearly names another: a confident verse-level match of a name other than
    the one TBESG says the tag's entry renders, in a verse whose Hebrew lacks
    that one (Brenton's Ἠλιού, matched to the NT's Ἡλίας, the Greek of
    Elijah, is Elihu in 1 Sam 1:1). An entry TBESG relates to no Hebrew name
    (its own Septuagint entry Ἄδαρ, the NT's Ἀνανίας) has nothing to be
    contradicted by, and stays."""
    gold = None
    table = DEFAULT_CALIBRATION
    stage = factory(set())
    if validate:
        gold, _ = evaluate_gold(factory, verses, stage.greek_to_hebrew)
        table = calibration(gold)
    learnt = self_train(stage, verses)
    found, reasons, consistent = stage.run(verses)
    everything = {**found, **consistent}
    associated = associate(stage, verses, everything)
    for position in associated:
        reasons.pop(position, None)
    everything.update(associated)
    decisions = decide(stage, verses, everything, confidence_from(table))
    applied: dict = {}
    notes: dict = dict(reasons)
    roles: dict = {}
    for number, verse in enumerate(verses):
        for position, view in enumerate(verse.tokens):
            role = stage.role(view)
            if role in ("candidate", "recheck"):
                roles[(number, position)] = role
    for position, decision in decisions.items():
        role = roles.get(position)
        view = verses[position[0]].tokens[position[1]]
        match = everything[position][0]
        if role == "candidate":
            applied[position] = decision
        elif role == "recheck":
            related = stage.greek_to_hebrew.get(view.strongs, set())
            target = stage.target(verses[position[0]].ref)
            present = {k for w in (stage.names_of(target) if target else []) for k in stage.unified(w)}
            if confirms(stage, view, match):
                notes[position] = f"the Hebrew ({match.hebrew.key} {match.hebrew.gloss}) confirms the tolerant match"
            elif not related:
                notes[position] = (f"the Hebrew suggests {match.hebrew.key} {match.hebrew.gloss}; TBESG gives the "
                                   "tolerant match no Hebrew name to contradict, so it is kept")
            elif stage.confident(match) and not related & present:
                applied[position] = decision
                notes[position] = f"replaces the tolerant match {view.strongs} {view.lemma}"
            else:
                notes[position] = (f"the Hebrew suggests {match.hebrew.key} {match.hebrew.gloss}, "
                                   "not surely enough to replace the tolerant match")
    for position, role in roles.items():
        if role == "recheck" and position not in notes:
            notes[position] = "no Hebrew name confirms or contradicts the tolerant match"
    run = NameRun(applied, notes, roles, everything, gold, table, learnt, stage, verses)
    run.relations = name_relations(run)
    return run


# ---------------------------------------------------------------------------
# Whether a tagged name transliterates the Hebrew name or stands for it

RELATION_KINDS = ("transliterates", "stands-for")
# A form "elsewhere renders another name" when at least this share of its
# confident verse-level matches are to one other Hebrew name (and two or more).
OTHER_NAME_SHARE = 0.8


# A form whose transliteration score against the name it is tagged with is
# below this reads as no rendering of it at all.
POOR_READING = 1.0


def name_relations(run: NameRun) -> dict[tuple[int, int], str]:
    """For every name the stage tagged, whether the Greek word transliterates
    the Hebrew name at that position or stands for it there without
    transliterating it (the Septuagint prints another name: 2 Chr 24:20
    Ἀζαρίαν where the Hebrew has Zechariah).

    The rule. A Greek form *stands for* the Hebrew name when
      * it was found by association (`associate`): no transliteration finds
        such a name (Ἀρταξέρξης for Ahasuerus, Αἰθιοπία for Cush);
      * or its transliteration score against that name is below
        `Settings.threshold` (only a list's order places such a word) and
        either below `POOR_READING`, or another Hebrew name of the chapter
        (and the chapters either side) reads at least `Settings.margin`
        better: the Greek is that other name (Ἀζαρίᾳ where 2 Kgs 15:13 has
        Uzziah, Μαρισά for Mesha). A weak reading with no better rival is a
        loose transliteration (Σαυά for Shua, Αἰνάν for Enaim);
      * or the same form's confident matches elsewhere are to another Hebrew
        name (`OTHER_NAME_SHARE` of them, two or more) which it reads at least
        `Settings.margin` better than this one: the Septuagint uses this
        spelling for that other person or place (Ἀβιμέλεχ for Ahimelech,
        Ἀμεσσαΐ for Abishai).
    A low score is excused when the form renders the same name (by key, or
    by TAHOT's gloss: the Aramaic Ezra is the Hebrew Ezra) in two other
    verses or more by transliteration: then it is the Septuagint's regular
    spelling of the name (Ἔσδρας, Μελχόλ for Michal, Ναυή for Nun).
    Otherwise it *transliterates* it. A form matched to one word of a Hebrew
    name of several (Μωάβ in Pahath-moab) is judged by its score alone.
    This describes the tag; it changes no tag."""
    stage = run.stage

    def identity(w) -> frozenset:
        return frozenset(stage.unified(w)) | {"gloss:" + w.gloss}

    confident: dict[str, Counter] = defaultdict(Counter)
    words: dict[tuple[str, frozenset], HebrewWord] = {}
    verses_of: dict[tuple[str, frozenset], set[int]] = defaultdict(set)
    for (verse, _), (match, unit) in run.found.items():
        keys = identity(match.hebrew)
        if not match.spelling and match.method != "hebrew-association":
            verses_of[(unit.letters, keys)].add(verse)
        if not match.spelling and stage.confident(match):
            confident[unit.letters][keys] += 1
            words.setdefault((unit.letters, keys), match.hebrew)
    by_identity: dict[str, set[int]] = defaultdict(set)
    for (letters, keys), numbers in verses_of.items():
        for key in keys:
            by_identity[(letters, key)] |= numbers
    out: dict[tuple[int, int], str] = {}
    for position in run.decisions:
        found = run.found.get(position)
        if found is None:
            continue
        match, unit = found
        mine = identity(match.hebrew)
        relation = "transliterates"
        if match.method == "hebrew-association":
            relation = "stands-for"
        elif match.score < stage.settings.threshold:
            others = set().union(*(by_identity.get((unit.letters, key), set()) for key in mine)) - {position[0]}
            if match.spelling or len(others) < 2:
                if match.score < POOR_READING or _better_rival(stage, run, position[0], unit.letters, mine,
                                                              match.score + stage.settings.margin):
                    relation = "stands-for"
        elif not match.spelling:
            counts = confident.get(unit.letters, Counter())
            total = sum(counts.values())
            for keys, n in counts.items():
                if keys & mine or total < 2 or n < OTHER_NAME_SHARE * total:
                    continue
                if stage.score(unit.letters, words[(unit.letters, keys)]) >= match.score + stage.settings.margin:
                    relation = "stands-for"
        out[position] = relation
    return out


def _better_rival(stage, run: NameRun, verse: int, letters: str, mine: frozenset, least: float) -> bool:
    """Whether a Hebrew name of the verse's chapter (and those either side),
    other than the tagged one, reads at least `least` as this Greek form."""
    if not run.verses:
        return False
    ref = run.verses[verse].ref
    refs = [r for r in ([stage.target(ref)] if stage.target(ref) else [])]
    refs += [r for r, _ in stage.nearby(ref)] + [r for r, _ in stage.chapter(ref)]
    seen: set[str] = set()
    for r in refs:
        for word in stage.names_of(r):
            if word.key in seen or stage.unified(word) & mine or ("gloss:" + word.gloss) in mine:
                continue
            seen.add(word.key)
            if stage.score(letters, word) >= least:
                return True
    return False


def views_from_results(results) -> list[VerseView]:
    """The stage's view of `tag_septuagint`'s results (book, chapter, verse
    record, [(TokenCandidates, Choice or None)])."""
    verses = []
    for _, _, verse, outcome in results:
        views = []
        for item, choice in outcome:
            chosen = choice.candidate if choice else None
            views.append(TokenView(
                item.token, item.tier, item.reason,
                any(candidate.kind == "common" for candidate, _ in item.set_aside),
                chosen.strongs if chosen else "", chosen.lemma if chosen else "",
                (chosen.morph or "") if chosen else "", choice.method if choice else ""))
        verses.append(VerseView(verse["ref"], views))
    return verses


def untagged_category(view: TokenView, note: str) -> str:
    """The report's reason for a tier-d candidate left untagged."""
    if view.set_aside_common and note.startswith(("no matching", "no Hebrew name")):
        return "a Greek word Brenton capitalises (a common reading was set aside), no Hebrew name"
    return note.split(";")[0]


def report(run: NameRun, top: int = 15) -> dict:
    """The report's `names` section (see `tag_septuagint`)."""
    stage, verses = run.stage, run.verses
    candidates = [(position, verses[position[0]].tokens[position[1]])
                  for position, role in run.roles.items() if role == "candidate"]
    recheck = [position for position, role in run.roles.items() if role == "recheck"]
    tagged = {position: decision for position, decision in run.decisions.items()}
    by_method: Counter = Counter(decision.method for decision in tagged.values())
    by_source: Counter = Counter(decision.key_source for decision in tagged.values())
    bands: dict[str, Counter] = defaultdict(Counter)
    for decision in tagged.values():
        bands[decision.method][score_band(decision.score)] += 1
    untagged: dict[str, Counter] = defaultdict(Counter)
    for position, view in candidates:
        if position in tagged:
            continue
        untagged[untagged_category(view, run.notes.get(position, "no matching Hebrew name"))][
            display_form(view.token.surface)] += 1
    recheck_outcomes: Counter = Counter()
    for position in recheck:
        note = run.notes.get(position, "")
        recheck_outcomes["replaced" if position in tagged else
                         "confirmed" if "confirms" in note else
                         "kept, TBESG gives it no Hebrew name" if "nothing to contradict" in note or
                         "no Hebrew name to contradict" in note else
                         "kept, not sure" if "not surely" in note else "kept, no Hebrew name"] += 1
    numbers: dict[str, Counter] = defaultdict(Counter)
    for (v, p), decision in tagged.items():
        numbers[display_form(verses[v].tokens[p].token.surface)][decision.strongs] += 1
    several = sorted(((sum(c.values()), form, c) for form, c in numbers.items() if len(c) > 1), reverse=True)
    before_forms = {display_form(view.token.surface) for _, view in candidates}
    after_forms = {display_form(view.token.surface) for position, view in candidates if position not in tagged}
    return {
        "keyMode": stage.key_mode,
        "sourceNote": ("Proper names the New Testament does not use are identified from STEPBible's tagged "
                       "Hebrew Old Testament (TAHOT and TBESH, CC BY 4.0): each carries the Strong's number "
                       "of the Hebrew name it renders in the same verse."),
        "methods": {
            "hebrew-verse": "matched to a name of the Hebrew verse the Septuagint's verse corresponds to",
            "hebrew-order": "placed by a list's order between matched names of that verse",
            "hebrew-nearby": "matched to a name of a neighbouring Hebrew verse, or of the same chapter",
            "hebrew-consistency": "the number the same Greek form takes everywhere it is matched",
            "hebrew-association": ("the Hebrew name the verses of an untagged Greek form nearly always have and "
                                   "no other Greek word of them accounts for (names no transliteration finds)"),
        },
        "tierD": {
            "before": {"tokens": len(candidates), "forms": len(before_forms)},
            "after": {"tokens": sum(1 for position, _ in candidates if position not in tagged),
                      "forms": len(after_forms)},
            "note": "candidates include capitalised tier-e words (see tierE)",
        },
        "tierE": {
            "candidates": sum(1 for _, view in candidates if view.tier == "e"),
            "tagged": sum(1 for position, view in candidates if view.tier == "e" and position in tagged),
        },
        "tagged": {
            "tokens": len(tagged),
            "fromTierD": sum(1 for position, _ in candidates if position in tagged),
            "byMethod": {m: by_method[m] for m in METHODS if by_method[m]},
            "byKey": {"hebrew (TAHOT)" if k == "tahot" else "NT name's Greek number (TBESG relation)": n
                      for k, n in sorted(by_source.items())},
            "byScoreBand": {m: dict(sorted(c.items())) for m, c in sorted(bands.items())},
        },
        "tolerantNameTags": {"checked": len(recheck), **dict(sorted(recheck_outcomes.items()))},
        "untagged": {
            reason: {"tokens": sum(c.values()), "forms": len(c),
                     "top": [{"form": f, "tokens": n} for f, n in sorted(c.items(), key=lambda i: (-i[1], i[0]))[:top]]}
            for reason, c in sorted(untagged.items(), key=lambda i: (-sum(i[1].values()), i[0]))
        },
        "consistency": {
            "forms": len(numbers),
            "oneNumber": sum(1 for c in numbers.values() if len(c) == 1),
            "severalNumbers": len(several),
            "largestWithSeveral": [{"form": form, "tokens": n, "numbers": dict(c.most_common())}
                                   for n, form, c in several[:top]],
        },
        "validation": run.gold.to_json() if run.gold else None,
        "calibration": {m: [{"fromScore": a, "confidence": b} for a, b in rows] for m, rows in run.calibration.items()},
        "learntFromOwnMatches": run.learnt,
        "relations": {
            "description": ("transliterates: the Greek word reads as the Hebrew name it is tagged with; "
                           "stands-for: the Septuagint prints another name at that position and the tag "
                           "says which Hebrew name it stands for: found by association, or a reading "
                           "below the verse threshold that is very poor or that another name of the "
                           "chapter reads better, or a form that elsewhere renders another Hebrew name "
                           "it reads better (names.name_relations)"),
            **{kind: sum(1 for r in run.relations.values() if r == kind) for kind in RELATION_KINDS},
            "standsForByMethod": dict(sorted(Counter(
                run.decisions[p].method for p, r in run.relations.items() if r == "stands-for").items())),
        },
        "settings": {k: v for k, v in sorted(vars(stage.settings).items())},
    }
