"""Tokenising Brenton's Septuagint and normalising its words to lookup keys.

Every equivalence the tagger applies between a printed word and a form of the
tagged Greek New Testament is defined here, so there is one place to read what
counts as "the same form" and why. There are three kinds:

  * Key rules fold differences that never distinguish two words. They are
    applied to both sides, Brenton's words and TAGNT's, so a key is a key.
  * Alternates are other keys tried only when a word's own key finds nothing,
    each with a licence: the reading the alternate finds must be of the kind
    the rule is about (a movable nu is accepted only on a verb's third person,
    a dative plural, or a -θεν adverb), so an alternate cannot turn one word
    into a different one.
  * Capitals' patterns: a word printed in capitals (chapter openings, titles,
    the Hebrew letters heading the acrostics) has lost the accents after its
    first letter, so it is looked up by letters alone, and the readings of every
    accentuation of those letters are kept, which leaves a genuinely lost
    distinction ambiguous rather than guessed.

KEY RULES (applied in this order; `Token.rules` names those that changed a word)

  editorial-marks  A dot below or macron is an editor's mark, not spelling:
                   removed. (Brenton has one dot below, Hab 3:1 Ὠ̣ΔΗΣ.)
  apostrophe-breathing
                   A word-initial apostrophe before a capital vowel or rho
                   (ʼΑγὰγ, ʼΑμαλὴκ, about 60 words) is how the digitisation
                   printed the breathing of a capital: it becomes a smooth
                   breathing on that letter, and only the capitals' pattern,
                   which ignores a breathing it cannot trust, is used for it.
  case             Lower-cased. Capitals are not lost: `Token.case` and
                   `Token.position` keep what the capital says (see below).
  elision          Every apostrophe-like mark at the end of a word (Brenton's
                   U+02BC ʼ, TAGNT's koronis U+1FBD, ’ and ') becomes U+2019.
  grave-tis        A grave on monosyllabic τις/τι (τὶς, τὶ) marks the
                   indefinite pronoun, never the interrogative, whose acute does
                   not turn grave; it becomes the unaccented τις/τι rather than
                   being folded into interrogative τίς/τί by the next rule.
  grave            Grave becomes acute: a grave is an oxytone's accent before
                   another word and says nothing about which word it is.
  enclitic-accent  A second accent, thrown back onto a word by the enclitic
                   after it (ὄνομά μου, δίδωσί σοι), is removed; no Greek word
                   has two accents of its own. The enclitic's own accent is
                   kept: accented and unaccented enclitics (τίς/τις) differ.
  rho-breathing    Breathings on rho are removed. Brenton prints ῤῥ inside
                   words (1,005 times) and the New Testament ρρ, and initial ῥ
                   is always rough, so the mark never distinguishes words.
  diaeresis        Removed: it marks pronunciation, not a different word
                   (Brenton's Μωυσῆς, the New Testament's Μωϋσῆς).
  crasis-subscript In crasis Brenton writes καί + ἐγώ as κᾀγώ, with an iota
                   subscript under the vowel bearing the coronis, where the New
                   Testament writes κἀγώ; the subscript is removed from a
                   non-initial vowel that carries a breathing (a coronis).
  final-sigma      ς is folded to σ.

  Kept distinct, because these marks do distinguish words: breathings on
  vowels (ὁ/ὅ, ἡ/ἥ), the placing and kind of accent (ἤ/ἦ/ᾗ, εἰς/εἷς, ἐν/ἕν,
  αὐτή/αὕτη, τίς/τις, ἄρα/ἆρα, ἀλλά/ἄλλα), and iota subscript (ἡ/ᾗ).

ALTERNATES (tried only when the key itself finds nothing; see `alternates`)

  movable-nu       A final ν after -σι (ξι, ψι), after -ε, and on ἐστι, with
                   and without: εἶπε/εἶπεν, ἐστί/ἐστίν, λέγουσι/λέγουσιν,
                   πᾶσι/πᾶσιν, ἔμπροσθε/ἔμπροσθεν. Not after -ει (λέγει is
                   not λέγειν). Licensed for a verb's third person, a dative
                   plural, εἴκοσι and πέρυσι, and adverbs and prepositions in
                   -θε(ν).
  movable-sigma    οὕτω/οὕτως, ἄχρι/ἄχρις, μέχρι/μέχρις. Licensed when the
                   reading's lemma is that word.
  ek-ex            ἐκ/ἐξ. Licensed when the lemma is ἐκ.
  ou-ouk-ouch      οὐ/οὐκ/οὐχ. Licensed when the lemma is οὐ. (οὐχί is a
                   different entry, G3780, and is not among them.)
  infinitive-subscript
                   Brenton writes the present infinitive of -άω verbs with an
                   iota subscript (ὁρᾷν, ἀγαπᾷν) where the New Testament has
                   ὁρᾶν, ἀγαπᾶν. Licensed for an infinitive.
  lemps            λήψομαι, ἐλήφθη, ἀντίληψις are Attic spellings of the New
                   Testament's λήμψομαι, ἐλήμφθη, ἀντίλημψις: a μ is inserted
                   before ψ or φ after λη. Licensed when the lemma is a
                   compound of λαμβάνω or a noun in -λημψις.

JOINED WORDS (split before lookup; see `JOINED`)

  Brenton's orthography prints some phrases as one word (διατοῦτο, ἀναμέσον,
  ἱνατί). Each listed joined word is split into its parts, each a token of
  its own with its own character span inside the printed word, so each part
  gets its own annotation. The list is explicit: a general splitter would cut
  real compounds (διαφθορά is not διά + φθορά).

MISPRINTS (see `MISPRINTS`)

  A few words of the digitised text are misprints no rule can read (γἱοὶ for
  υἱοί, Δέγει for Λέγει, θυλάτηρ for θυγάτηρ). They are never corrected
  silently: each is listed, printed form and intended form, after reading it
  in its verse, and a token whose key the table changes carries the rule
  "misprint" (the decisions file's `reason` says which correction was
  assumed). The printed text itself is never changed. A form not certainly
  a misprint is not listed and stays untagged.

CAPITALS AND POSITION

  `Token.case` is "lower", "capital" (first letter only) or "caps" (all
  letters of a word of two or more). `Token.position` is "initial" (first word
  of the verse, or after . ; or an opening quotation mark or bracket),
  "pause" (after , or ·) or "medial" (after a space alone). Brenton
  capitalises the first word of a sentence and of most verses, proper names,
  and the titles of God (Θεός, Κύριος, Ὕψιστος). So a capital in medial
  position is informative: the word is a name or a title (`name_like`). After
  a pause it is not: Brenton also capitalises the first word of quoted speech
  after a comma (εἶπεν, Ἰδοὺ ...), and names follow commas in lists. At the
  start of a sentence a capital says nothing. How the candidate providers use
  this is described in `candidates.py`.
"""

from __future__ import annotations

import functools
import re
import unicodedata
from dataclasses import dataclass, field

ACUTE, GRAVE, CIRCUMFLEX = "\u0301", "\u0300", "\u0342"
SMOOTH, ROUGH = "\u0313", "\u0314"
DIAERESIS, SUBSCRIPT = "\u0308", "\u0345"
ACCENTS = frozenset((ACUTE, GRAVE, CIRCUMFLEX))
BREATHINGS = frozenset((SMOOTH, ROUGH))
# Marks an editor adds and spelling does not have.
EDITORIAL = frozenset(("\u0323", "\u0304", "\u0306"))   # dot below, macron, breve
ELISION = "\u2019"
# Brenton's ʼ (U+02BC), TAGNT's koronis (U+1FBD), psili (U+1FBF), and the
# typographic and plain apostrophes.
APOSTROPHES = frozenset(("\u02bc", "\u2019", "\u0027", "\u2018", "\u1fbd", "\u1fbf"))
VOWELS = frozenset("αεηιουω")

# Punctuation before a word that makes it the start of a sentence, and
# punctuation that is only a pause. NFC turns the Greek question mark (U+037E)
# into ";" and the ano teleia (U+0387) into "·".
SENTENCE_BREAKS = frozenset(".;!?“‘([")
PAUSES = frozenset(",·:")


def _decompose(word: str) -> list[list[str]]:
    """A word as [base letter, *marks] clusters, in NFD."""
    clusters: list[list[str]] = []
    for char in unicodedata.normalize("NFD", word):
        if unicodedata.combining(char) and clusters:
            clusters[-1].append(char)
        else:
            clusters.append([char])
    return clusters


def _compose(clusters: list[list[str]]) -> str:
    return unicodedata.normalize("NFC", "".join("".join(cluster) for cluster in clusters))


@functools.lru_cache(maxsize=1 << 17)
def normalize_with_rules(word: str, skip: frozenset[str] = frozenset()) -> tuple[str, tuple[str, ...]]:
    """The lookup key of a printed word, and the key rules that changed it.

    See the module docstring for the rules. The key is NFC, lower case, with
    σ for every sigma and U+2019 for an elision mark. `skip` names rules to
    leave out, which is only for measuring what each rule contributes.
    """
    rules: list[str] = []
    clusters = _decompose(word)

    if "editorial-marks" not in skip and any(mark in EDITORIAL for cluster in clusters for mark in cluster[1:]):
        rules.append("editorial-marks")
        clusters = [[cluster[0], *(m for m in cluster[1:] if m not in EDITORIAL)] for cluster in clusters]

    if ("apostrophe-breathing" not in skip and len(clusters) > 1 and clusters[0][0] in APOSTROPHES and clusters[1][0].isupper()
            and clusters[1][0].lower() in VOWELS | {"ρ"}
            and not any(mark in BREATHINGS for mark in clusters[1][1:])):
        rules.append("apostrophe-breathing")
        clusters = [[clusters[1][0], SMOOTH, *clusters[1][1:]], *clusters[2:]]

    lowered = clusters if "case" in skip else [[cluster[0].lower(), *cluster[1:]] for cluster in clusters]
    if lowered != clusters:
        rules.append("case")
    clusters = lowered

    if "elision" not in skip and clusters and clusters[-1][0] in APOSTROPHES:
        if clusters[-1][0] != ELISION:
            rules.append("elision")
        clusters[-1] = [ELISION]

    letters = "".join(cluster[0] for cluster in clusters)
    if "grave-tis" not in skip and letters in ("τις", "τι") and any(GRAVE in cluster for cluster in clusters):
        rules.append("grave-tis")
        clusters = [[cluster[0], *(m for m in cluster[1:] if m not in ACCENTS)] for cluster in clusters]

    if "grave" not in skip and any(GRAVE in cluster for cluster in clusters):
        rules.append("grave")
        clusters = [[cluster[0], *(ACUTE if m == GRAVE else m for m in cluster[1:])] for cluster in clusters]

    accented = [index for index, cluster in enumerate(clusters) if any(m in ACCENTS for m in cluster[1:])]
    if "enclitic-accent" not in skip and len(accented) > 1:
        rules.append("enclitic-accent")
        for index in accented[1:]:
            clusters[index] = [clusters[index][0], *(m for m in clusters[index][1:] if m not in ACCENTS)]

    if "rho-breathing" not in skip and any(cluster[0] == "ρ" and len(cluster) > 1 for cluster in clusters):
        rules.append("rho-breathing")
        clusters = [[c[0], *(m for m in c[1:] if m not in BREATHINGS)] if c[0] == "ρ" else c for c in clusters]

    if "diaeresis" not in skip and any(DIAERESIS in cluster for cluster in clusters):
        rules.append("diaeresis")
        clusters = [[cluster[0], *(m for m in cluster[1:] if m != DIAERESIS)] for cluster in clusters]

    for index, cluster in enumerate(clusters[1:], 1):
        if "crasis-subscript" not in skip and SUBSCRIPT in cluster and any(m in BREATHINGS for m in cluster[1:]):
            rules.append("crasis-subscript")
            clusters[index] = [cluster[0], *(m for m in cluster[1:] if m != SUBSCRIPT)]

    if "final-sigma" not in skip:
        clusters = [["σ" if cluster[0] == "ς" else cluster[0], *cluster[1:]] for cluster in clusters]
    return _compose(clusters), tuple(rules)


def normalize(word: str) -> str:
    """The lookup key of a printed word. See `normalize_with_rules`."""
    return normalize_with_rules(word)[0]


@functools.lru_cache(maxsize=1 << 17)
def bare(key: str) -> str:
    """Letters alone: every accent, breathing, diaeresis and subscript removed."""
    return "".join(cluster[0] for cluster in _decompose(key))


def other_breathing(key: str) -> str | None:
    """The key with its initial vowel's breathing reversed (ἁβραάμ -> ἀβραάμ),
    or None if it does not start with a breathing. Only for names, whose
    breathings vary by edition: in other words the breathing distinguishes
    (ὁ/ὅ, ἡ/ἥ)."""
    clusters = _decompose(key)
    if not clusters or clusters[0][0] not in VOWELS:
        return None
    # A diphthong's breathing sits on its second vowel (αὐ, οἱ, εἰ).
    index = 1 if len(clusters) > 1 and any(m in BREATHINGS for m in clusters[1][1:]) else 0
    marks = clusters[index][1:]
    if SMOOTH in marks:
        swap = ROUGH
    elif ROUGH in marks:
        swap = SMOOTH
    else:
        return None
    clusters[index] = [clusters[index][0], *(swap if m in BREATHINGS else m for m in marks)]
    return _compose(clusters)


def caps_pattern(key: str, first_marks_known: bool) -> str:
    """The pattern a word printed in capitals is looked up by.

    In capitals Brenton prints the breathing and accent of an initial vowel
    before it (ἘΝ, ἜΤΟΥΣ, ὍΡΑΣΙΣ) and no other accent, so the pattern is the
    letters alone, plus the initial vowel's marks when they were printed.
    `first_marks_known` is false for a capital word with no mark at all on its
    first letter (ΑΛΕΦ, ΤΟΥ), whose initial breathing, if any, is lost too.
    """
    clusters = _decompose(key)
    letters = "".join(cluster[0] for cluster in clusters)
    if not first_marks_known or not clusters or clusters[0][0] not in VOWELS:
        return letters
    marks = "".join(sorted(m for m in clusters[0][1:] if m in ACCENTS | BREATHINGS))
    return f"{letters}|{marks}"


# ---------------------------------------------------------------------------
# Alternates

@dataclass(frozen=True)
class Alternate:
    """Another key to try when a word's own key finds nothing."""

    key: str
    rule: str


_MOVABLE_SIGMA = {"ουτω": "ουτωσ", "αχρι": "αχρισ", "μεχρι": "μεχρισ"}
_EK = ("ἐκ", "ἐξ")
_OU = ("οὐ", "οὐκ", "οὐχ")


def _takes_movable_nu(letters: str) -> bool:
    if letters.endswith(("σι", "ξι", "ψι")) or letters == "εστι":
        return True
    return letters.endswith("ε") and not letters.endswith("ει")


def alternates(key: str) -> list[Alternate]:
    """The alternate keys for `key`, each named by its rule, in a fixed order."""
    found: list[Alternate] = []
    letters = bare(key)
    # A movable nu follows -σι (ξι, ψι), a third person's -ε, and ἐστι; not
    # -ει, where λέγει and λέγειν are different forms.
    if _takes_movable_nu(letters):
        found.append(Alternate(key + "ν", "movable-nu"))
    if letters.endswith("ν") and len(letters) > 3 and _takes_movable_nu(letters[:-1]):
        found.append(Alternate(key[:-1], "movable-nu"))
    for short, long in _MOVABLE_SIGMA.items():
        if letters == short:
            found.append(Alternate(key + "σ", "movable-sigma"))
        elif letters == long:
            found.append(Alternate(key[:-1], "movable-sigma"))
    if key in _EK:
        found.extend(Alternate(other, "ek-ex") for other in _EK if other != key)
    if key in _OU:
        found.extend(Alternate(other, "ou-ouk-ouch") for other in _OU if other != key)
    if key.endswith("ᾷν"):
        found.append(Alternate(key[:-2] + "ᾶν", "infinitive-subscript"))
    nfd = unicodedata.normalize("NFD", key)
    inserted = re.sub(r"λη([\u0301\u0342]?)([ψφ])", r"λη\1μ\2", nfd)
    if inserted != nfd:
        found.append(Alternate(unicodedata.normalize("NFC", inserted), "lemps"))
    return found


def licensed(rule: str, alternate_key: str, lemma: str, morph: str) -> bool:
    """Whether a reading found through an alternate is of the kind its rule is about."""
    lemma_letters = bare(normalize(lemma.split(",")[0].strip())) if lemma else ""
    if rule == "movable-nu":
        if re.match(r"V-.*-3[SP]$", morph) or re.search(r"-DP[MFN]?\b", morph):
            return True
        if lemma_letters in ("εικοσι", "περυσι"):
            return True
        return morph.split("-")[0] in ("ADV", "PREP") and bare(alternate_key).endswith(("θε", "θεν"))
    if rule == "movable-sigma":
        return lemma_letters in set(_MOVABLE_SIGMA) | set(_MOVABLE_SIGMA.values())
    if rule == "ek-ex":
        return lemma_letters == "εκ"
    if rule == "ou-ouk-ouch":
        return lemma_letters == "ου"
    if rule == "infinitive-subscript":
        return morph.startswith("V-") and morph.endswith("N")
    if rule == "lemps":
        return lemma_letters.endswith("λαμβανω") or lemma_letters.endswith("λημψισ")
    return False


# ---------------------------------------------------------------------------
# Joined words

# Brenton's joined spellings, as key -> the keys of the parts. Each part is
# accented as the separate word is in the New Testament (διὰ τοῦτο); the
# printed spans are found by counting letters. Counts are occurrences in
# Brenton's Hebrew-canon books.
JOINED: dict[str, tuple[str, ...]] = {}


def _joined(*parts: str) -> None:
    keys = tuple(normalize(part) for part in parts)
    # Keyed on letters alone: which accents Brenton keeps on a run-together
    # word (ἱνατί, διατοῦτο) varies, and no other word has these letters.
    JOINED[bare("".join(keys))] = keys


for _parts in (
    ("διὰ", "τοῦτο"),       # διατοῦτο ×200
    ("ἀνὰ", "μέσον"),       # ἀναμέσον ×246
    ("ἵνα", "τί"),          # ἱνατί ×134 (TAGNT joins it too, with a compound tag)
    ("διὰ", "παντός"),      # διαπαντός ×98
    ("τὸ", "πρωΐ"),         # τοπρωΐ ×64
    ("διὰ", "τί"),          # διατί ×45
    ("ἐπὶ", "τὸ", "αὐτό"),  # ἐπιτοαυτό ×31
    ("τὸ", "τάχος"),        # τοτάχος ×6 "quickly" (Num 17:11, Deut 7:4)
    ("πρὶν", "ἤ"),          # πρινή ×6 "before" (Gen 29:26, Isa 7:15)
    ("τὸ", "πρότερον"),     # τοπρότερον ×5 "formerly" (Deut 2:20, Josh 11:10)
    ("εἴ", "τις"),          # εἴτις ×4 "if any(one)" (Gen 19:12, Exod 33:11)
    ("εἴ", "τι"),           # εἴτι ×1 (Job 14:17)
    ("τὸ", "δεύτερον"),     # τοδεύτερον ×3 "a second time"
    ("τὸ", "ἑσπέρας"),      # τοεσπέρας ×3 "at evening"
    ("τὸ", "αὐτό"),         # τοαυτό ×2 "together"
    ("τὸ", "δείλης"),       # τοδείλης ×2 "in the afternoon"
):
    _joined(*_parts)


def split_joined(surface: str) -> list[tuple[int, int, str]] | None:
    """A joined word's parts as (start, end, key) inside `surface`, or None.

    Matched on letters alone, so Brenton's accentuation of the run-together
    word (ἱνατί, Ἱνατί, διατί) does not matter; each part's key is the
    separate word's. Offsets count code points of `surface`, marks included.
    """
    pieces = _split_joined(surface)
    return list(pieces) if pieces is not None else None


@functools.lru_cache(maxsize=1 << 16)
def _split_joined(surface: str) -> tuple[tuple[int, int, str], ...] | None:
    key = normalize(surface)
    parts = JOINED.get(bare(key))
    if not parts:
        return None
    lengths = [len(bare(part)) for part in parts]
    spans: list[tuple[int, int, str]] = []
    letter, start, part = 0, 0, 0
    boundary = lengths[0]
    for offset, char in enumerate(surface):
        # A precomposed letter is one code point; a stray combining mark
        # belongs to the letter before it.
        if unicodedata.combining(char):
            continue
        if letter == boundary and part < len(parts) - 1:
            spans.append((start, offset, parts[part]))
            start, part = offset, part + 1
            boundary += lengths[part]
        letter += 1
    spans.append((start, len(surface), parts[part]))
    return tuple(spans) if len(spans) == len(parts) else None


# ---------------------------------------------------------------------------
# Tokens

@dataclass(frozen=True)
class Token:
    """One word of a verse, or one part of a joined word.

    `start`/`end` are code-point offsets into the verse text, as annotations
    use them; `surface` is `text[start:end]`. A joined word's parts share
    `word_start`/`word_end`, the printed word, and are numbered by `part` of
    `parts`.
    """

    index: int
    start: int
    end: int
    surface: str
    key: str
    rules: tuple[str, ...] = ()
    word_key: str = ""          # the printed word's key; differs from `key` for a part
    word_start: int = 0
    word_end: int = 0
    part: int = 0
    parts: int = 1
    case: str = "lower"         # "lower" | "capital" | "caps"
    position: str = "medial"    # "initial" | "pause" | "medial"
    first_marks_known: bool = True
    elided: bool = False

    @property
    def name_like(self) -> bool:
        """A capital Brenton would not print here unless the word were a name or title."""
        return self.case == "capital" and self.position == "medial"

    @property
    def case_informative(self) -> bool:
        """Whether the capital or its absence says anything about the word."""
        return self.position == "medial" and self.case != "caps"

    @property
    def joined(self) -> bool:
        return self.parts > 1


def _is_word_char(char: str) -> bool:
    return unicodedata.category(char)[0] in "LM" or char in APOSTROPHES


def _case(surface: str) -> str:
    letters = [char for char in surface if char.isalpha() and char not in APOSTROPHES]
    if not letters or not letters[0].isupper():
        return "lower"
    if len(letters) > 1 and all(char.isupper() for char in letters):
        return "caps"
    return "capital"


def _position(gap: str, first: bool) -> str:
    if first or any(char in SENTENCE_BREAKS for char in gap):
        return "initial"
    if any(char in PAUSES for char in gap):
        return "pause"
    return "medial"


def word_spans(text: str) -> list[tuple[int, int]]:
    """The printed words of a verse as (start, end): runs of letters, marks
    and apostrophes. Punctuation and spaces separate words."""
    spans, start = [], None
    for offset, char in enumerate(text):
        if _is_word_char(char):
            if start is None:
                start = offset
        elif start is not None:
            spans.append((start, offset))
            start = None
    if start is not None:
        spans.append((start, len(text)))
    # An apostrophe on its own is not a word.
    return [(s, e) for s, e in spans if any(char.isalpha() for char in text[s:e])]


# Printed form -> intended form, each read in its verse (round 2). Keys are
# normalised as every word is, so Δέγει stands for δέγει.
MISPRINT_TABLE: tuple[tuple[str, str, str], ...] = (
    # (printed, intended, where it was read)
    ("γἱοὶ", "υἱοὶ", "Gen 46:9, 1 Chr 23:8 and 36 more"),
    ("Δέγει", "Λέγει", "Gen 15:2"),
    ("λέλει", "λέγει", "2 Kgs 7:1, Ezek 28:2"),
    ("Ἀποκοιθεὶς", "Ἀποκριθεὶς", "Gen 27:39"),
    ("μον", "μου", "Gen 38:17, Lev 19:30"),
    ("τῶς", "τῶν", "Exod 28:11, 28:12"),
    ("θυλάτηρ", "θυγάτηρ", "2 Kgs 9:34, 22:1"),
    ("τεσσράκοντα", "τεσσαράκοντα", "1 Chr 29:27, Job 42:16"),
    ("εἶπν", "εἶπεν", "Esth 6:4, 7:8"),
    ("ἑπτκόσια", "ἑπτακόσια", "Gen 5:10"),
    ("αὐτοὺδ", "αὐτοὺς", "Gen 13:6"),
    ("ὑπήκουσελ", "ὑπήκουσεν", "Gen 16:2"),
    ("ἐτιδών", "ἐπιδών", "Gen 16:13"),
    ("ἐρχομενεην", "ἐρχομένην", "Gen 18:21"),
    ("ἀδφή", "ἀδελφή", "Gen 20:5"),
    ("ἐνατίον", "ἐναντίον", "Gen 21:11"),
    ("ἀγαπτοῦ", "ἀγαπητοῦ", "Gen 22:16"),
    ("σφάδρα", "σφόδρα", "Gen 29:17"),
    ("σφόδπα", "σφόδρα", "Exod 1:20"),
    ("ἑαντοῦ", "ἑαυτοῦ", "Gen 31:23"),
    ("ἀπηγγέλε", "ἀπηγγέλη", "Gen 38:13"),
    ("εὐώδο", "εὐώδου", "Gen 39:23"),
    ("συνέκριεν", "συνέκρινεν", "Gen 40:16"),
    ("τερέινθον", "τερέβινθον", "Gen 43:11"),
    ("κατέβηεν", "κατέβημεν", "Gen 43:20"),
    ("θάψτέ", "θάψατέ", "Gen 49:29"),
    ("ἐσπέπασαν", "ἐσκέπασαν", "Exod 2:2"),
    ("δετέρου", "δευτέρου", "Exod 4:8"),
    ("ἀλλ", "ἀλλ’", "Exod 5:7"),
    ("πάσηῃ", "πάσῃ", "Exod 10:15"),
    ("ἀτοῦ", "αὐτοῦ", "Exod 11:3"),
    ("θαλάσοης", "θαλάσσης", "Exod 14:27"),
    ("αήμερον", "σήμερον", "Exod 16:25"),
    ("γνς", "γῆς", "Exod 23:19"),
    ("ποιήεις", "ποιήσεις", "Exod 25:8"),
    ("πεπὶ", "περὶ", "Exod 28:12"),
    ("Ἰσρσήλ", "Ἰσραήλ", "Exod 28:11"),
    ("τρεῖν", "τρεῖς", "Num 12:5"),
    ("εἰν", "εἰς", "Num 12:5, 2 Sam 11:25"),
    ("ἐπ", "ἐπ’", "Ps 77:27, Jer 14:17"),
    ("Θυμὸ", "Θυμὸς", "Prov 16:14"),
    ("Ὅτ", "Ὅτι", "Eccl 11:8"),
)
MISPRINTS: dict[str, str] = {}


def _misprint(printed: str, intended: str) -> None:
    MISPRINTS[normalize(printed)] = normalize(intended)


def tokenize(text: str) -> list[Token]:
    """A verse's tokens, in order, with offsets into `text`."""
    tokens: list[Token] = []
    previous_end = 0
    for number, (start, end) in enumerate(word_spans(text)):
        surface = text[start:end]
        position = _position(text[previous_end:start], number == 0)
        previous_end = end
        case = _case(surface)
        first_marks_known = True
        if case == "caps":
            first = _decompose(surface.lstrip("".join(APOSTROPHES)))[0] if surface else [""]
            first_marks_known = any(m in ACCENTS | BREATHINGS for m in first[1:])
        elided = surface[-1] in APOSTROPHES
        word_key, word_rules = normalize_with_rules(surface)
        pieces = split_joined(surface)
        if pieces:
            for part, (part_start, part_end, key) in enumerate(pieces):
                part_surface = surface[part_start:part_end]
                tokens.append(Token(
                    index=len(tokens), start=start + part_start, end=start + part_end,
                    surface=part_surface, key=key,
                    rules=normalize_with_rules(part_surface)[1] + ("joined",), word_key=word_key,
                    word_start=start, word_end=end, part=part, parts=len(pieces),
                    case=_case(part_surface) if part == 0 else "lower",
                    position=position if part == 0 else "medial",
                    first_marks_known=first_marks_known, elided=False,
                ))
            continue
        key, rules = word_key, word_rules
        if key in MISPRINTS:
            key, rules = MISPRINTS[key], rules + ("misprint",)
        tokens.append(Token(
            index=len(tokens), start=start, end=end, surface=surface, key=key, rules=rules,
            word_key=key, word_start=start, word_end=end, case=case, position=position,
            first_marks_known=first_marks_known and "apostrophe-breathing" not in rules,
            elided=elided,
        ))
    return tokens


for _printed, _intended, _ in MISPRINT_TABLE:
    _misprint(_printed, _intended)
