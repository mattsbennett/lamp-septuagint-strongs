"""Greek sounds and spelling for the morphological analyser.

The analyser builds forms from stems and endings and compares them with
Brenton's words. This module holds the rules of Greek phonology and
orthography it needs, each a fact of the grammar:

  * syllables and vowel quantity, so the accent can be placed;
  * the rules of accentuation: recessive (finite verbs), persistent (nouns,
    adjectives, participles), fixed by the ending (second aorist
    infinitives, perfect participles ...), and the compound verb's limit
    (the accent of a compound verb goes back no further than the augment,
    nor past the last syllable of its preverb: ἀπῆλθον, ἀπόδος);
  * vowel contraction (ποιέ-ομεν > ποιοῦμεν, τιμά-ει > τιμᾷ, γένε-ος > γένους),
    with the accent rule (an accented first vowel gives a circumflex);
  * the augment (ἐ-, temporal lengthening ἀ>ἠ, ὀ>ὠ, αἰ>ᾐ, οἰ>ᾠ ...) and
    reduplication (λέ-λυκα, ἔ-σταλκα, γέ-γραφα, τέ-θεικα), and the
    consonant changes before σ, θ and the perfect middle's endings;
  * preverbs: the forms a preposition takes before a consonant, a vowel and
    a rough breathing (ἀπο-, ἀπ-, ἀφ-; συν-, συμ-, συλ-, συγ-, συ-; ἐκ-, ἐξ-),
    how to join one to a verb form, and how to find them in a lemma.

Strings here are "plain": NFC, lower case, σ for every sigma, with
breathings, iota subscript and diaeresis but no accent. A dichronic vowel
(α, ι, υ) may carry a macron (long) or breve (short) when its quantity is
known, which only the accentuation reads; `finish` removes it.
"""

from __future__ import annotations

import functools
import unicodedata

ACUTE, GRAVE, CIRCUMFLEX = "́", "̀", "͂"
SMOOTH, ROUGH = "̓", "̔"
DIAERESIS, SUBSCRIPT = "̈", "ͅ"
MACRON, BREVE = "̄", "̆"
ACCENTS = frozenset((ACUTE, GRAVE, CIRCUMFLEX))
BREATHINGS = frozenset((SMOOTH, ROUGH))
QUANTITY = frozenset((MACRON, BREVE))
VOWELS = frozenset("αεηιουω")
LONG_VOWELS = frozenset("ηω")
SHORT_VOWELS = frozenset("εο")
DIPHTHONGS = frozenset(("αι", "ει", "οι", "υι", "αυ", "ευ", "ου", "ηυ", "ωυ"))
LABIALS, VELARS, DENTALS = frozenset("πβφ"), frozenset("κγχ"), frozenset("τδθ")
LIQUIDS = frozenset("λρμν")
STOPS = LABIALS | VELARS | DENTALS
CONSONANTS = frozenset("βγδζθκλμνξπρστφχψ")
DEASPIRATE = {"φ": "π", "θ": "τ", "χ": "κ"}
ASPIRATE = {"π": "φ", "τ": "θ", "κ": "χ"}


def nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


Cluster = list  # [base letter, *combining marks]


def clusters(text: str) -> list[Cluster]:
    """A string as [letter, *marks] clusters (NFD), final sigma folded."""
    found: list[Cluster] = []
    for char in nfd(text):
        if unicodedata.combining(char) and found:
            found[-1].append(char)
        else:
            found.append(["σ" if char == "ς" else char])
    return found


def compose(parts: list[Cluster]) -> str:
    return nfc("".join("".join(cluster) for cluster in parts))


@functools.lru_cache(maxsize=1 << 18)
def plain(text: str) -> str:
    """Accents removed; breathings, subscripts, diaeresis and quantity kept;
    lower case, σ. A circumflex on α, ι or υ shows the vowel is long, so it
    leaves a macron."""
    out = []
    for cluster in clusters(text.lower()):
        base, marks = cluster[0], cluster[1:]
        if CIRCUMFLEX in marks and base in "αιυ" and MACRON not in marks:
            marks = [MACRON, *marks]
        out.append([base, *(m for m in marks if m not in ACCENTS)])
    return compose(out)


@functools.lru_cache(maxsize=1 << 18)
def finish(text: str) -> str:
    """A generated form as a lookup key, by `text.normalize`'s conventions:
    quantity marks and diaeresis removed, no breathing on ρ, NFC, σ."""
    out = []
    for cluster in clusters(text):
        marks = [m for m in cluster[1:] if m not in QUANTITY and m != DIAERESIS
                 and not (cluster[0] == "ρ" and m in BREATHINGS)]
        out.append([cluster[0], *marks])
    return compose(out)


@functools.lru_cache(maxsize=1 << 18)
def letters(text: str) -> str:
    """Letters alone (as `text.bare`), σ folded."""
    return "".join(cluster[0] for cluster in clusters(text))


def strip_breathing(text: str) -> str:
    return compose([[c[0], *(m for m in c[1:] if m not in BREATHINGS)] for c in clusters(text)])


def initial_breathing(text: str) -> str | None:
    """SMOOTH, ROUGH or None for a word's initial vowel (on a diphthong's second letter)."""
    parts = clusters(text)
    for cluster in parts[:2]:
        for mark in cluster[1:]:
            if mark in BREATHINGS:
                return mark
    return None


def set_breathing(text: str, breathing: str | None) -> str:
    """The word with its initial vowel (or diphthong) given this breathing."""
    parts = [[c[0], *(m for m in c[1:] if m not in BREATHINGS)] for c in clusters(text)]
    if not parts or parts[0][0] not in VOWELS or breathing is None:
        return compose(parts)
    index = 1 if (len(parts) > 1 and parts[0][0] + parts[1][0] in DIPHTHONGS
                  and DIAERESIS not in parts[1] and SUBSCRIPT not in parts[0]) else 0
    parts[index].insert(1, breathing)
    return compose(parts)


def starts_with_vowel(text: str) -> bool:
    return bool(text) and letters(text)[:1] in VOWELS


# ---------------------------------------------------------------------------
# Syllables and quantity

def _nuclei(parts: list[Cluster]) -> list[tuple[int, int]]:
    """(first, last) cluster indexes of each vowel nucleus."""
    found: list[tuple[int, int]] = []
    index = 0
    while index < len(parts):
        base = parts[index][0]
        if base in VOWELS:
            if (index + 1 < len(parts) and base + parts[index + 1][0] in DIPHTHONGS
                    and DIAERESIS not in parts[index + 1] and SUBSCRIPT not in parts[index]
                    and not (QUANTITY & set(parts[index][1:]))):
                found.append((index, index + 1))
                index += 2
                continue
            found.append((index, index))
        index += 1
    return found


def _quantity(parts: list[Cluster], nucleus: tuple[int, int]) -> str:
    """"L" long, "S" short or "?" for a nucleus."""
    first, last = nucleus
    if first != last:
        return "L"
    base, marks = parts[first][0], parts[first][1:]
    if base in LONG_VOWELS or SUBSCRIPT in marks or CIRCUMFLEX in marks or MACRON in marks:
        return "L"
    if base in SHORT_VOWELS or BREVE in marks:
        return "S"
    return "?"


def _final_quantity(parts: list[Cluster], nuclei: list[tuple[int, int]], ultima: str | None) -> str:
    if ultima:
        return ultima
    first, last = nuclei[-1]
    quantity = _quantity(parts, nuclei[-1])
    # Final -αι and -οι count short for the accent (but not in the optative,
    # whose endings say so with `ultima="L"`).
    if first != last and last == len(parts) - 1 and parts[first][0] + parts[last][0] in ("αι", "οι"):
        return "S"
    return quantity


def syllable_count(text: str) -> int:
    return len(_nuclei(clusters(text)))


def _accent_index(parts: list[Cluster], nucleus: tuple[int, int]) -> int:
    """The cluster an accent on this nucleus is written on (a diphthong's second vowel)."""
    return nucleus[1]


def _with_accent(parts: list[Cluster], nucleus: tuple[int, int], mark: str) -> str:
    out = [[c[0], *(m for m in c[1:] if m not in ACCENTS)] for c in parts]
    target = out[_accent_index(out, nucleus)]
    # Breathing and diaeresis come before the accent in canonical order.
    position = 1
    while position < len(target) and target[position] in BREATHINGS | {DIAERESIS} | QUANTITY:
        position += 1
    target.insert(position, mark)
    return compose(out)


def _kinds(parts, nuclei, position_from_end: int, ultima_q: str, kind: str | None) -> list[str]:
    nucleus = nuclei[len(nuclei) - 1 - position_from_end]
    quantity = _quantity(parts, nucleus)
    if position_from_end >= 2:
        return [ACUTE]
    if position_from_end == 1:
        if ultima_q == "L":
            return [ACUTE]
        if quantity == "L":
            return [CIRCUMFLEX]
        if quantity == "S":
            return [ACUTE]
        return [ACUTE, CIRCUMFLEX]
    # The ultima.
    if kind == CIRCUMFLEX and quantity != "S":
        return [CIRCUMFLEX]
    if kind is None and len(nuclei) == 1:
        return {"L": [CIRCUMFLEX], "S": [ACUTE]}.get(quantity, [ACUTE, CIRCUMFLEX])
    return [ACUTE]


@functools.lru_cache(maxsize=1 << 18)
def accentuate(text: str, mode: str = "recessive", ultima: str | None = None, floor: int = 0,
               syllable: int = -1, kind: str | None = None) -> tuple[str, ...]:
    """The accented spellings of an unaccented form (several where a vowel's
    quantity is unknown and decides between acute and circumflex).

    mode
      recessive  as far back as the ultima allows, but not before nucleus
                 `floor` (a compound verb's augment or preverb)
      syllable   on nucleus `syllable` counted from the start (a noun's
                 persistent accent), moved forward as the ultima requires;
                 a negative number counts from the end (-1 the ultima)
      fixed      the form already carries its accent: returned as it is
    `ultima` overrides the final syllable's quantity ("L" or "S"); `kind`
    is the accent wanted on an accented ultima (CIRCUMFLEX for the
    perispomena θεοῦ, ποιῶ), acute by default.
    """
    parts = clusters(text)
    if mode == "fixed" or any(m in ACCENTS for c in parts for m in c[1:]):
        return (compose(parts),)
    nuclei = _nuclei(parts)
    if not nuclei:
        return (compose(parts),)
    count = len(nuclei)
    ultima_q = _final_quantity(parts, nuclei, ultima)
    limit = 1 if ultima_q == "L" else 2    # furthest position back from the end
    if mode == "recessive":
        position = min(limit, count - 1, max(0, count - 1 - floor))
    else:
        index = syllable if syllable >= 0 else count + syllable
        index = max(0, min(index, count - 1))
        position = min(count - 1 - index, limit)
    found = []
    for mark in _kinds(parts, nuclei, position, ultima_q, kind):
        found.append(_with_accent(parts, nuclei[count - 1 - position], mark))
    return tuple(dict.fromkeys(found))


def accent_position(text: str) -> tuple[int, str] | None:
    """(nucleus index from the start, accent mark) of a form's (first) accent."""
    parts = clusters(text)
    for number, (first, last) in enumerate(_nuclei(parts)):
        for index in range(first, last + 1):
            for mark in parts[index][1:]:
                if mark in ACCENTS:
                    return number, CIRCUMFLEX if mark == CIRCUMFLEX else ACUTE
    return None


def nucleus_count(text: str) -> int:
    return len(_nuclei(clusters(text)))


def preverb_floor(prefix: str) -> int:
    """The floor (see `accentuate`) for a compound whose preverb part is
    `prefix`: the index of the preverb's last syllable."""
    count = nucleus_count(prefix)
    return max(0, count - 1)


# ---------------------------------------------------------------------------
# Contraction

# (stem vowel, ending's first vowel or diphthong) -> the contracted vowel.
CONTRACTIONS = {
    "α": {"ε": "ᾱ", "η": "ᾱ", "ει": "ᾳ", "ῃ": "ᾳ", "ο": "ω", "ω": "ω", "ου": "ω", "οι": "ῳ", "α": "ᾱ"},
    "ε": {"ε": "ει", "ει": "ει", "η": "η", "ῃ": "ῃ", "ο": "ου", "ου": "ου", "ω": "ω", "οι": "οι",
          "α": "η", "ι": "ει", "αι": "αι"},
    "ο": {"ε": "ου", "ει": "οι", "η": "ω", "ῃ": "οι", "ο": "ου", "ου": "ου", "ω": "ω", "οι": "οι",
          "α": "ω"},
}
# ζάω, χράομαι, πεινάω, διψάω contract to η where other -άω verbs have ᾱ.
ETA_CONTRACTIONS = {**CONTRACTIONS["α"], "ε": "η", "ει": "ῃ", "η": "η", "ῃ": "ῃ"}


def _vowel_group(parts: list[Cluster], index: int) -> tuple[str, int]:
    """The vowel or diphthong starting at `index` (with ῃ for η + subscript), and its length in clusters."""
    if index >= len(parts) or parts[index][0] not in VOWELS:
        return "", 0
    base = parts[index][0]
    if (index + 1 < len(parts) and base + parts[index + 1][0] in ("ει", "ου", "οι", "αι")
            and DIAERESIS not in parts[index + 1]):
        return base + parts[index + 1][0], 2
    if base == "η" and SUBSCRIPT in parts[index]:
        return "ῃ", 1
    return base, 1


def contract(text: str, boundary: int, table: dict[str, str] | None = None) -> str | None:
    """Contract the vowel before cluster `boundary` with the vowel(s) after it.

    `text` is an accented uncontracted form; the result carries the accent
    as the rule gives it: a circumflex if the first vowel was accented, the
    second vowel's own accent if it was. None if the two do not contract
    (no vowel on one side, or a pair the table lacks).
    """
    parts = clusters(text)
    if boundary <= 0 or boundary >= len(parts):
        return None
    left = parts[boundary - 1]
    if left[0] not in "αεο":
        return None
    group, size = _vowel_group(parts, boundary)
    if not group:
        return None
    outcome = (table or CONTRACTIONS[left[0]]).get(group)
    if outcome is None:
        return None
    left_accent = next((m for m in left[1:] if m in ACCENTS), None)
    right_accent = next((m for c in parts[boundary:boundary + size] for m in c[1:] if m in ACCENTS), None)
    breathing = [m for m in left[1:] if m in BREATHINGS]
    merged = clusters(outcome)
    for cluster in merged:
        cluster[1:] = [m for m in cluster[1:] if m != MACRON] + ([MACRON] if MACRON in cluster[1:] else [])
    if breathing:
        merged[-1][1:1] = breathing
    accent = CIRCUMFLEX if left_accent else right_accent
    if accent:
        target = merged[-1]
        position = 1
        while position < len(target) and target[position] in BREATHINGS | QUANTITY:
            position += 1
        target.insert(position, accent)
    return compose(parts[:boundary - 1] + merged + parts[boundary + size:])


# ---------------------------------------------------------------------------
# Augment and reduplication

# Temporal augment of an initial vowel or diphthong (letters alone), and
# the unaugmented spelling Koine often keeps instead.
TEMPORAL = {
    "α": ("η",), "ε": ("η",), "ο": ("ω",), "ι": ("ι",), "υ": ("υ",), "η": ("η",), "ω": ("ω",),
    "αι": ("ῃ",), "οι": ("ῳ", "οι"), "αυ": ("ηυ", "αυ"), "ευ": ("ηυ", "ευ"), "ει": ("ει", "ῃ"),
    "ου": ("ου",), "ᾳ": ("ᾳ",), "ῃ": ("ῃ",), "ῳ": ("ῳ",),
}


def _initial_vowels(text: str) -> tuple[str, str, str | None]:
    """(initial vowel group letters incl. subscript as ᾳ/ῃ/ῳ, the rest, breathing)."""
    parts = clusters(text)
    breathing = initial_breathing(text)
    if not parts or parts[0][0] not in VOWELS:
        return "", text, breathing
    if (len(parts) > 1 and parts[0][0] + parts[1][0] in DIPHTHONGS and DIAERESIS not in parts[1]
            and SUBSCRIPT not in parts[0]):
        return parts[0][0] + parts[1][0], compose(parts[2:]), breathing
    group = parts[0][0]
    if SUBSCRIPT in parts[0]:
        group = {"α": "ᾳ", "η": "ῃ", "ω": "ῳ"}[group]
    return group, compose(parts[1:]), breathing


def augment(stem: str) -> list[tuple[str, float]]:
    """The augmented spellings of a verb stem, with a weight for each.

    A consonant takes ἐ- (ῥ- doubles: ἔρριψα); a vowel is lengthened, and
    οἰ-, αὐ-, εὐ-, εἰ- are often left unaugmented in Koine.
    """
    if not stem:
        return []
    group, rest, breathing = _initial_vowels(stem)
    if not group:
        if letters(stem).startswith("ρ"):
            return [("ἐρρ" + strip_breathing(stem)[1:], 1.0), ("ἐρ" + strip_breathing(stem)[1:], 0.3)]
        return [("ἐ" + stem, 1.0)]
    found = []
    for number, outcome in enumerate(TEMPORAL.get(group, (group,))):
        found.append((set_breathing(outcome + rest, breathing or SMOOTH), 1.0 if number == 0 else 0.6))
    return found


def reduplicate(stem: str) -> list[tuple[str, float]]:
    """The reduplicated (perfect) spellings of a verb stem, weighted.

    A single consonant, or a stop before a liquid or nasal, is repeated with
    ε (an aspirate as its plain stop: τέθεικα, πεφύτευκα); ῥ, a double
    consonant (ζ, ξ, ψ) and other groups take ἐ- (ἔσταλκα, ἔγνωκα), though
    πτ, κτ and μν repeat (πέπτωκα, κέκτημαι, μέμνημαι); a vowel lengthens as
    in the augment.
    """
    if not stem:
        return []
    bare = letters(stem)
    if bare[0] in VOWELS:
        return augment(stem)
    if bare[0] == "ρ":
        return [("ἐρρ" + strip_breathing(stem)[1:], 1.0), ("ἐρ" + strip_breathing(stem)[1:], 0.3)]
    first = bare[0]
    second = bare[1] if len(bare) > 1 else ""
    repeated = DEASPIRATE.get(first, first) + "ε" + stem
    if first in "ζξψ":
        return [("ἐ" + stem, 1.0)]
    if second in VOWELS or (first in STOPS and second in LIQUIDS):
        return [(repeated, 1.0)]
    if bare[:2] in ("πτ", "κτ", "μν"):
        return [(repeated, 1.0), ("ἐ" + stem, 0.5)]
    return [("ἐ" + stem, 1.0), (repeated, 0.3)]


# ---------------------------------------------------------------------------
# Consonant changes

def before_sigma(stem: str) -> list[str]:
    """A stem's final consonant joined to σ (future, first aorist):
    labial > ψ, velar > ξ, dental (and ζ) dropped, ν dropped."""
    bare = letters(stem)
    if not bare:
        return [stem + "σ"]
    last = bare[-1]
    head = stem[:-1]
    if last in LABIALS:
        return [head + "ψ"]
    if last in VELARS:
        return [head + "ξ"]
    if last in DENTALS or last == "ζ":
        return [head + "σ"]
    if last == "ν":
        return [head + "σ"]
    return [stem + "σ"]


def before_theta(stem: str) -> str:
    """A stem's final consonant before θ (aorist passive): labial > φ, velar
    > χ, dental (and ζ) > σ."""
    bare = letters(stem)
    if not bare:
        return stem
    last = bare[-1]
    head = stem[:-1]
    if last in LABIALS:
        return head + "φ"
    if last in VELARS:
        return head + "χ"
    if last in DENTALS or last == "ζ":
        return head + "σ"
    return stem


def perfect_middle(stem: str, ending: str) -> str:
    """A perfect middle stem joined to its ending, with the consonant
    changes (γέγραμμαι, γέγραψαι, γέγραπται, γεγράφθαι; δεδίωγμαι,
    δεδίωξαι, δεδίωκται; ἥγνισμαι, ἥγνισται; ἔσταλμαι, ἐστάλθαι)."""
    bare = letters(stem)
    last = bare[-1] if bare else ""
    head = stem[:-1]
    if last == "σ" and ending.startswith("σ"):
        return stem + ending[1:]       # ἥγνισ-σαι > ἥγνισαι, ἡγνίσ-σθαι > ἡγνίσθαι
    if last not in CONSONANTS or last == "σ":
        return stem + ending
    first = letters(ending)[:1]
    rest = ending
    if ending.startswith("σθ"):
        rest = ending[1:]          # σ is lost between consonants
        first = "θ"
    if last in LABIALS:
        joined = {"μ": "μ", "σ": "ψ", "τ": "π", "θ": "φ"}.get(first)
        if joined == "ψ":
            return head + "ψ" + ending[1:]
        return head + (joined or last) + rest
    if last in VELARS:
        joined = {"μ": "γ", "σ": "ξ", "τ": "κ", "θ": "χ"}.get(first)
        if joined == "ξ":
            return head + "ξ" + ending[1:]
        return head + (joined or last) + rest
    if last in DENTALS or last == "ζ":
        if first == "σ":
            return head + ending
        return head + "σ" + rest
    if last == "ν":
        if first == "μ":
            return head + "μ" + rest
        return stem + rest
    return stem + rest          # λ, ρ


# ---------------------------------------------------------------------------
# Preverbs

class Preverb:
    """A preposition as a verb's prefix, in the forms it takes."""

    def __init__(self, name: str, consonant: str, vowel: str, rough: str, elides: bool = True):
        self.name = name                # the preposition: "ἀπό"
        self.consonant = consonant      # before a consonant: "ἀπο"
        self.vowel = vowel              # before a smooth vowel: "ἀπ"
        self.rough = rough              # before a rough vowel: "ἀφ"
        self.elides = elides

    def __repr__(self) -> str:
        return f"Preverb({self.name})"

    def before(self, form: str) -> list[tuple[str, float]]:
        """The preverb's spellings before `form` (plain), weighted."""
        bare = letters(form)
        if not bare:
            return [(self.consonant, 1.0)]
        if bare[0] in VOWELS:
            if self.name == "πρό" and bare[0] == "ε":
                return [(self.vowel, 1.0), ("πρ", 0.3)]     # προέβην, προὔβην (crasis)
            return [(self.rough if initial_breathing(form) == ROUGH else self.vowel, 1.0)]
        return [(spelling, 1.0) for spelling in self.assimilated(bare)]

    def assimilated(self, bare: str) -> list[str]:
        """Before a consonant: ἐν and σύν assimilate (ἐμ-, συλ-, συ-)."""
        first = bare[0]
        if self.consonant in ("ἐν", "συν"):
            stem = self.consonant[:-1]
            if first in "βπφψμ":
                return [stem + "μ"]
            if first in "γκχξ":
                return [stem + "γ"]
            if first == "λ":
                return [stem + "λ"]
            if first == "ρ":
                return [stem + "ρ", self.consonant]
            if first == "σ" and self.consonant == "συν":
                return ["συ", "συσ"] if len(bare) > 1 and bare[1] in VOWELS else ["συ"]
            if first == "ζ" and self.consonant == "συν":
                return ["συ"]
        return [self.consonant]


PREVERBS: tuple[Preverb, ...] = (
    Preverb("ἀμφί", "ἀμφι", "ἀμφ", "ἀμφ"),
    Preverb("ἀνά", "ἀνα", "ἀν", "ἀν"),
    Preverb("ἀντί", "ἀντι", "ἀντ", "ἀνθ"),
    Preverb("ἀπό", "ἀπο", "ἀπ", "ἀφ"),
    Preverb("διά", "δια", "δι", "δι"),
    Preverb("εἰς", "εἰσ", "εἰσ", "εἰσ", elides=False),
    Preverb("ἐκ", "ἐκ", "ἐξ", "ἐξ", elides=False),
    Preverb("ἐν", "ἐν", "ἐν", "ἐν", elides=False),
    Preverb("ἐπί", "ἐπι", "ἐπ", "ἐφ"),
    Preverb("κατά", "κατα", "κατ", "καθ"),
    Preverb("μετά", "μετα", "μετ", "μεθ"),
    Preverb("παρά", "παρα", "παρ", "παρ"),
    Preverb("περί", "περι", "περι", "περι", elides=False),
    Preverb("πρό", "προ", "προ", "προ", elides=False),
    Preverb("πρός", "προσ", "προσ", "προσ", elides=False),
    Preverb("σύν", "συν", "συν", "συν", elides=False),
    Preverb("ὑπέρ", "ὑπερ", "ὑπερ", "ὑπερ", elides=False),
    Preverb("ὑπό", "ὑπο", "ὑπ", "ὑφ"),
)
PREVERB_BY_NAME = {preverb.name: preverb for preverb in PREVERBS}


def attach(chain: tuple[Preverb, ...], form: str) -> list[tuple[str, int, float]]:
    """A base form with its preverbs (innermost last), as (word, number of
    the preverb part's last syllable, weight) for each spelling.

    The base loses its own breathing; a vowel-final preverb doubles an
    initial ρ as often as not (διαρρήγνυμι, ἀπορίπτω).
    """
    results = [(form, -1, 1.0)]
    for preverb in reversed(chain):
        extended = []
        for word, _, weight in results:
            bare = letters(word)
            for spelling, spelling_weight in preverb.before(word):
                body = strip_breathing(word)
                if bare[:1] == "ρ" and letters(spelling)[-1:] in VOWELS:
                    variants = [("ρ" + body[1:], 0.6), ("ρρ" + body[1:], 0.6)]
                elif letters(spelling)[-1:] == "σ" and bare[:1] == "σ" and bare[1:2] in CONSONANTS:
                    variants = [(body[1:], 1.0)]      # προσ-σχές > πρόσχες
                else:
                    variants = [(body, 1.0)]
                for tail, tail_weight in variants:
                    joined = spelling + tail
                    extended.append((joined, nucleus_count(spelling) - 1, weight * spelling_weight * tail_weight))
        results = extended
    return results


def preverb_spellings() -> list[tuple[str, Preverb, str]]:
    """Every spelling of every preverb, as (spelling, preverb, kind), longest first;
    kind is "consonant", "vowel" or "rough"."""
    found = set()
    for preverb in PREVERBS:
        for spelling in {preverb.consonant, *preverb.assimilated("β"), *preverb.assimilated("γ"),
                         *preverb.assimilated("λ"), *preverb.assimilated("ρ"), *preverb.assimilated("σα"),
                         *preverb.assimilated("στ")}:
            found.add((spelling, preverb, "consonant"))
        found.add((preverb.vowel, preverb, "vowel"))
        found.add((preverb.rough, preverb, "rough"))
    return sorted(found, key=lambda item: (-len(letters(item[0])), item[0], item[1].name, item[2]))


_SPELLINGS = None


def split_preverbs(word: str, depth: int = 3) -> list[tuple[tuple[Preverb, ...], str]]:
    """Every way to read a plain word as preverbs + base, the unsplit word
    included: [((ἐπί, σύν), "αγω"), ...]. The base gets the breathing the
    last preverb's spelling shows (ἀφ- before a rough vowel), smooth
    otherwise. A split is kept only where the sandhi agrees: a
    consonant-form preverb before a consonant (ἐμ- before a labial), a
    vowel-form before a vowel."""
    global _SPELLINGS
    if _SPELLINGS is None:
        _SPELLINGS = preverb_spellings()
    found: list[tuple[tuple[Preverb, ...], str]] = [((), word)]
    if depth == 0:
        return found
    bare = letters(word)
    smooth_word = strip_breathing(word)
    for spelling, preverb, kind in _SPELLINGS:
        spelt = letters(spelling)
        if not bare.startswith(spelt) or len(bare) - len(spelt) < 3:
            continue
        # The spelling's breathing must be the word's.
        if starts_with_vowel(spelling) and initial_breathing(spelling) != initial_breathing(word):
            continue
        rest_letters = bare[len(spelt):]
        rest = _drop_letters(smooth_word, len(spelt))
        nxt = rest_letters[0]
        if kind == "consonant":
            if nxt in VOWELS or spelling not in preverb.assimilated(rest_letters):
                continue
        elif kind in ("vowel", "rough"):
            if nxt not in VOWELS:
                continue
            rest = set_breathing(rest, ROUGH if kind == "rough" and preverb.rough != preverb.vowel else SMOOTH)
        if letters(rest)[:2] == "ρρ" and letters(spelling)[-1:] in VOWELS:
            rest = rest[1:]
        for chain, base in split_preverbs(rest, depth - 1):
            found.append(((preverb, *chain), base))
    unique = {}
    for chain, base in found:
        unique.setdefault((tuple(p.name for p in chain), base), (chain, base))
    return list(unique.values())


def _drop_letters(word: str, count: int) -> str:
    parts = clusters(word)
    return compose(parts[count:])
