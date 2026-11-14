"""How likely a Greek spelling is as the Septuagint's rendering of a Hebrew name.

THE MODEL

A stochastic transducer (a memoryless edit model, after Ristad and Yianilos)
reads the Hebrew name's sounds (`hebrew_names.sounds`: consonants, and vowels
from the pointing) and writes Greek letters. Before each Hebrew sound, and
after the last, it may insert a Greek letter (probability `insert`, the letter
from `inserted`); each Hebrew sound then becomes one Greek letter, the same
Greek letter doubled (גּ -> γγ, the doubled consonants of Σάρρα and
Ἀσσούρ), or nothing. Every way of writing the Greek is summed (the forward
algorithm), so the model gives P(Greek | Hebrew), a proper distribution over
Greek strings.

A name's score is the log-likelihood ratio of that against a null model, the
Greek letters' own frequencies (P0): how much more likely the Greek spelling is
as this Hebrew name's rendering than as a name chosen at random. It grows with
the length of what agrees, so a long name that matches is surer than a short
one, and a Greek name compared with two Hebrew names is compared with the same
denominator, so the difference of two scores is their likelihood ratio.

Greek letters are compared without accents or breathings (Brenton's are not
the Hebrew's), with ου as one letter (it writes u, o and w). The model starts
from the conventions every Septuagint translator follows (`CONVENTIONS`: ב
β, ג γ, ד δ, ה nothing or ε, ו ου/υ/β, ז ζ, ח χ or nothing, ט τ, י ι, כ χ,
ל λ, מ μ, נ ν, ס σ, ע nothing or γ, פ φ, צ σ, ק κ, ר ρ, ש σ, ת θ; vowels
loosely; μ/β/δ glides inserted; ד and ר, which the translators' Hebrew
confused, read for each other now and then: Ἄδερ for הֲדַד), as prior counts, and is fitted by expectation
maximisation to pairs of Greek and Hebrew spellings of one name (`fit`): the
New Testament's names whose TBESG entry gives their Hebrew number, and,
in later rounds, names the stage itself matched with confidence.

GREEK ENDINGS

A Greek name may carry a Greek case ending (Ἀζαρίας, Ἀζαρίου; Μαρδοχαῖος,
-ου, -ῳ, -ον) or a gentilic suffix (Χαναναῖος, Λευίτης, Ἀσσυρίων) that the
Hebrew lacks. `decompositions` lists the ways a form can be split into a
stem and such an ending, each with a prior probability; the score sums over
them, for the Hebrew name and for the null model alike, so an ending is never
evidence by itself. A gentilic suffix pairs with a Hebrew gentilic without its
-î (כְּנַעֲנִי, Χανανα-ῖος), and less readily with a place (אַשּׁוּר,
Ἀσσυρ-ίων). The split that contributes most is the one the stage reads the
name's case from (`best_split`).
"""

from __future__ import annotations

import functools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from lxx_tagging.text import bare

HEBREW_SOUNDS = tuple("ʔbgdhwzḥṭyklmnsʕpṣqršśt") + tuple("aeiouə")
GREEK_LETTERS = tuple("αβγδεζηθικλμνξοπρστυφχψω") + ("ου",)
DELETE = ""

# The Septuagint's conventions, as prior counts: (Hebrew sound, Greek letter
# or "" for nothing, weight). Doubling is allowed for every consonant with a
# small prior (`DOUBLE_PRIOR`).
CONVENTIONS: tuple[tuple[str, str, float], ...] = (
    ("ʔ", "", 8), ("ʔ", "α", 0.5), ("ʔ", "ε", 0.5),
    ("b", "β", 8), ("b", "μ", 0.3), ("b", "φ", 0.3),
    ("g", "γ", 8), ("g", "κ", 0.5),
    ("d", "δ", 8), ("d", "θ", 0.5), ("d", "τ", 0.3), ("d", "ρ", 0.5),
    ("h", "", 6), ("h", "ε", 0.5), ("h", "α", 0.5),
    ("w", "ου", 4), ("w", "υ", 3), ("w", "β", 2), ("w", "ο", 1), ("w", "ω", 1), ("w", "", 1),
    ("z", "ζ", 8), ("z", "σ", 1),
    ("ḥ", "χ", 6), ("ḥ", "", 4), ("ḥ", "κ", 0.3),
    ("ṭ", "τ", 8), ("ṭ", "θ", 1),
    ("y", "ι", 6), ("y", "", 2), ("y", "ε", 0.5), ("y", "η", 0.3),
    ("k", "χ", 6), ("k", "κ", 3),
    ("l", "λ", 8), ("l", "ρ", 0.3),
    ("m", "μ", 8), ("m", "ν", 1),
    ("n", "ν", 8), ("n", "μ", 0.3), ("n", "", 0.3),
    ("s", "σ", 8), ("s", "ξ", 0.3), ("s", "ζ", 0.3),
    ("ʕ", "", 6), ("ʕ", "γ", 2), ("ʕ", "α", 0.3), ("ʕ", "ε", 0.3),
    ("p", "φ", 6), ("p", "π", 2),
    ("ṣ", "σ", 8), ("ṣ", "ζ", 0.5), ("ṣ", "ξ", 0.3),
    ("q", "κ", 6), ("q", "χ", 2), ("q", "γ", 0.3),
    ("r", "ρ", 8), ("r", "λ", 0.3), ("r", "δ", 0.5),
    ("š", "σ", 8), ("š", "ξ", 0.3),
    ("ś", "σ", 8),
    ("t", "θ", 6), ("t", "τ", 3),
    ("a", "α", 8), ("a", "ε", 2), ("a", "η", 1.5), ("a", "ο", 0.5), ("a", "ω", 0.5), ("a", "", 1.5),
    ("e", "ε", 5), ("e", "η", 3), ("e", "α", 2), ("e", "ι", 1), ("e", "", 1.5),
    ("i", "ι", 6), ("i", "ε", 2), ("i", "η", 1), ("i", "", 1),
    ("o", "ω", 5), ("o", "ο", 4), ("o", "ου", 2), ("o", "α", 0.5), ("o", "", 1),
    ("u", "ου", 6), ("u", "ο", 2), ("u", "υ", 1), ("u", "ω", 1), ("u", "", 0.5),
    ("ə", "", 6), ("ə", "ε", 3), ("ə", "α", 2), ("ə", "ο", 0.5), ("ə", "ι", 0.3),
)
# What the translators insert: vowels most, the glides μ, β, δ (Μαμβρῆ,
# Ἔσδρας), and γ before γ/κ/χ (a nasal).
INSERTED_PRIOR = {"α": 3, "ε": 3, "ι": 2, "ο": 1, "η": 1, "ω": 0.5, "ου": 0.5, "υ": 0.5,
                  "μ": 0.5, "β": 0.5, "δ": 0.5, "ν": 0.5, "σ": 0.5, "γ": 0.3, "θ": 0.2, "χ": 0.2, "ρ": 0.2}
BACKGROUND = 0.02        # the prior count of every other substitution
DOUBLE_PRIOR = 0.3       # of doubling a consonant's usual letter
INSERT_PRIOR = (1.0, 9.0)  # pseudo-counts for (insert, do not insert)
MIN_PROBABILITY = 1e-6


def greek_units(letters: str) -> tuple[str, ...]:
    """Greek letters (`text.bare` of a key) as the model's units: ου is one."""
    units: list[str] = []
    index = 0
    while index < len(letters):
        if letters.startswith("ου", index):
            units.append("ου")
            index += 2
            continue
        char = letters[index]
        units.append("σ" if char == "ς" else char)
        index += 1
    return tuple(units)


# ---------------------------------------------------------------------------
# Greek endings

# Case endings a Greek name can take (letters, without accents), and the
# gentilic suffixes, each ending as the stem's last letters plus a case
# ending. Priors are shares of all forms: most of the Septuagint's Hebrew
# names are indeclinable.
CASE_ENDINGS = (
    "σ", "ν", "ου", "οσ", "ω", "ον", "οι", "ων", "οισ", "ουσ", "ασ", "α", "αν", "ησ", "η", "ην", "ε", "αι",
    "αισ", "ισ", "ιν", "ιδοσ", "εωσ", "ι",
)
GENTILIC_ENDINGS = (
    "αιοσ", "αιου", "αιω", "αιον", "αιοι", "αιων", "αιοισ", "αιουσ", "αια", "αιαν", "αιασ",
    "ιτησ", "ιτου", "ιτη", "ιτην", "ιται", "ιτων", "ιταισ", "ιτασ", "ιτισ", "ιτιδοσ",
    "ιοσ", "ιου", "ιω", "ιον", "ιοι", "ιων", "ιοισ", "ιουσ", "ηνοσ", "ηνοι", "ηνων",
)
P_INDECLINABLE = 0.55
P_CASE = 0.38
P_GENTILIC = 0.07


@dataclass(frozen=True)
class Split:
    stem: tuple[str, ...]
    ending: str          # the Greek letters stripped ("" for none)
    gentilic: bool


@functools.lru_cache(maxsize=1 << 16)
def decompositions(letters: str) -> tuple[tuple[Split, float], ...]:
    """The ways a Greek form (letters) splits into stem and ending, with each
    split's prior probability; a stem keeps at least two letters."""
    found: list[tuple[Split, float]] = [(Split(greek_units(letters), "", False), P_INDECLINABLE)]
    cases = [e for e in CASE_ENDINGS if letters.endswith(e) and len(letters) - len(e) >= 2]
    for ending in cases:
        found.append((Split(greek_units(letters[:-len(ending)]), ending, False), P_CASE / len(CASE_ENDINGS)))
    gentilics = [e for e in GENTILIC_ENDINGS if letters.endswith(e) and len(letters) - len(e) >= 2]
    for ending in gentilics:
        found.append((Split(greek_units(letters[:-len(ending)]), ending, True), P_GENTILIC / len(GENTILIC_ENDINGS)))
    return tuple(found)


# Hebrew forms the Greek may follow, by kind of name: (variant name,
# whether it pairs with a gentilic suffix, prior).
def hebrew_pairings(kind: str, variants: Sequence[tuple[tuple[str, ...], str]]):
    """(sounds, gentilic suffix expected?, prior) for each Hebrew variant."""
    out = []
    plain = [v for v in variants if v[1] != "gentilic"]
    for sounds_, name in plain:
        share = 1.0 / len(plain)
        out.append((sounds_, False, share))
        if kind == "place":
            out.append((sounds_, True, share * 0.5))
    for sounds_, name in variants:
        if name == "gentilic":
            out.append((sounds_, True, 1.0))
    return out


# ---------------------------------------------------------------------------
# The transducer

@dataclass
class Model:
    """Edit probabilities: for each Hebrew sound, P(Greek letter), P(the letter
    doubled), P(nothing); the insertion probability and inserted letters; and
    the null model's letter frequencies and stop probability."""

    substitute: dict[str, dict[str, float]] = field(default_factory=dict)
    double: dict[str, dict[str, float]] = field(default_factory=dict)
    delete: dict[str, float] = field(default_factory=dict)
    insert: float = 0.1
    inserted: dict[str, float] = field(default_factory=dict)
    background: dict[str, float] = field(default_factory=dict)
    stop: float = 0.15

    def __post_init__(self) -> None:
        self._cache: dict = {}

    # -- probabilities ------------------------------------------------------
    def log_p(self, greek: tuple[str, ...], hebrew: tuple[str, ...]) -> float:
        """log P(greek | hebrew), summed over every alignment."""
        key = (greek, hebrew)
        found = self._cache.get(key)
        if found is None:
            forward = self._forward(greek, hebrew)
            total = forward[len(hebrew)][len(greek)] * (1 - self.insert)
            found = math.log(total) if total > 0 else -1e9
            self._cache[key] = found
        return found

    def log_p0(self, greek: tuple[str, ...]) -> float:
        """log P0(greek): the letters' own frequencies, and a stop."""
        floor = MIN_PROBABILITY
        return (sum(math.log(self.background.get(g, floor)) for g in greek)
                + len(greek) * math.log(1 - self.stop) + math.log(self.stop))

    def _forward(self, greek, hebrew):
        n, m = len(hebrew), len(greek)
        ins, keep = self.insert, 1 - self.insert
        inserted = self.inserted
        table = [[0.0] * (m + 1) for _ in range(n + 1)]
        table[0][0] = 1.0
        for i in range(n + 1):
            row = table[i]
            if i:
                h = hebrew[i - 1]
                sub = self.substitute.get(h, {})
                dbl = self.double.get(h, {})
                dele = self.delete.get(h, MIN_PROBABILITY)
                above = table[i - 1]
            for j in range(m + 1):
                value = row[j]
                if i:
                    value += above[j] * keep * dele
                    if j:
                        g = greek[j - 1]
                        value += above[j - 1] * keep * sub.get(g, MIN_PROBABILITY)
                        if j > 1 and greek[j - 2] == g:
                            value += above[j - 2] * keep * dbl.get(g, MIN_PROBABILITY)
                if j:
                    value += row[j - 1] * ins * inserted.get(greek[j - 1], MIN_PROBABILITY)
                row[j] = value
        return table

    def _backward(self, greek, hebrew):
        n, m = len(hebrew), len(greek)
        ins, keep = self.insert, 1 - self.insert
        table = [[0.0] * (m + 1) for _ in range(n + 1)]
        table[n][m] = 1 - self.insert
        for i in range(n, -1, -1):
            row = table[i]
            if i < n:
                h = hebrew[i]
                sub = self.substitute.get(h, {})
                dbl = self.double.get(h, {})
                dele = self.delete.get(h, MIN_PROBABILITY)
                below = table[i + 1]
            for j in range(m, -1, -1):
                if i == n and j == m:
                    continue
                value = 0.0
                if j < m:
                    value += ins * self.inserted.get(greek[j], MIN_PROBABILITY) * row[j + 1]
                if i < n:
                    value += keep * dele * below[j]
                    if j < m:
                        value += keep * sub.get(greek[j], MIN_PROBABILITY) * below[j + 1]
                        if j + 1 < m and greek[j + 1] == greek[j]:
                            value += keep * dbl.get(greek[j], MIN_PROBABILITY) * below[j + 2]
                row[j] = value
        return table

    # -- scores -------------------------------------------------------------
    def score(self, letters: str, kind: str, variants) -> float:
        """The log-likelihood ratio of a Greek form (letters, see `text.bare`)
        as a rendering of the Hebrew name (its kind and `sound_variants`)
        against the null model, summed over the Greek form's splits."""
        return self.explain(letters, kind, variants)[0]

    def explain(self, letters: str, kind: str, variants) -> tuple[float, Split | None]:
        """The score and the split that contributes most to it."""
        key = ("explain", letters, kind, tuple(variants))
        found = self._cache.get(key)
        if found is not None:
            return found
        splits = decompositions(letters)
        numerator = 0.0
        best, best_value = None, 0.0
        for split, prior in splits:
            for sounds_, gentilic, share in hebrew_pairings(kind, variants):
                if split.gentilic != gentilic:
                    continue
                value = prior * share * math.exp(self.log_p(split.stem, sounds_))
                numerator += value
                if value > best_value:
                    best, best_value = split, value
        null = sum(prior * math.exp(self.log_p0(split.stem)) for split, prior in splits)
        if numerator <= 0 or null <= 0:
            found = (-1e9, None)
        else:
            found = (math.log(numerator) - math.log(null), best)
        self._cache[key] = found
        return found

    # -- persistence ----------------------------------------------------------
    def to_json(self) -> dict:
        return {"substitute": self.substitute, "double": self.double, "delete": self.delete,
                "insert": self.insert, "inserted": self.inserted, "background": self.background,
                "stop": self.stop}

    @classmethod
    def from_json(cls, data: dict) -> "Model":
        return cls(data["substitute"], data["double"], data["delete"], data["insert"], data["inserted"],
                   data["background"], data["stop"])


# ---------------------------------------------------------------------------
# Fitting

def prior_counts() -> tuple[dict, dict, dict, Counter, list[float]]:
    """The conventions as pseudo-counts (substitute, double, delete, inserted, insert)."""
    substitute: dict[str, Counter] = {h: Counter({g: BACKGROUND for g in GREEK_LETTERS}) for h in HEBREW_SOUNDS}
    double: dict[str, Counter] = {h: Counter({g: BACKGROUND / 10 for g in GREEK_LETTERS}) for h in HEBREW_SOUNDS}
    delete: Counter = Counter({h: BACKGROUND for h in HEBREW_SOUNDS})
    for h, g, weight in CONVENTIONS:
        if g == DELETE:
            delete[h] += weight
        else:
            substitute[h][g] += weight
            if h not in "aeiouə" and g not in ("ου",):
                double[h][g] += DOUBLE_PRIOR * weight / 8
    inserted = Counter({g: BACKGROUND for g in GREEK_LETTERS})
    inserted.update(INSERTED_PRIOR)
    return substitute, double, delete, inserted, list(INSERT_PRIOR)


def _normalise(substitute, double, delete, inserted, insert, background, stop) -> Model:
    model = Model()
    for h in HEBREW_SOUNDS:
        total = sum(substitute[h].values()) + sum(double[h].values()) + delete[h]
        model.substitute[h] = {g: c / total for g, c in substitute[h].items()}
        model.double[h] = {g: c / total for g, c in double[h].items()}
        model.delete[h] = delete[h] / total
    total = sum(inserted.values())
    model.inserted = {g: c / total for g, c in inserted.items()}
    model.insert = insert[0] / (insert[0] + insert[1])
    model.background = background
    model.stop = stop
    return model


def null_model(forms: Iterable[str]) -> tuple[dict[str, float], float]:
    """Letter frequencies and the stop probability of Greek name forms (letters)."""
    counts: Counter = Counter({g: 0.5 for g in GREEK_LETTERS})
    words = 0
    for letters in forms:
        counts.update(greek_units(letters))
        words += 1
    total = sum(counts.values())
    stop = (words + 1) / (total + words + 2)
    return {g: c / total for g, c in counts.items()}, stop


def initial(background: dict[str, float], stop: float) -> Model:
    return _normalise(*prior_counts(), background, stop)


def fit(pairs: Sequence[tuple[str, str, Sequence]], background: dict[str, float], stop: float,
        iterations: int = 8, model: Model | None = None) -> Model:
    """Fit by expectation maximisation to (Greek letters, Hebrew kind, Hebrew
    `sound_variants`) pairs, from the conventions' prior counts. Each pair's
    Greek split and Hebrew variant are latent too: its expected counts are
    shared among them by their posterior weight."""
    model = model or initial(background, stop)
    for _ in range(iterations):
        substitute, double, delete, inserted, insert = prior_counts()
        for letters, kind, variants in pairs:
            options = []
            for split, prior in decompositions(letters):
                for sounds_, gentilic, share in hebrew_pairings(kind, variants):
                    if split.gentilic != gentilic or not split.stem:
                        continue
                    options.append((prior * share * math.exp(model.log_p(split.stem, sounds_)), split.stem, sounds_))
            total = sum(weight for weight, _, _ in options)
            if total <= 0:
                continue
            for weight, greek, hebrew in options:
                share_ = weight / total
                if share_ < 1e-3:
                    continue
                _expected_counts(model, greek, hebrew, share_, substitute, double, delete, inserted, insert)
        model = _normalise(substitute, double, delete, inserted, insert, background, stop)
    return model


def _expected_counts(model: Model, greek, hebrew, weight, substitute, double, delete, inserted, insert) -> None:
    forward = model._forward(greek, hebrew)
    backward = model._backward(greek, hebrew)
    n, m = len(hebrew), len(greek)
    total = forward[n][m] * (1 - model.insert)
    if total <= 0:
        return
    scale = weight / total
    ins, keep = model.insert, 1 - model.insert
    for i in range(n + 1):
        for j in range(m + 1):
            f = forward[i][j]
            if f == 0:
                continue
            if j < m:
                g = greek[j]
                c = f * ins * model.inserted.get(g, MIN_PROBABILITY) * backward[i][j + 1] * scale
                inserted[g] += c
                insert[0] += c
            if i < n:
                h = hebrew[i]
                moved = 0.0
                c = f * keep * model.delete.get(h, MIN_PROBABILITY) * backward[i + 1][j] * scale
                delete[h] += c
                moved += c
                if j < m:
                    g = greek[j]
                    c = f * keep * model.substitute[h].get(g, MIN_PROBABILITY) * backward[i + 1][j + 1] * scale
                    substitute[h][g] += c
                    moved += c
                    if j + 1 < m and greek[j + 1] == g:
                        c = f * keep * model.double[h].get(g, MIN_PROBABILITY) * backward[i + 1][j + 2] * scale
                        double[h][g] += c
                        moved += c
                insert[1] += moved
    # The final "stop inserting" step.
    insert[1] += weight


def letters_of(key: str) -> str:
    """A Greek key's letters, for the model: `text.bare`, final sigma folded."""
    return bare(key).replace("ς", "σ")
