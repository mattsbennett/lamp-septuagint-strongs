"""Greek text helpers for the private CCAT validation harness.

Part of the scoring-only harness around ``validate_ccat.py``: nothing in the
tagger may import this module (see the firewall notes in ``validate_ccat.py``).

What is here:

* ``beta_to_unicode``: TLG/CCAT beta code to NFC polytonic Greek.
* ``strip_greek``: an accent-, breathing-, case- and final-sigma-insensitive
  form, used for every comparison the harness makes.  The harness keeps its
  own normaliser on purpose so that its measurement does not move when the
  tagger's normaliser changes.
* ``form_key`` / ``loose_key``: matching keys for aligning Brenton's words
  with Rahlfs' words (movable nu, elision, οὐκ/οὐχ, ἐκ/ἐξ, οὕτως/οὕτω, ...).
* ``tokenize_brenton``: a minimal word tokenizer for verses the tagger has
  not tokenised.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------
# Beta code
# --------------------------------------------------------------------------

_BETA_LETTERS = {
    "A": "α", "B": "β", "G": "γ", "D": "δ", "E": "ε", "Z": "ζ", "H": "η",
    "Q": "θ", "I": "ι", "K": "κ", "L": "λ", "M": "μ", "N": "ν", "C": "ξ",
    "O": "ο", "P": "π", "R": "ρ", "S": "σ", "T": "τ", "U": "υ", "F": "φ",
    "X": "χ", "Y": "ψ", "W": "ω", "V": "ϝ", "J": "ς",
}
_BETA_MARKS = {
    ")": "̓",  # smooth breathing
    "(": "̔",  # rough breathing
    "/": "́",  # acute
    "\\": "̀",  # grave
    "=": "͂",  # circumflex
    "|": "ͅ",  # iota subscript
    "+": "̈",  # diaeresis
}
_BETA_PUNCT = {
    "'": "’",  # elision mark
    ":": "·",  # middle dot
    "#": "ʹ",  # numeral sign
    "-": "-",
    ",": ",",
    ".": ".",
    ";": ";",
}


_MARK_ORDER = {"\u0308": 0, "\u0313": 1, "\u0314": 1, "\u0301": 2, "\u0300": 2, "\u0342": 2, "\u0345": 3}


def _ordered(marks: list[str]) -> str:
    """Diaeresis, then breathing, then accent, then iota subscript (NFC order)."""
    return "".join(sorted(marks, key=lambda m: _MARK_ORDER.get(m, 9)))


def beta_to_unicode(beta: str) -> str:
    """Convert TLG/CCAT beta code (upper- or lower-case) to NFC Greek.

    ``*`` marks a capital; the breathing/accent may come between ``*`` and the
    letter (``*)A``) or after the letter.  ``S`` becomes final sigma at the end
    of a word.  Unknown characters pass through unchanged.
    """
    out: list[str] = []
    i = 0
    text = beta.upper()
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "*":
            i += 1
            pre_marks = []
            while i < n and text[i] in _BETA_MARKS:
                pre_marks.append(_BETA_MARKS[text[i]])
                i += 1
            if i < n and text[i] in _BETA_LETTERS:
                letter = "Σ" if text[i] in "SJ" else _BETA_LETTERS[text[i]].upper()
                i += 1
                post = []
                while i < n and text[i] in _BETA_MARKS:
                    post.append(_BETA_MARKS[text[i]])
                    i += 1
                out.append(letter + _ordered(pre_marks + post))
            else:
                out.append("".join(pre_marks))
            continue
        if ch in _BETA_LETTERS:
            letter = _BETA_LETTERS[ch]
            i += 1
            marks = []
            while i < n and text[i] in _BETA_MARKS:
                marks.append(_BETA_MARKS[text[i]])
                i += 1
            # Digits after S (S1/S2/S3) select sigma shapes in some TLG texts.
            if ch == "S" and i < n and text[i] in "123":
                letter = {"1": "σ", "2": "ς", "3": "ϲ"}[text[i]]
                i += 1
            elif ch == "S":
                nxt = text[i] if i < n else ""
                if not nxt or not (nxt in _BETA_LETTERS or nxt == "*"):
                    letter = "ς"
            out.append(letter + _ordered(marks))
            continue
        if ch in _BETA_MARKS:  # stray mark: attach to what precedes
            out.append(_BETA_MARKS[ch])
            i += 1
            continue
        out.append(_BETA_PUNCT.get(ch, ch))
        i += 1
    return unicodedata.normalize("NFC", "".join(out))


def beta_initial_rough(beta: str) -> bool:
    """True when a beta-code word starts with a vowel carrying rough breathing."""
    s = beta.lstrip("*")
    # breathing may precede (capital) or follow the first letter(s)
    m = re.match(r"([()/\\=|+]*)([AEHIOUWR]{1,2})([()/\\=|+]*)", s.upper())
    if not m:
        return False
    return "(" in m.group(1) or "(" in m.group(3)


def beta_initial_vowel(beta: str) -> bool:
    s = beta.lstrip("*").lstrip(")(/\\=|+")
    return bool(s) and s[0].upper() in "AEHIOUW"


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_ELISION_MARKS = "’᾽᾿'ʼ̓‘´"
_GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")


def strip_greek(text: str) -> str:
    """Lower-case, strip all diacritics and elision marks, final sigma -> σ.

    Keeps only Greek letters (and Latin letters/digits, unchanged), so
    ``"Ἐφʼ"`` -> ``"εφ"`` and ``"ᾠδῇ"`` -> ``"ωδη"``.
    """
    decomposed = unicodedata.normalize("NFD", text)
    out = []
    for ch in decomposed:
        if unicodedata.category(ch) == "Mn":
            continue
        if ch in _ELISION_MARKS:
            continue
        low = ch.lower()
        if low == "ς" or low == "ϲ":
            low = "σ"
        if low.isalnum():
            out.append(low)
    return "".join(out)


def contains_greek(text: str) -> bool:
    return bool(_GREEK_RE.search(text))


_ELIDED_ASPIRATE = {"φ": "π", "θ": "τ", "χ": "κ"}
_FORM_FIXED = {
    "ουκ": "ου", "ουχ": "ου",
    "εξ": "εκ",
    "ουτωσ": "ουτω",
    "μεχρισ": "μεχρι", "αχρισ": "αχρι",
}


def _is_elided(text: str) -> bool:
    t = text.rstrip()
    return bool(t) and t[-1] in _ELISION_MARKS


def form_key(text: str) -> str:
    """Matching key for one word form (Brenton or Rahlfs).

    Accent-insensitive, plus the purely orthographic alternations that both
    editions print differently by context: movable nu, elision (with the
    aspirated consonant before a rough breathing), οὐκ/οὐχ/οὐ, ἐκ/ἐξ,
    οὕτως/οὕτω, μέχρι(ς)/ἄχρι(ς).
    """
    elided = _is_elided(text)
    k = strip_greek(text)
    if not k:
        return k
    if elided:
        if k[-1] in _ELIDED_ASPIRATE:
            k = k[:-1] + _ELIDED_ASPIRATE[k[-1]]
        return k + "'"
    k = _FORM_FIXED.get(k, k)
    # movable nu after ε/ι (ἐστίν, ἐποίησεν, πᾶσιν); both sides get the same
    # treatment so the occasional real -εν/-ιν word only loses distinctness.
    if len(k) > 3 and k.endswith("ν") and k[-2] in "ει":
        k = k[:-1]
    return k


_LOOSE_SUBS = [
    ("λημψ", "ληψ"), ("λημφ", "ληφ"), ("γιγν", "γιν"), ("ττ", "σσ"),
    ("ει", "ι"), ("ρρ", "ρ"), ("λλ", "λ"), ("μμ", "μ"), ("νν", "ν"),
    ("σσ", "σ"), ("ππ", "π"), ("κκ", "κ"), ("ββ", "β"),
]


def loose_key(text_or_key: str, *, is_key: bool = False) -> str:
    """A looser key for second-pass pairing inside differing stretches.

    Adds spelling alternations between Rahlfs and Brenton's printing
    (λημψ/ληψ, γιγν/γιν, ττ/σσ, ει/ι, single/double consonants) and treats an
    elided form as matching its full form.  Pairs found only this way are
    counted as "orthographic variant" matches.
    """
    k = text_or_key if is_key else form_key(text_or_key)
    if k.endswith("'"):
        k = k[:-1]
    for a, b in _LOOSE_SUBS:
        k = k.replace(a, b)
    return k


def elision_compatible(a_key: str, b_key: str) -> bool:
    """True when one key is an elided form of the other (ἀπʼ ~ ἀπό)."""
    for el, full in ((a_key, b_key), (b_key, a_key)):
        if el.endswith("'") and not full.endswith("'"):
            stem = el[:-1]
            if full.startswith(stem) and len(full) == len(stem) + 1 and full[-1] in "αεοι":
                return True
            # aspirated elision: ἀφʼ (key "απ'") ~ ἀπό
    return False


# --------------------------------------------------------------------------
# Brenton tokenizer (fallback when the tagger has not tokenised a verse)
# --------------------------------------------------------------------------

_WORD_RE = re.compile("[^\\W\\d_][\\ẁ-ͯ]*[ʼ’᾽᾿']?")


def tokenize_brenton(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, surface) for each word, offsets in the given string."""
    return [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]
