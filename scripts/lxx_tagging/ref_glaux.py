"""GLAUx Septuagint reader and AGDT-postag -> feature-bundle mapping.

Scoring-only, private (see ``validate_refs.py``).

SOURCE AND TERMS
  https://github.com/alekkeersmaekers/glaux (branch main), only the
  Hebrew-canon Septuagint files ``xml/0527-NNN.xml`` (listed in
  ``GLAUX_FILES``), fetched into ``.cache/lxx_validation/glaux/``.  README:
  "GLAUx is generally available under a CC BY-SA 4.0 license. Please note
  though that some of the texts and/or the manual annotations may have a more
  restrictive license."  metadata.txt gives the Septuagint text's source as
  https://el.wikisource.org, "CC-BY-SA 3.0".  The annotation is automatic
  (README: morphology 97.2%, lemmas 98.8%), so GLAUx is used for the
  three-way comparison with CCAT, not as gold.

MAPPING (AGDT 9-position postag)
  1 pos      n noun, v verb (t participle), a adj, d adv, l article,
             g particle, c and b conj (b: GLAUx's coordinating καί/δέ), r prep,
             p pronoun, m numeral, i/e interj.; u punctuation is not a word;
             z (lemma "E") is unmapped
  2 person   1 2 3;   3 number s p d;   4 tense p i r l f a (t fut. perfect
             -> unmapped);   5 mood i s o n m p;   6 voice a m p, e (medio-
             passive) -> either;   7 gender m f n, c (common) -> none;
             8 case n g d a v;
  9 degree   c s.
  A proper name is a noun whose lemma is capitalised.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .ccat_codes import Feats, Unmapped, compare_lemma_text, lemma_alternatives
    from .ref_common import RefToken, group_verses
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_codes import Feats, Unmapped, compare_lemma_text, lemma_alternatives
    from ref_common import RefToken, group_verses

GLAUX_BASE = "https://raw.githubusercontent.com/alekkeersmaekers/glaux/main/"
# Brenton book -> {text name: GLAUx file}; Daniel has two texts.
GLAUX_FILES = {
    "Gen": {"Gen": "0527-001"}, "Exod": {"Exod": "0527-002"}, "Lev": {"Lev": "0527-003"},
    "Num": {"Num": "0527-004"}, "Deut": {"Deut": "0527-005"}, "Josh": {"Josh": "0527-006"},
    "Judg": {"Judg": "0527-009"}, "Ruth": {"Ruth": "0527-010"}, "1Sam": {"1Sam": "0527-011"},
    "2Sam": {"2Sam": "0527-012"}, "1Kgs": {"1Kgs": "0527-013"}, "2Kgs": {"2Kgs": "0527-014"},
    "1Chr": {"1Chr": "0527-015"}, "2Chr": {"2Chr": "0527-016"}, "Ezra": {"2Esdr": "0527-018"},
    "Esth": {"Esth": "0527-019"}, "Ps": {"Ps": "0527-027"}, "Prov": {"Prov": "0527-029"},
    "Eccl": {"Eccl": "0527-030"}, "Song": {"Song": "0527-031"}, "Job": {"Job": "0527-032"},
    "Hos": {"Hos": "0527-036"}, "Amos": {"Amos": "0527-037"}, "Mic": {"Mic": "0527-038"},
    "Joel": {"Joel": "0527-039"}, "Obad": {"Obad": "0527-040"}, "Jonah": {"Jonah": "0527-041"},
    "Nah": {"Nah": "0527-042"}, "Hab": {"Hab": "0527-043"}, "Zeph": {"Zeph": "0527-044"},
    "Hag": {"Hag": "0527-045"}, "Zech": {"Zech": "0527-046"}, "Mal": {"Mal": "0527-047"},
    "Isa": {"Isa": "0527-048"}, "Jer": {"Jer": "0527-049"}, "Lam": {"Lam": "0527-051"},
    "Ezek": {"Ezek": "0527-053"}, "Dan": {"Dan-a": "0527-056", "Dan-b": "0527-057"},
}

_POS = {"n": "noun", "v": "verb", "t": "verb", "a": "adj", "d": "adv", "l": "art", "g": "ptcl",
        "c": "conj", "b": "conj", "r": "prep", "p": "pron", "m": "num", "i": "intj", "e": "intj"}
_TENSE = {"p": "P", "i": "I", "r": "R", "l": "L", "f": "F", "a": "A"}
_MOOD = {"i": "I", "s": "S", "o": "O", "n": "N", "m": "M", "p": "P"}
_VOICE = {"a": "A", "m": "M", "p": "P", "e": "MP"}
_CASE = {"n": "N", "g": "G", "d": "D", "a": "A", "v": "V"}
_NUMBER = {"s": "S", "p": "P", "d": "D"}
_GENDER = {"m": "M", "f": "F", "n": "N", "c": ""}  # c: common (1st/2nd-person pronouns)


def _get(table, ch):
    if ch in ("-", "_", ""):
        return ""
    if ch not in table:
        raise Unmapped
    return table[ch]


def decompose_agdt(postag: str, lemma: str = "") -> Feats:
    t = (postag or "").ljust(9, "-")
    if t[0] not in _POS:
        raise Unmapped
    f = Feats(pos=_POS[t[0]])
    if t[0] == "t":
        f.mood = "P"
    if f.pos == "noun" and lemma[:1].isupper():
        f.proper, f.sub = True, "proper"
    f.person = _get({"1": "1", "2": "2", "3": "3"}, t[1])
    f.number = _get(_NUMBER, t[2])
    if f.pos == "verb":
        f.tense = _get(_TENSE, t[3])
        f.mood = f.mood or _get(_MOOD, t[4])
        f.voice = _get(_VOICE, t[5])
        if not f.tense or not f.mood:
            raise Unmapped
    f.gender = _get(_GENDER, t[6])
    f.case = _get(_CASE, t[7])
    f.degree = _get({"c": "C", "s": "S"}, t[8])
    return f


def read_glaux_file(path: Path):
    """(chapter, verse, RefToken) per word, punctuation skipped."""
    root = ET.parse(path).getroot()
    for w in root.iter("word"):
        tag = w.get("postag", "")
        if not tag or tag[0] == "u":
            continue
        sec = w.get("div_section", "")
        m = re.match(r"^(\d+)\.(\d+)", sec)
        if m:
            ch, v = int(m.group(1)), int(m.group(2))
        elif re.match(r"^\d+$", sec):
            ch, v = 1, int(sec)
        else:
            continue
        yield ch, v, RefToken(w.get("form", ""), lemma=w.get("lemma", ""), tag=tag)


def read_glaux(directory: Path, book: str) -> dict | None:
    """{text name: RefVerses} for one Brenton book, or None if not fetched."""
    out = {}
    for name, fid in GLAUX_FILES.get(book, {}).items():
        path = Path(directory) / f"{fid}.xml"
        if path.exists():
            out[name] = group_verses(read_glaux_file(path))
    return out or None


class GlauxAdapter:
    name = "GLAUx"
    unmapped = "unmapped GLAUx tag"

    @staticmethod
    def feats(tok):
        return decompose_agdt(tok.tag, tok.lemma)

    @staticmethod
    def lemma(our_lemma, tok, *, proper=False, ours_compound=False):
        return compare_lemma_text(our_lemma, tok.lemma, proper=proper, ours_compound=ours_compound)

    @staticmethod
    def lemma_set(tok):
        return lemma_alternatives(tok.lemma)
