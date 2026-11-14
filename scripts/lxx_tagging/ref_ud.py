"""UD_Ancient_Greek-PTNK reader and UD -> feature-bundle mapping.

Scoring-only, private (see ``validate_refs.py``).

SOURCE AND TERMS
  https://github.com/UniversalDependencies/UD_Ancient_Greek-PTNK (master),
  files grc_ptnk-ud-{train,dev,test}.conllu, README.md, LICENSE.txt, fetched
  into ``.cache/lxx_validation/ptnk/``.  LICENSE.txt: "The treebank is
  licensed under the Creative Commons License Attribution-ShareAlike 4.0
  International."  README: "UD Ancient Greek PTNK contains portions of the
  Septuagint according to the Codex Alexandrinus" (Genesis and Ruth);
  "Lemmas: manual native / UPOS: manual native / Features: manual native".
  Being hand-checked, it is reported separately from CCAT.

MAPPING (UD -> the harness's Feats, see ccat_codes)
  UPOS   NOUN -> noun, PROPN -> noun (proper), ADJ -> adj, NUM -> num,
         DET with PronType=Art -> art, other DET and PRON -> pron (PronType
         Prs -> P, with Reflex=Yes -> F, Poss=Yes -> S; Dem -> D, Rel -> R,
         Int -> I, Ind -> X, Rcp -> C), VERB/AUX -> verb, ADP -> prep,
         ADV -> adv, CCONJ/SCONJ -> conj, PART -> ptcl, INTJ -> intj;
         PUNCT is not a word; X/SYM are unmapped.
  verbs  Tense/Aspect: Fut -> F; Pqp -> L; Past+Imp -> imperfect;
         Past+Perf or Perf without Tense -> aorist; Pres+Perf -> perfect;
         Pres (or Imp without Tense) -> present.  Mood Ind/Sub/Opt/Imp ->
         I/S/O/M, VerbForm Inf -> N, Part -> P.  Voice Act/Mid/Pass -> A/M/P,
         "Mid,Pass" -> either (the same convention as STEPBible's E).
  values Case, Number, Gender, Person, Degree map one to one; a UD value
         listing alternatives ("Acc,Nom", "Masc,Neut") is underspecified:
         our value among them counts as convention ("underspecified"),
         outside them as a disagreement.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    from .ccat_codes import Feats, Unmapped, compare_lemma_text, lemma_alternatives
    from .ref_common import RefToken, group_verses
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_codes import Feats, Unmapped, compare_lemma_text, lemma_alternatives
    from ref_common import RefToken, group_verses

PTNK_BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_Ancient_Greek-PTNK/master/"
PTNK_FILES = ["README.md", "LICENSE.txt", "grc_ptnk-ud-train.conllu", "grc_ptnk-ud-dev.conllu",
              "grc_ptnk-ud-test.conllu"]
UD_BOOKS = {"GEN": "Gen", "RUT": "Ruth", "RUTH": "Ruth", "EXO": "Exod", "LEV": "Lev", "NUM": "Num", "DEU": "Deut"}

_CASE = {"Nom": "N", "Gen": "G", "Dat": "D", "Acc": "A", "Voc": "V"}
_NUMBER = {"Sing": "S", "Plur": "P", "Dual": "D"}
_GENDER = {"Masc": "M", "Fem": "F", "Neut": "N"}
_MOOD = {"Ind": "I", "Sub": "S", "Opt": "O", "Imp": "M"}
_VOICE = {"Act": "A", "Mid": "M", "Pass": "P"}
_DEGREE = {"Cmp": "C", "Sup": "S"}
_PRON = {"Prs": "P", "Dem": "D", "Rel": "R", "Int": "I", "Ind": "X", "Rcp": "C"}


def _vals(value: str | None, table: dict) -> str:
    """Map a UD value (possibly "A,B") to our letters, alternatives joined by |."""
    if not value:
        return ""
    out = []
    for v in value.split(","):
        if v not in table:
            raise Unmapped
        out.append(table[v])
    return "|".join(sorted(set(out)))


def parse_feats(text: str) -> dict:
    if not text or text == "_":
        return {}
    return dict(x.split("=", 1) for x in text.split("|") if "=" in x)


def _tense(tense: str | None, aspect: str | None) -> str:
    if tense == "Fut":
        return "F"
    if tense == "Pqp":
        return "L"
    if tense == "Past":
        return "I" if aspect == "Imp" else "A"
    if tense == "Pres":
        return "R" if aspect == "Perf" else "P"
    if aspect == "Perf":
        return "A"
    if aspect == "Imp":
        return "P"
    return ""


def decompose_ud(upos: str, feats: dict) -> Feats:
    """UD UPOS + FEATS -> Feats (raises ``Unmapped``)."""
    if upos in ("NOUN", "PROPN"):
        f = Feats(pos="noun", proper=upos == "PROPN")
        if f.proper:
            f.sub = "proper"
    elif upos == "ADJ":
        f = Feats(pos="adj")
    elif upos == "NUM":
        f = Feats(pos="num")
    elif upos == "DET" and feats.get("PronType") == "Art":
        f = Feats(pos="art")
    elif upos in ("DET", "PRON"):
        sub = _PRON.get(feats.get("PronType", ""), "")
        if feats.get("Reflex") == "Yes":
            sub = "F"
        if feats.get("Poss") == "Yes":
            sub = "S"
        f = Feats(pos="pron", sub=sub)
    elif upos in ("VERB", "AUX"):
        f = Feats(pos="verb")
        form = feats.get("VerbForm")
        f.tense = _tense(feats.get("Tense"), feats.get("Aspect"))
        f.voice = _vals(feats.get("Voice"), _VOICE).replace("M|P", "MP")
        if form == "Inf":
            f.mood = "N"
        elif form == "Part":
            f.mood = "P"
        else:
            f.mood = _vals(feats.get("Mood"), _MOOD)
    else:
        simple = {"ADP": "prep", "ADV": "adv", "CCONJ": "conj", "SCONJ": "conj", "PART": "ptcl",
                  "INTJ": "intj"}
        if upos not in simple:
            raise Unmapped
        f = Feats(pos=simple[upos])
    f.case = _vals(feats.get("Case"), _CASE)
    f.number = _vals(feats.get("Number"), _NUMBER)
    f.gender = _vals(feats.get("Gender"), _GENDER)
    f.person = feats.get("Person", "") if f.pos in ("verb", "pron") else ""
    if f.pos == "verb" and f.mood in ("N", "P"):
        f.person = ""
    if f.pos == "verb" and f.mood not in ("P",):
        f.case = f.gender = ""
        if f.mood == "N":
            f.number = ""
    f.degree = _vals(feats.get("Degree"), _DEGREE)
    return f


_REF_RE = re.compile(r"(?:^|\|)Ref=([A-Z0-9]+)_(\d+)\.(\d+)")


def parse_conllu(text: str):
    """(book, chapter, verse, RefToken) for each word (punctuation skipped;
    a multiword token becomes one token whose ``parts`` are its words)."""
    pending_mwt = None  # (last id, token)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        c = line.split("\t")
        if len(c) < 10:
            continue
        m = _REF_RE.search(c[9])
        if "-" in c[0]:
            lo, hi = c[0].split("-")
            pending_mwt = (int(hi), RefToken(c[1]), m)
            continue
        if "." in c[0]:
            continue
        tok = RefToken(c[1], lemma=c[2], tag=f"{c[3]}|{c[5]}")
        if pending_mwt is not None:
            hi, mwt, mm = pending_mwt
            mwt.parts.append(tok)
            if int(c[0]) == hi:
                mwt.lemma = "+".join(p.lemma for p in mwt.parts)
                ref = mm or m
                if ref:
                    yield ref.group(1), int(ref.group(2)), int(ref.group(3)), mwt
                pending_mwt = None
            continue
        if c[3] == "PUNCT" or not m:
            continue
        yield m.group(1), int(m.group(2)), int(m.group(3)), tok


def read_ptnk(directory: Path) -> dict | None:
    """Brenton book id -> RefVerses, from all three PTNK files."""
    directory = Path(directory)
    files = [directory / f for f in PTNK_FILES if f.endswith(".conllu")]
    if not all(f.exists() for f in files):
        return None
    rows: dict[str, list] = {}
    for f in files:
        for book, ch, v, tok in parse_conllu(f.read_text(encoding="utf-8")):
            bid = UD_BOOKS.get(book)
            if bid:
                rows.setdefault(bid, []).append((ch, v, tok))
    out = {}
    for bid, items in rows.items():
        items.sort(key=lambda r: (r[0], r[1]))  # the files split Genesis by chapters
        out[bid] = group_verses(items)
    return out


class UdAdapter:
    name = "PTNK"
    unmapped = "unmapped UD tag"

    @staticmethod
    def feats(tok):
        if tok.parts:
            parts = [UdAdapter.feats(p) for p in tok.parts]
            return Feats(pos="crasis", parts=parts)
        upos, _, feats = tok.tag.partition("|")
        return decompose_ud(upos, parse_feats(feats))

    @staticmethod
    def lemma(our_lemma, tok, *, proper=False, ours_compound=False):
        return compare_lemma_text(our_lemma, tok.lemma, proper=proper, ours_compound=ours_compound)

    @staticmethod
    def lemma_set(tok):
        return lemma_alternatives(tok.lemma)
