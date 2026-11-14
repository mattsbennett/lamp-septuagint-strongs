"""Reader for the CCAT/CATSS morphologically analysed Septuagint (.mlxx).

Scoring-only: part of the private harness around ``validate_ccat.py``.  The
CCAT files are read from ``.cache/lxx_validation/ccat/`` and never copied
anywhere tracked; nothing in the tagger may import this module.

File format (fixed columns, Latin-1, one word per line, blank line between
verses)::

    <Book> <chapter>:<verse>[letter]          verse header
    <form, 25 cols><type, 4 cols><parse, 7 cols><lemma> [verb prefixes...]

* form and lemma are beta code (``*`` capital, ``)`` ``(`` breathings,
  ``/`` ``\\`` ``=`` accents, ``|`` iota subscript, ``+`` diaeresis);
* type: part of speech plus inflection class (``N2``, ``A1B``, ``VAI``,
  ``RA``, ``C``, ...; crasis is written ``C+RP`` and may run into the parse
  column);
* parse: case/number/gender, or tense/voice/mood/person/number(/participle
  case-number-gender);
* verbs list their preverbs after the simplex lemma (outermost first).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .ccat_greek import beta_to_unicode, form_key
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_greek import beta_to_unicode, form_key

# Brenton book id -> list of candidate CCAT texts; each text is a list of
# files read in order.  Where CCAT has two Greek texts of one book the
# harness measures which one Brenton prints (``choose_text``).
CCAT_TEXTS: dict[str, dict[str, list[str]]] = {
    "Gen": {"Gen": ["01.Gen.1.mlxx", "02.Gen.2.mlxx"]},
    "Exod": {"Exod": ["03.Exod.mlxx"]},
    "Lev": {"Lev": ["04.Lev.mlxx"]},
    "Num": {"Num": ["05.Num.mlxx"]},
    "Deut": {"Deut": ["06.Deut.mlxx"]},
    # JoshA is a short alternative text of parts of chs. 15-19 only.
    "Josh": {"JoshB": ["07.JoshB.mlxx"]},
    "Judg": {"JudgB": ["09.JudgesB.mlxx"], "JudgA": ["10.JudgesA.mlxx"]},
    "Ruth": {"Ruth": ["11.Ruth.mlxx"]},
    "1Sam": {"1Sam": ["12.1Sam.mlxx"]},
    "2Sam": {"2Sam": ["13.2Sam.mlxx"]},
    "1Kgs": {"1Kgs": ["14.1Kings.mlxx"]},
    "2Kgs": {"2Kgs": ["15.2Kings.mlxx"]},
    "1Chr": {"1Chr": ["16.1Chron.mlxx"]},
    "2Chr": {"2Chr": ["17.2Chron.mlxx"]},
    "Ezra": {"2Esdr": ["19.2Esdras.mlxx"]},
    "Esth": {"Esth": ["20.Esther.mlxx"]},
    "Job": {"Job": ["34.Job.mlxx"]},
    "Ps": {"Ps": ["28.Psalms1.mlxx", "29.Psalms2.mlxx"]},
    "Prov": {"Prov": ["31.Proverbs.mlxx"]},
    "Eccl": {"Qoh": ["32.Qoheleth.mlxx"]},
    "Song": {"Cant": ["33.Canticles.mlxx"]},
    "Isa": {"Isa": ["50.Isaiah1.mlxx", "51.Isaiah2.mlxx"]},
    "Jer": {"Jer": ["52.Jer1.mlxx", "53.Jer2.mlxx"]},
    "Lam": {"Lam": ["56.Lam.mlxx"]},
    "Ezek": {"Ezek": ["57.Ezek1.mlxx", "58.Ezek2.mlxx"]},
    "Dan": {"DanTh": ["62.DanielTh.mlxx"], "DanOG": ["61.DanielOG.mlxx"]},
    "Hos": {"Hos": ["38.Hosea.mlxx"]},
    "Joel": {"Joel": ["41.Joel.mlxx"]},
    "Amos": {"Amos": ["40.Amos.mlxx"]},
    "Obad": {"Obad": ["43.Obadiah.mlxx"]},
    "Jonah": {"Jonah": ["42.Jonah.mlxx"]},
    "Mic": {"Mic": ["39.Micah.mlxx"]},
    "Nah": {"Nah": ["44.Nahum.mlxx"]},
    "Hab": {"Hab": ["45.Habakkuk.mlxx"]},
    "Zeph": {"Zeph": ["46.Zeph.mlxx"]},
    "Hag": {"Hag": ["47.Haggai.mlxx"]},
    "Zech": {"Zech": ["48.Zech.mlxx"]},
    "Mal": {"Mal": ["49.Malachi.mlxx"]},
}


@dataclass
class CcatToken:
    form_beta: str
    type_code: str
    parse_code: str
    lemma_beta: str
    prefixes: tuple[str, ...] = ()

    @property
    def form(self) -> str:
        return beta_to_unicode(self.form_beta)

    @property
    def key(self) -> str:
        k = self.__dict__.get("_key")
        if k is None:
            k = self.__dict__["_key"] = form_key(beta_to_unicode(self.form_beta))
        return k


@dataclass
class CcatVerse:
    label: str          # header as printed after the book name, e.g. "3:13a"
    chapter: int | None
    verse: int | None
    letter: str
    tokens: list[CcatToken] = field(default_factory=list)


_LABEL_RE = re.compile(r"^(?:(\d+):)?(\d+)([a-z]*)")


def _parse_label(label: str) -> tuple[int | None, int | None, str]:
    m = _LABEL_RE.match(label.strip())
    if not m:
        return None, None, ""
    ch = int(m.group(1)) if m.group(1) else 1  # one-chapter books ("Obad 9")
    return ch, int(m.group(2)), m.group(3)


_CRASIS_TYPE_RE = re.compile(r"((?:[A-Z]+\+)+(?:R[ADIPRX]|[A-Z]))(\S*)")


def parse_token_line(line: str) -> CcatToken | None:
    """Parse one word line (fixed columns; see the module docstring)."""
    if len(line.split()) < 3:
        return None
    form = line[:25].strip()
    rest = line[25:]
    first = rest.split()[0] if rest.split() else ""
    if "+" in first:  # crasis: "C+RP  NS" or glued "C+RPNS"
        m = _CRASIS_TYPE_RE.fullmatch(first)
        type_code = m.group(1) if m else first
        glued = m.group(2) if m else ""
        idx = rest.index(first) + len(first)
        if glued:
            parse_code = glued
            fields = rest[idx:].split()
        else:
            parse_code = rest[idx:11].strip() if idx < 11 else ""
            fields = rest[max(idx, 11):].split()
    else:
        type_code = rest[:4].strip()
        # keep leading blanks: the parse is positional (an indeclinable
        # name may give only its number, in the second column)
        parse_code = rest[4:11].rstrip()
        fields = rest[11:].split()
    if not fields or not form:
        return None
    return CcatToken(form, type_code, parse_code, fields[0], tuple(fields[1:]))


def parse_mlxx(text: str) -> list[CcatVerse]:
    """Parse one .mlxx file's text into verses (file order kept).

    A header is any non-blank line with at most two fields ("Gen 1:1",
    "Obad 9", "1/3Kgs 2:35a", "Jer 7:27/28", or a bare book name); every
    other non-blank line is a word.
    """
    verses: list[CcatVerse] = []
    current: CcatVerse | None = None
    for raw in text.split("\n"):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        if len(line.split()) <= 2:
            parts = line.split()
            label = parts[1] if len(parts) == 2 else ""
            ch, v, letter = _parse_label(label)
            current = CcatVerse(label, ch, v, letter)
            verses.append(current)
            continue
        tok = parse_token_line(line)
        if tok is None:
            continue
        if current is None:
            current = CcatVerse("", None, None, "")
            verses.append(current)
        current.tokens.append(tok)
    return [v for v in verses if v.tokens]


def read_ccat_text(ccat_dir: Path, files: list[str]) -> list[CcatVerse] | None:
    verses: list[CcatVerse] = []
    for name in files:
        path = Path(ccat_dir) / name
        if not path.exists():
            return None
        verses.extend(parse_mlxx(path.read_text(encoding="latin-1")))
    return verses
