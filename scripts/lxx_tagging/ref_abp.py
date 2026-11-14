"""Apostolic Bible Polyglot (ABP) reader and Strong's-number scorer.

Scoring-only, private (see ``validate_refs.py`` for the firewall).  ABP is
the one reference that numbers the Septuagint's words Strong's-style, so it
is the only direct check of our Strong's numbers (CCAT has lemmas only).

SOURCE AND TERMS
  CrossWire's SWORD module ABPGRK ("Apostolic Bible Polyglot Greek Text",
  version 1.2, 2010-02-22; "History_1.2=Added OT text"), downloaded once from
  https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/ABPGRK.zip
  into ``.cache/lxx_validation/abp/``.  Its .conf says:
      Copyright=Copyright 1996 Charles Van der Pool
      DistributionLicense=Copyrighted; Permission to distribute granted to CrossWire
  and the publisher's site (apostolicbible.com/text.htm) says "The Old & New
  Testament text is free" ("may be freely obtained").  So one copy is kept
  for private measurement; nothing from it is redistributed or written to a
  tracked file, and no ABP number for any word appears in any report.

FORMAT
  zText (BlockType=BOOK, zlib), KJV versification: ``ot.bzs`` block index
  (offset, size, uncompressed size: 3 x uint32), ``ot.bzv`` entry index
  (block, offset, size: uint32, uint32, uint16), ``ot.bzz`` the blocks.
  Entries 0-1 are module/testament headings; then per book a heading entry
  (``<div ... type="book" sID=...>``), per chapter a heading entry
  (``<chapter osisID="Gen.1" sID=...>``), then one entry per verse.  Words
  are ``<w lemma="strong:G1722" src="1">εν</w>``; several numbers for a
  crasis, ``G*`` for a proper name, decimals for ABP's own numbers.

NUMBERS (documented normalisation)
  ours   STEPBible key "G0976"/"G2264G" -> 976 / 2264 (leading zeros and the
         disambiguation letter dropped); ranges: 1-5624 classic Strong's,
         6000-6099 NASB extras, 6100-9979 "Extra Greek from Apostolic Bible
         Project" (TBESG), >= 10000 Greek forms of Hebrew names.
  ABP    "1722" -> 1722; "1510.7.3" (ABP's sub-numbers for forms of εἰμί)
         -> 1510; "180.2" (a single decimal) -> an ABP addition, i.e. a word
         not in Strong's; "*" -> a proper name without number.
  TBESG documents that G6100-G9979 come from ABP but gives no crosswalk from
  ABP's decimals to them, and the two lists do not line up by rank (checked:
  3,723 ABP decimals vs 3,739 TBESG entries, order drifting within the first
  few hundred).  So ABP additions are not number-checked: "both sides say it
  is an LXX-only word" is reported as its own (uncertain) bucket.
"""

from __future__ import annotations

import re
import struct
import zipfile
import zlib
from pathlib import Path

try:
    from .ccat_codes import compare_lemma_sets, lemma_alternatives
    from .ref_common import RefToken, RefVerse
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_codes import compare_lemma_sets, lemma_alternatives
    from ref_common import RefToken, RefVerse

ABP_ZIP_URL = "https://www.crosswire.org/ftpmirror/pub/sword/packages/rawzip/ABPGRK.zip"
ABP_CONF_URL = "https://www.crosswire.org/ftpmirror/pub/sword/raw/mods.d/abpgrk.conf"
MODULE_PATH = "modules/texts/ztext/abpgrk/"

# OSIS book -> (Brenton book, chapter offset).  Brenton's Ezra is 2 Esdras,
# with Nehemiah as its chapters 11-23.
OSIS_TO_BRENTON = {
    "Gen": ("Gen", 0), "Exod": ("Exod", 0), "Lev": ("Lev", 0), "Num": ("Num", 0),
    "Deut": ("Deut", 0), "Josh": ("Josh", 0), "Judg": ("Judg", 0), "Ruth": ("Ruth", 0),
    "1Sam": ("1Sam", 0), "2Sam": ("2Sam", 0), "1Kgs": ("1Kgs", 0), "2Kgs": ("2Kgs", 0),
    "1Chr": ("1Chr", 0), "2Chr": ("2Chr", 0), "Ezra": ("Ezra", 0), "Neh": ("Ezra", 10),
    "Esth": ("Esth", 0), "Job": ("Job", 0), "Ps": ("Ps", 0), "Prov": ("Prov", 0),
    "Eccl": ("Eccl", 0), "Song": ("Song", 0), "Isa": ("Isa", 0), "Jer": ("Jer", 0),
    "Lam": ("Lam", 0), "Ezek": ("Ezek", 0), "Dan": ("Dan", 0), "Hos": ("Hos", 0),
    "Joel": ("Joel", 0), "Amos": ("Amos", 0), "Obad": ("Obad", 0), "Jonah": ("Jonah", 0),
    "Mic": ("Mic", 0), "Nah": ("Nah", 0), "Hab": ("Hab", 0), "Zeph": ("Zeph", 0),
    "Hag": ("Hag", 0), "Zech": ("Zech", 0), "Mal": ("Mal", 0),
}

_BOOK_RE = re.compile(r'<div\b[^>]*\btype="book"[^>]*>')
_CHAPTER_RE = re.compile(r'<chapter\b[^>]*\bsID="[^"]*"[^>]*>')
_OSIS_RE = re.compile(r'osisID="([^"]+)"')
_W_RE = re.compile(r'<w\b([^>]*)>(.*?)</w>', re.S)
_LEMMA_RE = re.compile(r'lemma="([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")


# ---------------------------------------------------------------------------
# Reading the SWORD module
# ---------------------------------------------------------------------------


def read_ztext_entries(bzs: bytes, bzv: bytes, bzz: bytes) -> list[str]:
    blocks = [struct.unpack("<III", bzs[i:i + 12]) for i in range(0, len(bzs) - 11, 12)]
    cache: dict[int, bytes] = {}
    out = []
    for i in range(0, len(bzv) - 9, 10):
        b, off, size = struct.unpack("<IIH", bzv[i:i + 10])
        if not size or b >= len(blocks):
            out.append("")
            continue
        if b not in cache:
            boff, bsize, _ = blocks[b]
            cache[b] = zlib.decompress(bzz[boff:boff + bsize])
        out.append(cache[b][off:off + size].decode("utf-8", "replace"))
    return out


def walk_entries(entries: list[str]):
    """(osis book, chapter, verse, entry text) for each verse entry.

    Headings are recognised by their milestones, so the walk needs no
    versification table: entries 0-1 are skipped, a book heading starts a
    book, a chapter heading starts a chapter, every other entry is the next
    verse (an empty entry is a verse ABP leaves out).
    """
    book, chapter, verse = None, 0, 0
    for i, text in enumerate(entries):
        if i < 2:
            continue
        m = _BOOK_RE.search(text)
        # A start has an sID. The marker that ends a book sits in its last
        # verse's entry with an eID, and its osisID= also contains "sID=".
        if m and re.search(r'\bsID="', m.group(0)):
            o = _OSIS_RE.search(m.group(0))
            book, chapter, verse = (o.group(1) if o else book), 0, 0
            continue
        m = _CHAPTER_RE.search(text)
        if m:
            o = _OSIS_RE.search(m.group(0))
            if o and "." in o.group(1):
                chapter = int(o.group(1).split(".")[1])
            else:
                chapter += 1
            verse = 0
            continue
        if book is None or chapter == 0:
            continue
        verse += 1
        yield book, chapter, verse, text


def parse_words(text: str) -> list[RefToken]:
    toks = []
    for m in _W_RE.finditer(text):
        lm = _LEMMA_RE.search(m.group(1))
        nums = []
        if lm:
            for part in lm.group(1).split():
                if part.startswith("strong:"):
                    nums.append(part[len("strong:"):].lstrip("G"))
        form = _TAG_RE.sub("", m.group(2)).strip()
        if form:
            toks.append(RefToken(form, numbers=tuple(nums)))
    return toks


def read_abp(path: Path) -> dict[str, list[RefVerse]] | None:
    """Brenton book id -> ABP's verses (KJV numbering, Nehemiah folded into Ezra)."""
    path = Path(path)
    if not path.exists():
        return None
    with zipfile.ZipFile(path) as z:
        entries = read_ztext_entries(z.read(MODULE_PATH + "ot.bzs"), z.read(MODULE_PATH + "ot.bzv"),
                                     z.read(MODULE_PATH + "ot.bzz"))
    out: dict[str, list[RefVerse]] = {}
    for osis, ch, v, text in walk_entries(entries):
        target = OSIS_TO_BRENTON.get(osis)
        if not target:
            continue
        book, offset = target
        toks = parse_words(text)
        if toks:
            out.setdefault(book, []).append(RefVerse(f"{ch + offset}:{v}", ch + offset, v, "", toks))
    return out


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

CLASSIC_MAX = 5624
ABP_EXTRA = (6100, 9979)

# Strong's (1890) gave some forms of one word their own numbers; STEPBible's
# TAGNT files them under the headword, ABP keeps Strong's numbers.  Each set
# is one headword.
FORM_NUMBER_CLASSES = [
    {1473, 1691, 1698, 1700, 3165, 3427, 3450, 2248, 2249, 2254, 2257},   # ἐγώ / ἡμεῖς
    {4771, 4571, 4671, 4675, 5209, 5210, 5213, 5216},                     # σύ / ὑμεῖς
    {1510, 1488, 1498, 1511, 1526, 2070, 2071, 2075, 2076, 2077, 2252, 2258, 5600, 2468},  # εἰμί
    {3004, 2036, 2046, 4483},                                             # λέγω / εἶπον / ἐρῶ
    {3708, 1492, 3700},                                                   # ὁράω / εἶδον / ὄψομαι
    {2068, 5315},                                                         # ἐσθίω / φαγεῖν
    {3756, 3780},                                                         # οὐ / οὐχί
]
_FORM_CLASS = {n: i for i, cls in enumerate(FORM_NUMBER_CLASSES) for n in cls}


def norm_ours(strongs: str) -> int | None:
    m = re.match(r"^[GH]?0*(\d+)", strongs.strip())
    return int(m.group(1)) if m else None


def classify_abp(num: str) -> tuple[str, int | None]:
    """("name"|"classic"|"extra"|"none", base number)."""
    num = num.strip()
    if num == "*":
        return "name", None
    if not num:
        return "none", None
    parts = num.split(".")
    if not parts[0].isdigit():
        return "none", None
    base = int(parts[0])
    if len(parts) == 1:
        return "classic", base
    if len(parts) >= 3 or base == 1510:
        return "classic", base   # ABP's sub-numbers for forms of one word (εἰμί)
    return "extra", base


def ours_is_name(n: int | None, morph: str, names: set | frozenset = frozenset()) -> bool:
    """A proper name by our morphology (-P/-L ...), by number range, or by
    TBESG (its part-of-speech column starts "N:" for names)."""
    if n is not None and (n >= 10000 or n in names):
        return True
    return bool(re.search(r"-(P|L|PG|LG|PRI)\b", morph or ""))


class Tbesg:
    """Strong's number -> TBESG headword, and the set of name numbers."""

    def __init__(self, lemmas: dict[int, str] | None = None, names: set[int] | None = None):
        self.lemmas = lemmas or {}
        self.names = names or set()

    def get(self, n, default=None):
        return self.lemmas.get(n, default)


def load_tbesg_lemmas(path: Path) -> Tbesg:
    """TBESG (open data, CC BY) headwords and name numbers, for conventions."""
    lemmas: dict[int, str] = {}
    names: set[int] = set()
    path = Path(path)
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            f = line.split("\t")
            if len(f) > 5 and re.match(r"^G\d{4,5}$", f[0]):
                n = int(f[0][1:])
                lemmas.setdefault(n, f[3])
                if f[5].startswith("N:"):
                    names.add(n)
    return Tbesg(lemmas, names)


def _headword_match(lemma_a: str | None, lemma_b: str | None) -> bool:
    if not lemma_a or not lemma_b:
        return False
    return compare_lemma_sets(lemma_a, lemma_alternatives(lemma_b))[0] in ("agree", "convention")


def _same_headword(a: int, b: int, tbesg, our_lemma: str = "") -> bool:
    """The two numbers name one headword: TBESG's headwords agree under the
    lemma conventions, or our own lemma is ABP's number's headword."""
    tb = tbesg.get(b)
    return _headword_match(tbesg.get(a), tb) or _headword_match(our_lemma, tb)


# ABP numbers every personal pronoun (μου, σου, αὐτοῦ, ὑμῶν ...) 1473.
ABP_PERSONAL = 1473
_PERSONAL_NUMBERS = FORM_NUMBER_CLASSES[0] | FORM_NUMBER_CLASSES[1] | {846}


def compare_numbers(our_strongs: str, our_morph: str, abp_numbers: tuple,
                    tbesg, our_lemma: str = "") -> tuple[str, str, str]:
    """(outcome, rule, range) for our Strong's key vs ABP's number(s).

    range is one of: classic, both-extra, abp-extra/ours-classic,
    abp-classic/ours-extra, name, compound, none.  ``tbesg`` is a ``Tbesg``
    (or a plain number -> headword dict).
    """
    if isinstance(tbesg, dict):
        tbesg = Tbesg(tbesg)
    n = norm_ours(our_strongs)
    if n is None:
        return "uncertain", "no number", "none"
    kinds = [classify_abp(x) for x in abp_numbers]
    kinds = [k for k in kinds if k[0] != "none"]
    if not kinds:
        return "uncertain", "no ABP number", "none"
    # the other numbers in our own compound code ("PRT-N + G2089=ADV")
    ours_all = {n} | {int(x) for x in re.findall(r"G0*(\d+)=", our_morph or "")}
    if len(kinds) > 1:  # crasis (κἀγώ = 2532 + 1473) or ABP's double coding
        bases = {b for k, b in kinds if b is not None}
        if ours_all & bases or any(_FORM_CLASS.get(x) is not None and _FORM_CLASS.get(x) == _FORM_CLASS.get(b)
                                   for x in ours_all for b in bases):
            return "convention", "compound number", "compound"
        return "uncertain", "compound number", "compound"
    kind, base = kinds[0]
    if our_strongs.strip().startswith("H"):
        # A name keyed by its Hebrew number shares no numbering with ABP: only
        # "is ABP's word a name at all" can be checked.
        if kind == "name":
            return "uncertain", "name (ABP gives no number)", "name"
        if kind == "extra" or base in tbesg.names:
            return "uncertain", "Hebrew-keyed name (no shared numbering)", "name"
        return "disagree", "ours a name, ABP a common word", "name"
    ours_extra = ABP_EXTRA[0] <= n <= ABP_EXTRA[1]
    is_name = ours_is_name(n, our_morph, tbesg.names)
    if kind == "name":
        if is_name:
            return "uncertain", "name (ABP gives no number)", "name"
        return "disagree", "ABP name, ours a common word", "name"
    if kind == "extra":
        if ours_extra:
            return "uncertain", "both LXX-only (no open number mapping)", "both-extra"
        if is_name:
            return "uncertain", "ABP addition vs our name number", "abp-extra/ours-classic"
        return "disagree", "ABP addition, ours a Strong's word", "abp-extra/ours-classic"
    # ABP classic number
    if n == base:
        return "agree", "", "classic"
    rng = "abp-classic/ours-extra" if n > CLASSIC_MAX else "classic"
    if base in ours_all:
        return "convention", "compound tag", rng
    if _FORM_CLASS.get(n) is not None and _FORM_CLASS.get(n) == _FORM_CLASS.get(base):
        return "convention", "form-numbered Strong's", rng
    if base == ABP_PERSONAL and n in _PERSONAL_NUMBERS:
        return "convention", "ABP personal pronoun 1473", rng
    if _same_headword(n, base, tbesg, our_lemma):
        return "convention", "same headword, other number", rng
    if (our_morph or "").count(" + ") and our_lemma and any(
            norm.startswith(o) for o in lemma_alternatives(our_lemma)
            for norm in lemma_alternatives(tbesg.get(base) or "")):
        return "convention", "compound tag", rng
    return "disagree", "", rng
