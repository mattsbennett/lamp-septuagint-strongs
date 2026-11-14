"""Parse-code decomposition and lemma comparison for the CCAT harness.

Scoring-only: part of the private harness around ``validate_ccat.py``.  The
tables below describe *notational conventions* of the two tagsets (taken from
CCAT's published coding sheet and STEPBible's documented code set), not any
word's analysis; nothing in the tagger may import this module.

Both tagsets are decomposed into the same feature bundle::

    pos     noun adj num art pron verb conj prep adv ptcl intj
    sub     pronoun class (P personal, D demonstrative, R relative,
            I interrogative/indefinite, X indefinite, F reflexive,
            S possessive, C reciprocal, K/Q correlative), or "proper"
    case number gender person tense voice mood degree
    second  2nd aorist/future-passive marking (informational only)

Each compared field ends in one of four outcomes:

``agree``       same value;
``convention``  differs only by a documented notational convention
                (``CONVENTIONS`` below);
``disagree``    a real difference;
``uncertain``   cannot be compared honestly (unmapped code, dual number,
                crasis or joined word against a single analysis, ...).

The full-parse outcome is the worst field outcome (disagree > uncertain >
convention > agree), with POS included.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from .ccat_greek import beta_initial_rough, beta_initial_vowel, beta_to_unicode, strip_greek
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_greek import beta_initial_rough, beta_initial_vowel, beta_to_unicode, strip_greek

POS_ORDER = ["noun", "adj", "num", "art", "pron", "verb", "conj", "prep", "adv", "ptcl", "intj"]
INDECLINABLE = {"conj", "prep", "adv", "ptcl", "intj"}
NOMINAL_FIELDS = ("case", "number", "gender")
VERB_FIELDS = ("tense", "voice", "mood", "person", "number")
PARTICIPLE_FIELDS = ("tense", "voice", "mood", "case", "number", "gender")
FIELDS = ("case", "number", "gender", "person", "tense", "voice", "mood", "degree")

# The documented convention differences (how each is scored).  Keys are the
# rule names used in reports.  The descriptions are transliterated so that a
# report can carry them without carrying Greek text (see the firewall in
# validate_ccat.py); the Greek examples are in the comments.
CONVENTIONS = {
    # POS-level
    "substantival adjective": "noun vs adjective with the same lemma: CCAT types an adjective "
    "consistently used as a noun as N; STEPBible keeps the lexical class (or the reverse).",
    "pronominal adjective": "pronoun vs adjective with the same lemma: possessives (emos, sos), "
    "correlatives (toioutos) and reciprocals are adjectives in CCAT, S/K/C/D pronouns in STEPBible.",
    "numeral class": "indeclinable numeral (CCAT M, STEPBible A-NUI) vs a declined adjective "
    "reading of the same number word.",
    "indeclinable class": "conjunction / particle / adverb / preposition / interjection with the "
    "same lemma: CCAT types by syntax (adverbial kai = D, heos/enopion P or D, de/gar/oun X), "
    "STEPBible by lexeme (de CONJ, ou PRT-N, hos PRT ...).",
    "adverbial adjective": "adverb vs adjective of the same stem, lemma or POS (proton / protos, "
    "makran / makros).",
    "frozen case form": "adverb or preposition vs the noun/adjective case form it froze from "
    "(kyklo as an adverb vs the dative of kyklos).",
    "substantival participle": "a noun headword that is a participle of the other side's verb "
    "(oikoumene / oikeo, archon / archo).",
    "interjection headword": "ide / idou / ophelon as interjection or particle vs the verb "
    "form of horao / opheilo.",
    "compound tag": "STEPBible's compound code (main part + G-number=part, e.g. for oukheti) vs "
    "CCAT's single analysis, or CCAT crasis vs a single tag: the main part agrees.",
    "middle/passive label": "M vs P in present, imperfect, perfect or pluperfect, where the forms "
    "are identical; STEPBible E (either) and the deponent labels D / O / N vs CCAT M or P.",
    "underspecified": "a reference value listing alternatives (UD Case=Acc,Nom) that "
    "includes ours.",
    "degree marking": "CCAT marks comparative/superlative only for irregular forms; a degree "
    "given by one side and not the other is not counted against either.",
    # lemma-level
    "pronoun headword": "personal pronouns under ego/sy in CCAT vs hemeis/hymeis, possessive "
    "S- codes under ego/sy in STEPBible; article, relative, demonstrative, reciprocal headwords.",
    "suppletive verb headword": "eidon/oida/horao, eipon/ero/lego, phago/esthio, pio/pino, "
    "dramo/trecho, helein/haireo, pletho/pimplemi filed under different headwords.",
    "negative headword": "oukhi filed under ou.",
    "impersonal headword": "dei filed under deo.",
    "degree headword": "comparative/superlative headwords (pleion, meizon, kreisson, elasson ...) "
    "vs the positive.",
    "Attic/Koine spelling": "gignomai/ginomai, gignosko/ginosko, -tt-/-ss-, -rr-/-rs-.",
    "spelling variant": "ei/i and single/double consonant spellings of a headword, gm/chm, and "
    "listed pairs (borras/boreas, chthes/echthes, hals/halas, thelo/ethelo, porro/proso, "
    "nipto/nizo, ileos/ilaos ...).",
    "substantive headword": "masculine/neuter or singular/plural headword of the same "
    "substantive (akron / akros, peteinon / peteinos).",
    "deponent headword": "-omai vs -o headword (and contracted -oumai vs -eo).",
    "-mi/-o headword": "ollymi/ollyo, deiknymi/deiknyo, anoignymi/anoigo etc.",
    "contracted headword": "chrysous/chryseos, chous/choos (contracted vs uncontracted).",
    "movable final": "houtos/houto, mechri(s), achri(s), -then/-the, heneka/heneken.",
    "crasis headword": "kago vs kai + ego.",
    "elided headword": "a reference headword printed elided (tot', deur') vs our full headword.",
    "name spelling": "proper names: ei/i, single/double consonants, one-letter differences "
    "(Rahlfs vs Brenton's printing vs the Strong's headword).",
}

# ---------------------------------------------------------------------------
# Feature bundles
# ---------------------------------------------------------------------------


@dataclass
class Feats:
    pos: str
    sub: str = ""
    case: str = ""
    number: str = ""
    gender: str = ""
    person: str = ""
    tense: str = ""
    voice: str = ""
    mood: str = ""
    degree: str = ""
    second: bool | None = None
    proper: bool = False
    parts: list["Feats"] = field(default_factory=list)  # crasis / compound

    def get(self, name: str) -> str:
        return getattr(self, name)


class Unmapped(Exception):
    """A code the decomposer does not understand (reported by shape only)."""


def code_shape(type_code: str, parse_code: str = "") -> str:
    """Anonymised shape of a CCAT code: first letter of the type kept, other
    letters -> x, digits -> 9, blanks -> _ (not tied to any word)."""
    t = type_code[:1] + re.sub(r"[A-Z]", "x", re.sub(r"\d", "9", type_code[1:]))
    p = re.sub(r"[A-Z]", "x", re.sub(r"\d", "9", parse_code)).replace(" ", "_")
    return f"{t}|{p}"


# ---------------------------------------------------------------------------
# CCAT
# ---------------------------------------------------------------------------

_CCAT_TENSE = {"P": "P", "I": "I", "F": "F", "A": "A", "X": "R", "Y": "L"}
_CCAT_MOOD = {"I": "I", "D": "M", "S": "S", "O": "O", "N": "N", "P": "P"}
_CASES = set("NGDAV")
_NUMBERS = set("SDP")
_GENDERS = set("MFN")
_FIRST_PERSON = {"εγω", "ημεισ", "εμοσ", "ημετεροσ", "εμαυτου"}
_SECOND_PERSON = {"συ", "υμεισ", "σοσ", "υμετεροσ", "σεαυτου"}


def _ccat_nominal(p: str, f: Feats, *, degree: bool = False) -> None:
    p = p.ljust(4)
    c, n, g = p[0], p[1], p[2]
    if c != " ":
        if c not in _CASES:
            raise Unmapped
        f.case = c
    if n != " ":
        if n not in _NUMBERS:
            raise Unmapped
        f.number = n
    if g != " ":
        if g not in _GENDERS:
            raise Unmapped
        f.gender = g
    if degree and p[3] in "CS":
        f.degree = p[3]
    elif p[3].strip():
        raise Unmapped


def decompose_ccat(type_code: str, parse_code: str, lemma_beta: str = "") -> Feats:
    """CCAT type + parse -> Feats.  Raises ``Unmapped``."""
    t = type_code.strip()
    p = parse_code.rstrip()
    if not t:
        raise Unmapped
    if "+" in t:  # crasis: the parse belongs to the last declinable part
        pieces = t.split("+")
        lemmas = lemma_beta.split("+") if "+" in lemma_beta else [lemma_beta] * len(pieces)
        parts = []
        for k, piece in enumerate(pieces):
            last = k == len(pieces) - 1
            parts.append(decompose_ccat(piece, p if last else "", lemmas[min(k, len(lemmas) - 1)]))
        return Feats(pos="crasis", parts=parts)
    head = t[0]
    lem = strip_greek(beta_to_unicode(lemma_beta))
    if head == "N":
        f = Feats(pos="noun", proper=lemma_beta.startswith("*"))
        if f.proper:
            f.sub = "proper"
        if p.strip():
            _ccat_nominal(p, f)
        return f
    if head == "A":
        f = Feats(pos="adj")
        if p.strip():
            _ccat_nominal(p, f, degree=True)
        return f
    if head == "R":
        if len(t) < 2 or t[1] not in "ADIPRX":
            raise Unmapped
        sub = t[1]
        if sub == "A":
            f = Feats(pos="art")
        else:
            f = Feats(pos="pron", sub=sub)
            if lem in _FIRST_PERSON:
                f.person = "1"
            elif lem in _SECOND_PERSON:
                f.person = "2"
            elif lem == "αυτοσ":
                f.sub = "P"  # αὐτός: RD/RP in CCAT, personal pronoun in STEPBible
        if p.strip():
            _ccat_nominal(p, f)
        return f
    if t in ("C", "X", "I", "P", "D"):
        f = Feats(pos={"C": "conj", "X": "ptcl", "I": "intj", "P": "prep", "D": "adv"}[t])
        if p.strip() and t != "D":
            _ccat_nominal(p, f)
        elif p.strip():
            if p.strip() in ("C", "S"):
                f.degree = p.strip()
            else:
                raise Unmapped
        return f
    if head == "M":
        f = Feats(pos="num")
        if p.strip():
            _ccat_nominal(p, f)
        return f
    if head == "V":
        if len(p) < 3 or p[0] not in _CCAT_TENSE or p[1] not in "AMP" or p[2] not in _CCAT_MOOD:
            raise Unmapped
        f = Feats(pos="verb", tense=_CCAT_TENSE[p[0]], voice=p[1], mood=_CCAT_MOOD[p[2]])
        rest = p[3:].ljust(3)
        if f.mood == "P":
            _ccat_nominal(rest, f)
            if not f.case:
                raise Unmapped
        elif f.mood == "N":
            if rest.strip():
                raise Unmapped
        else:
            if rest[0] not in "123" or rest[1] not in _NUMBERS:
                raise Unmapped
            f.person, f.number = rest[0], rest[1]
        stem = t[1:3].rstrip("I") if len(t) > 1 else ""
        if f.tense == "A" or (f.tense == "F" and f.voice == "P"):
            if stem in ("B", "Z", "D"):
                f.second = True
            elif stem in ("A", "C", "V", "S", "Q"):
                f.second = False
        return f
    raise Unmapped


# ---------------------------------------------------------------------------
# STEPBible / Robinson-style codes
# ---------------------------------------------------------------------------

_ROB_VOICE = {"A": "A", "M": "M", "P": "P", "D": "M", "O": "P", "N": "MP", "E": "MP"}
_ROB_TENSE = {"P": "P", "I": "I", "F": "F", "A": "A", "R": "R", "L": "L"}
_ROB_MOOD = {"I": "I", "S": "S", "O": "O", "M": "M", "N": "N", "P": "P"}
_ROB_PRON = set("PRDIXFSCKQ")
_CNG_RE = re.compile(r"^([NGDAV])([SP])([MFN])?$")


def _rob_cng(s: str, f: Feats) -> None:
    m = _CNG_RE.match(s)
    if not m:
        raise Unmapped
    f.case, f.number, f.gender = m.group(1), m.group(2), m.group(3) or ""


def _decompose_robinson_part(code: str) -> Feats:
    code = code.strip()
    if "=" in code:  # "G1565=D-NSM" inside a compound
        code = code.split("=", 1)[1]
    bits = code.split("-")
    head = bits[0]
    simple = {
        "CONJ": "conj", "COND": "conj", "PREP": "prep", "ADV": "adv",
        "PRT": "ptcl", "INJ": "intj", "HEB": "noun", "ARAM": "noun",
    }
    if head in simple:
        f = Feats(pos=simple[head])
        for b in bits[1:]:
            if b in ("C", "S") and head == "ADV":
                f.degree = b
        return f
    if head == "N":
        f = Feats(pos="noun")
        if len(bits) >= 2 and bits[1] in ("PRI", "LI", "OI"):
            f.proper = bits[1] == "PRI"
            if f.proper:
                f.sub = "proper"
            return f
        if len(bits) < 2:
            raise Unmapped
        _rob_cng(bits[1], f)
        if any(b in ("P", "L", "PG", "LG", "T", "G", "LI") for b in bits[2:]):
            if any(b in ("P", "L", "PG", "LG", "G") for b in bits[2:]):
                f.proper, f.sub = True, "proper"
        return f
    if head == "A":
        if len(bits) >= 2 and bits[1] == "NUI":
            return Feats(pos="num")
        f = Feats(pos="adj")
        if len(bits) < 2:
            raise Unmapped
        _rob_cng(bits[1], f)
        for b in bits[2:]:
            if b in ("C", "S"):
                f.degree = b
        return f
    if head == "T":
        f = Feats(pos="art")
        if len(bits) < 2:
            raise Unmapped
        _rob_cng(bits[1], f)
        return f
    if head in _ROB_PRON and len(head) == 1:
        f = Feats(pos="pron", sub=head)
        if len(bits) < 2:
            raise Unmapped
        body = bits[1]
        if head == "S":  # S-1SGSN: person, possessor number, then case-number-gender
            m = re.match(r"^([123])([SP])(.+)$", body)
            if not m:
                raise Unmapped
            f.person = m.group(1)
            _rob_cng(m.group(3), f)
            return f
        m = re.match(r"^([123])(.+)$", body)
        if m:
            f.person = m.group(1)
            body = m.group(2)
        _rob_cng(body, f)
        return f
    if head == "V":
        if len(bits) < 2:
            raise Unmapped
        tvm = bits[1]
        second = tvm.startswith("2")
        tvm = tvm.lstrip("2")
        if len(tvm) != 3 or tvm[0] not in _ROB_TENSE or tvm[1] not in _ROB_VOICE or tvm[2] not in _ROB_MOOD:
            raise Unmapped
        f = Feats(pos="verb", tense=_ROB_TENSE[tvm[0]], voice=_ROB_VOICE[tvm[1]], mood=_ROB_MOOD[tvm[2]])
        f.sub = "deponent" if tvm[1] in "DON" else ""
        if f.tense == "A" or (f.tense == "F" and f.voice in ("P",)):
            f.second = second
        rest = [b for b in bits[2:] if b not in ("ATT", "ARAM", "HEB", "C", "M")]
        if f.mood == "P":
            if not rest:
                raise Unmapped
            _rob_cng(rest[0], f)
        elif f.mood == "N":
            pass
        else:
            if not rest:
                raise Unmapped
            m = re.match(r"^([123])([SP])$", rest[0])
            if not m:
                raise Unmapped
            f.person, f.number = m.group(1), m.group(2)
        return f
    raise Unmapped


def decompose_robinson(code: str) -> Feats:
    """STEPBible/Robinson morphology code -> Feats (``Unmapped`` if unknown).

    Compound codes (``"P-1NS + G2532=CONJ"``) come back with ``pos="crasis"``
    and the parts in the order written.
    """
    pieces = [x for x in code.split(" + ") if x.strip()]
    if not pieces:
        raise Unmapped
    if len(pieces) == 1:
        return _decompose_robinson_part(pieces[0])
    return Feats(pos="crasis", parts=[_decompose_robinson_part(x) for x in pieces])


# ---------------------------------------------------------------------------
# Lemma comparison
# ---------------------------------------------------------------------------

_PRONOUN_CLASSES = {
    "εγω": {"εγω", "ημεισ", "εμου", "μου", "εμοι", "μοι", "εμε", "με", "ημων", "ημιν", "ημασ",
            "εμοσ", "ημετεροσ", "εμαυτου"},
    "συ": {"συ", "υμεισ", "σου", "σοι", "σε", "υμων", "υμιν", "υμασ", "σοσ", "υμετεροσ", "σεαυτου"},
    "ο": {"ο", "η", "το"},
    "ουτοσ": {"ουτοσ", "αυτη", "τουτο"},
    "οδε": {"οδε", "ηδε", "τοδε"},
    "εαυτου": {"εαυτου", "αυτου"},
}
_SUPPLETIVE = {
    "οραω": ["οραω", "ειδω", "οιδα", "ειδον", "οπτομαι", "οπτανομαι", "οπτανω", "οψομαι"],
    "λεγω": ["λεγω", "ειπον", "επω", "ερεω", "ερω", "ειρω", "ρεω"],
    "εσθιω": ["εσθιω", "εσθω", "φαγω", "φαγομαι", "φαγειν"],
    "πινω": ["πινω", "πιω", "πιομαι", "ποω"],
    "τρεχω": ["τρεχω", "δρεμω", "δραμω"],
    "αιρεω": ["αιρεω", "ελω", "ελειν"],
    "ερχομαι": ["ερχομαι", "ελευθομαι", "ελευσομαι"],
    "φερω": ["φερω", "οιω", "ενεγκω"],
    "πιμπλημι": ["πιμπλημι", "πιπλημι", "πληθω"],
}
# Whole-word headword classes: (rule, canonical, members).
_HEADWORD_CLASSES = [
    ("negative headword", "ου", {"ουχι"}),
    ("interjection headword", "οραω", {"ιδε", "ιδου"}),
    ("interjection headword", "οφειλω", {"οφελον"}),
    ("impersonal headword", "δεω", {"δει"}),
    ("degree headword", "πολυσ", {"πλειων", "πλειστοσ", "πλεον"}),
    ("degree headword", "μεγασ", {"μειζων", "μεγιστοσ"}),
    ("degree headword", "αγαθοσ", {"κρεισσων", "κρειττων", "βελτιων", "αριστοσ", "κρατιστοσ",
                                  "βελτιστοσ", "αμεινων"}),
    ("degree headword", "κακοσ", {"χειρων", "χειριστοσ", "ησσων", "ηττων"}),
    ("degree headword", "ελαχυσ", {"ελασσων", "ελαττων", "ελαχιστοσ"}),
    ("degree headword", "νεοσ", {"νεωτεροσ"}),
    ("degree headword", "πρεσβυσ", {"πρεσβυτεροσ"}),
    ("spelling variant", "θελω", {"εθελω"}),
    ("spelling variant", "πορρω", {"προσω"}),
    ("spelling variant", "πορρωθεν", {"προσωθεν"}),
    ("spelling variant", "ρυομαι", {"ερυω"}),
    ("spelling variant", "νιπτω", {"νιζω"}),
    ("spelling variant", "φωσ", {"φαοσ"}),
    ("spelling variant", "ιλεωσ", {"ιλαοσ"}),
    ("spelling variant", "ωδιν", {"ωδισ"}),
    ("name spelling", "ιερουσαλημ", {"ιεροσολυμα"}),
    ("spelling variant", "ουδεισ", {"ουθεισ"}),
    ("spelling variant", "μηδεισ", {"μηθεισ"}),
    ("movable final", "εικοσι", {"εικοσιν"}),
    ("pronoun headword", "αλληλων", {"αλληλω", "αλληλοι"}),
    ("spelling variant", "βορρασ", {"βορεασ"}),
    ("spelling variant", "αλσ", {"αλασ"}),
    ("spelling variant", "χειμαρροσ", {"χειμαρρουσ"}),
    ("spelling variant", "χθεσ", {"εχθεσ"}),
    ("movable final", "ενεκα", {"ενεκεν", "εινεκεν", "εινεκα"}),
]
_PREFIX_SPELLINGS = sorted({
    "απ", "αφ", "απο", "εκ", "εξ", "κατ", "καθ", "κατα", "ανα", "αν", "επ", "εφ", "επι", "συν",
    "συμ", "συγ", "συλ", "συσ", "συρ", "συ", "εν", "εμ", "εγ", "ελ", "δια", "δι", "παρ", "παρα",
    "προσ", "εισ", "υπ", "υφ", "υπο", "περι", "αντ", "ανθ", "αντι", "προ", "υπερ", "μετ", "μεθ",
    "μετα", "αμφ", "αμφι",
}, key=len, reverse=True)


def _is_prefix_chain(s: str) -> bool:
    if not s:
        return True
    for p in _PREFIX_SPELLINGS:
        if s.startswith(p) and _is_prefix_chain(s[len(p):]):
            return True
    return False


def norm_lemma(text: str) -> str:
    return re.sub(r"[^a-zα-ω]", "", strip_greek(text))


def lemma_alternatives(text: str) -> set[str]:
    """Our lemma may list alternatives ("ἔπω, ἐρῶ, εἶπον", "ὅς, ἥ")."""
    out = {norm_lemma(x) for x in re.split(r"[,/;]| or ", text) if norm_lemma(x)}
    return out


def canonical_lemma(s: str) -> tuple[str, tuple[str, ...]]:
    """Apply the dictionary-form convention classes; return (form, rules)."""
    rules: list[str] = []
    for key, members in _PRONOUN_CLASSES.items():
        if s in members and s != key:
            s = key
            rules.append("pronoun headword")
            break
    for rule, key, members in _HEADWORD_CLASSES:
        if s in members:
            s = key
            rules.append(rule)
            break
    for key, members in _SUPPLETIVE.items():
        done = False
        for m in sorted(members, key=len, reverse=True):
            if s.endswith(m) and _is_prefix_chain(s[: -len(m)]):
                if m != key:
                    s = s[: -len(m)] + key
                    rules.append("suppletive verb headword")
                done = True
                break
        if done:
            break
    for a, b, rule in (("γιγν", "γιν", "Attic/Koine spelling"), ("ττ", "σσ", "Attic/Koine spelling"),
                       ("ρσ", "ρρ", "Attic/Koine spelling")):
        if a in s:
            s = s.replace(a, b)
            rules.append(rule)
    for ending, repl, rule, minlen in (
        ("ουμαι", "εω", "deponent headword", 6),
        ("ωμαι", "αω", "deponent headword", 6),
        ("ομαι", "ω", "deponent headword", 6),
        ("γνυμι", "γω", "-mi/-o headword", 6),   # ἀνοίγνυμι / ἀνοίγω
        ("υμι", "υω", "-mi/-o headword", 5),
        ("εοσ", "ουσ", "contracted headword", 6),   # χρύσεος / χρυσοῦς
        ("οοσ", "ουσ", "contracted headword", 4),   # χόος / χοῦς, νόος / νοῦς
    ):
        if len(s) >= minlen and s.endswith(ending):
            s = s[: -len(ending)] + repl
            rules.append(rule)
            break
    for a, b in (("ουτωσ", "ουτω"), ("μεχρισ", "μεχρι"), ("αχρισ", "αχρι")):
        if s == a:
            s = b
            rules.append("movable final")
    if len(s) > 4 and s.endswith("θεν"):
        s = s[:-1]
        rules.append("movable final")
    return s, tuple(rules)


_ASPIRATE = {"π": "φ", "τ": "θ", "κ": "χ"}
# preverbs as CCAT occasionally spells them -> the full form
_PREFIX_FULL = {"εξ": "εκ", "κατ": "κατα", "απ": "απο", "επ": "επι", "αν": "ανα", "υπε": "υπερ",
                "μετ": "μετα", "παρ": "παρα", "δι": "δια", "υπ": "υπο", "αντ": "αντι"}


def _combine(prefix: str, stem: str, rough: bool, vowel: bool) -> set[str]:
    p = _PREFIX_FULL.get(prefix, prefix)
    out = {p + stem}
    if not stem:
        return out
    if vowel:
        if p in ("περι", "προ", "αμφι"):
            out.add(p + stem)
        elif p[-1] in "αεοι":
            base = p[:-1]
            # aspirate before a rough breathing; offered for smooth ones too,
            # since some compounds aspirate irregularly (καθεύδω)
            if base[-1] in _ASPIRATE:
                out.add(base[:-1] + _ASPIRATE[base[-1]] + stem)
            out.add(base + stem)
        elif p == "εκ":
            out.add("εξ" + stem)
    else:
        c = stem[0]
        if p in ("εν", "συν"):
            b = p[:-1]
            if c in "πβφψμ":
                out.add(b + "μ" + stem)
            elif c in "κγχξ":
                out.add(b + "γ" + stem)
            elif c in "λρ":
                out.add(b + c + stem)
            elif c in "σζ" and p == "συν":
                out.add("συ" + stem)
        if c == "ρ" and p[-1] in "αεοιυ":
            out.add(p + "ρ" + stem)
    return out


def ccat_lemma_candidates(lemma_beta: str, prefixes: tuple[str, ...] = ()) -> set[str]:
    """Normalised CCAT headword(s), with verb preverbs recombined (sandhi)."""
    base = lemma_beta
    cands = {norm_lemma(beta_to_unicode(base))}
    rough, vowel = beta_initial_rough(base), beta_initial_vowel(base)
    for pre in reversed(prefixes):
        pnorm = norm_lemma(beta_to_unicode(pre))
        new: set[str] = set()
        for c in cands:
            new |= _combine(pnorm, c, rough, vowel)
        cands = new
        rough, vowel = beta_initial_rough(pre), beta_initial_vowel(pre)
    return cands


def _crasis_join(parts: list[str]) -> set[str]:
    """καί + ἐγώ -> κἀγώ (stripped: καγω); καί + ἄν -> κἄν."""
    if len(parts) != 2 or parts[0] != "και":
        return set()
    second = parts[1]
    return {"κ" + ("α" + second[1:] if second[:1] in "εα" else second), "κ" + second}


def _name_key(s: str) -> str:
    s = s.replace("ει", "ι").replace("γμ", "χμ")
    s = re.sub(r"(.)\1", r"\1", s)
    return s


def _within_one_edit(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    edits = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return edits + (len(b) - j) + (len(a) - i) <= 1


_SUBST_END = re.compile(r"(οσ|ον|η|α|οι|αι|ων)$")
_ADV_END = re.compile(r"(εωσ|ωσ|οσ|ον|ην|αν|ησ|η|α|υσ|υ|εσ|ου|ω|ι)$")
_PTCP_END = re.compile(r"(μενοσ|μενη|μενον|ων|ουσα|ον|ασ|εισ)$")


def _stem(s: str, pat: re.Pattern) -> str:
    return pat.sub("", s)


def compare_lemma(our_lemma: str, ccat_lemma_beta: str, prefixes: tuple[str, ...] = (),
                  *, proper: bool = False, ours_compound: bool = False) -> tuple[str, str]:
    """CCAT wrapper of ``compare_lemma_sets`` (beta-code headword + preverbs)."""
    if "+" in ccat_lemma_beta:
        parts = [norm_lemma(beta_to_unicode(x)) for x in ccat_lemma_beta.split("+")]
        return _compare_crasis(lemma_alternatives(our_lemma), parts)
    return compare_lemma_sets(our_lemma, ccat_lemma_candidates(ccat_lemma_beta, prefixes),
                              proper=proper or ccat_lemma_beta.startswith("*"),
                              ours_compound=ours_compound)


def _compare_crasis(ours: set[str], parts: list[str]) -> tuple[str, str]:
    if not ours:
        return "uncertain", "no lemma"
    canon_parts = [canonical_lemma(x)[0] for x in parts]
    joined = _crasis_join(parts) | _crasis_join(canon_parts) | {"".join(parts)}
    if ours & joined or {canonical_lemma(o)[0] for o in ours} & joined:
        return "convention", "crasis headword"
    if ours & set(parts) or {canonical_lemma(o)[0] for o in ours} & set(canon_parts):
        return "uncertain", "crasis part"
    return "disagree", ""


_ELIDED_END = ("\u2019", "\u1fbd", "\u1fbf", "'", "\u02bc")


def compare_lemma_text(our_lemma: str, ref_lemma: str, *, proper: bool = False,
                       ours_compound: bool = False) -> tuple[str, str]:
    """Our lemma vs a reference headword given as Greek text (UD, GLAUx).

    A reference headword printed elided (τότ᾿, δεῦρ᾿) matches our full
    headword with one more final vowel ("elided headword")."""
    ref_lemma = ref_lemma.strip()
    if ref_lemma.endswith(_ELIDED_END):
        stem = norm_lemma(ref_lemma)
        for o in lemma_alternatives(our_lemma):
            if o.startswith(stem) and len(o) == len(stem) + 1:
                return "convention", "elided headword"
    if "+" in ref_lemma:
        return _compare_crasis(lemma_alternatives(our_lemma),
                               [norm_lemma(x) for x in ref_lemma.split("+")])
    return compare_lemma_sets(our_lemma, lemma_alternatives(ref_lemma), proper=proper,
                              ours_compound=ours_compound)


def compare_lemma_sets(our_lemma: str, theirs: set[str], *, proper: bool = False,
                       ours_compound: bool = False) -> tuple[str, str]:
    """Return (outcome, rule) for our lemma vs CCAT's headword.

    Order: exact (accent-stripped, with CCAT's preverbs recombined) ->
    headword-convention classes (``canonical_lemma``) -> spelling variant
    (ει/ι, single/double consonants) -> substantive headword (-ος/-ον/-η/-α/
    plural) -> compound tag (our compound code's main lemma begins CCAT's
    headword: οὐ for οὐκέτι) -> name spelling (one edit, proper names).
    """
    ours = lemma_alternatives(our_lemma)
    if not ours:
        return "uncertain", "no lemma"
    if not theirs:
        return "uncertain", "no reference lemma"
    if ours & theirs:
        return "agree", ""
    for t in theirs:  # our crasis headword vs the reference's second part (κἀγώ / ἐγώ)
        if ours & _crasis_join(["και", t]):
            return "convention", "crasis headword"
    ours_c = {canonical_lemma(x) for x in ours}
    theirs_c = {canonical_lemma(x) for x in theirs}
    for oc, orules in ours_c:
        for tc, trules in theirs_c:
            if oc == tc:
                only = [r for r in trules if r not in orules] + [r for r in orules if r not in trules]
                rules = only or list(orules + trules)
                return "convention", (rules[0] if rules else "")
    for oc, _ in ours_c:
        for tc, _ in theirs_c:
            if len(oc) >= 4 and _name_key(oc) == _name_key(tc):
                return "convention", "spelling variant"
            so, st = _stem(_name_key(oc), _SUBST_END), _stem(_name_key(tc), _SUBST_END)
            if len(st) >= 3 and so == st:
                return "convention", "substantive headword"
    if ours_compound:
        for o in ours:
            for t in theirs:
                if len(o) >= 2 and t.startswith(o) and len(t) > len(o):
                    return "convention", "compound tag"
    if proper:
        for o in ours:
            for t in theirs:
                if len(o) >= 3 and _within_one_edit(_name_key(o), _name_key(t)):
                    return "convention", "name spelling"
    return "disagree", ""


def adverb_stem_related(our_lemma: str, ccat_lemma_beta: str = "", *, theirs: set[str] | None = None) -> bool:
    """πρῶτον ~ πρῶτος, καλῶς ~ καλός, μακράν ~ μακρός, κύκλῳ ~ κύκλος:
    the same stem once the ending is dropped.  ``theirs``: the reference's
    normalised headword(s), instead of a CCAT beta-code lemma."""
    if theirs is None:
        theirs = {norm_lemma(beta_to_unicode(ccat_lemma_beta))}
    stems = {_stem(t, _ADV_END) for t in theirs}
    return any(_stem(o, _ADV_END) in stems and len(_stem(o, _ADV_END)) >= 3
               for o in lemma_alternatives(our_lemma))


def participle_headword(our_lemma: str, ccat_lemma_beta: str = "", prefixes: tuple[str, ...] = (),
                        *, theirs: set[str] | None = None) -> bool:
    """οἰκουμένη ~ οἰκέω, ἄρχων ~ ἄρχω (and the reverse, εὐλογέω ~
    εὐλογημένος): one lexicon's noun/adjective headword is a participle of
    the other's verb (same first three letters, participial ending)."""
    if theirs is None:
        theirs = ccat_lemma_candidates(ccat_lemma_beta, prefixes)
    ours = lemma_alternatives(our_lemma)
    for a_side, b_side in ((ours, theirs), (theirs, ours)):
        for o in a_side:
            if _PTCP_END.search(o) and any(len(t) >= 3 and o[:3] == t[:3] for t in b_side):
                return True
    return False


def adverb_shaped(lemma: str) -> bool:
    return any(re.search(r"(ωσ|ω|θεν|θε|αν|ην)$", o) for o in lemma_alternatives(lemma))


# ---------------------------------------------------------------------------
# Feature comparison
# ---------------------------------------------------------------------------

_RANK = {"agree": 0, "convention": 1, "uncertain": 2, "disagree": 3}


def worst(outcomes) -> str:
    out = "agree"
    for o in outcomes:
        if _RANK[o] > _RANK[out]:
            out = o
    return out


def compare_pos(ours: Feats, theirs: Feats, lemma_outcome: str, adverb_related: bool = False,
                lemma_rule: str = "") -> tuple[str, str]:
    a, b = ours.pos, theirs.pos
    lemma_ok = lemma_outcome in ("agree", "convention")
    if a == b:
        return "agree", ""
    pair = {a, b}
    if lemma_rule == "interjection headword" and pair & {"intj", "ptcl"} and "verb" in pair:
        return "convention", "interjection headword"
    if lemma_rule == "substantival participle":
        return "convention", "substantival participle"
    if pair == {"noun", "adj"} and lemma_ok:
        return "convention", "substantival adjective"
    if pair == {"adj", "num"}:
        return ("convention", "numeral class") if lemma_ok else ("disagree", "")
    if pair == {"pron", "adj"} and lemma_ok:
        return "convention", "pronominal adjective"
    if a in INDECLINABLE and b in INDECLINABLE and lemma_ok:
        return "convention", "indeclinable class"
    if pair == {"adv", "adj"} and (lemma_ok or adverb_related):
        return "convention", "adverbial adjective"
    if pair & {"adv", "prep"} and pair & {"noun", "adj"} and adverb_related:
        return "convention", "frozen case form"
    if pair == {"art", "pron"}:
        return "uncertain", "article/pronoun"
    return "disagree", ""


def _fields_for(f: Feats) -> tuple[str, ...]:
    if f.pos == "verb":
        if f.mood == "P":
            return PARTICIPLE_FIELDS
        if f.mood == "N":
            return ("tense", "voice", "mood")
        return VERB_FIELDS
    if f.pos in ("noun", "adj", "art", "num"):
        return NOMINAL_FIELDS + ("degree",)
    if f.pos == "pron":
        return NOMINAL_FIELDS + ("person",)
    if f.pos == "adv":
        return ("degree",)
    return ()


def compare_field(name: str, ours: Feats, theirs: Feats) -> tuple[str, str] | None:
    """Outcome for one field, or None when not compared (one side lacks it)."""
    a, b = ours.get(name), theirs.get(name)
    if name == "degree":
        if not a and not b:
            return None
        if a and b:
            return ("agree", "") if a == b else ("disagree", "")
        return "convention", "degree marking"
    if not a or not b:
        return None
    if name == "number" and "D" in (a, b):
        return "uncertain", "dual"
    if "|" in a or "|" in b:  # a reference listing alternatives (UD "Acc,Nom")
        if a == b:
            return "agree", ""
        if set(a.split("|")) & set(b.split("|")):
            return "convention", "underspecified"
        return "disagree", ""
    if name == "voice":
        if a == b:
            return "agree", ""
        if a == "MP" and b in ("M", "P") or b == "MP" and a in ("M", "P"):
            return "convention", "middle/passive label"
        if ours.sub == "deponent" and b in ("M", "P", "A"):
            # STEPBible's deponent labels are lexical; a reference may call a
            # deponent's form middle, passive or (UD, for ἔσομαι) active
            return "convention", "middle/passive label"
        if {a, b} <= {"M", "P"} and ours.tense in ("P", "I", "R", "L"):
            return "convention", "middle/passive label"
        return "disagree", ""
    return ("agree", "") if a == b else ("disagree", "")


@dataclass
class Comparison:
    pos: str
    pos_rule: str
    fields: dict  # name -> (outcome, rule, ours value, theirs value)
    full: str
    second: str | None  # agree / disagree / None (informational)


def compare_feats(ours: Feats, theirs: Feats, lemma_outcome: str, adverb_related: bool = False,
                  lemma_rule: str = "") -> Comparison:
    if ours.pos == "crasis" or theirs.pos == "crasis":
        if ours.pos == "crasis" and theirs.pos == "crasis" and len(ours.parts) == len(theirs.parts):
            best = None
            for order in (theirs.parts, list(reversed(theirs.parts))):
                subs = [compare_feats(o, t, "agree") for o, t in zip(ours.parts, order)]
                full = worst(s.full for s in subs)
                if best is None or _RANK[full] < _RANK[best[0]]:
                    best = (full, subs)
            full, subs = best
            pos = worst(s.pos for s in subs)
            return Comparison(pos, "crasis parts", {}, worst([full, "convention"]), None)
        # one side compound (TAGNT's "PRT-N + G2089=ADV" for οὐκέτι, CCAT's
        # C+RP for κἀγώ), the other a single analysis: compare the single one
        # with the compound's main part (ours: the first; CCAT's: the part of
        # the same class, else the last); agreement there is a convention.
        if ours.pos == "crasis":
            main, other = ours.parts[0], theirs
            sub = compare_feats(main, other, lemma_outcome, adverb_related, lemma_rule)
        else:
            cand = [p for p in theirs.parts if p.pos == ours.pos] or theirs.parts[-1:]
            sub = compare_feats(ours, cand[0], lemma_outcome, adverb_related, lemma_rule)
        if _RANK[sub.full] <= _RANK["convention"]:
            return Comparison(worst([sub.pos, "convention"]), "compound tag", sub.fields, "convention", None)
        return Comparison("uncertain", "compound vs single", {}, "uncertain", None)
    pos, rule = compare_pos(ours, theirs, lemma_outcome, adverb_related, lemma_rule)
    fields = {}
    names = []
    for n in _fields_for(ours) + _fields_for(theirs):
        if n not in names:
            names.append(n)
    for n in names:
        r = compare_field(n, ours, theirs)
        if r is not None:
            fields[n] = (r[0], r[1], ours.get(n), theirs.get(n))
    outcomes = [pos] + [v[0] for v in fields.values()]
    full = worst(outcomes)
    second = None
    if ours.second is not None and theirs.second is not None:
        second = "agree" if ours.second == theirs.second else "disagree"
    return Comparison(pos, rule, fields, full, second)
