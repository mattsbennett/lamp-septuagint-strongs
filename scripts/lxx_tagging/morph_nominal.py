"""Noun and adjective paradigms for the morphological analyser.

A nominal is generated from hypotheses (`Declension`): a table of endings,
the stem it is added to, and a weight. Which tables a lemma may take is read
off its nominative, gender and (when a definition gives it) genitive
(`noun_hypotheses`, `adjective_hypotheses`); where the nominative does not
decide (a stem in -ξ may be -κ-, -γ- or -χ-; -ις may be πόλις -εως or ἐλπίς
-ίδος), every possibility is a hypothesis, and the analyser keeps those the
New Testament's forms of the word confirm.

The accent is persistent: it stays on the syllable the nominative has it on,
as far as the final syllable allows (ἄνθρωπος, ἀνθρώπου; ὄνομα, ὀνόματος),
an accented long final syllable of the genitive and dative of the first and
second declensions takes the circumflex (θεοῦ, φωνῇ), the first
declension's genitive plural is always -ῶν, third-declension words with a
monosyllabic nominative accent the ending in the genitive and dative (σαρκός,
σαρκί, σαρκῶν), and -ευς nouns the ending throughout (βασιλέως, βασιλεῖ).
Contracted types (γένος, -ους; ἀληθής, -οῦς; χρυσοῦς; ὀστοῦν) are built
uncontracted and contracted by rule.

Codes are TAGNT's: N-{case}{number}{gender}, A-{case}{number}{gender}, with
-C for a comparative and -S for a superlative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from lxx_tagging import morph_phon as ph

CASES = "NGDAV"


def _rows(text: str) -> tuple:
    """`NS:ος GS:ου^ ...` -> ((case-number, ending, flags), ...). Flags: ^
    circumflex if the accent falls on this (long) ending; L/S quantity of
    the ultima; c contract at the stem boundary; w a rarer variant; r
    recessive accent (the neuter of a barytone -ης or -ων adjective)."""
    rows = []
    for item in text.split():
        slot, _, ending = item.partition(":")
        flags = ""
        while ending and ending[-1] in "^LScwr":
            flags += ending[-1]
            ending = ending[:-1]
        rows.append((slot, "" if ending == "-" else ending, flags))
    return tuple(rows)


# First declension.
A_PURE = _rows("NS:ᾱ GS:ᾱσ^ DS:ᾳ^ AS:ᾱν VS:ᾱ NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
A_SHORT = _rows("NS:ᾰ GS:ᾱσ^ DS:ᾳ^ AS:ᾰν VS:ᾰ NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
A_MIXED = _rows("NS:ᾰ GS:ησ^ DS:ῃ^ AS:ᾰν VS:ᾰ NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
ETA = _rows("NS:η GS:ησ^ DS:ῃ^ AS:ην VS:η NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
MASC_AS = _rows("NS:ᾱσ GS:ου^ DS:ᾳ^ AS:ᾱν VS:ᾱ NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
MASC_ES = _rows("NS:ησ GS:ου^ DS:ῃ^ AS:ην VS:ᾰ VS:η NP:αι GP:ῶν DP:αισ^ AP:ᾱσ VP:αι")
# Second declension.
O_MASC = _rows("NS:οσ GS:ου^ DS:ῳ^ AS:ον VS:ε NP:οι GP:ων^ DP:οισ^ AP:ουσ VP:οι")
O_NEUT = _rows("NS:ον GS:ου^ DS:ῳ^ AS:ον VS:ον NP:ᾰ GP:ων^ DP:οισ^ AP:ᾰ VP:ᾰ")
O_CONTRACT_MASC = _rows("NS:οῦσ GS:οῦ DS:ῷ AS:οῦν VS:οῦ NP:οῖ GP:ῶν DP:οῖσ AP:οῦσ VP:οῖ")
O_CONTRACT_NEUT = _rows("NS:οῦν GS:οῦ DS:ῷ AS:οῦν VS:οῦν NP:ᾶ GP:ῶν DP:οῖσ AP:ᾶ VP:ᾶ")
# Third declension, consonant stems: the stem is the oblique stem.
CONS = _rows("GS:οσ DS:ι AS:α NP:εσ VP:εσ GP:ων DP:ι DP:ιν AP:ασ")
CONS_NEUT = _rows("GS:οσ DS:ι NP:ᾰ VP:ᾰ AP:ᾰ GP:ων DP:ι DP:ιν")
# γένος, -ους: stem in -ε-, contracted.
S_NEUT = _rows("GS:οσc DS:ϊc NP:αc VP:αc AP:αc GP:ωνc GP:ων DP:σι DP:σιν")
# πόλις, -εως; πῆχυς, -εως (stem without the vowel).
I_STEM = _rows("NS:ισ GS:εωσS DS:ει AS:ιν VS:ι NP:εισ VP:εισ GP:εωνS DP:εσι DP:εσιν AP:εισ")
U_EWS = _rows("NS:υσ GS:εωσS GS:εοσ DS:ει AS:υν VS:υ NP:εισ VP:εισ GP:εωνS GP:ῶν DP:εσι DP:εσιν AP:εισ")
U_STEM = _rows("NS:υσ GS:υοσ DS:υι AS:υν VS:υ NP:υεσ VP:υεσ GP:υων DP:υσι DP:υσιν AP:υασ AP:ῡσ")
EU_STEM = _rows("NS:ευσ GS:εωσS DS:ει^ AS:εα VS:ευ^ NP:εισ^ VP:εισ^ GP:εωνS DP:ευσι DP:ευσιν AP:εισ^ AP:εασw")


@dataclass(frozen=True)
class Declension:
    """One hypothesis for how a nominal declines (in one gender)."""

    table: tuple
    stem: str              # plain; the table's endings are added to it
    gender: str            # "M", "F", "N"
    weight: float = 1.0
    source: str = "rule"
    accent: str = "persistent"   # "persistent", "ending", "monosyllable"
    note: str = ""
    nominative: str = ""   # the lemma's own form (accented key) for NS, and AS/VS of neuters
    plural_only: bool = False


@dataclass
class NominalForm:
    key: str
    case_number: str      # "NS", "GP" ...
    gender: str
    weight: float
    note: str


def _lemma_accent(lemma: str) -> tuple[int, str] | None:
    return ph.accent_position(lemma)


def decline(decl: Declension) -> Iterator[NominalForm]:
    """Every form of one declension hypothesis."""
    position = _lemma_accent(decl.nominative) if decl.nominative else None
    syllable = position[0] if position else 0
    oxytone = bool(position) and position[0] == ph.nucleus_count(decl.nominative) - 1
    perispomenon = oxytone and position[1] == ph.CIRCUMFLEX      # γῆ, γῆς, γῆν
    stem_nuclei = ph.nucleus_count(decl.stem)
    seen_ns = False
    for slot, ending, flags in decl.table:
        weight = decl.weight * (0.5 if "w" in flags else 1.0)
        if slot == "NS":
            seen_ns = True
        if decl.plural_only and slot.endswith("S"):
            weight *= 0.5
        word = decl.stem + ending
        ultima = "L" if "L" in flags else ("S" if "S" in flags else None)
        if decl.nominative and (slot == "NS" or (decl.gender == "N" and slot in ("AS", "VS"))) \
                and not decl.plural_only and ph.letters(word) == ph.letters(decl.nominative):
            # The lemma is its own nominative, accent and all.
            yield NominalForm(decl.nominative, slot, decl.gender, weight, decl.note)
            continue
        if any(m in ph.ACCENTS for m in ph.nfd(ending)):
            accented = (word,)
        elif decl.accent in ("ending", "stemfinal"):
            kind = ph.CIRCUMFLEX if "^" in flags else ph.ACUTE
            index = stem_nuclei if decl.accent == "ending" else stem_nuclei - 1
            accented = ph.accentuate(word, "syllable", ultima, 0, index, kind)
        elif "r" in flags:
            accented = ph.accentuate(word, "recessive", ultima)
        elif decl.accent == "monosyllable" and slot[0] in "GD":
            kind = ph.CIRCUMFLEX if slot == "GP" else ph.ACUTE
            accented = ph.accentuate(word, "syllable", ultima, 0, -1, kind)
        else:
            kind = ph.CIRCUMFLEX if ("^" in flags and oxytone) or perispomenon else None
            if oxytone and not kind:
                kind = ph.ACUTE
            accented = ph.accentuate(word, "syllable", ultima, 0, syllable, kind)
        for form in accented:
            if "c" in flags:
                contracted = ph.contract(form, len(ph.clusters(decl.stem)))
                if contracted is None:
                    continue
                form = contracted
            yield NominalForm(ph.finish(form), slot, decl.gender, weight, decl.note)
    if decl.nominative and not seen_ns:
        yield NominalForm(decl.nominative, "NS", decl.gender, decl.weight, decl.note)
        if decl.gender == "N":
            yield NominalForm(decl.nominative, "AS", decl.gender, decl.weight, decl.note)
            yield NominalForm(decl.nominative, "VS", decl.gender, decl.weight, decl.note)
        else:
            yield NominalForm(decl.nominative, "VS", decl.gender, decl.weight * 0.5, decl.note)


# ---------------------------------------------------------------------------
# Hypotheses from the lemma

def _cut(plain: str, count: int) -> str:
    """`plain` without its last `count` letters (marks go with their letter)."""
    parts = ph.clusters(plain)
    return ph.compose(parts[:len(parts) - count]) if count else plain


def _dat_plural(stem: str) -> str:
    """A third-declension stem joined to the dative plural's -σι, without the
    final ι: stops join σ (φύλαξι, ἄραψι, ἐλπίσι), -ντ- is lost with
    lengthening (ἄρχουσι, πᾶσι, -εῖσι), ν is lost (αἰῶσι, ποιμέσι)."""
    bare = ph.letters(stem)
    if bare.endswith("οντ"):
        return _cut(stem, 3) + "ουσ"
    if bare.endswith("αντ"):
        return _cut(stem, 3) + "ᾱσ"
    if bare.endswith("εντ"):
        return _cut(stem, 3) + "εισ"
    if bare.endswith("ντ"):
        return _cut(stem, 2) + "σ"
    if bare[-1:] in ph.DENTALS or bare[-1:] == "ν":
        return _cut(stem, 1) + "σ"
    if bare[-1:] in ph.LABIALS:
        return _cut(stem, 1) + "ψ"
    if bare[-1:] in ph.VELARS:
        return _cut(stem, 1) + "ξ"
    return stem + "σ"


def consonant_table(stem: str, neuter: bool, accusative_nu: bool = False) -> tuple:
    """A consonant stem's endings, the dative plural joined to the stem
    (written as the stem's replacement: the table's stem is then '')."""
    dative = _dat_plural(stem)
    rows = []
    for slot, ending, flags in (CONS_NEUT if neuter else CONS):
        if slot == "DP":
            rows.append((slot, dative + ending, flags))
        elif slot == "AS" and accusative_nu:
            rows.append((slot, "", "N"))     # replaced below
        else:
            rows.append((slot, stem + ending, flags))
    return tuple(rows)


def third_declension(nominative_plain: str, nominative_key: str, oblique: str, gender: str, weight: float,
                     note: str, accusative_nu: bool = False) -> Declension:
    """A consonant-stem hypothesis: nominative as the lemma, oblique stem `oblique`."""
    neuter = gender == "N"
    rows = []
    for slot, ending, flags in consonant_table(oblique, neuter):
        if slot == "AS" and accusative_nu:
            continue
        rows.append((slot, ending, flags))
    if accusative_nu:
        rows.append(("AS", _cut(nominative_plain, 1) + "ν", ""))
    if not neuter:
        rows.append(("VS", nominative_plain, ""))
    monosyllable = ph.nucleus_count(nominative_plain) == 1
    return Declension(tuple(rows), "", gender, weight, "rule", "monosyllable" if monosyllable else "persistent",
                      note, nominative_key)


def _alpha_quantity(lemma_key: str) -> str:
    """"L", "S" or "?" for the final -α of a first-declension nominative, from its accent."""
    position = ph.accent_position(lemma_key)
    count = ph.nucleus_count(lemma_key)
    if not position:
        return "?"
    index, mark = position
    if index == count - 1:
        return "L"
    if index == count - 3:
        return "S"
    if index == count - 2:
        if mark == ph.CIRCUMFLEX:
            return "S"
        parts = ph.clusters(lemma_key)
        nuclei = ph._nuclei(parts)
        if ph._quantity(parts, nuclei[index]) == "L":
            return "L"
    return "?"


def noun_hypotheses(lemma_key: str, gender: str | None, genitive: str | None = None) -> list[Declension]:
    """The declension hypotheses of a noun lemma (an accented key)."""
    plain = ph.plain(lemma_key)
    bare = ph.letters(plain)
    found: list[Declension] = []

    def add(table, cut, g, weight, note, accent="persistent", plural_only=False):
        found.append(Declension(table, _cut(plain, cut), g, weight, "rule", accent, note, lemma_key, plural_only))

    def third(oblique_cut: int, addition: str, g: str, weight: float, note: str, accusative_nu=False):
        oblique = _cut(plain, oblique_cut) + addition
        found.append(third_declension(plain, lemma_key, oblique, g, weight, note, accusative_nu))

    gen = ph.letters(genitive) if genitive else ""
    genders = [gender] if gender in ("M", "F", "N") else []

    if bare.endswith("μα") and gender in (None, "N"):
        third(0, "τ", "N", 1.0, "-μα -ματος")
        return found
    if bare.endswith("ευσ"):
        add(EU_STEM, 3, gender or "M", 1.0, "-ευς -έως", accent="ending")
        return found
    if bare.endswith("ουσ") and gender != "F":
        add(O_CONTRACT_MASC, 3, gender or "M", 0.6, "-οῦς contracted")
        third(3, "οδ", gender or "M", 0.4, "-ους -οδος")
        third(3, "οντ", gender or "M", 0.3, "-ους -οντος")
        return found
    if bare.endswith("εον") and gender in (None, "N"):
        # ὀστέον: the uncontracted lemma, contracted forms too (ὀστᾶ, ὀστῶν, ὀστοῦν).
        add(O_NEUT, 2, "N", 1.0, "-εον -έου")
        add(O_CONTRACT_NEUT, 3, "N", 0.8, "-εον contracted -οῦν")
        return found
    if bare.endswith("ουν") and gender in (None, "N"):
        add(O_CONTRACT_NEUT, 3, "N", 1.0, "-οῦν contracted")
        open_stem = _cut(plain, 3) + "ε"
        open_nominative = ph.finish(ph.accentuate(open_stem + "ον", "syllable", None, 0,
                                                  ph.nucleus_count(open_stem) - 1)[0])
        found.append(Declension(O_NEUT, open_stem, "N", 0.6, "rule", "persistent", "-εον uncontracted",
                                open_nominative))
        return found
    if bare.endswith("οσ"):
        if gender == "N":
            found.append(Declension(S_NEUT, _cut(plain, 2) + "ε", "N", 1.0, "rule", "persistent",
                                    "-ος -ους", lemma_key))
            # TBESG files some second-declension neuters under -ος (πετεινός, ἱλαστήριος),
            # and the LXX declines ἔλεος, σκότος also as masculines.
            found.append(Declension(O_NEUT, _cut(plain, 2), "N", 0.6, "rule", "persistent", "-ον -ου",
                                    lemma_key[:-1] + "ν"))
            add(O_MASC, 2, "M", 0.15, "-ος -ου (masculine)")
        else:
            for g in genders or ["M"]:
                add(O_MASC, 2, g, 1.0, "-ος -ου")
            if not genders:
                found.append(Declension(S_NEUT, _cut(plain, 2) + "ε", "N", 0.2, "rule", "persistent",
                                        "-ος -ους", lemma_key))
                # A noun TBESG gives no gender may be an adjective's feminine too (πρεσβυτέρα).
                fem = A_PURE if _feminine_alpha(ph.letters(_cut(plain, 2))) else ETA
                fem = tuple((s, "ων^" if s == "GP" else e, f) for s, e, f in fem)
                add(fem, 2, "F", 0.3, "-ος -α/-η (feminine)")
        return found
    if bare.endswith("ον"):
        add(O_NEUT, 2, "N", 1.0, "-ον -ου")
        if gender in ("M", "F"):
            add(O_NEUT, 2, gender, 0.5, "-ον, gender as TBESG gives it")
        return found
    if bare.endswith("η"):
        # (TBESG gives some -η nouns a wrong gender, συμβολή N-N: the ending decides.)
        add(ETA, 1, "F", 1.0, "-η -ης")
        return found
    if bare.endswith("α") and gender in (None, "F"):
        quantity = _alpha_quantity(lemma_key)
        pure = bare[-2:-1] in ("ε", "ι", "ρ")
        if gen.endswith("ησ"):
            add(A_MIXED, 1, "F", 1.0, "-α -ης")
            if bare[-2:-1] == "ρ":
                # The NT's μαχαίρης is Brenton's (Attic) μαχαίρας, μαχαίρᾳ.
                add(A_SHORT if quantity != "L" else A_PURE, 1, "F", 0.8, "-ρα -ρας")
        elif gen.endswith("ασ"):
            add(A_SHORT if quantity == "S" else A_PURE, 1, "F", 1.0, "-α -ας")
        elif pure:
            if quantity != "S":
                add(A_PURE, 1, "F", 1.0, "-ᾱ -ας")
            if quantity != "L":
                add(A_SHORT, 1, "F", 1.0, "-ᾰ -ας")
                if bare[-2:-1] == "ρ":
                    add(A_MIXED, 1, "F", 0.8, "-ρα -ρης")
        else:
            add(A_MIXED, 1, "F", 1.0, "-α -ης")
            if quantity == "L":
                add(A_PURE, 1, "F", 0.5, "-ᾱ -ας")
        return found
    if bare.endswith("α") and gender == "N":
        add(O_NEUT, 1, "N", 0.6, "neuter plural lemma", plural_only=True)
        third(0, "τ", "N", 0.3, "-α -ατος")
        return found
    if bare.endswith("οι") and gender in (None, "M"):
        add(O_MASC, 2, "M", 0.8, "plural lemma -οι", plural_only=True)
        return found
    if bare.endswith("αι") and gender in (None, "F"):
        add(ETA, 2, "F", 0.6, "plural lemma -αι", plural_only=True)
        add(A_PURE, 2, "F", 0.6, "plural lemma -αι", plural_only=True)
        return found
    if bare.endswith("εσ") and len(bare) > 4:
        third(2, "", gender or "M", 0.6, "plural lemma -ες")
        return found
    if bare.endswith("ησ"):
        if gen.endswith("ουσ") or gen.endswith("ητοσ"):
            pass
        else:
            add(MASC_ES, 2, gender if gender in ("M", "F") else "M", 1.0, "-ης -ου")
        if gen.endswith("ητοσ") or not gen:
            third(1, "τ", gender or "M", 0.3 if not gen else 1.0, "-ης -ητος")
        if gen.endswith("ουσ"):
            found.append(Declension(S_NEUT, _cut(plain, 2) + "ε", gender or "M", 1.0, "rule", "persistent",
                                    "-ης -ους", lemma_key))
        return found
    if bare.endswith("ασ"):
        if gender == "N":
            third(1, "τ", "N", 0.8, "-ας -ατος")
            third(1, "", "N", 0.2, "-ας -αος")
        elif gender == "F":
            third(1, "δ", "F", 1.0, "-ας -αδος")
        else:
            add(MASC_AS, 2, "M", 1.0, "-ας -ου")
            third(1, "ντ", "M", 0.3, "-ας -αντος")
            third(1, "δ", gender or "F", 0.2, "-ας -αδος")
        return found
    if bare.endswith("ισ"):
        position = ph.accent_position(lemma_key)
        oxytone = bool(position) and position[0] == ph.nucleus_count(lemma_key) - 1
        g = gender or "F"
        if bare.endswith(("σισ", "ψισ", "ξισ")):
            add(I_STEM, 2, g, 1.0, "-σις -σεως")
            return found
        add(I_STEM, 2, g, 0.3 if oxytone else 0.9, "-ις -εως")
        third(1, "δ", g, 0.9 if oxytone else 0.4, "-ις -ιδος")
        third(1, "τ", g, 0.2, "-ις -ιτος", accusative_nu=not oxytone)
        third(1, "θ", g, 0.1, "-ις -ιθος")
        return found
    if bare.endswith("υσ"):
        g = gender or "M"
        add(U_EWS, 2, g, 0.6, "-υς -εως")
        add(U_STEM, 2, g, 0.6, "-υς -υος")
        third(1, "δ", g, 0.1, "-υς -υδος")
        return found
    if bare.endswith("ξ") or bare.endswith("ψ"):
        g = gender or "M"
        options = (("κ", 0.7), ("γ", 0.5), ("χ", 0.4)) if bare.endswith("ξ") else (("π", 0.6), ("β", 0.4), ("φ", 0.5))
        for consonant, weight in options:
            third(1, consonant, g, weight, f"-{bare[-1]} -{consonant}ος")
        return found
    if bare.endswith("ων"):
        position = ph.accent_position(lemma_key)
        oxytone = bool(position) and position[0] == ph.nucleus_count(lemma_key) - 1
        g = gender or "M"
        if oxytone:
            found.append(third_declension(plain, lemma_key, plain, g, 0.8, "-ών -ῶνος"))
            found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "ον", g, 0.4, "-ών -όνος"))
        else:
            found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "ον", g, 0.7, "-ων -ονος"))
            found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "οντ", g, 0.5, "-ων -οντος"))
            found.append(third_declension(plain, lemma_key, plain, g, 0.4, "-ων -ωνος"))
        return found
    if bare.endswith("ην"):
        g = gender or "M"
        found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "εν", g, 0.6, "-ην -ενος"))
        found.append(third_declension(plain, lemma_key, plain, g, 0.6, "-ην -ηνος"))
        return found
    if bare.endswith("ηρ"):
        g = gender or "M"
        found.append(third_declension(plain, lemma_key, plain, g, 0.7, "-ηρ -ηρος"))
        found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "ερ", g, 0.4, "-ηρ -ερος"))
        return found
    if bare.endswith("ωρ"):
        g = gender or "M"
        if g == "N":
            found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "ατ", "N", 0.8, "-ωρ -ατος"))
        else:
            found.append(third_declension(plain, lemma_key, _cut(plain, 2) + "ορ", g, 0.8, "-ωρ -ορος"))
            found.append(third_declension(plain, lemma_key, plain, g, 0.4, "-ωρ -ωρος"))
        return found
    if bare.endswith("αρ"):
        g = gender or "N"
        found.append(third_declension(plain, lemma_key, _cut(plain, 1) + "τ", g, 0.7, "-αρ -ατος"))
        found.append(third_declension(plain, lemma_key, plain, g, 0.4, "-αρ -αρος"))
        return found
    if bare.endswith(("υρ", "ορ", "ερ", "ιρ")):
        found.append(third_declension(plain, lemma_key, plain, gender or "M", 0.6, "-ρ -ρος"))
        return found
    if bare.endswith("ωσ"):
        g = gender or "M"
        found.append(third_declension(plain, lemma_key, _cut(plain, 1) + "τ", g, 0.6, "-ως -ωτος"))
        found.append(third_declension(plain, lemma_key, _cut(plain, 1), g, 0.4, "-ως -ωος"))
        return found
    if bare.endswith("ι") and gender == "N":
        add(_rows("GS:εωσS DS:ει NP:ηc GP:εωνS DP:εσι"), 1, "N", 0.5, "-ι -εως")
        found.append(third_declension(plain, lemma_key, plain + "τ", "N", 0.3, "-ι -ιτος"))
        return found
    if bare.endswith("υ") and gender == "N":
        found.append(third_declension(plain, lemma_key, plain, "N", 0.5, "-υ -υος"))
        found.append(third_declension(plain, lemma_key, plain + "τ", "N", 0.3, "-υ -υτος"))
        return found
    if bare.endswith("σ"):
        # -ς after a stop's loss (ἐσθής -ῆτος, ὄρνις handled above): dental stems.
        for consonant, weight in (("τ", 0.5), ("δ", 0.4), ("θ", 0.2)):
            third(1, consonant, gender or "M", weight, f"-ς -{consonant}ος")
        return found
    return found


# ---------------------------------------------------------------------------
# Adjectives

def _feminine_alpha(bare_stem: str) -> bool:
    return bare_stem[-1:] in ("ε", "ι", "ρ")


def adjective_hypotheses(lemma_key: str, two_termination: float = 0.3) -> list[tuple[Declension, ...]]:
    """The hypotheses of an adjective lemma, each a tuple of Declensions for
    its genders (a two-termination adjective's feminine is its masculine's
    forms). `two_termination` weights that reading of an -ος adjective."""
    plain = ph.plain(lemma_key)
    bare = ph.letters(plain)
    found: list[tuple[Declension, ...]] = []
    if (bare.endswith("ουσ") and len(bare) > 4) or (bare.endswith("εοσ") and len(bare) > 5):
        if bare.endswith("εοσ"):
            # χρύσεος, πορφύρεος, χάλκεος: the uncontracted lemma, contracted forms (χρυσοῦς, χαλκᾶ).
            found += adjective_hypotheses(lemma_key[:-0] if False else ph.finish(_cut(plain, 3) + "οῦσ"), 0)
            stem = _cut(plain, 3) + "ε"
            fem_table = A_PURE if _feminine_alpha(ph.letters(stem)) else ETA
            found.append((Declension(O_MASC, stem, "M", 0.6, "rule", "persistent", "-εος uncontracted", lemma_key),
                          Declension(fem_table, stem, "F", 0.6, "rule", "persistent", "-εος uncontracted", lemma_key),
                          Declension(O_NEUT, stem, "N", 0.6, "rule", "persistent", "-εος uncontracted", lemma_key)))
            return found
        stem = _cut(plain, 3)
        fem = _rows("NS:ῆ GS:ῆσ DS:ῇ AS:ῆν NP:αῖ GP:ῶν DP:αῖσ AP:ᾶσ")
        if _feminine_alpha(ph.letters(stem)):
            fem = _rows("NS:ᾶ GS:ᾶσ DS:ᾷ AS:ᾶν NP:αῖ GP:ῶν DP:αῖσ AP:ᾶσ")
        found.append((Declension(O_CONTRACT_MASC, stem, "M", 1.0, "rule", "persistent", "-οῦς contracted"),
                      Declension(fem, stem, "F", 1.0, "rule", "persistent", "-οῦς contracted"),
                      Declension(O_CONTRACT_NEUT, stem, "N", 1.0, "rule", "persistent", "-οῦς contracted")))
        return found
    if bare.endswith("οσ"):
        stem = _cut(plain, 2)
        fem_table = A_PURE if _feminine_alpha(ph.letters(stem)) else ETA
        fem_table = tuple((s, "ων^" if s == "GP" else e, f) for s, e, f in fem_table)
        found.append((Declension(O_MASC, stem, "M", 1.0, "rule", "persistent", "-ος -η -ον", lemma_key),
                      Declension(fem_table, stem, "F", 1.0, "rule", "persistent", "-ος -η -ον", lemma_key),
                      Declension(O_NEUT, stem, "N", 1.0, "rule", "persistent", "-ος -η -ον", lemma_key)))
        if two_termination > 0:
            found.append((Declension(O_MASC, stem, "F", two_termination, "rule", "persistent", "-ος -ον (two endings)",
                                     lemma_key),))
        return found
    if bare.endswith("οι") and len(bare) > 3:
        stem = _cut(plain, 2)
        fem_table = A_PURE if _feminine_alpha(ph.letters(stem)) else ETA
        fem_table = tuple((s, "ων^" if s == "GP" else e, f) for s, e, f in fem_table)
        found.append((Declension(O_MASC, stem, "M", 1.0, "rule", "persistent", "plural -οι -αι -α", "", True),
                      Declension(fem_table, stem, "F", 1.0, "rule", "persistent", "plural -οι -αι -α", "", True),
                      Declension(O_NEUT, stem, "N", 1.0, "rule", "persistent", "plural -οι -αι -α", "", True)))
        return found
    if bare.endswith("ησ"):
        stem = _cut(plain, 2) + "ε"
        position = ph.accent_position(lemma_key)
        oxytone = bool(position) and position[0] == ph.nucleus_count(lemma_key) - 1
        masc = _rows("GS:οσc DS:ϊc AS:αc NP:εσc VP:εσc GP:ωνc DP:σι DP:σιν AP:εσc")
        neut = _rows("NS:σ AS:σ VS:σ GS:οσc DS:ϊc NP:αc VP:αc AP:αc GP:ωνc DP:σι DP:σιν")
        if not oxytone:
            neut = tuple((s, e, f + ("r" if s in ("NS", "AS", "VS") else "")) for s, e, f in neut)
        accent = "stemfinal" if oxytone else "persistent"
        decls = []
        for g in ("M", "F"):
            decls.append(Declension(masc, stem, g, 1.0, "rule", accent, "-ης -ες", lemma_key))
        decls.append(Declension(neut, stem, "N", 1.0, "rule", accent, "-ης -ες", ""))
        found.append(tuple(decls))
        return found
    if bare.endswith("υσ"):
        stem = _cut(plain, 2)
        masc = _rows("NS:υσ GS:εοσ DS:ει^ AS:υν VS:υ NP:εισ^ VP:εισ^ GP:εων DP:εσι DP:εσιν AP:εισ^")
        fem = _rows("NS:εια GS:ειᾱσ DS:ειᾳ AS:ειαν NP:ειαι GP:ειῶν DP:ειαισ AP:ειᾱσ")
        neut = _rows("NS:υ AS:υ VS:υ GS:εοσ DS:ει^ NP:εα VP:εα AP:εα GP:εων DP:εσι DP:εσιν")
        found.append((Declension(masc, stem, "M", 1.0, "rule", "ending", "-υς -εια -υ"),
                      Declension(fem, stem, "F", 1.0, "rule", "ending", "-υς -εια -υ"),
                      Declension(neut, stem, "N", 1.0, "rule", "ending", "-υς -εια -υ")))
        return found
    if bare.endswith("ων"):
        stem = _cut(plain, 2) + "ον"
        masc, neut = consonant_table(stem, False), consonant_table(stem, True)
        decls = [Declension(masc, "", g, 1.0, "rule", "persistent", "-ων -ον", lemma_key) for g in ("M", "F")]
        decls.append(Declension(neut, "", "N", 1.0, "rule", "persistent", "-ων -ον", lemma_key))
        # The comparative's contracted forms: μείζω, μείζους.
        short = _cut(plain, 2)
        for g in ("M", "F"):
            decls.append(Declension(_rows("AS:ω NP:ουσ AP:ουσ"), short, g, 0.5, "rule", "persistent",
                                    "-ων -ω -ους", lemma_key))
        decls.append(Declension(_rows("NP:ω AP:ω"), short, "N", 0.5, "rule", "persistent", "-ων -ω", lemma_key))
        neuter_nominative = ph.accentuate(_cut(plain, 2) + "ον", "recessive")
        decls.append(Declension(_rows("NS:- AS:- VS:-"), neuter_nominative[0], "N", 1.0, "rule", "persistent",
                                "-ων -ον", ""))
        found.append(tuple(decls))
        return found
    if bare.endswith("πουσ"):
        # τετράπους, -ποδος (two endings)
        stem = _cut(plain, 3) + "οδ"
        decls = [third_declension(plain, lemma_key, stem, g, 1.0, "-πους -ποδος") for g in ("M", "F")]
        decls.append(Declension(CONS_NEUT, stem, "N", 1.0, "rule", "persistent", "-πουν", lemma_key))
        found.append(tuple(decls))
        return found
    if bare.endswith(("ξ", "ψ")):
        # ἅρπαξ, ἄζυξ: a consonant stem, masculine and feminine alike.
        for consonant, weight in ((("κ", 0.7), ("γ", 0.5)) if bare.endswith("ξ") else (("π", 0.6), ("β", 0.4))):
            oblique = _cut(plain, 1) + consonant
            found.append(tuple(third_declension(plain, lemma_key, oblique, g, weight, f"-ξ -{consonant}ος")
                               for g in ("M", "F")))
        return found
    if bare.endswith("ασ") and bare.endswith("λασ"):
        # μέλας, μέλαινα, μέλαν
        stem = _cut(plain, 2)
        masc = third_declension(plain, lemma_key, stem + "αν", "M", 1.0, "-ας -αινα -αν")
        fem = Declension(A_MIXED, stem + "αιν", "F", 1.0, "rule", "persistent", "-ας -αινα -αν",
                         ph.finish(ph.accentuate(stem + "αινα", "recessive")[0]))
        neut = Declension(_rows("NS:- AS:- VS:-") + CONS_NEUT, stem + "αν", "N", 1.0, "rule", "persistent",
                          "-ας -αινα -αν", lemma_key)
        found.append((masc, fem, neut))
        return found
    if bare.endswith("ην"):
        stem = _cut(plain, 2) + "εν"
        decls = [third_declension(plain, lemma_key, stem, g, 1.0, "-ην -εν") for g in ("M", "F")]
        decls.append(Declension(CONS_NEUT, stem, "N", 1.0, "rule", "persistent", "-ην -εν", lemma_key))
        found.append(tuple(decls))
        return found
    return found


# Comparison: (lemma ending, comparative suffix, superlative suffix).
def comparison_stems(lemma_key: str) -> list[tuple[str, str, float]]:
    """(stem + suffix as a plain -ος adjective nominative, "C" or "S", weight)."""
    plain = ph.plain(lemma_key)
    bare = ph.letters(plain)
    found = []
    if bare.endswith("οσ"):
        stem = _cut(plain, 2)
        for vowel, weight in (("ο", 0.6), ("ω", 0.6)):
            found.append((stem + vowel + "τεροσ", "C", weight))
            found.append((stem + vowel + "τατοσ", "S", weight))
    elif bare.endswith("ησ"):
        stem = _cut(plain, 2)
        found += [(stem + "εστεροσ", "C", 0.8), (stem + "εστατοσ", "S", 0.8)]
    elif bare.endswith("υσ"):
        stem = _cut(plain, 2)
        found += [(stem + "υτεροσ", "C", 0.8), (stem + "υτατοσ", "S", 0.8)]
    elif bare.endswith("ων"):
        stem = _cut(plain, 2)
        found += [(stem + "ονεστεροσ", "C", 0.6), (stem + "ονεστατοσ", "S", 0.6)]
    return found
