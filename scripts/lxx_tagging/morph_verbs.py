"""Verb paradigms for the morphological analyser.

A verb is generated from its tense stems (`Stem`), one per tense system and
hypothesis, and the endings of that system. The tables below are the
regular Greek endings, written unaccented (the accent is placed by rule,
see `morph_phon.accentuate`) except where an ending fixes the accent
(second aorist infinitive -εῖν, aorist passive subjunctive -θῶ, perfect
participle -κώς ...). Endings marked "rare" are real Koine/LXX variants
generated with a lower weight: -οσαν third plurals (ἐλέγοσαν, ἤλθοσαν),
first-aorist endings on second-aorist stems (εἶπαν, ἦλθαν, ἔπεσαν),
future middle second singular -ει and -εσαι (φάγεσαι), -έτωσαν
imperatives, the "Aeolic" aorist optative (ποιήσειε).

Systems (the `Stem.system` values):

  pres    present stem: present and imperfect, active and middle/passive
          (`contract` "α", "ε", "ο", "η" or "ε1" for the contract classes)
  fut     future (sigmatic: λύσ-ω); `contract` "ε" for the liquid and Attic
          futures (κρινῶ, ἐλπιῶ)
  aor1    first aorist (λύσ-α, ἔκριν-α), active and middle
  aor2    second (thematic) aorist (ἔλαβ-ον), active and middle
  aorp    aorist passive in -θη- (ἐλύθ-ην), and the future passive from it
  aorp2   second aorist passive in -η- (ἐγράφ-ην)
  perf    perfect active (λέλυκ-α, γέγραφ-α), and pluperfect
  perfm   perfect middle/passive (λέλυ-μαι, γέγραμ-μαι), and pluperfect

The codes are TAGNT's: `V-{tense}{voice}{mood}-{person}{number}` and
`V-{tense}{voice}P-{case}{number}{gender}`, where TAGNT writes a second
aorist, second perfect or second future with a leading 2 and uses its own
voice letters (A active, M middle, P passive, D middle deponent, O passive
deponent, N middle or passive deponent, E middle or passive). Which voice
letters and which "2" convention a verb's forms carry is the lexeme's
business (`VerbLexeme.voices`, `VerbLexeme.second`), learned from TAGNT where
the verb or its simplex is there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from lxx_tagging import morph_phon as ph

RARE, VERY_RARE = 0.5, 0.25


def _t(*rows):
    """Rows of (person, ending[, flags]) as (person, ending, ultima, weight)."""
    out = []
    for row in rows:
        person, ending = row[0], row[1]
        flags = row[2] if len(row) > 2 else ""
        ultima = "L" if "L" in flags else ("S" if "S" in flags else None)
        weight = VERY_RARE if "x" in flags else (RARE if "w" in flags else 1.0)
        out.append((person, ending, ultima, weight))
    return tuple(out)


# Thematic: present, future, second aorist (non-indicative), imperfect.
ACT_PRIMARY = _t(("1S", "ω"), ("2S", "εισ"), ("3S", "ει"), ("1P", "ομεν"), ("2P", "ετε"),
                 ("3P", "ουσι"), ("3P", "ουσιν"))
ACT_SUBJ = _t(("1S", "ω"), ("2S", "ῃσ"), ("3S", "ῃ"), ("1P", "ωμεν"), ("2P", "ητε"),
              ("3P", "ωσι"), ("3P", "ωσιν"))
ACT_OPT = _t(("1S", "οιμι"), ("2S", "οισ"), ("3S", "οι", "L"), ("1P", "οιμεν"), ("2P", "οιτε"), ("3P", "οιεν"))
ACT_OPT_CONTRACT = ACT_OPT + _t(("1S", "οιην", "w"), ("2S", "οιησ", "w"), ("3S", "οιη", "w"))
ACT_IMP = _t(("2S", "ε"), ("3S", "ετω"), ("2P", "ετε"), ("3P", "ετωσαν"), ("3P", "οντων", "x"))
ACT_SECONDARY = _t(("1S", "ον"), ("2S", "εσ"), ("3S", "ε"), ("3S", "εν"), ("1P", "ομεν"), ("2P", "ετε"),
                   ("3P", "ον"), ("3P", "οσαν", "w"))
MID_PRIMARY = _t(("1S", "ομαι"), ("2S", "ῃ"), ("2S", "ει", "w"), ("3S", "εται"), ("1P", "ομεθα"),
                 ("2P", "εσθε"), ("3P", "ονται"))
MID_SUBJ = _t(("1S", "ωμαι"), ("2S", "ῃ"), ("3S", "ηται"), ("1P", "ωμεθα"), ("2P", "ησθε"), ("3P", "ωνται"))
MID_OPT = _t(("1S", "οιμην"), ("2S", "οιο"), ("3S", "οιτο"), ("1P", "οιμεθα"), ("2P", "οισθε"), ("3P", "οιντο"))
MID_IMP = _t(("2S", "ου"), ("3S", "εσθω"), ("2P", "εσθε"), ("3P", "εσθωσαν"))
MID_SECONDARY = _t(("1S", "ομην"), ("2S", "ου"), ("3S", "ετο"), ("1P", "ομεθα"), ("2P", "εσθε"), ("3P", "οντο"),
                   ("3P", "οσαν", "x"))

# First aorist.
A1_ACT_IND = _t(("1S", "α"), ("2S", "ασ"), ("3S", "ε"), ("3S", "εν"), ("1P", "αμεν"), ("2P", "ατε"), ("3P", "αν"))
A1_ACT_OPT = _t(("1S", "αιμι"), ("2S", "αισ"), ("2S", "ειασ", "w"), ("3S", "αι", "L"), ("3S", "ειε", "w"),
                ("3S", "ειεν", "w"), ("1P", "αιμεν"), ("2P", "αιτε"), ("3P", "αιεν"), ("3P", "ειαν", "w"))
A1_ACT_IMP = _t(("2S", "ον"), ("3S", "ατω"), ("2P", "ατε"), ("3P", "ατωσαν"))
A1_MID_IND = _t(("1S", "αμην"), ("2S", "ω"), ("3S", "ατο"), ("1P", "αμεθα"), ("2P", "ασθε"), ("3P", "αντο"))
A1_MID_OPT = _t(("1S", "αιμην"), ("2S", "αιο"), ("3S", "αιτο"), ("1P", "αιμεθα"), ("2P", "αισθε"), ("3P", "αιντο"))
A1_MID_IMP = _t(("2S", "αι"), ("3S", "ασθω"), ("2P", "ασθε"), ("3P", "ασθωσαν"))
# The LXX's first-aorist endings on a second-aorist stem (εἶπα, ἦλθαν, ἔπεσαν).
A2_ALPHA = _t(("1S", "α", "w"), ("2S", "ασ", "w"), ("1P", "αμεν", "w"), ("2P", "ατε", "w"), ("3P", "αν", "w"))
A2_MID_ALPHA = _t(("1S", "αμην", "x"), ("3S", "ατο", "x"), ("3P", "αντο", "x"))

# Aorist passive (θη and η); the subjunctive and optative carry their accent.
AP_IND = _t(("1S", "ην"), ("2S", "ησ"), ("3S", "η"), ("1P", "ημεν"), ("2P", "ητε"), ("3P", "ησαν"))
AP_SUBJ = _t(("1S", "ῶ"), ("2S", "ῇσ"), ("3S", "ῇ"), ("1P", "ῶμεν"), ("2P", "ῆτε"), ("3P", "ῶσι"), ("3P", "ῶσιν"))
AP_OPT = _t(("1S", "είην"), ("2S", "είησ"), ("3S", "είη"), ("1P", "εῖμεν"), ("1P", "είημεν", "w"), ("2P", "εῖτε"),
            ("3P", "εῖεν"), ("3P", "είησαν", "w"))
AP_IMP = _t(("3S", "ητω"), ("2P", "ητε"), ("3P", "ητωσαν"))

# Perfect and pluperfect.
PF_ACT_IND = _t(("1S", "α"), ("2S", "ασ"), ("3S", "ε"), ("3S", "εν"), ("1P", "αμεν"), ("2P", "ατε"),
                ("3P", "ασι"), ("3P", "ασιν"), ("3P", "αν", "w"))
PF_ACT_SUBJ = _t(("1S", "ω", "w"), ("2S", "ῃσ", "w"), ("3S", "ῃ", "w"), ("1P", "ωμεν", "w"), ("2P", "ητε", "w"),
                 ("3P", "ωσιν", "w"))
PLPF_ACT = _t(("1S", "ειν"), ("2S", "εισ"), ("3S", "ει"), ("1P", "ειμεν"), ("2P", "ειτε"), ("3P", "εισαν"),
              ("1S", "η", "x"), ("3S", "ειν", "x"))
PF_MID_IND = _t(("1S", "μαι"), ("2S", "σαι"), ("3S", "ται"), ("1P", "μεθα"), ("2P", "σθε"), ("3P", "νται"))
PF_MID_IMP = _t(("2S", "σο"), ("3S", "σθω"), ("2P", "σθε"))
PLPF_MID = _t(("1S", "μην"), ("2S", "σο"), ("3S", "το"), ("1P", "μεθα"), ("2P", "σθε"), ("3P", "ντο"))

# Participles: (case-number-gender, ending, ultima, weight).
def _ptc_third(m_nom: str, m_stem: str, f_stem: str, n_nom: str, dat_pl: str) -> tuple:
    """A participle whose masculine and neuter are third declension (stem in
    -ντ- or -οτ-) and feminine first declension in -σα/-ια, as (cng, ending,
    ultima, weight). Dichronic vowels carry their quantity (λύσᾱς, -ᾱσα)."""
    rows = [
        ("NSM", m_nom, "L"), ("VSM", m_nom, "L"), ("GSM", m_stem + "οσ"), ("DSM", m_stem + "ι"),
        ("ASM", m_stem + "α"), ("NPM", m_stem + "εσ"), ("VPM", m_stem + "εσ"), ("GPM", m_stem + "ων"),
        ("DPM", dat_pl), ("DPM", dat_pl + "ν"), ("APM", m_stem + "ασ"),
        ("NSN", n_nom), ("VSN", n_nom), ("ASN", n_nom), ("GSN", m_stem + "οσ"), ("DSN", m_stem + "ι"),
        ("NPN", m_stem + "α"), ("VPN", m_stem + "α"), ("APN", m_stem + "α"), ("GPN", m_stem + "ων"),
        ("DPN", dat_pl), ("DPN", dat_pl + "ν"),
    ]
    fem_long = f_stem.endswith("ι")       # -υῖα: genitive -υίας
    rows += [
        ("NSF", f_stem + "α", "S"), ("VSF", f_stem + "α", "S"), ("ASF", f_stem + "αν", "S"),
        ("GSF", f_stem + ("ᾱσ" if fem_long else "ησ")), ("DSF", f_stem + ("ᾳ" if fem_long else "ῃ")),
        ("NPF", f_stem + "αι"), ("VPF", f_stem + "αι"), ("GPF", f_stem + "ῶν"), ("DPF", f_stem + "αισ"),
        ("APF", f_stem + "ᾱσ"),
    ]
    return tuple((row[0], row[1], row[2] if len(row) > 2 else None, 1.0) for row in rows)


# The accent of these stays on the ending's first syllable (persistent there):
# λαβών -όντος -οῦσα, λυθείς -έντος -εῖσα, λελυκώς -ότος -υῖα.
PTC_ONT = _ptc_third("ων", "οντ", "ουσ", "ον", "ουσι")     # λύων (present, future; recessive)
PTC_ANT = _ptc_third("ᾱσ", "αντ", "ᾱσ", "αν", "ᾱσι")       # λύσας (first aorist; recessive)
PTC_ENT = _ptc_third("εισ", "εντ", "εισ", "εν", "εισι")     # λυθείς (on the ending)
PTC_OT = _ptc_third("ωσ", "οτ", "υι", "οσ", "οσι")          # λελυκώς (on the ending)
PTC_ONT_FIXED = PTC_ONT                                    # λαβών (on the ending)


def _second_declension(stem: str) -> tuple:
    """-ος -η -ον endings (participles in -μενος)."""
    rows = [("NSM", "οσ"), ("GSM", "ου"), ("DSM", "ῳ"), ("ASM", "ον"), ("VSM", "ε"),
            ("NPM", "οι"), ("VPM", "οι"), ("GPM", "ων"), ("DPM", "οισ"), ("APM", "ουσ"),
            ("NSF", "η"), ("VSF", "η"), ("GSF", "ησ"), ("DSF", "ῃ"), ("ASF", "ην"),
            ("NPF", "αι"), ("VPF", "αι"), ("GPF", "ων"), ("DPF", "αισ"), ("APF", "ᾱσ"),
            ("NSN", "ον"), ("VSN", "ον"), ("ASN", "ον"), ("GSN", "ου"), ("DSN", "ῳ"),
            ("NPN", "α"), ("VPN", "α"), ("APN", "α"), ("GPN", "ων"), ("DPN", "οισ")]
    return tuple((cng, stem + ending, None, 1.0) for cng, ending in rows)


PTC_MEN = _second_declension("ομεν")      # λυόμενος
PTC_AMEN = _second_declension("αμεν")     # λυσάμενος
PTC_PMEN = _second_declension("μεν")      # λελυμένος (accent fixed on -μέν-)


# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Stem:
    """One hypothesis for a tense stem of a verb's base (no preverb).

    `stem` is unaugmented (a perfect's is reduplicated); `augmented` its
    augmented spellings for the indicative of the past tenses, with weights.
    `second` is TAGNT's "2" convention for the system's codes.
    """

    system: str
    stem: str
    augmented: tuple[tuple[str, float], ...] = ()
    weight: float = 1.0
    source: str = "rule"
    contract: str | None = None
    second: bool = False
    note: str = ""
    # For system "table": (tense, slot, mood, rest, accented simplex form, fixed accent, weight).
    forms: tuple = ()


@dataclass
class Form:
    key: str          # the lookup key (accented, NFC, σ)
    code: str         # TAGNT code
    weight: float
    note: str


def _code(tense: str, voice: str, mood: str, rest: str) -> str:
    return f"V-{tense}{voice}{mood}" + (f"-{rest}" if rest else "")


@dataclass
class VerbLexeme:
    """A verb as the generator needs it.

    `chain` are its preverbs, outermost first; `systems` the stem
    hypotheses of its base; `voices` maps a formal slot to TAGNT voice
    letters with weights:
      "act" active forms, "mp" present/perfect middle-passive forms,
      "fm" future middle forms, "am" aorist middle forms, "ap" aorist
      passive forms, "fp" future passive forms.
    `second` maps a system to the share of TAGNT codes with a leading "2"
    where TAGNT's own use for this verb differs from the system's default.
    """

    strongs: str
    lemma: str
    chain: tuple = ()
    systems: dict = field(default_factory=dict)
    voices: dict = field(default_factory=dict)
    second: dict = field(default_factory=dict)
    weight: float = 1.0
    note: str = ""
    augment_initial: bool = False   # augment before the preverb (denominatives: ἐπροφήτευσα)


DEFAULT_VOICES = {
    "act": (("A", 1.0),),
    "mp": (("P", 1.0), ("M", 0.5)),
    "fm": (("M", 1.0), ("D", 1.0)),
    "am": (("M", 1.0), ("D", 0.4)),
    "ap": (("P", 1.0), ("O", 0.2)),
    "fp": (("P", 1.0),),
}
DEPONENT_VOICES = {
    "act": (("A", 1.0),),
    "mp": (("N", 1.0), ("M", 0.2)),
    "fm": (("D", 1.0),),
    "am": (("D", 1.0), ("M", 0.2)),
    "ap": (("O", 1.0), ("P", 0.8)),
    "fp": (("O", 1.0), ("P", 1.0)),
}


class Generator:
    """Generates every form of a VerbLexeme, as `Form`s (key, TAGNT code,
    weight, the rule path in `note`).

    `vocabulary`, if given, is the set of letters (`text.bare`) of the forms
    wanted: a form with other letters is dropped before it is accented,
    which is most of the generation's cost. Codes are not filtered here (the
    analyser maps a parse TAGNT lacks to its nearest code)."""

    def __init__(self, vocabulary: set[str] | None = None):
        self.vocabulary = vocabulary

    def forms(self, lexeme: VerbLexeme) -> Iterator[Form]:
        for stems in lexeme.systems.values():
            for stem in stems:
                yield from self._system(lexeme, stem)

    # -- helpers -----------------------------------------------------------

    def _voices(self, lexeme: VerbLexeme, slot: str):
        if slot in lexeme.voices:
            return lexeme.voices[slot]
        return DEFAULT_VOICES[slot]

    def _tenses(self, lexeme: VerbLexeme, stem: Stem, letter: str, mood: str) -> list[tuple[str, float]]:
        """TAGNT's tense field for this stem's forms: "A" or "2A" (etc.), weighted."""
        default = "2" if stem.second else ""
        share = lexeme.second.get((stem.system, mood), lexeme.second.get((stem.system, "")))
        if share is None:
            return [(default + letter, 1.0)]
        found = []
        if share > 0:
            found.append(("2" + letter, share))
        if share < 1:
            found.append((letter, 1 - share))
        return found

    def _emit(self, lexeme, stem, words, codes_weights, base_weight, note):
        for word, weight in words:
            key = ph.finish(word)
            for code, code_weight in codes_weights:
                yield Form(key, code, base_weight * weight * code_weight * stem.weight * lexeme.weight,
                           f"{stem.source}:{stem.system}{(' ' + note) if note else ''}")

    def _build(self, lexeme: VerbLexeme, body: str, ending: str, *, augmented: bool, mode: str = "recessive",
               ultima=None, contract: str | None = None, syllable_from_body: int | None = None,
               kind: str | None = None) -> list[tuple[str, float]]:
        """The accented spellings of preverbs + body + ending, weighted.

        `syllable_from_body`: for a persistent or fixed-position accent, the
        nucleus index counted from the start of `body`.
        """
        results = []
        chain = lexeme.chain
        if lexeme.augment_initial and augmented and chain:
            # Augment before the whole compound: the body was built unaugmented.
            chain = ()
        for word, preverb_last, weight in ph.attach(chain, body + ending):
            prefix_nuclei = preverb_last + 1
            body_start = len(ph.clusters(word)) - len(ph.clusters(body + ending))
            boundary = body_start + len(ph.clusters(body))
            if self.vocabulary is not None:
                test = (ph.contract(word, boundary, _contract_table(contract)) or word) if contract else word
                if ph.letters(test) not in self.vocabulary:
                    continue
            if mode == "recessive":
                floor = prefix_nuclei if (augmented and chain) else max(0, prefix_nuclei - 1)
                accented = ph.accentuate(word, "recessive", ultima, floor, -1, kind)
            elif mode == "syllable":
                accented = ph.accentuate(word, "syllable", ultima, 0,
                                         prefix_nuclei + (syllable_from_body or 0), kind)
            elif mode == "penult":
                accented = ph.accentuate(word, "syllable", ultima, 0, -2)
            else:
                accented = ph.accentuate(word, "fixed")
            for form in accented:
                if contract:
                    contracted = ph.contract(form, boundary, _contract_table(contract))
                    if contracted is None and contract != "ε1":
                        continue
                    form = contracted or form
                results.append((form, weight))
        return results

    def _bodies(self, lexeme: VerbLexeme, stem: Stem, augmented: bool) -> list[tuple[str, float]]:
        if not augmented:
            return [(stem.stem, 1.0)]
        if lexeme.augment_initial and lexeme.chain:
            # ἐ-προφήτευσα: the whole compound takes the augment.
            found = []
            for word, _, weight in ph.attach(lexeme.chain, stem.stem):
                for aug, aug_weight in ph.augment(word):
                    found.append((aug, weight * aug_weight))
            return found
        # The augments TAGNT shows for this stem, and the rule's (an
        # unaugmented Koine οἰκοδομήθη does not rule out ᾠκοδομήθη).
        found = list(stem.augmented)
        seen = {ph.letters(text) for text, _ in found}
        for text, weight in ph.augment(stem.stem):
            if ph.letters(text) not in seen:
                found.append((text, weight * (0.8 if stem.augmented else 1.0)))
        return found

    # -- systems -----------------------------------------------------------

    def _system(self, lexeme: VerbLexeme, stem: Stem) -> Iterator[Form]:
        system = stem.system
        if system == "pres":
            yield from self._thematic(lexeme, stem, "P", "I", primary=True)
        elif system == "fut":
            yield from self._thematic(lexeme, stem, "F", None, primary=False)
        elif system == "aor1":
            yield from self._aorist1(lexeme, stem)
        elif system == "aor2":
            yield from self._aorist2(lexeme, stem)
        elif system in ("aorp", "aorp2"):
            yield from self._passive(lexeme, stem)
        elif system == "perf":
            yield from self._perfect(lexeme, stem)
        elif system == "perfm":
            yield from self._perfect_middle(lexeme, stem)
        elif system == "table":
            yield from self._table(lexeme, stem)

    def _finite(self, lexeme, stem, rows, tense_letter, mood, slot, augmented, contract=None, mode="recessive",
                note=""):
        for body, body_weight in self._bodies(lexeme, stem, augmented):
            for person, ending, ultima, weight in rows:
                words = self._build(lexeme, body, ending, augmented=augmented, mode=mode, ultima=ultima,
                                    contract=contract)
                codes = [(_code(tense, voice, mood, person), tw * vw)
                         for tense, tw in self._tenses(lexeme, stem, tense_letter, mood)
                         for voice, vw in self._voices(lexeme, slot)]
                yield from self._emit(lexeme, stem, words, codes, weight * body_weight, note)

    def _nonfinite(self, lexeme, stem, ending, tense_letter, slot, *, mode, ultima=None, contract=None, note=""):
        words = self._build(lexeme, stem.stem, ending, augmented=False, mode=mode, ultima=ultima, contract=contract)
        codes = [(_code(tense, voice, "N", ""), tw * vw)
                 for tense, tw in self._tenses(lexeme, stem, tense_letter, "N")
                 for voice, vw in self._voices(lexeme, slot)]
        yield from self._emit(lexeme, stem, words, codes, 1.0, note)

    def _participle(self, lexeme, stem, table, tense_letter, slot, *, accent: str, contract=None,
                    body: str | None = None):
        """accent: "recessive" (persistent from the recessive NSM), "ending"
        (persistent on the ending's first syllable), or "men" (on -μέν-)."""
        body = stem.stem if body is None else body
        codes_for = lambda cng: [(_code(tense, voice, "P", cng), tw * vw)
                                 for tense, tw in self._tenses(lexeme, stem, tense_letter, "P")
                                 for voice, vw in self._voices(lexeme, slot)]
        syllable = None
        if accent == "recessive":
            nominative = next(ending for cng, ending, _, _ in table if cng == "NSM")
            nominative_ultima = next(ultima for cng, _, ultima, _ in table if cng == "NSM")
            # The NSM's recessive accent, as a nucleus index from the body's start.
            word = body + nominative
            accented = ph.accentuate(word, "recessive", nominative_ultima)[0]
            position = ph.accent_position(accented)
            syllable = position[0] if position else 0
        elif accent == "men":
            syllable = ph.nucleus_count(body + "με") - 1
        elif accent == "ending":
            syllable = ph.nucleus_count(body)
        for cng, ending, ultima, weight in table:
            if any(m in ph.ACCENTS for m in ph.nfd(ending)):
                words = self._build(lexeme, body, ending, augmented=False, mode="fixed", contract=contract)
            else:
                words = self._build(lexeme, body, ending, augmented=False, mode="syllable", ultima=ultima,
                                    contract=contract, syllable_from_body=syllable,
                                    kind=ph.ACUTE if accent == "ending" else None)
            yield from self._emit(lexeme, stem, words, codes_for(cng), weight, "")

    def _thematic(self, lexeme, stem, tense, past, primary):
        contract = stem.contract
        if tense == "F" and contract:
            contract = "ε"
        inf = "εν" if contract else "ειν"
        inf_ultima = "L" if contract else None
        opt = ACT_OPT_CONTRACT if contract else ACT_OPT
        act = self._voices(lexeme, "act")
        if act:
            yield from self._finite(lexeme, stem, ACT_PRIMARY, tense, "I", "act", False, contract)
            if primary:
                yield from self._finite(lexeme, stem, ACT_SUBJ, tense, "S", "act", False, contract)
                yield from self._finite(lexeme, stem, ACT_IMP, tense, "M", "act", False, contract)
                yield from self._finite(lexeme, stem, ACT_SECONDARY, "I", "I", "act", True, contract)
            yield from self._finite(lexeme, stem, opt, tense, "O", "act", False, contract)
            yield from self._nonfinite(lexeme, stem, inf, tense, "act", mode="recessive", ultima=inf_ultima,
                                       contract=contract)
            yield from self._participle(lexeme, stem, PTC_ONT, tense, "act", accent="recessive", contract=contract)
        slot = "mp" if primary else "fm"
        middle_primary = MID_PRIMARY if primary else MID_PRIMARY + _t(("2S", "εσαι", "x"))
        yield from self._finite(lexeme, stem, middle_primary, tense, "I", slot, False, contract)
        if primary:
            yield from self._finite(lexeme, stem, MID_SUBJ, tense, "S", slot, False, contract)
            yield from self._finite(lexeme, stem, MID_IMP, tense, "M", slot, False, contract)
            yield from self._finite(lexeme, stem, MID_SECONDARY, "I", "I", slot, True, contract)
        yield from self._finite(lexeme, stem, MID_OPT, tense, "O", slot, False, contract)
        yield from self._nonfinite(lexeme, stem, "εσθαι", tense, slot, mode="recessive", contract=contract)
        yield from self._participle(lexeme, stem, PTC_MEN, tense, slot, accent="recessive", contract=contract)

    def _aorist1(self, lexeme, stem):
        if self._voices(lexeme, "act"):
            yield from self._finite(lexeme, stem, A1_ACT_IND, "A", "I", "act", True)
            yield from self._finite(lexeme, stem, ACT_SUBJ, "A", "S", "act", False)
            yield from self._finite(lexeme, stem, A1_ACT_OPT, "A", "O", "act", False)
            yield from self._finite(lexeme, stem, A1_ACT_IMP, "A", "M", "act", False)
            yield from self._nonfinite(lexeme, stem, "αι", "A", "act", mode="penult")
            yield from self._participle(lexeme, stem, PTC_ANT, "A", "act", accent="recessive")
        yield from self._finite(lexeme, stem, A1_MID_IND, "A", "I", "am", True)
        yield from self._finite(lexeme, stem, MID_SUBJ, "A", "S", "am", False)
        yield from self._finite(lexeme, stem, A1_MID_OPT, "A", "O", "am", False)
        yield from self._finite(lexeme, stem, A1_MID_IMP, "A", "M", "am", False)
        yield from self._nonfinite(lexeme, stem, "ασθαι", "A", "am", mode="recessive")
        yield from self._participle(lexeme, stem, PTC_AMEN, "A", "am", accent="recessive")

    def _aorist2(self, lexeme, stem):
        if self._voices(lexeme, "act"):
            yield from self._finite(lexeme, stem, ACT_SECONDARY + A2_ALPHA, "A", "I", "act", True)
            yield from self._finite(lexeme, stem, ACT_SUBJ, "A", "S", "act", False)
            yield from self._finite(lexeme, stem, ACT_OPT, "A", "O", "act", False)
            yield from self._finite(lexeme, stem, ACT_IMP + _t(("2P", "ατε", "w"), ("3S", "ατω", "w"),
                                                               ("3P", "ατωσαν", "w")), "A", "M", "act", False)
            # εἰπέ, ἐλθέ, εὑρέ, ἰδέ, λαβέ: the five oxytone imperatives.
            if ph.letters(stem.stem) in ("ειπ", "ελθ", "ευρ", "ιδ", "λαβ") and not lexeme.chain:
                yield from self._finite(lexeme, stem, _t(("2S", "έ")), "A", "M", "act", False, mode="fixed")
            yield from self._nonfinite(lexeme, stem, "εῖν", "A", "act", mode="fixed")
            yield from self._participle(lexeme, stem, PTC_ONT_FIXED, "A", "act", accent="ending")
        yield from self._finite(lexeme, stem, MID_SECONDARY + A2_MID_ALPHA, "A", "I", "am", True)
        yield from self._finite(lexeme, stem, MID_SUBJ, "A", "S", "am", False)
        yield from self._finite(lexeme, stem, MID_OPT, "A", "O", "am", False)
        yield from self._finite(lexeme, stem, _t(("2S", "οῦ")), "A", "M", "am", False, mode="fixed")
        yield from self._finite(lexeme, stem, _t(("3S", "εσθω"), ("2P", "εσθε"), ("3P", "εσθωσαν")),
                                "A", "M", "am", False)
        yield from self._nonfinite(lexeme, stem, "έσθαι", "A", "am", mode="fixed")
        yield from self._participle(lexeme, stem, PTC_MEN, "A", "am", accent="recessive")

    def _passive(self, lexeme, stem):
        slot = "ap"
        yield from self._finite(lexeme, stem, AP_IND, "A", "I", slot, True)
        yield from self._finite(lexeme, stem, AP_SUBJ, "A", "S", slot, False, mode="fixed")
        yield from self._finite(lexeme, stem, AP_OPT, "A", "O", slot, False, mode="fixed")
        imperative = "ητι" if ph.letters(stem.stem).endswith("θ") else "ηθι"
        yield from self._finite(lexeme, stem, _t(("2S", imperative)) + AP_IMP, "A", "M", slot, False)
        yield from self._nonfinite(lexeme, stem, "ῆναι", "A", slot, mode="fixed")
        yield from self._participle(lexeme, stem, PTC_ENT, "A", slot, accent="ending")
        # The future passive: the aorist passive stem + -ήσ- + middle endings.
        future = Stem("futp", stem.stem + "ησ", (), stem.weight, stem.source, None, stem.second, stem.note)
        yield from self._finite(lexeme, future, MID_PRIMARY, "F", "I", "fp", False)
        yield from self._finite(lexeme, future, MID_OPT, "F", "O", "fp", False)
        yield from self._nonfinite(lexeme, future, "εσθαι", "F", "fp", mode="recessive")
        yield from self._participle(lexeme, future, PTC_MEN, "F", "fp", accent="recessive")

    def _perfect(self, lexeme, stem):
        yield from self._finite(lexeme, stem, PF_ACT_IND, "R", "I", "act", False)
        yield from self._finite(lexeme, stem, PF_ACT_SUBJ, "R", "S", "act", False)
        yield from self._finite(lexeme, stem, _t(("2S", "ε", "w"), ("2P", "ετε", "w")), "R", "M", "act", False)
        yield from self._nonfinite(lexeme, stem, "έναι", "R", "act", mode="fixed")
        yield from self._participle(lexeme, stem, PTC_OT, "R", "act", accent="ending")
        # The pluperfect, with and (as often in Koine) without the augment.
        pluperfect = Stem("perf", stem.stem, stem.augmented or tuple(
            (aug, w * 0.6) for aug, w in ph.augment(stem.stem) if not ph.starts_with_vowel(stem.stem)),
            stem.weight, stem.source, None, stem.second, stem.note)
        yield from self._finite(lexeme, Stem("perf", stem.stem, (), stem.weight, stem.source, None, stem.second),
                                PLPF_ACT, "L", "I", "act", False)
        if pluperfect.augmented:
            yield from self._finite(lexeme, pluperfect, PLPF_ACT, "L", "I", "act", True)

    def _perfect_middle(self, lexeme, stem):
        consonant = ph.letters(stem.stem)[-1:] in ph.CONSONANTS
        mp = self._voices(lexeme, "mp")

        def codes(tense, mood, person):
            return [(_code(t, voice, mood, person), tw * vw)
                    for t, tw in self._tenses(lexeme, stem, tense, mood) for voice, vw in mp]

        for mood, table in (("I", PF_MID_IND), ("M", PF_MID_IMP)):
            for person, ending, ultima, weight in table:
                if consonant and person == "3P":
                    continue        # periphrastic (γεγραμμένοι εἰσίν)
                words = self._build(lexeme, ph.perfect_middle(stem.stem, ending), "", augmented=False,
                                    ultima=ultima)
                yield from self._emit(lexeme, stem, words, codes("R", mood, person), weight, "")
        words = self._build(lexeme, ph.perfect_middle(stem.stem, "σθαι"), "", augmented=False, mode="penult")
        yield from self._emit(lexeme, stem, words, codes("R", "N", ""), 1.0, "")
        body = ph.perfect_middle(stem.stem, "μεν")[:-3]
        yield from self._participle(lexeme, stem, PTC_PMEN, "R", "mp", accent="men", body=body)
        # The pluperfect, unaugmented (as often in Koine) and augmented.
        bodies = [(stem.stem, 1.0, False)]
        if not ph.starts_with_vowel(stem.stem):
            bodies += [(aug, w * 0.6, True) for aug, w in (stem.augmented or ph.augment(stem.stem))]
        for body_stem, body_weight, augmented in bodies:
            for person, ending, ultima, weight in PLPF_MID:
                if consonant and person == "3P":
                    continue
                words = self._build(lexeme, ph.perfect_middle(body_stem, ending), "", augmented=augmented,
                                    ultima=ultima)
                yield from self._emit(lexeme, stem, words, codes("L", "I", person), weight * body_weight, "")

    def _table(self, lexeme, stem):
        """Explicit forms (`Stem.forms`): the athematic verbs and other
        paradigms written out. A finite form's accent is recessive and so
        placed again in a compound (δίδωσι, ἀποδίδωσι; δός, ἀπόδος), unless
        the table marks it fixed; other forms keep theirs (δοῦναι, ἀποδοῦναι)."""
        for entry in stem.forms:
            tense, slot, mood, rest, form, fixed, weight = entry
            simplex = ph.plain(form)
            recessive = mood in "IM" and not fixed
            augmented = mood == "I" and tense.lstrip("2") in "IAL" and ph.starts_with_vowel(simplex)
            position = ph.accent_position(form)
            # A simplex accented on its last syllable keeps that accent's kind
            # if the compound's accent cannot go further back (ἦν, παρῆν).
            kind = position[1] if position and position[0] == ph.nucleus_count(form) - 1 else None
            for word, preverb_last, attach_weight in ph.attach(lexeme.chain, simplex if recessive else form):
                if self.vocabulary is not None and ph.letters(word) not in self.vocabulary:
                    continue
                if not lexeme.chain:
                    accented = (form,)
                elif recessive:
                    prefix_nuclei = preverb_last + 1
                    floor = prefix_nuclei if augmented else max(0, prefix_nuclei - 1)
                    accented = ph.accentuate(word, "recessive", None, floor, -1, kind)
                else:
                    accented = (word,)
                words = [(a, attach_weight) for a in accented]
                codes = [(_code(tense, voice, mood, rest), vw) for voice, vw in self._voices(lexeme, slot)]
                yield from self._emit(lexeme, stem, words, codes, weight, "")


def _contract_table(contract: str):
    if contract == "η":
        return ph.ETA_CONTRACTIONS
    if contract == "ε1":
        return {"ε": "ει", "ει": "ει"}
    return ph.CONTRACTIONS[contract]
