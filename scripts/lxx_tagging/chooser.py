"""The contextual chooser: one reading per token, chosen in its verse.

WHAT IT DOES

A token reaches the chooser with its candidates (`TokenCandidates`): the
readings the form table, the lexicon, a tolerant lookup or an analyser offer,
after the case filter. Tier a has one reading; tiers b (one word, several
parses), c (several words) and u (tolerant matches only) have several or
unconfirmed ones. The chooser picks the best path through the verse's
*candidate lattice*, one candidate per token, with a linear model, and gives
every choice a calibrated confidence. It never invents a reading: it only
chooses among the candidates it is given, so it works the same on readings
the NT never attests (the analyser's).

THE MODEL

A structured linear model scored over candidates (not over free tags), learnt
with the averaged structured perceptron and decoded with a beam search over
the lattice in which hypotheses that agree on the last candidate and on a
small summary of the context so far are recombined (so the search is exact
whenever the beam holds every distinct state). A path's score is the sum, at
each token, of three kinds of feature:

  unary    the candidate in its observed context: its full code; its lemma
           with its code; the printed form with its code and with its Strong's
           number (memorised preferences for forms the NT has); the share of
           the token's evidence it carries, its rank, its count, and whether
           it was added by the paradigm rules with no evidence of its own; its
           source and kind against the token's capital and position; its
           last letters with its case, number, gender or its tense, voice,
           mood, person (how an unseen form's ending bears on its parse); the
           neighbouring printed words and punctuation; whether the next
           tokens can agree with it in case, number and gender (looking on
           past an article or adjective that agrees, up to three tokens:
           τοῦ ἀγαθοῦ λόγου).
  pairwise the previous token's chosen candidate with this one: tag bigram
           (part of speech, and full codes); case/number/gender agreement by
           pair of parts of speech; a preposition's lemma with the case it
           governs; a verb's lemma with the case after it; a closed-class
           word's lemma with the next code; a nominative's number and person
           with a following finite verb.
  context  a summary of the path so far: the parts of speech of the two
           previous tokens (tag trigram); the open articles (one not yet
           closed by its noun, and the one outside it: ὁ τοῦ θεοῦ λόγος; a
           word in another case inside the phrase leaves it open: τοῖς
           ἐκείνου γράμμασιν) and whether this candidate agrees with them, or
           is the infinitive one nominalises (τοῦ/ἐν τῷ + infinitive); an
           open preposition and the case this candidate has under it; the
           sentence's last finite verb (person, number, mood, lemma) against
           a nominative, vocative or another finite verb, forgotten at a
           subordinating conjunction, whose clause has its own; the
           sentence's last nominative against a finite verb; coordination:
           the first nominal word after καί, ἤ, δέ ... against the case and
           number of the last nominal word before it (a possessive genitive
           pronoun between does not count: τὰ κτήνη αὐτῶν καὶ πάντα).
  sentence what the rest of the sentence certainly is, read from the other
           tokens' candidates (no decision needed): the nearest finite verb
           on each side of a nominal, with what stands between (a relative
           pronoun or subordinating conjunction: Τὸ θηρίον ὃ εἶδες ἦν; a
           coordinator; a sentence break), its person, number, voice and
           lemma; whether another word is certainly nominative (a plural
           nominative/vocative counts) or accusative; the nearest finite
           verb's arguments: whether another noun phrase about it (between
           the finite verbs either side; a run of tokens that can agree,
           `_chunks`; not a preposition's object) must, can or cannot be
           nominative, and the same for accusative, with the verb's voice
           and how often its lemma takes an object in the text being tagged
           (`verb_objects`, an unsupervised count like `certain_lexemes`),
           for a nominative or accusative reading; for a finite verb, the
           nearest conjunction or particle before it in its clause (ἵνα,
           ἐάν, μή, οὐ μή, ὅτι ...) and the other verbs' moods; for a
           vocative, the moods about it; for a third-person pronoun or a
           relative, its antecedent (the nearest earlier noun of the same
           number) and whether the genders match; for a name tagged without
           its case, the gender TBESG gives it; for a choice between words,
           how often each is certain elsewhere in the same text
           (`certain_lexemes`, an unsupervised count).
  rules    one rule of grammar is imposed, not learnt: after a proper
           preposition the word is not nominative or vocative.

Clauses for these features are cut at sentence breaks only, not at commas:
Brenton prints nearly twice as many commas as the NA text the model learns
from, and features that read them shifted the LXX's choices (more
nominatives) without helping the NT.

Every feature is decomposed from the morphology code (part of speech; case,
number, gender; person; tense, voice, mood), so what is learnt about one word
or form carries over to forms and lemmas the NT never has.

TRAINING DATA

TAGNT only (`ntcorpus`): the NA text set in Brenton's capitals, its tokens
looked up by the tagger's own providers (`pipeline.TaggerSpec`, the CLI's
own) against a form table built from the other folds' books (every
edition's rows), the morphological analyser included and built over that
same table, with the paradigm expansion and the case filter exactly as for
the LXX. The folds' candidate sets are cached under
`.cache/lxx_tagging/candidate_sets/` (an analyser takes a minute or so to
build per fold; they are built in parallel). So the training tokens are ambiguous,
or unseen, the way the LXX's are. Where the true reading is not among a
token's candidates the token constrains nothing (the gold path is the best
path through the rest; the token's choice there is free); its candidates
still serve as context.

CONFIDENCE

A choice's margin is how much worse the best path is that chooses otherwise
at that token, the tokens around it free to change with it (a window of
`WINDOW` tokens either side is re-decoded, and the rest of the path rescored
until it agrees again), so a noun and its article flipping together count
as one alternative. Margins become confidences through a calibration table
measured on held-out NT folds: accuracy of the full reading (Strong's,
lemma, morphology) by margin band, per method, tier, source family (TAGNT
form, lexicon, analyser) and number of readings chosen from (2, 3, 4+),
smoothed and made monotone. A unique reading (tier a) gets the measured
accuracy of tier a for its source family.

METHODS (`Choice.method`)

  unique    one reading: tier a. Not the model's decision.
  model     the model chose among taggable candidates (tiers b, c, or any
            token an analyser gave several readings).
  tolerant  the model chose among tolerant matches only (tier u). A match
            whose case an iota subscript on the printed form contradicts
            (ἐν κληρονομίᾳ found as κληρονομία, N-NSF) is set aside when
            another fits (`tolerant_fits`).
  tolerant-mismatch
            no tolerant match fits the printed form; the best is still
            tagged (the policy tags every token with a candidate), with
            confidence at most `MISMATCH_CONFIDENCE`: its Strong's number
            and lemma are usually right, its case is not.

Tokens with no candidate the case filter kept (tiers d and e, and the u
tokens whose only readings are names for a lower-case word, mostly common
words such as βόες and χλόης that only a name in the NT spells alike) are
not tagged.

WHAT WAS TRIED AND LEFT OUT

Self-training on the LXX (retraining with the model's own confident LXX
choices, margin >= 40, as extra examples) changed 3,391 LXX decisions; of 40
of them read by hand the retrained model was right 18 times and the
original 17, and held-out NT accuracy fell slightly (91.1% to 91.0% on
fold 1's ambiguous tokens). So the model is trained on TAGNT alone, and the
LXX text is not an input to it. A wider beam (16) and more epochs (10) made
no measurable difference.

Round 2 (NT cross-validation, ambiguous tokens, from 90.67%): the verb's
arguments 90.73, with coordination and the subordinate-clause reset 90.78,
with the longer agreement lookahead 90.87. Also tried and left out: a
lookback from an adjective or participle to the noun before it, past a
genitive or article (90.72, worse). Nouns whose gender the candidates leave
open are rare in the Septuagint (about 1,200 tokens, mostly παῖς, πατριά,
ὄνος), so no gender lexicon was built for them.

PERSISTENCE

`load_or_train` trains on first use and caches the model under
`.cache/lxx_tagging/chooser/`, keyed by a hash of the TAGNT and TBESG files
and of the code that makes the training data and the model; the same inputs
give the same model, byte for byte, and the same choices.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import time
import unicodedata
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, NamedTuple, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging.candidates import Candidate, Choice, TokenCandidates  # noqa: E402
from lxx_tagging.lexicon import split_variants  # noqa: E402
from lxx_tagging.text import bare, normalize  # noqa: E402

REPOSITORY_ROOT = SCRIPTS.parent
CACHE_DIR = REPOSITORY_ROOT / ".cache" / "lxx_tagging" / "chooser"
FORMAT = 1

BEAM = 8
WINDOW = 3
EPOCHS = 6
FOLDS = 5
SEED = 0


# ---------------------------------------------------------------------------
# Morphology codes, decomposed

class Attrs(NamedTuple):
    pos: str        # the code's head: N A T P R C D K I X Q F S V ADV CONJ ...; NAME for no code
    cls: str        # coarse class: VF finite verb, VP participle, VN infinitive, else pos
    case: str
    num: str
    gen: str
    person: str
    tense: str
    voice: str
    mood: str
    suffix: str
    compound: bool

    @property
    def cng(self) -> str:
        return self.case + self.num + self.gen

    @property
    def tvm(self) -> str:
        return self.tense + self.voice + self.mood + self.person + self.num


_CNG = re.compile(r"^([NGDAV])([SP])([MFN]?)$")
_NOMINAL_HEADS = frozenset("NATRCDKIXQ")
_VERB = re.compile(r"^(2?)([PIFARL])([AMPEDON])([ISOMNP])$")
NONE_ATTRS = Attrs("UNK", "UNK", "", "", "", "", "", "", "", "", False)
NAME_ATTRS = Attrs("NAME", "NAME", "", "", "", "", "", "", "", "", False)


@lru_cache(maxsize=None)
def parse_morph(code: str | None) -> Attrs:
    """A TAGNT code's fields. `None` (a name or loanword tagged without its
    case) is NAME. A compound code (κἀγώ `P-1NS + G2532=CONJ`, κἀκεῖνος
    `CONJ + G1565=D-NSM`) takes its head from the first part and its case,
    number and gender from the first part that has them."""
    if code is None:
        return NAME_ATTRS
    parts = [part.split("=")[-1] for part in code.split(" + ")]
    head = _parse_simple(parts[0])
    if len(parts) > 1:
        if not head.case:
            for part in parts[1:]:
                other = _parse_simple(part)
                if other.case:
                    head = head._replace(case=other.case, num=other.num, gen=other.gen)
                    break
        head = head._replace(compound=True)
    return head


def _parse_simple(code: str) -> Attrs:
    fields = code.split("-")
    first = fields[0]
    rest = fields[1:]
    case = num = gen = person = tense = voice = mood = suffix = ""
    cls = first
    if first == "V" and rest:
        match = _VERB.match(rest[0])
        if match:
            second, tense, voice, mood = match.groups()
            tense = second + tense
            if mood == "P" and len(rest) > 1:
                cng = _CNG.match(rest[1])
                if cng:
                    case, num, gen = cng.groups()
                cls = "VP"
            elif mood == "N":
                cls = "VN"
            else:
                if len(rest) > 1 and len(rest[1]) == 2:
                    person, num = rest[1][0], rest[1][1]
                cls = "VF"
            suffix = rest[2] if len(rest) > 2 else ""
    elif first in _NOMINAL_HEADS and rest:
        cng = _CNG.match(rest[0])
        if cng:
            case, num, gen = cng.groups()
            suffix = rest[1] if len(rest) > 1 else ""
        else:
            suffix = "-".join(rest)
    elif first in ("P", "F") and rest:
        body = rest[0]
        if body[:1] in "123":
            person, body = body[0], body[1:]
        cng = _CNG.match(body)
        if cng:
            case, num, gen = cng.groups()
        elif len(body) == 2 and body[0] in "NGDAV":
            case, num = body[0], body[1]
        suffix = "-".join(rest[1:])
    elif first == "S" and rest:
        body = rest[0]
        person = body[:2]
        cng = _CNG.match(body[2:])
        if cng:
            case, num, gen = cng.groups()
    else:
        suffix = "-".join(rest)
    return Attrs(first, cls, case, num, gen, person, tense, voice, mood, suffix, False)


def _agree(a: Attrs, b: Attrs) -> str:
    """Case, number, gender flags: 1 same, 0 different, - unknown."""
    return "".join(
        "-" if not x or not y else ("1" if x == y else "0")
        for x, y in ((a.case, b.case), (a.num, b.num), (a.gen, b.gen))
    )


# Parts of speech that head a nominal phrase, closing an open article or preposition.
_HEADS = frozenset(("N", "NAME", "P", "D", "I", "X", "R", "F", "C", "K", "Q", "UNKNAME"))
_OPEN_CLASS = frozenset(("N", "A", "V", "NAME"))


@lru_cache(maxsize=None)
def lemma_key(lemma: str) -> str:
    variants = split_variants(lemma)
    return normalize(variants[0] if variants else lemma)


def family(source: str) -> str:
    """A candidate source's family, for features and calibration."""
    head = source.split("+")[0].split(":")[0]
    if head in ("nt-form", "nt-caps", "nt-compound"):
        return "nt"
    if head in ("tbesg",):
        return "tbesg"
    if head in ("nt-tolerant", "tbesg-tolerant"):
        return "tolerant"
    return head or "other"


# ---------------------------------------------------------------------------
# The lattice

@dataclass
class Option:
    """One distinct reading of a token: the candidates offering it merged."""

    candidate: Candidate | None       # the representative (most weight); None for UNK
    reading: tuple
    attrs: Attrs
    lemma: str
    weight: float
    sources: tuple[str, ...]
    taggable: bool
    rid: int = 0
    unary: tuple[int, ...] = ()


@dataclass
class Slot:
    """A token's place in the lattice."""

    item: TokenCandidates | None
    options: list[Option]
    initial: bool
    choosable: bool                  # the token is to be tagged (some option is a reading)
    method: str = ""
    unary_scores: list[float] = field(default_factory=list)
    sentence: "Sentence | None" = None
    counts: dict | None = None        # the text's certain-lexeme counts (`certain_lexemes`)
    nouns: list | None = None         # the verse's certain nouns: (index, gender, number)


_SUBSCRIPTED = re.compile("[ᾳᾴᾲᾷῃῄῂῇῳῴῲῷ]$")
# The confidence of a tolerant match whose case the printed form contradicts.
MISMATCH_CONFIDENCE = 0.02


def tolerant_fits(key: str, attrs: Attrs) -> bool:
    """Whether a tolerant match (same letters, other marks) can be the
    printed form's reading: an iota subscript on the last vowel is a dative
    (or a verb's ending), so a nominal match in another case cannot be it
    (ἐν κληρονομίᾳ is not κληρονομία, N-NSF)."""
    return not (attrs.case and attrs.case != "D" and _SUBSCRIPTED.search(key))


def options_for(item: TokenCandidates, name_genders: dict | None = None) -> tuple[list[Option], str]:
    """The options of a token and the method a choice among them will have:
    "unique", "model", "tolerant", or "tolerant-mismatch" when no tolerant
    match fits the printed form (`tolerant_fits`); none for tiers d and e."""
    if item.tier in ("d", "e") or not item.candidates:
        return [], ""
    usable = [c for c in item.candidates if c.taggable]
    method = "model"
    if not usable:
        usable = list(item.candidates)
        method = "tolerant"
    merged: dict[tuple, list[Candidate]] = {}
    for candidate in usable:
        merged.setdefault(candidate.reading + (candidate.kind,), []).append(candidate)
    options = []
    for key, group in merged.items():
        best = max(group, key=lambda c: (c.weight, c.taggable, -_SOURCE_ORDER.get(family(c.source), 9), c.source))
        attrs = parse_morph(best.morph)
        if best.morph is None and name_genders and best.strongs in name_genders:
            # A name tagged without its case still has the gender TBESG
            # gives it, for the article's agreement (τοῦ Ἀβραάμ, τῆς Σάρρας).
            attrs = attrs._replace(gen=name_genders[best.strongs])
        options.append(Option(
            candidate=best, reading=key, attrs=attrs, lemma=lemma_key(best.lemma),
            weight=sum(c.weight for c in group),
            sources=tuple(sorted({c.source for c in group})),
            taggable=any(c.taggable for c in group),
        ))
    options.sort(key=lambda o: (-o.weight, o.reading[0], o.reading[1], o.reading[2] or "", o.reading[3]))
    if method == "tolerant":
        fitting = [o for o in options if tolerant_fits(item.token.key, o.attrs)]
        if fitting:
            options = fitting
        else:
            method = "tolerant-mismatch"
    if len(options) == 1 and method == "model":
        method = "unique"
    return options, method


_SOURCE_ORDER = {"nt": 0, "tbesg": 1, "analyser": 2, "tolerant": 3}


def _share_bucket(share: float) -> str:
    if share <= 0:
        return "0"
    for edge, name in ((0.02, "a"), (0.1, "b"), (0.3, "c"), (0.5, "d"), (0.7, "e"), (0.9, "f"), (0.98, "g")):
        if share < edge:
            return name
    return "h"


def _count_bucket(weight: float) -> str:
    if weight <= 0:
        return "0"
    return str(min(int(math.log2(weight + 1)), 12))


def _signature(attrs: Attrs) -> str:
    """The parts of a parse that context bears on most."""
    if attrs.case:
        return attrs.case + attrs.num
    if attrs.cls == "VF":
        return "VF" + attrs.mood + attrs.person + attrs.num
    return attrs.cls


class Vocabulary:
    """Feature strings to ids; frozen, unknown strings are dropped."""

    def __init__(self) -> None:
        self.ids: dict[str, int] = {}
        self.frozen = False

    def __len__(self) -> int:
        return len(self.ids)

    def get(self, name: str) -> int | None:
        found = self.ids.get(name)
        if found is None and not self.frozen:
            found = self.ids[name] = len(self.ids)
        return found

    def map(self, names: Iterable[str]) -> tuple[int, ...]:
        ids = self.ids
        if self.frozen:
            return tuple(i for i in (ids.get(n) for n in names) if i is not None)
        out = []
        for name in names:
            found = ids.get(name)
            if found is None:
                found = ids[name] = len(ids)
            out.append(found)
        return tuple(out)


class Readings:
    """Reading ids: the pairwise and context features depend on a candidate
    only through its reading, so they are memoised by id."""

    def __init__(self) -> None:
        self.ids: dict[tuple, int] = {("UNK",): 0, ("UNKNAME",): 1, ("BOS",): 2}
        self.attrs: list[Attrs] = [NONE_ATTRS, NONE_ATTRS._replace(pos="UNKNAME", cls="UNKNAME"),
                                   NONE_ATTRS._replace(pos="BOS", cls="BOS")]
        self.lemmas: list[str] = ["", "", ""]
        self.morphs: list[str] = ["UNK", "UNKNAME", "BOS"]
        self.strongs: list[str] = ["", "", ""]

    def get(self, option: Option) -> int:
        key = option.reading
        found = self.ids.get(key)
        if found is None:
            found = self.ids[key] = len(self.attrs)
            self.attrs.append(option.attrs)
            self.lemmas.append(option.lemma)
            self.morphs.append(option.reading[2] or "NONE")
            self.strongs.append(option.reading[0])
        return found


def unary_features(slots: list[Slot], index: int, option: Option) -> list[str]:
    """The observed-context features of one option. See the module docstring."""
    slot = slots[index]
    token = slot.item.token
    attrs = option.attrs
    morph = option.reading[2] or "NONE"
    pos = attrs.pos
    sig = _signature(attrs)
    key = token.key
    total = sum(o.weight for o in slot.options) or 0.0
    share = option.weight / total if total else 0.0
    rank = 1 + sum(1 for o in slot.options if o.weight > option.weight)
    feats = [
        "M:" + morph,
        "LM:" + option.lemma + "|" + morph,
        "LP:" + option.lemma + "|" + pos,
        "KM:" + key + "|" + morph,
        "KS:" + key + "|" + option.reading[0],
        "SH:" + _share_bucket(share),
        "SHP:" + _share_bucket(share) + "|" + pos,
        "RK:" + str(min(rank, 4)) + "|" + str(min(len(slot.options), 4)),
        "LC:" + _count_bucket(option.weight) + "|" + pos,
        "KC:" + option.reading[3] + "|" + token.case + "|" + token.position,
        "PP:" + token.position + "|" + sig,
    ]
    if option.weight <= 0:
        feats.append("W0:" + pos + "|" + sig)
    for source in option.sources:
        fam = family(source)
        feats.append("SRC:" + fam + "|" + pos)
        if "+paradigm" in source:
            feats.append("PAR:" + sig)
        if fam not in ("nt", "tbesg", "tolerant"):
            feats.append("AW:" + fam + "|" + _count_bucket(option.weight * 8))
    if not option.taggable:
        feats.append("TOL:" + pos)
    if slot.counts is not None and len({o.reading[0] for o in slot.options}) > 1:
        # Which word: how often each is certain elsewhere in the same text.
        counts = [slot.counts.get(o.reading[0], 0) for o in slot.options]
        mine = slot.counts.get(option.reading[0], 0)
        feats.append("TC:" + str(min(int(math.log2(mine + 1)), 12)))
        feats.append("TCR:" + str(1 + sum(1 for c in counts if c > mine)) + "|" + pos)
        total = sum(counts)
        feats.append("TCS:" + _share_bucket(mine / total if total else 0.0))
    if option.candidate is not None and option.candidate.notes and family(option.candidate.source) not in ("nt", "tbesg", "tolerant"):
        # An analyser's notes name the rule it used; let them carry weight.
        feats.append("NOTE:" + option.candidate.notes.split(";")[0][:40])
    letters = bare(key)
    if attrs.case or attrs.cls in ("VF", "VN", "VP"):
        detail = attrs.cng if attrs.case else attrs.tvm
        feats.append("E2:" + letters[-2:] + "|" + detail)
        feats.append("E3:" + letters[-3:] + "|" + detail)
        if attrs.cls == "VP":
            feats.append("E3T:" + letters[-3:] + "|" + attrs.tvm)
    if token.elided:
        feats.append("EL:" + pos)
    if index == 0:
        feats.append("BOS:" + sig)
    prev = slots[index - 1].item.token if index > 0 else None
    if prev is not None:
        feats.append("PK:" + prev.key + "|" + sig)
    if index + 1 < len(slots):
        following = slots[index + 1]
        nxt = following.item.token
        feats.append("NK:" + nxt.key + "|" + sig)
        feats.append("NP:" + nxt.position + "|" + sig)
        if pos in ("T", "A", "D", "P", "R") and attrs.case:
            # The next word's ending, for a noun the tables do not know.
            ending = bare(nxt.key)
            feats.append("NE2:" + ending[-2:] + "|" + attrs.cng)
            feats.append("NE3:" + ending[-3:] + "|" + attrs.cng)
            feats.append("NE4:" + ending[-4:] + "|" + attrs.cng)
        for distance in (1, 2, 3):
            if index + distance >= len(slots):
                break
            other = slots[index + distance]
            if not attrs.case:
                break
            cased = [o.attrs for o in other.options if o.attrs.case]
            if not other.options:
                feats.append(f"R{distance}:{pos}>none")
                break
            if not cased:
                poses = "/".join(sorted({o.attrs.cls for o in other.options}))
                feats.append(f"R{distance}:{pos}>{poses}")
                break
            agree = max((_agree(attrs, b) for b in cased), key=lambda f: (f.count("1"), f))
            heads = "/".join(sorted({b.pos for b in cased}))
            feats.append(f"R{distance}:{pos}>{heads}:{agree}")
            if distance < 3 and all(b.pos in ("T", "A") for b in cased) and agree[0] != "0":
                # An article or adjective between (τοῦ ἀγαθοῦ λόγου): look on
                # to the noun.
                continue
            break
    else:
        feats.append("EOS:" + sig)
    if pos in ("P", "R", "D") and not attrs.person and attrs.gen and slot.nouns is not None:
        # The antecedent: the nearest noun before it of the same number.
        # Its gender against this reading's; with none, a default the
        # weights learn (most such pronouns are masculine).
        before = [n for n in slot.nouns if n[0] < index and n[2] == attrs.num]
        if before:
            position, gender, _ = before[-1]
            distance = index - position
            feats.append("ANT:" + pos + ("1" if gender == attrs.gen else "0") + "|" + attrs.gen
                         + str(min(distance, 6) // 3))
        else:
            feats.append("ANT:" + pos + "-|" + attrs.gen)
    if attrs.case in ("N", "A", "V") and slot.sentence is not None:
        feats += _sentence_features(slot.sentence, index, attrs)
    if attrs.cls == "VF" and slot.sentence is not None:
        feats += _verb_context_features(slots, slot.sentence, index, attrs)
    return feats


@dataclass
class Sentence:
    """What a sentence's other tokens certainly are: its finite verbs (every
    option a finite verb of one person and number) and its tokens that are
    certainly nominative or accusative."""

    verbs: list[tuple[int, str, str, str]]   # (index, person+number, lemma, voice+mood)
    nominatives: list[int]
    accusatives: list[int]
    verbals: list[tuple[int, str, str, str]]  # any verb form: (index, class, voice, lemma)
    clauses: list[int] = field(default_factory=list)   # the first token of each clause
    particles: list[tuple[int, str]] = field(default_factory=list)  # certain conjunctions and particles: (index, lemma)
    moods: list[tuple[int, str]] = field(default_factory=list)      # certain finite verbs: (index, person+mood)
    relatives: list[int] = field(default_factory=list)              # certain relative pronouns
    # Noun phrases: runs of tokens that can agree in case, number and gender
    # (`_chunks`); per token its chunk's index (-1 for none), and per chunk
    # (start, stop, cases it can have, cases it must have).
    chunk_of: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)
    objects: dict | None = None      # the text's verb lemmas' objects (`verb_objects`)

    def clause_of(self, index: int) -> tuple[int, int]:
        """The span of the clause (between punctuation) a token is in."""
        start = max((c for c in self.clauses if c <= index), default=0)
        later = [c for c in self.clauses if c > index]
        return start, (later[0] if later else 1 << 30)


def _sentences(slots: list[Slot], objects: dict | None = None) -> None:
    start = 0
    for index in range(len(slots) + 1):
        if index == len(slots) or (index > start and slots[index].initial):
            verbs, nominatives, accusatives, verbals = [], [], [], []
            clauses, particles, moods, relatives = [], [], [], []
            for position in range(start, index):
                if position == start or slots[position].item.token.position in CLAUSE_BREAKS:
                    clauses.append(position)
                options = [o for o in slots[position].options if o.candidate is not None]
                if not options:
                    continue
                cases = {o.attrs.case for o in options}
                if all(o.attrs.pos == "R" for o in options):
                    relatives.append(position)
                if all(o.attrs.pos in _PARTICLES for o in options) and len({o.lemma for o in options}) == 1:
                    particles.append((position, options[0].lemma))
                if all(o.attrs.cls == "VF" for o in options) and len({o.attrs.person + o.attrs.mood for o in options}) == 1:
                    moods.append((position, options[0].attrs.person + options[0].attrs.mood))
                if all(o.attrs.pos == "V" for o in options) and len({o.lemma for o in options}) == 1:
                    classes = {o.attrs.cls for o in options}
                    voices = {o.attrs.voice for o in options}
                    verbals.append((position, classes.pop() if len(classes) == 1 else "V",
                                    voices.pop() if len(voices) == 1 else "?", options[0].lemma))
                if all(o.attrs.cls == "VF" for o in options):
                    first = options[0].attrs
                    if all(o.attrs.person + o.attrs.num == first.person + first.num for o in options):
                        voices = {o.attrs.voice + o.attrs.mood for o in options}
                        verbs.append((position, first.person + first.num, options[0].lemma,
                                      voices.pop() if len(voices) == 1 else "?"))
                elif "N" in cases and cases <= {"N", "V"}:
                    # Nominative and vocative are one form in the plural
                    # (οἱ υἱοί): a subject all the same, or an address.
                    nominatives.append(position)
                elif cases == {"A"}:
                    accusatives.append(position)
            sentence = Sentence(verbs, nominatives, accusatives, verbals, clauses, particles, moods, relatives)
            sentence.chunk_of, sentence.chunks = _chunks(slots, start, index)
            sentence.objects = objects
            for position in range(start, index):
                slots[position].sentence = sentence
            start = index


_CHUNK_POS = frozenset(("T", "A", "N", "P", "D", "R", "K", "Q", "X", "I", "C", "F", "S", "NAME"))


def _chunks(slots: list[Slot], start: int, stop: int) -> tuple[dict, list]:
    """The sentence's noun phrases, read from the candidates alone: a run of
    nominal tokens (articles, adjectives, nouns, pronouns, participles) each
    of which can agree with the one before it in case, number and gender
    (ὁ λόγος ὁ ἀγαθός; τὸ πνεῦμα τὸ ἅγιον). A genitive between (τὸ πνεῦμα
    τοῦ θεοῦ) ends the run. Returns {position: chunk index} and, per chunk,
    (start, stop, the cases some token can have, the cases a token must have)."""
    chunk_of: dict[int, int] = {}
    chunks: list[list] = []
    previous: list[Attrs] = []
    governed = False
    for position in range(start, stop):
        options = [o.attrs for o in slots[position].options if o.candidate is not None]
        cased = [a for a in options if a.case and (a.pos in _CHUNK_POS or a.cls == "VP")]
        if not cased:
            previous = []
            continue
        joins = previous and any(
            a.case == b.case and a.num == b.num and (not a.gen or not b.gen or a.gen == b.gen)
            for a in previous for b in cased)
        if not joins:
            before = [o.attrs.pos for o in slots[position - 1].options] if position > start else []
            # A preposition's object is no verb's subject or object.
            governed = bool(before) and all(pos == "PREP" for pos in before)
            if not governed:
                chunks.append([position, position + 1, set(), None])
        previous = cased
        if governed:
            continue
        chunk = chunks[-1]
        chunk[1] = position + 1
        cases = {a.case for a in cased}
        chunk[2] |= cases
        if len(cases) == 1:
            chunk[3] = (chunk[3] or set()) | cases
        chunk_of[position] = len(chunks) - 1
    return chunk_of, [(a, b, frozenset(c), frozenset(d or ())) for a, b, c, d in chunks]


def _region(sentence: Sentence, verb: int) -> tuple[int, int]:
    """The stretch a finite verb's subject and object are looked for in:
    between the finite verbs before and after it."""
    positions = [v[0] for v in sentence.verbs]
    low = max((p for p in positions if p < verb), default=-1)
    high = min((p for p in positions if p > verb), default=1 << 30)
    return low, high


def _arguments(sentence: Sentence, index: int, verb: int) -> str:
    """Whether some other noun phrase near the verb must be (c), can be (p)
    or cannot be (0) its nominative, and the same for an accusative."""
    low, high = _region(sentence, verb)
    mine = sentence.chunk_of.get(index, -1)
    nom = acc = "0"
    for number, (a, b, can, must) in enumerate(sentence.chunks):
        if number == mine or b <= low or a >= high:
            continue
        if "N" in must or must == frozenset("V") and "N" in can:
            nom = "c"
        elif "N" in can and nom != "c":
            nom = "p"
        if "A" in must:
            acc = "c"
        elif "A" in can and acc != "c":
            acc = "p"
    return nom + acc


def _transitivity(counts: dict | None, lemma: str) -> str:
    if not counts:
        return "?"
    seen, objects = counts.get(lemma, (0, 0))
    if seen < 3:
        return "u"
    share = objects / seen
    return "0" if share < 0.1 else "1" if share < 0.25 else "2" if share < 0.5 else "3"


def _sentence_features(sentence: Sentence, index: int, attrs: Attrs) -> list[str]:
    case = attrs.case
    feats = []
    if sentence.verbs and case in ("N", "A") and sentence.chunks:
        # The nearest finite verb's need of a subject and an object: the
        # other noun phrases about it that must or can be nominative or
        # accusative, its voice, and how often its lemma takes an object in
        # this text (`verb_objects`).
        position, pn, lemma, voice = min(sentence.verbs, key=lambda v: (abs(v[0] - index), v[0]))
        args = _arguments(sentence, index, position)
        side = "L" if position < index else "R"
        feats.append("ARG:" + case + "|" + pn + "|" + args + "|" + attrs.gen)
        feats.append("ARGV:" + case + side + "|" + voice[:1] + "|" + args)
        feats.append("ARGT:" + case + side + "|" + voice[:1] + _transitivity(sentence.objects, lemma) + "|" + args[1])
        if attrs.gen == "N" and attrs.num == "P":
            feats.append("ARGNP:" + case + "|" + pn + "|" + args[0])
    if sentence.verbs:
        position, pn, lemma, voice = min(sentence.verbs, key=lambda v: (abs(v[0] - index), v[0]))
        side = "L" if position < index else "R"
        feats.append("SV:" + case + side + pn + "|" + (attrs.person or "3") + attrs.num + attrs.gen)
        feats.append("SVL:" + lemma + "|" + case + side)
        feats.append("SVD:" + case + side + str(min(abs(position - index), 5)))
    else:
        feats.append("SV:" + case + "-")
    others = [v for v in sentence.verbals if v[0] != index]
    if others:
        position, cls, voice, lemma = min(others, key=lambda v: (abs(v[0] - index), v[0]))
        side = "L" if position < index else "R"
        feats.append("SVK:" + case + side + cls + voice + "|" + attrs.num + attrs.gen)
        feats.append("SVKL:" + lemma + "|" + case + side + cls)
    nom = any(p != index for p in sentence.nominatives)
    acc = any(p != index for p in sentence.accusatives)
    feats.append("SN:" + case + "|" + ("n" if nom else "-") + ("a" if acc else "-") + ("v" if sentence.verbs else "-"))
    # The same within the clause: a finite verb there wants one subject; a
    # first or second person verb has it already; another certain
    # nominative fills it.
    start, stop = sentence.clause_of(index)
    verbs = [v for v in sentence.verbs if start <= v[0] < stop and v[0] != index]
    cnom = any(start <= p < stop and p != index for p in sentence.nominatives)
    cacc = any(start <= p < stop and p != index for p in sentence.accusatives)
    if verbs:
        position, pn, lemma, voice = min(verbs, key=lambda v: (abs(v[0] - index), v[0]))
        feats.append("CV:" + case + ("L" if position < index else "R") + pn + voice[-1:] + "|" + attrs.num)
        feats.append("CVL:" + lemma + "|" + case)
    feats.append("CN:" + case + "|" + ("n" if cnom else "-") + ("a" if cacc else "-") + ("v" if verbs else "-")
                 + ("0" if index == start else ""))
    # The nearest finite verb on each side, and what stands between: a
    # relative pronoun or subordinating conjunction (the verb is another
    # clause's: Τὸ θηρίον ὃ εἶδες ἦν), a coordinator (καί, δέ), or
    # punctuation.
    for side, verbs_on_side in (("L", [v for v in sentence.verbs if v[0] < index]),
                                ("R", [v for v in sentence.verbs if v[0] > index])):
        if not verbs_on_side:
            feats.append("V" + side + ":" + case + "-")
            continue
        position, pn, lemma, voice = verbs_on_side[-1] if side == "L" else verbs_on_side[0]
        low, high = (position, index) if side == "L" else (index, position)
        flags = ""
        if any(low < p < high for p in sentence.relatives):
            flags += "r"
        between = [lemma_ for p, lemma_ in sentence.particles if low < p < high]
        if any(word in _SUBORDINATORS for word in between):
            flags += "s"
        if any(word in _COORDINATORS for word in between):
            flags += "k"
        if any(low < c <= high for c in sentence.clauses):
            flags += "p"
        feats.append("V" + side + ":" + case + pn + "|" + attrs.num + attrs.gen + "|" + flags)
        feats.append("V" + side + "L:" + lemma + "|" + case + "|" + ("x" if flags else ""))
    if case == "V":
        moods = [m for p, m in sentence.moods if p != index]
        feats.append("VOC:" + ("/".join(sorted(set(moods)))[:12] if moods else "-"))
    return feats


# Where a clause starts (`Token.position`). Commas are left out: Brenton
# prints nearly twice as many as the Nestle-Aland text the model learns
# from (12.6% of tokens follow a comma or ano teleia, against 6.9%), so a
# clause cut at commas would mean different things in the two.
CLAUSE_BREAKS = ("initial",)

# Parts of speech of the words that open a clause and set a verb's mood.
_PARTICLES = frozenset(("CONJ", "COND", "PRT", "ADV"))
_SUBORDINATORS = frozenset(normalize(w) for w in (
    "ὅτι", "ἵνα", "ὅπως", "ἐάν", "εἰ", "ὅταν", "ὅτε", "καθώς", "ὡς", "ὥσπερ", "ἕως", "διότι", "ἐπεί",
    "πρίν", "ὥστε", "καθότι", "ἡνίκα", "ὅθεν", "οὗ", "ὅπου"))
_COORDINATORS = frozenset(normalize(w) for w in ("καί", "δέ", "τε", "ἀλλά", "ἤ", "οὐδέ", "μηδέ"))


def _verb_context_features(slots: list[Slot], sentence: Sentence, index: int, attrs: Attrs) -> list[str]:
    """What sets a finite verb's mood and tense: the nearest conjunction or
    particle before it in its sentence (ἵνα, ὅπως, ἐάν, μή, ὅταν with the
    subjunctive; ὅτι, καθώς with the indicative; οὐ μή with the aorist
    subjunctive or future), whether it opens its clause (an imperative
    often does), and the other finite verbs' person and mood."""
    feats = []
    mt = attrs.mood + attrs.tense[-1:]
    before = [(p, lemma) for p, lemma in sentence.particles if p < index and index - p <= 6]
    # Stop at a finite verb between: its clause is another.
    between_verbs = {p for p, _ in sentence.moods if p < index}
    before = [(p, lemma) for p, lemma in before if not any(p < v < index for v in between_verbs)]
    if before:
        position, lemma = before[-1]
        feats.append("SUB:" + lemma + "|" + mt)
        feats.append("SUBM:" + lemma + "|" + attrs.mood + attrs.person)
        if len(before) > 1:
            feats.append("SUB2:" + before[-2][1] + "+" + lemma + "|" + mt)
    else:
        feats.append("SUB:-|" + mt)
    start, _ = sentence.clause_of(index)
    feats.append("CL0:" + ("1" if index == start else "0") + "|" + attrs.mood + attrs.person)
    others = sorted({m for p, m in sentence.moods if p != index})
    if others:
        feats.append("OM:" + "/".join(others)[:12] + "|" + attrs.person + attrs.mood)
    return feats


def pair_features(readings: Readings, prev: int, cur: int) -> list[str]:
    """Features of two adjacent readings. See the module docstring."""
    a = readings.attrs[prev]
    b = readings.attrs[cur]
    am, bm = readings.morphs[prev], readings.morphs[cur]
    feats = ["B:" + a.cls + "|" + b.cls, "BM:" + am + "|" + bm]
    if a.case and b.case:
        feats.append("AG:" + a.pos + ">" + b.pos + ":" + _agree(a, b))
    if a.pos == "PREP":
        feats.append("GOV:" + readings.lemmas[prev] + "|" + (b.case or b.cls))
    elif a.cls in ("VF", "VN", "VP"):
        feats.append("VG:" + readings.lemmas[prev] + "|" + (b.case or b.cls))
    if a.pos not in _OPEN_CLASS and a.pos not in ("UNK", "UNKNAME", "BOS"):
        feats.append("PL:" + readings.lemmas[prev] + "|" + bm)
    if b.pos not in _OPEN_CLASS and b.pos not in ("UNK", "UNKNAME"):
        feats.append("CL:" + am + "|" + readings.lemmas[cur])
    if b.pos == "NAME" and a.pos == "T":
        feats.append("TNG:" + a.gen + "|" + (b.gen or "-"))
    if b.pos == "NAME" or b.suffix in ("P", "L", "PG", "LG"):
        # Which name: a person, a people, a place (γῆ Ἰούδα is the land).
        feats.append("PLS:" + readings.lemmas[prev] + "|" + readings.strongs[cur])
    if a.case == "N" and b.cls == "VF":
        feats.append("NV1:" + (a.person or "3") + a.num + "|" + b.person + b.num)
    if a.cls == "VF" and b.case in ("N", "V"):
        feats.append("VN1:" + a.person + a.num + "|" + b.case + (b.person or "3") + b.num)
    return feats


# The summary of a path so far (see the module docstring), as a tuple:
#   0 cls1     class of the last token
#   1 cls2     class of the token before it
#   2 article  an open article's case, number and gender ("" if none)
#   3 prep     an open preposition's lemma
#   4 verb     the sentence's last finite verb: person, number, mood
#   5 nom      the sentence's last nominative: person and number
#   6 vlemma   the sentence's last finite verb's lemma
#   7 head     the verse's last noun: gender and number
#   8 outer    the last finite verb before this sentence: person, number, mood
#   9 article2 the open article outside it (ὁ in ὁ τοῦ θεοῦ λόγος)
#  10 last     the sentence's last nominal word: case and number
#  11 coord    `last` when a coordinator (καί, ἤ, οὐδέ ...) followed it, until
#              the next nominal word: what that word is coordinated with
INITIAL_STATE = ("BOS", "BOS", "", "", "", "", "", "", "", "", "", "")


def state_features(readings: Readings, state: tuple, cur: int) -> list[str]:
    b = readings.attrs[cur]
    cls1, cls2, article, prep, verb, nom, vlemma, head, outer, article2, last, coord = state
    feats = ["TRI:" + cls2 + "|" + cls1 + "|" + b.cls]
    if coord and b.case and b.pos in _COORD_POS:
        # X καὶ Y: the second member of a pair in the first's case.
        feats.append("CO:" + b.pos + "|" + coord[:1] + b.case)
        feats.append("CON:" + b.pos + "|" + ("1" if coord[:1] == b.case else "0") + coord[1:] + b.num)
    if article2 and b.case:
        feats.append("OA2:" + b.pos + ":" + _agree(_art_attrs(article2), b))
    if article:
        if b.pos == "NAME":
            feats.append("OAN:" + article[2:] + "|" + (b.gen or "-"))
        if b.case:
            feats.append("OA:" + b.pos + ":" + _agree(_art_attrs(article), b))
        elif b.cls == "VN":
            feats.append("OAV:" + article)
        else:
            feats.append("OAX:" + b.cls)
    if prep:
        if b.case:
            feats.append("GP:" + prep + "|" + b.case)
            feats.append("GPP:" + prep + "|" + b.pos + b.case)
        elif b.cls == "VN":
            feats.append("GPV:" + prep + "|" + (article[:1] if article else "-"))
    if b.case in ("N", "A") and (b.pos in ("N", "A", "P", "D", "T", "R", "F", "I", "X", "S") or b.cls == "VP"):
        feats.append("NS:" + ("n" if nom else "-") + ("v" if verb else "-") + "|" + b.pos + b.case)
        if b.case == "N":
            feats.append("VS:" + (verb[:2] or "-") + "|" + (b.person or "3") + b.num)
    if b.case and vlemma:
        feats.append("VL:" + vlemma + "|" + b.case)
    if b.case == "V":
        feats.append("VV:" + (verb or "-"))
    if b.person and b.pos in ("P", "F", "S"):
        feats.append("PER:" + (verb[:2] or outer[:2] or "-") + "|" + b.person + b.num)
    if b.pos == "R":
        feats.append("REL:" + (head or "-") + "|" + b.gen + b.num)
    if b.cls == "VF":
        feats.append("NV:" + (nom or "-") + "|" + b.person + b.num)
        if verb:
            feats.append("VV2:" + verb + "|" + b.person + b.num + b.mood)
        else:
            feats.append("OV:" + (outer or "-") + "|" + b.person + b.num + b.mood)
    return feats


_ART: dict[str, Attrs] = {}
_COORD_POS = frozenset(("T", "N", "A", "P", "D", "NAME", "R", "F", "S", "X", "I", "K", "Q", "C"))
RULE_PENALTY = -1000.0
_PROPER_PREPOSITIONS = frozenset(normalize(w) for w in (
    "ἐν", "εἰς", "ἐκ", "ἀπό", "διά", "ἐπί", "μετά", "παρά", "περί", "πρό", "πρός", "σύν", "ὑπό", "ἀντί",
    "ἀμφί"))


def _art_attrs(cng: str) -> Attrs:
    found = _ART.get(cng)
    if found is None:
        found = _ART[cng] = Attrs("T", "T", cng[:1], cng[1:2], cng[2:3], "", "", "", "", "", False)
    return found


def next_state(readings: Readings, state: tuple, cur: int) -> tuple:
    b = readings.attrs[cur]
    cls1, _, article, prep, verb, nom, vlemma, head, outer, article2, last, coord = state
    pos = b.pos
    closes = pos in _HEADS or b.cls in ("VF", "VN") or pos in ("CONJ", "COND")
    if pos == "T" and b.case:
        # A second article inside an open one (ὁ τοῦ θεοῦ λόγος) opens a
        # phrase within it; the outer one waits for its own noun.
        if article and article[:1] != b.case:
            article2 = article
        article = b.cng
    elif closes:
        if b.cls in ("VF", "VN") or pos in ("CONJ", "COND"):
            article = article2 = ""
        elif article and b.case and b.case != article[:1]:
            # A word in another case (τοῖς ἐκείνου γράμμασιν) is inside the
            # phrase: the article is still open.
            pass
        else:
            article, article2 = article2, ""
    if pos == "PREP":
        prep = readings.lemmas[cur]
    elif closes:
        prep = ""
    if b.cls == "VF":
        verb = b.person + b.num + b.mood
        vlemma = readings.lemmas[cur]
    if b.case == "N" and pos in ("N", "P", "D", "A", "R", "NAME"):
        nom = (b.person or "3") + b.num
    if pos == "N" and b.num:
        head = b.gen + b.num
    if b.case == "G" and pos in ("P", "F") and last:
        pass    # a possessive (τὰ κτήνη αὐτῶν καὶ ...) leaves its noun the member
    elif b.case and pos in _COORD_POS:
        last, coord = b.case + b.num, ""
    elif pos == "CONJ" and readings.lemmas[cur] in _COORDINATORS and last:
        coord = last
    elif b.cls in ("VF", "VN") or pos == "PREP":
        last = coord = ""
    if pos in ("CONJ", "COND") and readings.lemmas[cur] in _SUBORDINATORS:
        # A subordinate clause: its verb and subject are its own.
        verb, nom, vlemma = "", "", ""
    return (b.cls, cls1, article, prep, verb, nom, vlemma, head, outer, article2, last, coord)


def reset_sentence(state: tuple) -> tuple:
    return (state[0], state[1], "", "", "", "", "", state[7], state[4] or state[8], "", "", "")


# ---------------------------------------------------------------------------
# The model

class Model:
    """Weights over feature ids, with the memoised pairwise and context
    features, the decoder and the margins."""

    def __init__(self, vocabulary: Vocabulary | None = None, weights: list[float] | None = None) -> None:
        self.vocabulary = vocabulary or Vocabulary()
        self.weights: list[float] = weights if weights is not None else []
        self.readings = Readings()
        self._pair: dict[tuple[int, int], tuple[int, ...]] = {}
        self._state: dict[tuple[tuple, int], tuple[tuple[int, ...], tuple]] = {}
        self._pair_score: dict[tuple[int, int], float] = {}
        self._state_score: dict[tuple[tuple, int], float] = {}
        self.frozen = False
        # How often each lexeme is certain in the text being tagged (see
        # `certain_lexemes`); None leaves the feature out.
        self.text_counts: dict | None = None
        # Strong's number -> gender of the names TBESG gives one (`name_genders`).
        self.name_genders: dict | None = None
        # Verb lemma -> (occurrences, with an object) in the text (`verb_objects`).
        self.verb_objects: dict | None = None

    # -- features ---------------------------------------------------------
    def _grow(self) -> None:
        missing = len(self.vocabulary) - len(self.weights)
        if missing > 0:
            self.weights.extend([0.0] * missing)

    def pair_ids(self, prev: int, cur: int) -> tuple[int, ...]:
        key = (prev, cur)
        found = self._pair.get(key)
        if found is None:
            found = self._pair[key] = self.vocabulary.map(pair_features(self.readings, prev, cur))
        return found

    def state_ids(self, state: tuple, cur: int) -> tuple[tuple[int, ...], tuple]:
        key = (state, cur)
        found = self._state.get(key)
        if found is None:
            found = self._state[key] = (self.vocabulary.map(state_features(self.readings, state, cur)),
                                        next_state(self.readings, state, cur))
        return found

    def lattice(self, verse: Sequence[TokenCandidates]) -> list[Slot]:
        """The verse's slots, with unary feature ids."""
        slots = []
        for item in verse:
            options, method = options_for(item, self.name_genders)
            choosable = bool(options)
            if not options:
                name_like = item.tier == "d"
                options = [Option(None, ("UNKNAME",) if name_like else ("UNK",),
                                  NONE_ATTRS._replace(pos="UNKNAME", cls="UNKNAME") if name_like else NONE_ATTRS,
                                  "", 0.0, (), False, rid=1 if name_like else 0)]
            slots.append(Slot(item, options, item.token.position == "initial", choosable, method))
        _sentences(slots, self.verb_objects)
        nouns = []
        for index, slot in enumerate(slots):
            found = {(o.attrs.gen, o.attrs.num) for o in slot.options if o.candidate is not None}
            if (found and all(o.attrs.pos in ("N", "NAME") for o in slot.options)
                    and len(found) == 1 and all(found.copy().pop())):
                nouns.append((index, *found.pop()))
        for slot in slots:
            slot.nouns = nouns
        if self.text_counts is not None:
            for slot in slots:
                slot.counts = self.text_counts
        for index, slot in enumerate(slots):
            for option in slot.options:
                if option.candidate is not None:
                    option.rid = self.readings.get(option)
                    option.unary = self.vocabulary.map(unary_features(slots, index, option))
        self._grow()
        return slots

    # -- scoring ----------------------------------------------------------
    def freeze(self) -> None:
        """Fix the weights: scores are memoised from here on."""
        self.frozen = True
        self.vocabulary.frozen = True
        self._pair_score.clear()
        self._state_score.clear()

    def _unary_scores(self, slots: list[Slot]) -> None:
        w = self.weights
        for slot in slots:
            slot.unary_scores = [sum(w[i] for i in option.unary) for option in slot.options]

    def _pair_score_of(self, prev: int, cur: int) -> float:
        if self.frozen:
            key = (prev, cur)
            found = self._pair_score.get(key)
            if found is None:
                w = self.weights
                found = self._pair_score[key] = (sum(w[i] for i in self.pair_ids(prev, cur))
                                                 + self._rule_penalty(prev, cur))
            return found
        ids = self.pair_ids(prev, cur)
        self._grow_lazily()
        w = self.weights
        return sum(w[i] for i in ids) + self._rule_penalty(prev, cur)

    def _rule_penalty(self, prev: int, cur: int) -> float:
        """Rules of grammar, not learnt: the word after a proper preposition
        (ἐν, εἰς, ἐκ, ὑπό ...) is in the case it governs, never nominative or
        vocative. Not for the adverbs TAGNT also tags PREP (ὅπου, ὡσεί,
        ἔξω, ἐγγύς, ἕως), nor distributive ἀνά (ἀνὰ εἷς), κατά (εἷς κατὰ
        εἷς) or ὑπέρ (ὑπὲρ ἐγώ), which the NT shows with a nominative."""
        a = self.readings.attrs[prev]
        if (a.pos == "PREP" and self.readings.lemmas[prev] in _PROPER_PREPOSITIONS
                and self.readings.attrs[cur].case in ("N", "V")):
            return RULE_PENALTY
        return 0.0

    def _state_of(self, state: tuple, cur: int) -> tuple[float, tuple]:
        if self.frozen:
            key = (state, cur)
            found = self._state_score.get(key)
            if found is None:
                ids, after = self.state_ids(state, cur)
                w = self.weights
                found = self._state_score[key] = (sum(w[i] for i in ids), after)
            return found
        ids, after = self.state_ids(state, cur)
        self._grow_lazily()
        w = self.weights
        return sum(w[i] for i in ids), after

    def _grow_lazily(self) -> None:
        if len(self.weights) < len(self.vocabulary):
            self._grow()

    # -- decoding ---------------------------------------------------------
    def _run(self, slots: list[Slot], start: int, stop: int, init: list[tuple], allowed, beam: int):
        """Beam search over slots[start:stop] from the hypotheses `init`, each
        (score, option index at start-1 or -1, state, parent). Returns the
        beams, one list per position, each sorted best first."""
        beams = []
        previous = init
        for index in range(start, stop):
            slot = slots[index]
            options = slot.options
            prev_slot = slots[index - 1] if index > 0 else None
            candidates: dict = {}
            choices = allowed(index) if allowed else range(len(options))
            unary = slot.unary_scores
            for parent, (score, prev_option, state, _) in enumerate(previous):
                prev_rid = prev_slot.options[prev_option].rid if prev_slot is not None and prev_option >= 0 else 2
                if slot.initial:
                    state = reset_sentence(state)
                for choice in choices:
                    rid = options[choice].rid
                    context, after = self._state_of(state, rid)
                    total = score + unary[choice] + self._pair_score_of(prev_rid, rid) + context
                    key = (choice, after)
                    held = candidates.get(key)
                    if held is None or total > held[0]:
                        candidates[key] = (total, choice, after, parent)
            ranked = sorted(candidates.values(), key=lambda h: -h[0])[:beam]
            beams.append(ranked)
            previous = ranked
        return beams

    @staticmethod
    def _backtrack(beams: list, which: int = 0) -> list[int]:
        path = []
        position = which
        for beam in reversed(beams):
            hypothesis = beam[position]
            path.append(hypothesis[1])
            position = hypothesis[3]
        return path[::-1]

    def decode(self, slots: list[Slot], allowed=None, beam: int = BEAM) -> list[int]:
        if not slots:
            return []
        self._unary_scores(slots)
        beams = self._run(slots, 0, len(slots), [(0.0, -1, INITIAL_STATE, -1)], allowed, beam)
        return self._backtrack(beams)

    def path_contributions(self, slots: list[Slot], path: Sequence[int]) -> tuple[list[float], list[tuple]]:
        """Each position's score and the state after it, along a path."""
        scores, states = [], []
        state = INITIAL_STATE
        prev_rid = 2
        for index, choice in enumerate(path):
            slot = slots[index]
            if slot.initial:
                state = reset_sentence(state)
            rid = slot.options[choice].rid
            context, after = self._state_of(state, rid)
            scores.append(slot.unary_scores[choice] + self._pair_score_of(prev_rid, rid) + context)
            states.append(after)
            state, prev_rid = after, rid
        return scores, states

    def margins(self, slots: list[Slot], path: list[int], beam: int = BEAM, window: int = WINDOW
                ) -> list[tuple[float, int | None]]:
        """For each position, (margin, runner-up option): how much the best
        path choosing otherwise there scores below `path`, its neighbours
        within `window` free to change with it. Positions with one option
        have margin inf and no runner-up."""
        n = len(slots)
        scores, states = self.path_contributions(slots, path)
        suffix = [0.0] * (n + 1)
        for index in range(n - 1, -1, -1):
            suffix[index] = suffix[index + 1] + scores[index]
        out: list[tuple[float, int | None]] = []
        for index in range(n):
            options = slots[index].options
            if len(options) < 2:
                out.append((math.inf, None))
                continue
            start = max(0, index - window)
            stop = min(n, index + window + 1)
            init_state = states[start - 1] if start > 0 else INITIAL_STATE
            init = [(0.0, path[start - 1] if start > 0 else -1, init_state, -1)]
            chosen = path[index]
            others = [o for o in range(len(options)) if o != chosen]

            def allowed(position, index=index, others=others):
                return others if position == index else range(len(slots[position].options))

            beams = self._run(slots, start, stop, init, allowed, beam)
            best_alt = -math.inf
            best_option = None
            last = beams[-1]
            for rank, (score, choice, state, _) in enumerate(last):
                total = score
                position = stop
                same_last = choice == path[stop - 1]
                prev_rid = slots[stop - 1].options[choice].rid
                # Rescore the rest of the path until it is back in the path's
                # own state on the path's own option; from there it is the path.
                while position < n and not (same_last and state == states[position - 1]):
                    slot = slots[position]
                    if slot.initial:
                        state = reset_sentence(state)
                    rid = slot.options[path[position]].rid
                    context, state = self._state_of(state, rid)
                    total += slot.unary_scores[path[position]] + self._pair_score_of(prev_rid, rid) + context
                    prev_rid = rid
                    position += 1
                    same_last = True
                if position < n:
                    total += suffix[position]
                if total > best_alt:
                    best_alt = total
                    best_option = self._backtrack(beams, rank)[index - start]
            out.append((suffix[start] - best_alt, best_option))
        return out

    # -- features of a path (for training) ---------------------------------
    def path_features(self, slots: list[Slot], path: Sequence[int]) -> Counter:
        counts: Counter = Counter()
        state = INITIAL_STATE
        prev_rid = 2
        for index, choice in enumerate(path):
            slot = slots[index]
            if slot.initial:
                state = reset_sentence(state)
            option = slot.options[choice]
            counts.update(option.unary)
            counts.update(self.pair_ids(prev_rid, option.rid))
            ids, state = self.state_ids(state, option.rid)
            counts.update(ids)
            prev_rid = option.rid
        return counts

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        names = sorted(self.vocabulary.ids, key=self.vocabulary.ids.get)
        return {name: self.weights[self.vocabulary.ids[name]] for name in sorted(names)
                if self.vocabulary.ids[name] < len(self.weights) and self.weights[self.vocabulary.ids[name]] != 0.0}

    @classmethod
    def from_json(cls, weights: dict[str, float]) -> "Model":
        vocabulary = Vocabulary()
        values = []
        for name in sorted(weights):
            vocabulary.ids[name] = len(values)
            values.append(float(weights[name]))
        model = cls(vocabulary, values)
        model.freeze()
        return model


def name_genders(lexicon) -> dict[str, str]:
    """Strong's number -> M/F/N for the names TBESG gives a gender
    (`N:N-M-P` a man, `N:N-F-P` a woman)."""
    found: dict[str, str] = {}
    for entry in lexicon.entries:
        if entry.is_name and entry.gender and entry.strongs not in found:
            found[entry.strongs] = entry.gender
    return found


def certain_lexemes(items: Iterable[TokenCandidates]) -> dict[str, int]:
    """Strong's number -> tokens of the text whose candidates are all that
    word (tiers a and b): an unsupervised count from the text itself, which
    says which of two words a form more likely is in this text."""
    counts: Counter = Counter()
    for item in items:
        if item.tier in ("a", "b"):
            numbers = {c.strongs for c in item.candidates if c.taggable}
            if len(numbers) == 1:
                counts[numbers.pop()] += 1
    return dict(counts)


OBJECT_WINDOW = 6


def verb_objects(items: Iterable[TokenCandidates]) -> dict[str, tuple[int, int]]:
    """Verb lemma -> (its certain finite occurrences, those followed within
    `OBJECT_WINDOW` tokens of its sentence, before another finite verb, by a
    word that is certainly accusative): an unsupervised count from the text
    itself of how often a verb takes an object, for verbs the NT lacks."""
    seen: Counter = Counter()
    objects: Counter = Counter()
    pending: list[list] = []
    for item in items:
        if item.token.position == "initial":
            pending = []
        attrs = [parse_morph(c.morph) for c in item.candidates if c.taggable]
        if attrs and all(a.cls == "VF" for a in attrs):
            lemmas = {lemma_key(c.lemma) for c in item.candidates if c.taggable}
            pending = []
            if len(lemmas) == 1:
                lemma = lemmas.pop()
                seen[lemma] += 1
                pending.append([lemma, OBJECT_WINDOW])
            continue
        if attrs and pending and all(a.case == "A" and a.pos in _CHUNK_POS for a in attrs):
            for lemma, _ in pending:
                objects[lemma] += 1
            pending = []
            continue
        for entry in pending:
            entry[1] -= 1
        pending = [entry for entry in pending if entry[1] > 0]
    return {lemma: (seen[lemma], objects[lemma]) for lemma in seen}


# ---------------------------------------------------------------------------
# Training

@dataclass
class Example:
    slots: list[Slot]
    gold: list[int | None]        # gold option per position, None if not among the options


def gold_index(slot: Slot, reading) -> int | None:
    """The option whose reading is TAGNT's (see `ntcorpus.same_reading`)."""
    if reading is None or not slot.choosable:
        return None
    from lxx_tagging.ntcorpus import same_reading
    exact = None
    loose = None
    for index, option in enumerate(slot.options):
        if option.candidate is None:
            continue
        if same_reading(option.reading[:3], reading):
            if option.reading[2] == reading.morph:
                exact = index
                break
            if loose is None:
                loose = index
    return exact if exact is not None else loose


def examples_from(model: Model, verses: Sequence, sets: Sequence[Sequence[TokenCandidates]]) -> list[Example]:
    """Training examples: each NT verse's lattice and the gold option per token."""
    examples = []
    for verse, items in zip(verses, sets):
        slots = model.lattice(items)
        examples.append(Example(slots, [gold_index(slot, reading) for slot, reading in zip(slots, verse.gold)]))
    return examples


def train(examples: list[Example], model: Model | None = None, epochs: int = EPOCHS, beam: int = BEAM,
          seed: int = SEED, log: Callable[[str], None] | None = None) -> Model:
    """The averaged structured perceptron. Deterministic for the same examples."""
    model = model or Model()
    model._grow()
    weights = model.weights
    totals = [0.0] * len(weights)        # sum over updates of step * delta, for averaging
    step = 1
    order = list(range(len(examples)))
    rng = random.Random(seed)
    for epoch in range(epochs):
        rng.shuffle(order)
        errors = tokens = 0
        started = time.time()
        for number in order:
            example = examples[number]
            slots = example.slots
            if not any(slot.choosable and len(slot.options) > 1 for slot in slots):
                step += 1
                continue
            predicted = model.decode(slots, beam=beam)
            gold = example.gold

            def constrained(position, gold=gold):
                return (gold[position],) if gold[position] is not None else range(len(slots[position].options))

            wrong = any(g is not None and g != p for g, p in zip(gold, predicted))
            if wrong:
                target = model.decode(slots, allowed=constrained, beam=beam)
                delta = model.path_features(slots, target)
                delta.subtract(model.path_features(slots, predicted))
                if len(weights) < len(model.vocabulary):
                    model._grow()
                if len(totals) < len(weights):
                    totals.extend([0.0] * (len(weights) - len(totals)))
                for feature, change in delta.items():
                    if change:
                        weights[feature] += change
                        totals[feature] += step * change
            for slot, g, p in zip(slots, gold, predicted):
                if g is not None and len(slot.options) > 1:
                    tokens += 1
                    errors += g != p
            step += 1
        if log:
            log(f"  epoch {epoch + 1}: {errors:,} of {tokens:,} ambiguous training tokens wrong "
                f"({time.time() - started:.0f}s, {len(model.vocabulary):,} features)")
    if len(totals) < len(weights):
        totals.extend([0.0] * (len(weights) - len(totals)))
    averaged = [w - t / step for w, t in zip(weights, totals)]
    # Rounded so the stored model and the model in memory are the same numbers.
    averaged = [float(f"{value:.6g}") for value in averaged]
    result = Model(model.vocabulary, averaged)
    result.readings = model.readings
    result.text_counts = model.text_counts
    result.name_genders = model.name_genders
    result.verb_objects = model.verb_objects
    result._pair, result._state = model._pair, model._state
    result.freeze()
    return result


# ---------------------------------------------------------------------------
# Calibration

BAND_EDGES = (0.5, 0.8, 0.9, 0.95, 0.99)
MIN_GROUP = 300


@dataclass
class Calibration:
    """Margin -> confidence, per group, measured on held-out NT tokens."""

    groups: dict[str, dict]          # group -> {"edges": [...], "confidence": [...], "n": [...], "correct": [...]}

    @staticmethod
    def group_keys(method: str, tier: str, source_family: str, options: int | None = None) -> list[str]:
        """Most specific first: method, tier, the chosen reading's source
        family and the number of readings it was chosen from (2, 3, 4+)."""
        keys = [f"{method}:{tier}:{source_family}", f"{method}:{tier}", method, "all"]
        if options is not None and options > 1:
            keys.insert(0, f"{method}:{tier}:{source_family}:{min(options, 4)}")
        return keys

    def confidence(self, method: str, tier: str, source_family: str, margin: float,
                   options: int | None = None) -> float:
        for key in self.group_keys(method, tier, source_family, options):
            table = self.groups.get(key)
            if table:
                index = bisect_right(table["edges"], margin) if table["edges"] else 0
                return table["confidence"][index]
        return 0.5

    @classmethod
    def fit(cls, records: Iterable[tuple], bins: int = 10) -> "Calibration":
        """From (method, tier, family, margin, correct[, options]) records."""
        pooled: dict[str, list[tuple[float, bool]]] = {}
        for record in records:
            method, tier, source_family, margin, correct = record[:5]
            options = record[5] if len(record) > 5 else None
            for key in cls.group_keys(method, tier, source_family, options):
                pooled.setdefault(key, []).append((margin, correct))
        groups = {}
        for key in sorted(pooled):
            data = sorted(pooled[key], key=lambda r: r[0])
            if len(data) < MIN_GROUP and key != "all" and key.count(":") > 0:
                continue
            finite = [m for m, _ in data if math.isfinite(m)]
            edges: list[float] = []
            if finite and len(set(finite)) > 1:
                count = min(bins, max(1, len(data) // 150))
                for b in range(1, count):
                    edge = finite[min(len(finite) - 1, b * len(finite) // count)]
                    edge = float(f"{edge:.4g}")
                    if not edges or edge > edges[-1]:
                        edges.append(edge)
            n = [0] * (len(edges) + 1)
            correct = [0] * (len(edges) + 1)
            for margin, ok in data:
                index = bisect_right(edges, margin)
                n[index] += 1
                correct[index] += ok
            # Pool adjacent violators: confidence never falls as the margin grows.
            blocks = [[correct[i], n[i], [i]] for i in range(len(n))]
            merged: list[list] = []
            for block in blocks:
                merged.append(block)
                while len(merged) > 1 and (merged[-2][0] + 1) / (merged[-2][1] + 2) > (merged[-1][0] + 1) / (merged[-1][1] + 2):
                    last = merged.pop()
                    merged[-1][0] += last[0]
                    merged[-1][1] += last[1]
                    merged[-1][2] += last[2]
            confidence = [0.0] * len(n)
            for c, total, members in merged:
                for member in members:
                    confidence[member] = round((c + 1) / (total + 2), 4)
            groups[key] = {"edges": edges, "confidence": confidence, "n": n, "correct": correct}
        return cls(groups)


def band(confidence: float) -> str:
    """A confidence's band, for reports: "<0.5", "0.5-0.8", ..., ">=0.99"."""
    lower = None
    for edge in BAND_EDGES:
        if confidence < edge:
            return f"<{edge:g}" if lower is None else f"{lower:g}-{edge:g}"
        lower = edge
    return f">={BAND_EDGES[-1]:g}"


BANDS = tuple(band(x) for x in (0.0, 0.6, 0.85, 0.92, 0.97, 1.0))


# ---------------------------------------------------------------------------
# The chooser

class ModelChooser:
    """A `Chooser`: the model's best path through each verse, with calibrated
    confidences. See the module docstring for the methods."""

    def __init__(self, model: Model, calibration: Calibration, beam: int = BEAM, window: int = WINDOW,
                 meta: dict | None = None):
        self.model = model
        self.calibration = calibration
        self.beam = beam
        self.window = window
        self.meta = meta or {}
        model.freeze()
        # Runner-ups of the last verse, by token index, for the decisions file.
        self.runner_up: dict[int, Candidate] = {}

    def prepare(self, items: Iterable[TokenCandidates], lexicon=None) -> None:
        """Read the whole text's candidates first: the lexeme counts the
        model uses to choose between words (`certain_lexemes`); and, from
        the lexicon, the names' genders (`name_genders`)."""
        items = list(items)
        self.model.text_counts = certain_lexemes(items)
        self.model.verb_objects = verb_objects(items)
        if lexicon is not None:
            self.model.name_genders = name_genders(lexicon)

    def decide(self, verse: Sequence[TokenCandidates]) -> list[tuple[Choice | None, Candidate | None]]:
        slots = self.model.lattice(verse)
        path = self.model.decode(slots, beam=self.beam)
        margins = self.model.margins(slots, path, beam=self.beam, window=self.window)
        out: list[tuple[Choice | None, Candidate | None]] = []
        for slot, choice, (margin, other) in zip(slots, path, margins):
            if not slot.choosable:
                out.append((None, None))
                continue
            option = slot.options[choice]
            method = slot.method
            tier = slot.item.tier
            fam = family(option.candidate.source)
            count = len(slot.options)
            if method == "tolerant-mismatch":
                confidence = min(MISMATCH_CONFIDENCE, self.calibration.confidence("tolerant", tier, fam, margin, count))
            else:
                confidence = self.calibration.confidence(method, tier, fam, margin, count)
            runner = slot.options[other].candidate if other is not None else None
            shown = margin if math.isfinite(margin) else math.inf
            out.append((Choice(option.candidate, confidence, float(f"{shown:.4g}") if math.isfinite(shown) else shown,
                               method), runner))
        return out

    def __call__(self, verse: Sequence[TokenCandidates]) -> list[Choice | None]:
        decided = self.decide(verse)
        self.runner_up = {item.token.index: runner for item, (_, runner) in zip(verse, decided) if runner}
        return [choice for choice, _ in decided]


# ---------------------------------------------------------------------------
# Training from the sources, cross-fitting, and the cache

def _log_to_stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def code_paths(extra: Iterable[Path] = ()) -> list[Path]:
    """The code whose behaviour the trained model depends on."""
    package = Path(__file__).resolve().parent
    return [package / "chooser.py"] + candidate_code_paths() + [Path(p) for p in extra]


def candidate_code_paths() -> list[Path]:
    """The code that makes the training data's candidate sets (not the model's)."""
    package = Path(__file__).resolve().parent
    names = ["ntcorpus.py", "candidates.py", "text.py", "ntforms.py", "lexicon.py", "pipeline.py", "analyser.py"]
    paths = [package / name for name in names if (package / name).exists()]
    paths += sorted(package.glob("morph_*.py"))
    paths.append(SCRIPTS / "stepbible_to_translation_schema.py")
    return paths


SETS_DIR = CACHE_DIR.parent / "candidate_sets"


def default_jobs() -> int:
    import os
    return max(1, min(9, (os.cpu_count() or 2) - 1))


def cached_candidates(sources: Path, factory, name: str, settings: dict, compute: Callable[[], object],
                      cache_dir: Path | None = SETS_DIR, log: Callable[[str], None] | None = None):
    """Candidate sets for the NT folds, computed once per version of the
    sources and of the code that makes them (pickled under `cache_dir`), so
    work on the model does not rebuild a fold's analyser every run."""
    import pickle
    from lxx_tagging import ntforms
    if cache_dir is None:
        return compute()
    factory_settings = factory.settings() if hasattr(factory, "settings") else {"factory": repr(factory)}
    key = cache_key(ntforms.tagnt_paths(sources) + [sources / "TBESG.txt"], candidate_code_paths(),
                    {"name": name, **settings, **factory_settings})
    path = cache_dir / f"{name}-{key}.pickle"
    if path.exists():
        with open(path, "rb") as handle:
            return pickle.load(handle)
    started = time.time()
    value = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)
    if log:
        log(f"  candidate sets ({name}) made in {time.time() - started:.0f}s; cached as {path.name}")
    return value


_FILE_DIGESTS: dict[tuple[str, bool], bytes] = {}


def _digest(path: Path, snapshot: bool) -> bytes:
    """A file's hash. Code files (`snapshot`) are hashed once per process, so
    a key describes the code this process loaded even if the file is edited
    while it runs."""
    key = (str(Path(path).resolve()), snapshot)
    if snapshot and key in _FILE_DIGESTS:
        return _FILE_DIGESTS[key]
    found = hashlib.sha256(Path(path).read_bytes()).digest()
    if snapshot:
        _FILE_DIGESTS[key] = found
    return found


def cache_key(sources: Sequence[Path], code: Sequence[Path], settings: dict) -> str:
    digest = hashlib.sha256()
    for path, snapshot in [(p, False) for p in sources] + [(p, True) for p in code]:
        digest.update(Path(path).name.encode("utf-8") + b"\0")
        digest.update(_digest(path, snapshot))
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:20]


@dataclass
class FoldResult:
    """Out-of-fold decisions for one fold: per token (verse index, position,
    tier, method, family, margin, chosen option reading, gold reading, baselines)."""

    fold: int
    records: list[dict]


def evaluate(model: Model, chooser: ModelChooser | None, verses, sets, fold: int) -> FoldResult:
    """Choose every held-out token and record what was chosen against TAGNT."""
    from lxx_tagging.ntcorpus import same_reading
    chooser = chooser or ModelChooser(model, Calibration({}))
    records = []
    for verse_index, (verse, items) in enumerate(zip(verses, sets)):
        decided = chooser.decide(items)
        for position, (item, gold, (choice, _)) in enumerate(zip(items, verse.gold, decided)):
            if gold is None:
                continue
            readings = item.readings()
            frequent = readings[0] if readings and item.tier not in ("d", "e") else None
            first = item.candidates[0].reading if item.candidates and item.tier not in ("d", "e") else None
            records.append({
                "fold": fold, "book": verse.book, "verse": verse_index, "position": position,
                "tier": item.tier, "method": choice.method if choice else "",
                "family": family(choice.candidate.source) if choice else "",
                "margin": choice.margin if choice else None,
                "chosen": choice.candidate.reading if choice else None,
                "gold": (gold.strongs, gold.lemma, gold.morph),
                "frequent": frequent, "first": first,
                "n_options": len(options_for(item)[0]),
                "gold_available": any(same_reading(c.reading, gold) for c in item.candidates),
                "unseen": item.candidates == [] or all(family(c.source) != "nt" for c in item.candidates),
            })
    return FoldResult(fold, records)


def correct(chosen, gold) -> bool:
    from lxx_tagging.ntcorpus import same_reading
    from lxx_tagging.ntforms import Reading
    return chosen is not None and same_reading(chosen, Reading(*gold))


def calibration_records(results: Iterable[FoldResult]):
    for result in results:
        for record in result.records:
            if record["chosen"] is None:
                continue
            margin = record["margin"]
            yield (record["method"], record["tier"], record["family"],
                   math.inf if margin is None else margin, correct(record["chosen"], record["gold"]),
                   record["n_options"])


@dataclass
class TrainingData:
    verses: list
    rows: list
    assignment: dict
    sets: dict[int, list]       # fold -> candidate sets of its verses (from the other folds' table)
    lexicon: object = None
    profile: object = None
    counts: dict | None = None
    objects: dict | None = None

    def fold_verses(self, fold: int) -> list:
        return [verse for verse in self.verses if verse.fold == fold]


def training_data(sources: Path, factory, folds: int = FOLDS, holdout: str = "stratified",
                  lexicon=None, jobs: int | None = None, cache_dir: Path | None = SETS_DIR,
                  log: Callable[[str], None] | None = None) -> TrainingData:
    """The NT verses, their folds, and each fold's candidate sets from a table
    (and analyser) built from the other folds."""
    from lxx_tagging import lexicon as lexicon_module
    from lxx_tagging import ntcorpus, ntforms
    verses, rows = ntcorpus.read(ntforms.tagnt_paths(sources))
    assignment = ntcorpus.assign_folds(verses, folds, holdout)
    lexicon = lexicon or lexicon_module.build(sources / "TBESG.txt")
    profile = ntcorpus.profile_of(verses)
    jobs = default_jobs() if jobs is None else jobs
    sets = cached_candidates(
        sources, factory, "jackknife", {"folds": folds, "holdout": holdout},
        lambda: ntcorpus.jackknife(verses, rows, assignment, lexicon, profile, factory, range(folds), jobs=jobs),
        cache_dir, log)
    data = TrainingData(verses, rows, assignment, sets)
    data.lexicon, data.profile = lexicon, profile
    data.counts = certain_lexemes(item for fold in sorted(sets) for verse in sets[fold] for item in verse)
    data.objects = verb_objects(item for fold in sorted(sets) for verse in sets[fold] for item in verse)
    return data


def nested_sets(data: TrainingData, sources: Path, factory, folds: int, holdout: str,
                jobs: int | None = None, cache_dir: Path | None = SETS_DIR,
                log: Callable[[str], None] | None = None) -> dict:
    """(held-out fold, training fold) -> the training fold's candidate sets
    from a table without either: training data for a model to be measured
    on the held-out fold that has seen nothing of it."""
    from lxx_tagging import ntcorpus
    tasks = [(fold, (held,)) for held in range(folds) for fold in range(folds) if fold != held]
    jobs = default_jobs() if jobs is None else jobs
    found = cached_candidates(
        sources, factory, "nested", {"folds": folds, "holdout": holdout},
        lambda: ntcorpus.jackknife(data.verses, data.rows, data.assignment, data.lexicon, data.profile, factory,
                                   (), jobs=jobs, tasks=tasks),
        cache_dir, log)
    return {(excluded[0], fold): sets for (fold, excluded), sets in found.items()}


def fit(data: TrainingData, folds: Sequence[int], epochs: int = EPOCHS, beam: int = BEAM,
        log: Callable[[str], None] | None = None) -> Model:
    model = Model()
    model.text_counts = data.counts
    model.verb_objects = data.objects
    model.name_genders = name_genders(data.lexicon) if data.lexicon is not None else None
    examples: list[Example] = []
    for fold in folds:
        examples += examples_from(model, data.fold_verses(fold), data.sets[fold])
    if log:
        log(f"  {len(examples):,} verses, {sum(len(e.slots) for e in examples):,} tokens")
    return train(examples, model, epochs=epochs, beam=beam, log=log)


def load_or_train(sources: Path, factory, cache_dir: Path = CACHE_DIR, extra_code: Iterable[Path] = (),
                  retrain: bool = False, log: Callable[[str], None] | None = _log_to_stderr,
                  epochs: int = EPOCHS, beam: int = BEAM, folds: int = FOLDS,
                  settings: dict | None = None, verbose: bool = False) -> ModelChooser:
    """The chooser trained on TAGNT, from the cache when the sources and code
    are unchanged; otherwise trained (and cached). Calibration comes from
    `folds`-fold cross-fitting: each fold chosen by a model trained on the
    others. `factory(forms, lexicon, profile)` makes the tagger whose
    candidates the training data has: the CLI's own, so training sees what
    tagging will; `settings` (say, its provider names) join the cache key."""
    from lxx_tagging import ntforms
    source_files = ntforms.tagnt_paths(sources) + [sources / "TBESG.txt"]
    settings = {"format": FORMAT, "epochs": epochs, "beam": beam, "window": WINDOW, "folds": folds, "seed": SEED,
                **(factory.settings() if hasattr(factory, "settings") else {}), **(settings or {})}
    key = cache_key(source_files, code_paths(extra_code), settings)
    path = cache_dir / f"chooser-{key}.json.gz"
    if path.exists() and not retrain:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
        return ModelChooser(Model.from_json(stored["weights"]), Calibration(stored["calibration"]),
                            beam=beam, meta=stored["meta"])
    log = log or (lambda message: None)
    detail = log if verbose else None
    started = time.time()
    log(f"training the chooser on TAGNT (cache {path.name} not found)")
    # The fold candidate sets live beside the model cache (and are reused
    # when only the model's code changes).
    data = training_data(sources, factory, folds, log=log, cache_dir=cache_dir.parent / "candidate_sets")
    results = []
    for held in range(folds):
        log(f" fold {held + 1} of {folds} held out")
        model = fit(data, [f for f in range(folds) if f != held], epochs, beam, detail)
        results.append(evaluate(model, ModelChooser(model, Calibration({}), beam=beam), data.fold_verses(held),
                                data.sets[held], held))
    calibration = Calibration.fit(calibration_records(results))
    log(" final model on every fold")
    model = fit(data, range(folds), epochs, beam, detail)
    meta = {"key": key, "settings": settings, "features": len(model.to_json())}
    stored = {"meta": meta, "weights": model.to_json(), "calibration": calibration.groups}
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with open(temporary, "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as handle:
            handle.write(json.dumps(stored, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    temporary.replace(path)
    log(f"trained in {time.time() - started:.0f}s; cached as {path}")
    loaded = Model.from_json(stored["weights"])
    return ModelChooser(loaded, calibration, beam=beam, meta=meta)


# The code this process runs, hashed as it starts (see `_digest`).
for _path in code_paths():
    _digest(_path, True)
