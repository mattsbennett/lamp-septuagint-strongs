"""Content-based alignment of Brenton's verses and words with CCAT's.

Scoring-only: part of the private harness around ``validate_ccat.py``.

Verse alignment
    Brenton and Rahlfs both number by the Septuagint but differ in places
    (Psalms, Proverbs' and Jeremiah's order, Exodus 35-40, 3 Kingdoms'
    lettered verses, Esther's additions merged into numbered verses).  No
    versification table is used.  For each Brenton verse the harness scores
    CCAT verses near the same chapter:verse and near where the previous verse
    aligned, by idf-weighted overlap of accent-stripped word keys; only if
    none is convincing does it search the whole book through an inverted
    index of rarer words.  A link is accepted when the overlap is high in both
    directions (Dice >= 0.5 locally, 0.6 for a book-wide hit), or when one
    verse is mostly contained in the other (a split or merged verse).
    Linked verses form components (many-to-many), which are aligned word by
    word; a component whose words barely match is dropped as unconfident.

Word alignment
    ``difflib.SequenceMatcher`` on ``form_key`` (accent-insensitive, movable
    nu, elision, οὐκ/οὐχ, ἐκ/ἐξ ...).  Inside the stretches where the
    sequences differ a small dynamic programme pairs words whose
    ``loose_key`` agrees (orthographic variants), an elided form with its full
    form, one Brenton word with two or three CCAT words (Brenton's joined
    διατοῦτο / ἀναμέσον / ἱνατί when not split by the tagger) and the reverse.
    Only paired words are ever scored.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

try:
    from .ccat_greek import beta_to_unicode, elision_compatible, form_key, loose_key
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_greek import beta_to_unicode, elision_compatible, form_key, loose_key


@dataclass
class Unit:
    """One Brenton scoring unit (a tagger token, or a word when untagged)."""

    ref: int
    index: int
    start: int
    end: int
    surface: str
    key: str = ""
    strongs: str = ""
    lemma: str = ""
    morph: str = ""
    tier: str = ""
    confidence: float | None = None
    method: str = ""
    tagged: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            self.key = form_key(self.surface)


@dataclass
class BrentonVerse:
    ref: int
    chapter: int
    verse: int
    units: list[Unit]


@dataclass
class Pair:
    unit: Unit
    ccat: list  # list[CcatToken]
    kind: str   # exact | variant | elision | joined | split


@dataclass
class BookAlignment:
    book: str
    text: str
    brenton_verses: int = 0
    aligned_verses: int = 0
    brenton_units: int = 0
    units_in_aligned_verses: int = 0
    ccat_verses: int = 0
    ccat_verses_used: int = 0
    ccat_tokens: int = 0
    kinds: Counter = field(default_factory=Counter)
    pairs: list[Pair] = field(default_factory=list)
    link_kinds: Counter = field(default_factory=Counter)

    def summary(self) -> dict:
        return {
            "text": self.text,
            "brenton_verses": self.brenton_verses,
            "aligned_verses": self.aligned_verses,
            "ccat_verses": self.ccat_verses,
            "ccat_verses_used": self.ccat_verses_used,
            "brenton_tokens": self.brenton_units,
            "tokens_in_aligned_verses": self.units_in_aligned_verses,
            "ccat_tokens": self.ccat_tokens,
            "tokens_form_matched": sum(self.kinds.values()),
            "match_kinds": dict(sorted(self.kinds.items())),
            "verse_link_kinds": dict(sorted(self.link_kinds.items())),
        }


# --------------------------------------------------------------------------
# Verse alignment
# --------------------------------------------------------------------------

LOCAL_DICE = 0.5
GLOBAL_DICE = 0.6
CONTAIN = 0.7
EXTEND_CONTAIN = 0.6
MIN_COMPONENT_MATCH = 0.3


def _ccat_key(tok) -> str:
    """Matching key of a reference token: any reference token may carry a
    precomputed ``key``; CCAT tokens are keyed from their beta code."""
    key = getattr(tok, "key", None)
    if key is not None:
        return key
    return form_key(beta_to_unicode(tok.form_beta))


class _Scorer:
    def __init__(self, ccat_bags: list[Counter]):
        n = len(ccat_bags)
        df: Counter = Counter()
        for bag in ccat_bags:
            df.update(bag.keys())
        self.idf = {k: math.log(1 + n / d) for k, d in df.items()}
        self.default = math.log(1 + n)
        self.bags = ccat_bags
        self.weights = [self.weight(b) for b in ccat_bags]
        self.inverted: dict[str, list[int]] = defaultdict(list)
        limit = max(20, n // 20)
        for pos, bag in enumerate(ccat_bags):
            for k in bag:
                if df[k] <= limit:
                    self.inverted[k].append(pos)

    def w(self, k: str) -> float:
        return self.idf.get(k, self.default)

    def weight(self, bag: Counter) -> float:
        return sum(self.w(k) * c for k, c in bag.items())

    def overlap(self, bag: Counter, pos: int) -> float:
        other = self.bags[pos]
        if len(bag) > len(other):
            return sum(self.w(k) * min(c, bag.get(k, 0)) for k, c in other.items())
        return sum(self.w(k) * min(c, other.get(k, 0)) for k, c in bag.items())

    def scores(self, bag: Counter, wb: float, pos: int) -> tuple[float, float, float]:
        ov = self.overlap(bag, pos)
        wc = self.weights[pos]
        dice = 2 * ov / (wb + wc) if wb + wc else 0.0
        return dice, (ov / wb if wb else 0.0), (ov / wc if wc else 0.0)

    def global_candidates(self, bag: Counter, top: int = 5) -> list[int]:
        acc: Counter = Counter()
        for k, c in bag.items():
            for pos in self.inverted.get(k, ()):
                acc[pos] += self.w(k) * min(c, self.bags[pos][k])
        return [p for p, _ in acc.most_common(top)]


def link_verses(brenton: list[BrentonVerse], ccat_verses: list) -> list[tuple[int, set[int], str]]:
    """Return, per Brenton verse index, the linked CCAT positions and how."""
    bags = [Counter(_ccat_key(t) for t in v.tokens) for v in ccat_verses]
    scorer = _Scorer(bags)
    by_cv: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pos, v in enumerate(ccat_verses):
        if v.chapter is not None and v.verse is not None:
            by_cv[(v.chapter, v.verse)].append(pos)
    n = len(ccat_verses)
    links: list[tuple[int, set[int], str]] = []
    last = -1
    bbags = [Counter(u.key for u in bv.units) for bv in brenton]
    for i, bv in enumerate(brenton):
        bag = bbags[i]
        wb = scorer.weight(bag)
        if not bag:
            links.append((i, set(), "empty"))
            continue
        cands: set[int] = set()
        for dv in range(-3, 4):
            for pos in by_cv.get((bv.chapter, bv.verse + dv), ()):
                cands.update(range(max(0, pos - 2), min(n, pos + 3)))
        if last >= 0:
            cands.update(range(max(0, last - 1), min(n, last + 5)))
        scored = {p: scorer.scores(bag, wb, p) for p in cands}
        best = max(scored, key=lambda p: (scored[p][0], -abs(p - last)), default=None)
        how = "local"
        ok = best is not None and (
            scored[best][0] >= LOCAL_DICE or scored[best][1] >= CONTAIN or scored[best][2] >= CONTAIN
        )
        if not ok:
            g = scorer.global_candidates(bag)
            gs = {p: scorer.scores(bag, wb, p) for p in g}
            gbest = max(gs, key=lambda p: (gs[p][0], -abs(p - last)), default=None)
            if gbest is not None and (gs[gbest][0] >= GLOBAL_DICE or min(gs[gbest][1], gs[gbest][2]) >= 0.5 and max(gs[gbest][1:]) >= 0.8):
                scored.update(gs)
                best, ok, how = gbest, True, "global"
        if not ok:
            links.append((i, set(), "none"))
            continue
        linked = {best}
        # extend over neighbouring CCAT verses that are mostly inside this one
        nxt_bag = bbags[i + 1] if i + 1 < len(brenton) else None
        nxt_w = scorer.weight(nxt_bag) if nxt_bag else 0.0
        for step in (1, -1):
            p = best + step
            while 0 <= p < n:
                d, sb, sc = scorer.scores(bag, wb, p)
                if sc < EXTEND_CONTAIN:
                    break
                if step == 1 and nxt_bag:
                    nd = scorer.scores(nxt_bag, nxt_w, p)
                    if nd[2] > sc:
                        break
                linked.add(p)
                p += step
        links.append((i, linked, how if len(linked) == 1 else how + "-merged"))
        last = max(linked)
    _fill_sandwiched(links, brenton, ccat_verses, bbags, scorer)
    return links


SANDWICH_DICE = 0.15


def _fill_sandwiched(links, brenton, ccat_verses, bbags, scorer) -> None:
    """Second pass: a run of unlinked Brenton verses between two linked ones,
    facing a run of unused CCAT verses of the same length, is linked in order
    (name lists and numbers, where spellings differ but the position is
    certain).  Each pair still needs some overlap or the same chapter:verse.
    """
    used = set()
    for _, linked, _ in links:
        used.update(linked)
    i = 0
    n = len(links)
    while i < n:
        if links[i][1] or links[i][2] == "empty":
            i += 1
            continue
        j = i
        while j < n and not links[j][1] and links[j][2] != "empty":
            j += 1
        prev = max(links[i - 1][1]) if i > 0 and links[i - 1][1] else None
        nxt = min(links[j][1]) if j < n and links[j][1] else None
        if prev is not None and nxt is not None and nxt - prev - 1 == j - i:
            gap = list(range(prev + 1, nxt))
            if not any(p in used for p in gap):
                for k, p in enumerate(gap):
                    bi = i + k
                    bag = bbags[bi]
                    d, _, _ = scorer.scores(bag, scorer.weight(bag), p)
                    cv = ccat_verses[p]
                    same = (cv.chapter, cv.verse) == (brenton[bi].chapter, brenton[bi].verse)
                    if d >= SANDWICH_DICE or same:
                        links[bi] = (bi, {p}, "sandwiched")
                        used.add(p)
        i = j


# --------------------------------------------------------------------------
# Word alignment
# --------------------------------------------------------------------------

_DP_LIMIT = 80


def _loose_eq(a: str, b: str) -> str | None:
    if a == b:
        return "exact"
    if elision_compatible(a, b):
        return "elision"
    if loose_key(a, is_key=True) == loose_key(b, is_key=True) and len(a) > 1:
        return "variant"
    return None


def _region_pairs(a: list[str], b: list[str]) -> list[tuple[list[int], list[int], str]]:
    """Pair words inside one differing stretch (small DP)."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0 or na > _DP_LIMIT or nb > _DP_LIMIT:
        return []
    bstrip = [x.rstrip("'") for x in b]
    astrip = [x.rstrip("'") for x in a]
    # dp[i][j]: best score for a[i:], b[j:]
    dp = [[0.0] * (nb + 1) for _ in range(na + 1)]
    move: list[list[tuple | None]] = [[None] * (nb + 1) for _ in range(na + 1)]
    for i in range(na - 1, -1, -1):
        for j in range(nb - 1, -1, -1):
            best, mv = dp[i + 1][j], ("skip_a",)
            if dp[i][j + 1] > best:
                best, mv = dp[i][j + 1], ("skip_b",)
            kind = _loose_eq(a[i], b[j])
            if kind:
                s = 1.0 + dp[i + 1][j + 1]
                if s > best:
                    best, mv = s, ("pair", kind)
            for k in (2, 3):
                if j + k <= nb and astrip[i] == "".join(bstrip[j:j + k]):
                    s = 0.9 * k + dp[i + 1][j + k]
                    if s > best:
                        best, mv = s, ("join", k)
                if i + k <= na and "".join(astrip[i:i + k]) == bstrip[j]:
                    s = 0.9 * k + dp[i + k][j + 1]
                    if s > best:
                        best, mv = s, ("split", k)
            dp[i][j], move[i][j] = best, mv
    out = []
    i = j = 0
    while i < na and j < nb:
        mv = move[i][j]
        if mv is None:
            break
        if mv[0] == "skip_a":
            i += 1
        elif mv[0] == "skip_b":
            j += 1
        elif mv[0] == "pair":
            out.append(([i], [j], mv[1]))
            i += 1
            j += 1
        elif mv[0] == "join":
            out.append(([i], list(range(j, j + mv[1])), "joined"))
            i += 1
            j += mv[1]
        else:
            out.append((list(range(i, i + mv[1])), [j], "split"))
            i += mv[1]
            j += 1
    return out


def align_words(a_keys: list[str], b_keys: list[str]) -> list[tuple[list[int], list[int], str]]:
    """Align two key sequences; returns (a indices, b indices, kind) groups."""
    sm = SequenceMatcher(None, a_keys, b_keys, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            out.extend(([i1 + k], [j1 + k], "exact") for k in range(i2 - i1))
        elif i2 > i1 and j2 > j1:
            for ai, bj, kind in _region_pairs(a_keys[i1:i2], b_keys[j1:j2]):
                out.append(([i1 + x for x in ai], [j1 + y for y in bj], kind))
    return out


def align_book(book: str, text: str, brenton: list[BrentonVerse], ccat_verses: list) -> BookAlignment:
    res = BookAlignment(book=book, text=text)
    res.brenton_verses = len(brenton)
    res.brenton_units = sum(len(v.units) for v in brenton)
    res.ccat_verses = len(ccat_verses)
    res.ccat_tokens = sum(len(v.tokens) for v in ccat_verses)
    links = link_verses(brenton, ccat_verses)

    # union-find over Brenton verse indices ("b", i) and CCAT positions ("c", p)
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, linked, how in links:
        res.link_kinds[how] += 1
        for p in linked:
            ra, rb = find(("b", i)), find(("c", p))
            if ra != rb:
                parent[ra] = rb
    comps: dict = defaultdict(lambda: ([], []))
    for i, linked, _ in links:
        if linked:
            comps[find(("b", i))][0].append(i)
            for p in linked:
                comps[find(("b", i))][1].append(p)
    used_ccat: set[int] = set()
    for root, (bis, cps) in comps.items():
        bis = sorted(set(bis))
        cps = sorted(set(cps))
        units = [u for i in bis for u in brenton[i].units]
        toks = [t for p in cps for t in ccat_verses[p].tokens]
        groups = _align_component(brenton, ccat_verses, bis, cps, links)
        matched = sum(len(g[0]) for g in groups)
        if not units or matched < 2 and len(units) > 2 or matched / max(1, len(units)) < MIN_COMPONENT_MATCH:
            res.link_kinds["dropped-unconfident"] += len(bis)
            continue
        res.aligned_verses += len(bis)
        res.units_in_aligned_verses += len(units)
        used_ccat.update(cps)
        for ui, tj, kind in groups:
            if kind == "split":
                for x in ui:
                    res.pairs.append(Pair(units[x], [toks[y] for y in tj], kind))
                res.kinds[kind] += len(ui)
            else:
                res.pairs.append(Pair(units[ui[0]], [toks[y] for y in tj], kind))
                res.kinds[kind] += 1
    res.ccat_verses_used = len(used_ccat)
    return res


def _align_component(brenton, ccat_verses, bis, cps, links):
    """Word-align one component; concatenate when the links are monotone."""
    link_of = {i: sorted(linked) for i, linked, _ in links if i in set(bis)}
    starts = [link_of[i][0] for i in bis if link_of.get(i)]
    units = [u for i in bis for u in brenton[i].units]
    toks = [t for p in cps for t in ccat_verses[p].tokens]
    if starts == sorted(starts):
        return align_words([u.key for u in units], [_ccat_key(t) for t in toks])
    # non-monotone: align verse by verse, first come first served on tokens
    offset_u = {}
    acc = 0
    for i in bis:
        offset_u[i] = acc
        acc += len(brenton[i].units)
    offset_c = {}
    acc = 0
    for p in cps:
        offset_c[p] = acc
        acc += len(ccat_verses[p].tokens)
    taken: set[int] = set()
    out = []
    for i in bis:
        lp = link_of.get(i) or []
        sub_toks = [(offset_c[p] + k, t) for p in lp for k, t in enumerate(ccat_verses[p].tokens)]
        groups = align_words([u.key for u in brenton[i].units], [_ccat_key(t) for _, t in sub_toks])
        for ui, tj, kind in groups:
            gidx = [sub_toks[y][0] for y in tj]
            if any(g in taken for g in gidx):
                continue
            taken.update(gidx)
            out.append(([offset_u[i] + x for x in ui], gidx, kind))
    return out
