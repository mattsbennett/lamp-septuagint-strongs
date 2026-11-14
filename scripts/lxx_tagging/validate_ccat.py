"""Private, scoring-only accuracy check of the Septuagint tagger against CCAT.

The only existing tagged Septuagint is the CCAT/CATSS "Morphologically
Analyzed Septuagint" (Rahlfs' text; Kraft, Taylor, Wheeler).  Its user
declaration forbids redistribution and requires controlled access, so the
tagger's tags must never be derived from it.  The project uses it only to
*measure* the approach, privately.  This harness is that measurement.

FIREWALL
  * CCAT files live only in ``.cache/lxx_validation/ccat/`` (gitignored);
    ``--fetch`` downloads them there from CCAT's own server, nothing else.
  * No CCAT line, lemma or parse is written to a tracked file.  Reports go
    to ``.cache/lxx_validation/reports/``; the harness refuses to write
    inside ``outputs/``, ``scripts/``, ``sources/``, ``modules/`` or
    ``tests/``.
  * Nothing else in ``scripts/lxx_tagging/`` may import this module or its
    ``ccat_*`` helpers (``tests/test_lxx_validation.py`` checks this).
  * Books are split, fixed in code (``DEV``/``TEST`` below).  For the test
    split only aggregate numbers are produced: no forms, lemmas or verse
    references (checked at run time; a violation raises ``FirewallError``).
    For the dev split the report adds category-level breakdowns and, with
    ``--samples N`` (at most 25), N randomly chosen disagreements per
    category showing *our* tag and only the category of the difference
    (``case differs: ours G, theirs D``), never CCAT's lemma or full parse.
    A sampled word appears under one category only, and the draw is fixed by
    ``--seed`` so that re-running does not reveal more.

WHAT IS MEASURED (per split; per book for dev; overall for test)
  * alignment coverage: Brenton verses and words aligned with CCAT's (see
    ``ccat_align.py``: verses by content, words by sequence alignment; only
    words whose form matches on both sides are scored);
  * recall: of aligned words, how many we tagged;
  * of tagged words: lemma, POS and full-parse agreement, each as
    agree / convention / uncertain / disagree (``ccat_codes.py`` documents
    the conventions), broken down by the tagger's decision tier or
    confidence bucket (from ``outputs/lxx_tagging/lxx_tagging_decisions.tsv``), and for
    dev by CCAT part of speech, with per-field confusion counts.

Strong's numbers cannot be compared (CCAT has none): the lemma attached to
our Strong's number is compared with CCAT's headword.

Usage::

    python3 scripts/lxx_tagging/validate_ccat.py --fetch          # once
    python3 scripts/lxx_tagging/validate_ccat.py                  # full run
    python3 scripts/lxx_tagging/validate_ccat.py --split dev --samples 10
    python3 scripts/lxx_tagging/validate_ccat.py --books Gen,Exod

Without ``outputs/lxx_tagging/LXX_tagged.json`` the run is a sanity check of the harness
itself: alignment coverage per book, which of CCAT's double texts Brenton
prints, and the CCAT codes the decomposer could not read (by anonymised
shape).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import ssl
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccat_align import BrentonVerse, Unit, align_book  # noqa: E402
from ccat_codes import (  # noqa: E402
    CONVENTIONS,
    Unmapped,
    adverb_shaped,
    adverb_stem_related,
    ccat_lemma_candidates,
    code_shape,
    compare_feats,
    compare_lemma,
    decompose_ccat,
    decompose_robinson,
    participle_headword,
)
from ccat_greek import contains_greek, tokenize_brenton  # noqa: E402
from ccat_reader import CCAT_TEXTS, read_ccat_text  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CCAT = REPO / ".cache" / "lxx_validation" / "ccat"
DEFAULT_REPORTS = REPO / ".cache" / "lxx_validation" / "reports"
DEFAULT_LXX = REPO / "outputs" / "lxx_tagging" / "LXX_source.json"
DEFAULT_TAGGED = REPO / "outputs" / "lxx_tagging" / "LXX_tagged.json"
DEFAULT_DECISIONS = REPO / "outputs" / "lxx_tagging" / "lxx_tagging_decisions.tsv"
CCAT_URL = "https://ccat.sas.upenn.edu/gopher/text/religion/biblical/lxxmorph/"
FORBIDDEN_REPORT_DIRS = ("outputs", "scripts", "sources", "modules", "modules_db", "tests", "schemas")

# ---------------------------------------------------------------------------
# Dev/test split (fixed).  Roughly half the words each, balanced by genre:
#   law       dev Gen Lev                      test Exod Num Deut
#   history   dev Judg 1-2Sam 1Chr Ezra(2Esd)  test Josh Ruth 1-2Kgs 2Chr Esth
#   poetry    dev Job Ps 1-75                  test Ps 76-151 Prov Eccl Song
#   prophets  dev Isa Ezek Hos Joel Amos Obad Jonah Mic
#             test Jer Lam Dan Nah Hab Zeph Hag Zech Mal
# (Brenton words: dev 235,035, test 243,656.)  Synoptic parallels stay
# together where they can (Samuel with 1 Chronicles, Kings with 2 Chronicles).
# ---------------------------------------------------------------------------
DEV = frozenset({"Gen", "Lev", "Judg", "1Sam", "2Sam", "1Chr", "Ezra", "Job", "Isa", "Ezek",
                 "Hos", "Joel", "Amos", "Obad", "Jonah", "Mic"})
TEST = frozenset({"Exod", "Num", "Deut", "Josh", "Ruth", "1Kgs", "2Kgs", "2Chr", "Esth", "Prov",
                  "Eccl", "Song", "Jer", "Lam", "Dan", "Nah", "Hab", "Zeph", "Hag", "Zech", "Mal"})
PSALMS_DEV_LAST = 75  # Ps 1-75 dev, 76-151 test


def split_of(book: str, chapter: int) -> str:
    if book == "Ps":
        return "dev" if chapter <= PSALMS_DEV_LAST else "test"
    if book in DEV:
        return "dev"
    if book in TEST:
        return "test"
    raise KeyError(book)


class FirewallError(RuntimeError):
    """The report would reveal something the firewall forbids."""


_REF_PATTERNS = (re.compile(r"\b\d{1,3}:\d{1,3}\b"), re.compile(r"\b\d{7,8}\b"))


def assert_aggregate_only(obj, where: str) -> None:
    """Test-split output may hold numbers and category names, nothing else."""
    text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    if contains_greek(text):
        raise FirewallError(f"{where}: Greek text in a test-split report")
    for pat in _REF_PATTERNS:
        if pat.search(text):
            raise FirewallError(f"{where}: verse reference in a test-split report")


# ---------------------------------------------------------------------------
# Fetching CCAT (only from CCAT's own server; sequential, polite)
# ---------------------------------------------------------------------------


def fetch_ccat(dest: Path, books: list[str] | None = None, delay: float = 1.0) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    names = ["0-readme.txt", "0-user-declaration.txt", "0-betacode.txt"]
    for book, texts in CCAT_TEXTS.items():
        if books and book not in books:
            continue
        for files in texts.values():
            names.extend(files)
    # The server's certificate chain is rejected by strict clients.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    fetched = []
    for name in names:
        path = dest / name
        if path.exists() and path.stat().st_size > 0:
            continue
        with urllib.request.urlopen(CCAT_URL + name, context=ctx, timeout=60) as r:
            path.write_bytes(r.read())
        fetched.append(name)
        time.sleep(delay)
    return fetched


# ---------------------------------------------------------------------------
# Our side: Brenton units with the tagger's annotations and decisions
# ---------------------------------------------------------------------------


def _tokenizer():
    """The tagger's own tokenizer when importable (so units are its tokens),
    else the harness's minimal one."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from lxx_tagging.text import tokenize  # type: ignore

        return lambda text: [(t.start, t.end, t.surface) for t in tokenize(text)]
    except Exception:  # pragma: no cover - fallback
        return tokenize_brenton


_COLS = {
    "ref": ("ref", "verse", "verse_ref", "verseref", "reference"),
    "index": ("token", "token_index", "index", "idx", "i"),
    "start": ("start",),
    "end": ("end",),
    "surface": ("surface", "text", "form", "word"),
    "tier": ("tier", "decision_tier", "rule", "decision"),
    "confidence": ("confidence", "score", "probability", "prob"),
    "margin": ("margin",),
    "method": ("method",),
    "strongs": ("strongs", "chosen_strongs", "strong"),
    "lemma": ("lemma", "chosen_lemma"),
    "morph": ("morph", "morphology", "chosen_morph", "chosen_morphology"),
}


def read_decisions(path: Path) -> dict:
    """Read the tagger's decisions TSV, adapting to its header names."""
    if not path or not Path(path).exists():
        return {}
    with open(path, encoding="utf-8", newline="") as fh:
        rows = csv.reader(fh, delimiter="\t")
        header = None
        for row in rows:
            if row and not row[0].startswith("#"):
                header = [h.strip().lower() for h in row]
                break
        if header is None:
            return {}
        col = {}
        for name, options in _COLS.items():
            for o in options:
                if o in header:
                    col[name] = header.index(o)
                    break
        if "ref" not in col:
            return {}
        out = {}
        for row in rows:
            if not row or row[0].startswith("#"):
                continue

            def get(name, row=row):
                i = col.get(name)
                return row[i].strip() if i is not None and i < len(row) else ""

            try:
                ref = int(get("ref"))
            except ValueError:
                continue
            rec = {k: get(k) for k in col}
            if "start" in col and "end" in col and get("start").isdigit():
                out[(ref, int(get("start")), int(get("end")))] = rec
            elif "index" in col and get("index").isdigit():
                out[(ref, "i", int(get("index")))] = rec
        return out


def _confidence(rec: dict, data: dict) -> float | None:
    for v in (rec.get("confidence") if rec else None, data.get("confidence"), data.get("score")):
        try:
            if v not in (None, ""):
                return float(v)
        except (TypeError, ValueError):
            pass
    return None


def build_units(verse: dict, tokenize, decisions: dict) -> list[Unit]:
    text = verse["content"]["text"]
    ref = verse["ref"]
    anns = sorted(
        (a for a in verse["content"].get("annotations", []) if a.get("type") == "strongs"),
        key=lambda a: (a["start"], a["end"]),
    )
    spans = [(s, e, w) for s, e, w in tokenize(text)]
    by_span = {(a["start"], a["end"]): a for a in anns}
    units: list[Unit] = []
    # tokens first (tagger tokens == annotation spans when tagged)
    for s, e, w in spans:
        inner = [a for a in anns if a["start"] >= s and a["end"] <= e]
        if (s, e) in by_span or not inner:
            units.append(Unit(ref, 0, s, e, w))
        else:  # an annotation splits this word (joined word tagged in parts)
            for a in inner:
                units.append(Unit(ref, 0, a["start"], a["end"], text[a["start"]:a["end"]]))
    # annotations whose span no token reproduces
    have = {(u.start, u.end) for u in units}
    for a in anns:
        if (a["start"], a["end"]) not in have and not any(
            u.start <= a["start"] and a["end"] <= u.end for u in units
        ):
            units.append(Unit(ref, 0, a["start"], a["end"], text[a["start"]:a["end"]]))
    units.sort(key=lambda u: (u.start, u.end))
    for i, u in enumerate(units):
        u.index = i
        a = by_span.get((u.start, u.end))
        rec = decisions.get((ref, u.start, u.end)) or decisions.get((ref, "i", i)) or {}
        data = (a or {}).get("data", {}) if a else {}
        # Only the shipped annotations count as tags; the decisions file
        # supplies the tier/confidence of each token, not tags of its own.
        if a:
            u.tagged = bool(data.get("strongs") or data.get("lemma"))
            u.strongs = data.get("strongs", "") or ""
            u.lemma = data.get("lemma", "") or ""
            u.morph = data.get("morphology", "") or ""
        u.tier = (rec.get("tier") if rec else "") or data.get("tier", "") or ("tagged" if u.tagged else "untagged")
        u.confidence = _confidence(rec, data)
        u.method = (rec.get("method") if rec else "") or data.get("method", "") or ""
    return units


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

OUTCOMES = ("agree", "convention", "uncertain", "disagree")
CONF_BUCKETS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99)


def conf_bucket(c: float | None) -> str:
    if c is None:
        return ""
    lo = 0.0
    for b in CONF_BUCKETS:
        if c < b:
            return f"[{lo:.2f},{b:.2f})"
        lo = b
    return f"[{lo:.2f},1.00]"


class Stats:
    """Aggregate counters for one slice (split, book, tier ...)."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.lemma: Counter = Counter()
        self.pos: Counter = Counter()
        self.full: Counter = Counter()
        self.rules: Counter = Counter()
        self.uncertain: Counter = Counter()
        self.fields: dict[str, Counter] = defaultdict(Counter)
        self.confusion: dict[str, Counter] = defaultdict(Counter)
        self.second: Counter = Counter()

    def core(self) -> dict:
        aligned = self.c["aligned"]
        tagged = self.c["tagged"]

        def rate(cnt: Counter) -> dict:
            n = sum(cnt.values())
            out = {k: cnt.get(k, 0) for k in OUTCOMES}
            out["scored"] = n
            decided = n - cnt.get("uncertain", 0)
            if decided:
                out["strict_rate"] = round(cnt.get("agree", 0) / decided, 4)
                out["with_conventions_rate"] = round(
                    (cnt.get("agree", 0) + cnt.get("convention", 0)) / decided, 4)
            return out

        return {
            "aligned_tokens": aligned,
            "tagged": tagged,
            "recall": round(tagged / aligned, 4) if aligned else None,
            "tagged_with_lemma": self.c["with_lemma"],
            "tagged_with_morph": self.c["with_morph"],
            "lemma": rate(self.lemma),
            "pos": rate(self.pos),
            "full_parse": rate(self.full),
        }

    def detailed(self) -> dict:
        out = self.core()
        out["convention_rules"] = dict(self.rules.most_common())
        out["uncertain_reasons"] = dict(self.uncertain.most_common())
        out["fields"] = {k: dict(v) for k, v in sorted(self.fields.items())}
        out["confusion"] = {
            k: {f"ours {a} / theirs {b}": n for (a, b), n in v.most_common()}
            for k, v in sorted(self.confusion.items())
        }
        out["second_aorist_marking"] = dict(self.second)
        return out


class CcatAdapter:
    """How the generic scorer reads one CCAT token (see ``score_against``)."""

    name = "CCAT"
    unmapped = "unmapped CCAT code"

    @staticmethod
    def feats(tok):
        return decompose_ccat(tok.type_code, tok.parse_code, tok.lemma_beta)

    @staticmethod
    def lemma(our_lemma, tok, *, proper=False, ours_compound=False):
        return compare_lemma(our_lemma, tok.lemma_beta, tok.prefixes, proper=proper,
                             ours_compound=ours_compound)

    @staticmethod
    def lemma_set(tok):
        return ccat_lemma_candidates(tok.lemma_beta, tok.prefixes)


def score_pair(pair) -> dict | None:
    """Compare one aligned Brenton unit with CCAT; None when untagged."""
    return score_against(pair, CcatAdapter)


def score_against(pair, adapter) -> dict | None:
    """Compare one aligned Brenton unit with a reference token's lemma and
    feature bundle (CCAT, UD, GLAUx ...); None when the unit is untagged.

    ``adapter`` supplies ``feats(tok)`` (raises ``Unmapped``),
    ``lemma(our_lemma, tok, proper=, ours_compound=)`` and ``lemma_set(tok)``.
    """
    u = pair.unit
    if not u.tagged:
        return None
    res = {"lemma": "uncertain", "lemma_rule": "", "pos": "uncertain", "pos_rule": "",
           "full": "uncertain", "fields": {}, "second": None, "ccat_pos": "", "reason": "",
           "has_lemma": bool(u.lemma), "has_morph": bool(u.morph), "unmapped_ours": False}
    if pair.kind == "split":
        res["reason"] = "word split differently"
        return res
    toks = pair.ccat
    ours = None
    if u.morph:
        try:
            ours = decompose_robinson(u.morph)
        except Unmapped:
            res["unmapped_ours"] = True
    if pair.kind == "joined":
        res["reason"] = "joined word"
        if ours is not None and ours.pos == "crasis" and len(ours.parts) == len(toks):
            lem = [adapter.lemma(u.lemma, t)[0] for t in toks]
            res["lemma"] = "agree" if "agree" in lem else ("convention" if "convention" in lem else "disagree")
            try:
                theirs = [adapter.feats(t) for t in toks]
            except Unmapped:
                res["reason"] = adapter.unmapped
                return res
            subs = [compare_feats(o, t, "agree") for o, t in zip(ours.parts, theirs)]
            res["pos"] = max((s.pos for s in subs), key=OUTCOMES.index)
            res["full"] = max((s.full for s in subs), key=OUTCOMES.index)
            res["ccat_pos"] = theirs[0].pos
        return res
    return score_core(res, u.lemma, ours, toks[0], adapter, ours_unmapped=bool(u.morph) and ours is None,
                      has_morph=bool(u.morph))


def score_core(res: dict, our_lemma: str, ours, tok, adapter, *, ours_unmapped: bool = False,
               has_morph: bool = True) -> dict:
    """Lemma + feature comparison of one analysis (``our_lemma``, ``ours``
    Feats) with one reference token.  Shared by the tagger-vs-reference
    scorer and the reference-vs-reference benchmark (validate_refs), where
    "ours" is the first reference's analysis."""
    try:
        theirs = adapter.feats(tok)
    except Unmapped:
        theirs = None
        res["reason"] = adapter.unmapped
    if theirs is not None:
        res["ccat_pos"] = theirs.pos
    proper = bool(theirs and theirs.proper) or bool(ours and ours.proper)
    if our_lemma:
        res["lemma"], res["lemma_rule"] = adapter.lemma(
            our_lemma, tok, proper=proper, ours_compound=bool(ours and ours.pos == "crasis"))
    else:
        res["lemma"], res["lemma_rule"] = "uncertain", "no lemma"
    if not has_morph:
        res["reason"] = res["reason"] or "no morphology"
        return res
    if ours is None:
        res["reason"] = "unmapped tagger code" if ours_unmapped else res["reason"]
        return res
    if theirs is None:
        return res
    adv_rel = False
    pos_pair = {ours.pos, theirs.pos}
    theirs_lemmas = adapter.lemma_set(tok)
    if res["lemma"] == "disagree" and our_lemma:
        if ((pos_pair & {"adv", "prep"} or adverb_shaped(our_lemma))
                and adverb_stem_related(our_lemma, theirs=theirs_lemmas)):
            adv_rel = True
            res["lemma"], res["lemma_rule"] = "convention", "adverbial adjective"
        elif ("verb" in pos_pair and pos_pair & {"noun", "adj"}
              and (theirs.mood == "P" or ours.mood == "P" or ours.pos != "verb" or theirs.pos != "verb")
              and participle_headword(our_lemma, theirs=theirs_lemmas)):
            res["lemma"], res["lemma_rule"] = "convention", "substantival participle"
    elif pos_pair & {"adv", "prep"} and pos_pair & {"adj", "noun"} and our_lemma:
        adv_rel = adverb_stem_related(our_lemma, theirs=theirs_lemmas)
    cmp = compare_feats(ours, theirs, res["lemma"], adv_rel, res["lemma_rule"])
    res["pos"], res["pos_rule"], res["full"] = cmp.pos, cmp.pos_rule, cmp.full
    res["fields"] = cmp.fields
    res["second"] = cmp.second
    res["ours_pos"] = ours.pos
    if cmp.pos == "uncertain":
        res["reason"] = cmp.pos_rule
    elif cmp.full == "uncertain":
        res["reason"] = next((v[1] for v in cmp.fields.values() if v[0] == "uncertain"), "field")
    return res


def add(stats: Stats, pair, res: dict | None, *, detailed: bool) -> None:
    stats.c["aligned"] += 1
    if res is None:
        return
    stats.c["tagged"] += 1
    if res["has_lemma"]:
        stats.c["with_lemma"] += 1
        stats.lemma[res["lemma"]] += 1
    if res["has_morph"]:
        stats.c["with_morph"] += 1
        stats.pos[res["pos"]] += 1
        stats.full[res["full"]] += 1
    if not detailed:
        return
    for rule in (res["lemma_rule"] if res["lemma"] == "convention" else "",
                 res["pos_rule"] if res["pos"] == "convention" else ""):
        if rule:
            stats.rules[rule] += 1
    for name, (outcome, rule, a, b) in res["fields"].items():
        stats.fields[name][outcome] += 1
        if outcome == "convention" and rule:
            stats.rules[rule] += 1
        stats.confusion[name][(a or "-", b or "-")] += 1
    if res["has_morph"] and res.get("ours_pos"):
        stats.confusion["pos"][(res["ours_pos"], res["ccat_pos"] or "-")] += 1
    if (res["has_morph"] and res["full"] == "uncertain") or (res["has_lemma"] and res["lemma"] == "uncertain"):
        stats.uncertain[res["reason"] or res["lemma_rule"] or "other"] += 1
    if res["second"]:
        stats.second[res["second"]] += 1


# ---------------------------------------------------------------------------
# Dev samples: our tag + the category of the difference only
# ---------------------------------------------------------------------------

MAX_SAMPLES = 25


def difference_categories(res: dict) -> list[tuple[str, str]]:
    """(category, description) for each difference; never CCAT's lemma or code."""
    out = []
    if res["lemma"] == "disagree":
        out.append(("lemma differs", "lemma differs"))
    if res["pos"] == "disagree":
        out.append(("POS differs", f"POS differs: ours {res.get('ours_pos', '?')}, theirs {res['ccat_pos'] or '?'}"))
    for name, (outcome, _rule, a, b) in res["fields"].items():
        if outcome == "disagree":
            out.append((f"{name} differs", f"{name} differs: ours {a or '-'}, theirs {b or '-'}"))
    return out


def draw_samples(candidates: dict[str, list], n: int, seed: int) -> dict[str, list]:
    n = max(0, min(n, MAX_SAMPLES))
    rng = random.Random(seed)
    used: set[int] = set()
    out: dict[str, list] = {}
    for cat in sorted(candidates):
        pool = [c for c in candidates[cat] if id(c[0]) not in used]
        rng.shuffle(pool)
        chosen = pool[:n]
        for c in chosen:
            used.add(id(c[0]))
        out[cat] = [
            {"book": book, "ref": u.ref, "surface": u.surface, "strongs": u.strongs,
             "lemma": u.lemma, "morph": u.morph, "tier": u.tier, "difference": desc}
            for (u, book, desc) in chosen
        ]
    return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def _check_reports_dir(path: Path) -> Path:
    path = Path(path).resolve()
    for name in FORBIDDEN_REPORT_DIRS:
        root = (REPO / name).resolve()
        if path == root or root in path.parents:
            raise FirewallError(f"refusing to write reports inside {name}/")
    return path


def load_brenton(lxx_path: Path, tagged_path: Path | None) -> tuple[dict, bool]:
    if tagged_path and Path(tagged_path).exists():
        return json.loads(Path(tagged_path).read_text(encoding="utf-8")), True
    return json.loads(Path(lxx_path).read_text(encoding="utf-8")), False


def run(args) -> dict:
    ccat_dir = Path(args.ccat_dir)
    reports = _check_reports_dir(Path(args.reports))
    module, tagged = load_brenton(Path(args.lxx), Path(args.tagged) if args.tagged else None)
    decisions = read_decisions(Path(args.decisions)) if (tagged and args.decisions) else {}
    tokenize = _tokenizer()
    wanted = set(args.books.split(",")) if args.books else None
    splits_wanted = {"dev", "test"} if args.split == "all" else {args.split}

    split_stats = {"dev": Stats(), "test": Stats()}
    tier_stats = {"dev": defaultdict(Stats), "test": defaultdict(Stats)}
    conf_stats = {"dev": defaultdict(Stats), "test": defaultdict(Stats)}
    tier_conf_stats = {"dev": defaultdict(Stats), "test": defaultdict(Stats)}
    method_stats = {"dev": defaultdict(Stats), "test": defaultdict(Stats)}
    dev_book = defaultdict(Stats)
    dev_book_tier = defaultdict(lambda: defaultdict(Stats))
    dev_pos = defaultdict(Stats)
    coverage: dict[str, dict] = {}
    text_choice: dict[str, dict] = {}
    unmapped_shapes: Counter = Counter()
    unmapped_ours = Counter()
    sample_pool: dict[str, list] = defaultdict(list)
    missing = []
    present: set[str] = set()

    for book in module["books"]:
        bid = book["id"]
        if bid not in CCAT_TEXTS or (wanted and bid not in wanted):
            continue
        book_splits = {split_of(bid, ch["chapter"]) for ch in book["chapters"]}
        if not (book_splits & splits_wanted):
            continue
        present |= book_splits & splits_wanted
        bverses = []
        for ch in book["chapters"]:
            for v in ch["verses"]:
                bverses.append(BrentonVerse(v["ref"], ch["chapter"], v["v"], build_units(v, tokenize, decisions)))
        best = None
        choice = {}
        for name, files in CCAT_TEXTS[bid].items():
            cverses = read_ccat_text(ccat_dir, files)
            if cverses is None:
                continue
            res = align_book(bid, name, bverses, cverses)
            matched = sum(res.kinds.values())
            choice[name] = {"tokens_form_matched": matched, "aligned_verses": res.aligned_verses}
            if best is None or matched > sum(best.kinds.values()):
                best = res
        if best is None:
            missing.append(bid)
            continue
        if len(choice) > 1:
            text_choice[bid] = {"chosen": best.text, "candidates": choice}
        coverage[bid] = best.summary()
        coverage[bid]["split"] = "/".join(sorted(book_splits))
        # unmapped CCAT codes (by anonymised shape)
        for p in best.pairs:
            for t in p.ccat:
                try:
                    decompose_ccat(t.type_code, t.parse_code, t.lemma_beta)
                except Unmapped:
                    unmapped_shapes[code_shape(t.type_code, t.parse_code)] += 1
        for pair in best.pairs:
            u = pair.unit
            sp = split_of(bid, (u.ref // 1000) % 1000)
            if sp not in splits_wanted:
                continue
            res = score_pair(pair)
            detailed = sp == "dev"
            add(split_stats[sp], pair, res, detailed=detailed)
            add(tier_stats[sp][u.tier or "-"], pair, res, detailed=False)
            if u.confidence is not None:
                add(conf_stats[sp][conf_bucket(u.confidence)], pair, res, detailed=False)
                add(tier_conf_stats[sp][f"{u.tier or '-'} x {conf_bucket(u.confidence)}"], pair, res,
                    detailed=False)
            if u.method:
                add(method_stats[sp][u.method], pair, res, detailed=False)
            if res and res["unmapped_ours"]:
                unmapped_ours[sp] += 1
            if sp == "dev":
                add(dev_book[bid], pair, res, detailed=True)
                add(dev_book_tier[bid][u.tier or "-"], pair, res, detailed=False)
                if res is not None:
                    add(dev_pos[res["ccat_pos"] or "n/a (split, joined or unmapped)"], pair, res, detailed=True)
                    for cat, desc in difference_categories(res):
                        sample_pool[cat].append((u, bid, desc))
                        break  # one category per word

    report = {
        "meta": {
            "harness": "validate_ccat.py",
            "tagged_input": bool(tagged),
            "decisions_input": bool(decisions),
            "splits": sorted(present),
            "split_definition": {"dev": sorted(DEV) + [f"Ps 1-{PSALMS_DEV_LAST}"],
                                 "test": sorted(TEST) + [f"Ps {PSALMS_DEV_LAST + 1}-151"]},
            "missing_ccat_books": missing,
            "note": "CCAT is used only to measure; nothing here may feed the tagger.",
        },
        "alignment": {"per_book": coverage, "overall": _coverage_total(coverage)},
        "double_texts": text_choice,
        "unmapped_ccat_code_shapes": dict(unmapped_shapes.most_common()),
        "unmapped_tagger_codes": dict(unmapped_ours),
        "conventions": CONVENTIONS,
    }
    if "dev" in present:
        report["dev"] = {
            "overall": split_stats["dev"].detailed(),
            "by_tier": {k: v.core() for k, v in sorted(tier_stats["dev"].items())},
            "by_confidence": {k: v.core() for k, v in sorted(conf_stats["dev"].items())},
            "by_tier_and_confidence": {k: v.core() for k, v in sorted(tier_conf_stats["dev"].items())},
            "by_method": {k: v.core() for k, v in sorted(method_stats["dev"].items())},
            "by_ccat_pos": {k: v.detailed() for k, v in sorted(dev_pos.items())},
            "per_book": {
                b: dict(s.detailed(), by_tier={k: v.core() for k, v in sorted(dev_book_tier[b].items())})
                for b, s in dev_book.items()
            },
        }
        if args.samples:
            report["dev"]["samples"] = draw_samples(sample_pool, args.samples, args.seed)
    if "test" in present:
        test = {
            "overall": split_stats["test"].core(),
            "by_tier": {k: v.core() for k, v in sorted(tier_stats["test"].items())},
            "by_confidence": {k: v.core() for k, v in sorted(conf_stats["test"].items())},
            "by_tier_and_confidence": {k: v.core() for k, v in sorted(tier_conf_stats["test"].items())},
            "by_method": {k: v.core() for k, v in sorted(method_stats["test"].items())},
        }
        assert_aggregate_only(test, "test split")
        report["test"] = test

    text = render_summary(report)
    reports.mkdir(parents=True, exist_ok=True)
    stem = args.name or "ccat_validation"
    (reports / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (reports / f"{stem}.txt").write_text(text, encoding="utf-8")
    print(text)
    return report


def _coverage_total(coverage: dict) -> dict:
    keys = ("brenton_verses", "aligned_verses", "brenton_tokens", "tokens_form_matched", "ccat_tokens")
    tot = {k: sum(c[k] for c in coverage.values()) for k in keys}
    if tot["brenton_tokens"]:
        tot["token_coverage"] = round(tot["tokens_form_matched"] / tot["brenton_tokens"], 4)
    if tot["brenton_verses"]:
        tot["verse_coverage"] = round(tot["aligned_verses"] / tot["brenton_verses"], 4)
    return tot


def _pct(x) -> str:
    return "-" if x is None else f"{100 * x:5.1f}%"


def _core_line(name: str, core: dict) -> str:
    def r(block):
        return f"{_pct(block.get('strict_rate'))} / {_pct(block.get('with_conventions_rate'))}"
    return (f"  {name:<22} aligned {core['aligned_tokens']:>7}  recall {_pct(core['recall'])}  "
            f"lemma {r(core['lemma'])}  POS {r(core['pos'])}  parse {r(core['full_parse'])}  "
            f"uncertain L{core['lemma']['uncertain']}/P{core['full_parse']['uncertain']}")


def render_summary(report: dict) -> str:
    lines = ["CCAT validation (private; scoring only)", ""]
    meta = report["meta"]
    lines.append(f"tagged input: {meta['tagged_input']}   decisions: {meta['decisions_input']}   "
                 f"splits: {', '.join(meta['splits'])}")
    if meta["missing_ccat_books"]:
        lines.append(f"CCAT files missing for: {', '.join(meta['missing_ccat_books'])} (run --fetch)")
    tot = report["alignment"]["overall"]
    lines += ["", "ALIGNMENT (Brenton words whose form matches a CCAT word)"]
    if tot.get("brenton_tokens"):
        lines.append(f"  overall: verses {tot['aligned_verses']}/{tot['brenton_verses']} "
                     f"({_pct(tot.get('verse_coverage'))}), tokens {tot['tokens_form_matched']}/"
                     f"{tot['brenton_tokens']} ({_pct(tot.get('token_coverage'))})")
    for b, c in report["alignment"]["per_book"].items():
        cov = c["tokens_form_matched"] / c["brenton_tokens"] if c["brenton_tokens"] else None
        lines.append(f"  {b:<6} [{c['split']:<8}] {c['text']:<6} verses {c['aligned_verses']:>5}/"
                     f"{c['brenton_verses']:<5} tokens {c['tokens_form_matched']:>6}/{c['brenton_tokens']:<6} "
                     f"{_pct(cov)}")
    if report["double_texts"]:
        lines += ["", "DOUBLE TEXTS (form-matched tokens per CCAT text)"]
        for b, c in report["double_texts"].items():
            alts = ", ".join(f"{k} {v['tokens_form_matched']}" for k, v in c["candidates"].items())
            lines.append(f"  {b}: Brenton prints {c['chosen']} ({alts})")
    shapes = report["unmapped_ccat_code_shapes"]
    lines += ["", f"UNMAPPED CCAT CODES among aligned tokens: {sum(shapes.values())} "
              f"({len(shapes)} shapes)"]
    for s, n in list(shapes.items())[:12]:
        lines.append(f"  {s:<16} {n}")
    if report["unmapped_tagger_codes"]:
        lines.append(f"  tagger codes not decomposed: {report['unmapped_tagger_codes']}")
    if "dev" in report:
        d = report["dev"]
        lines += ["", "DEV (rates: strict / with conventions, over decided = not uncertain)",
                  _core_line("overall", d["overall"])]
        for b, s in d["per_book"].items():
            lines.append(_core_line(b, s))
        lines.append("  by tier:")
        for k, s in d["by_tier"].items():
            lines.append(_core_line(k, s))
        for key, title in (("by_confidence", "by confidence"), ("by_tier_and_confidence", "by tier x confidence"),
                           ("by_method", "by method")):
            if d.get(key):
                lines.append(f"  {title}:")
                for k, s in d[key].items():
                    lines.append(_core_line(k, s))
        lines.append("  by CCAT part of speech:")
        for k, s in d["by_ccat_pos"].items():
            lines.append(_core_line(k, s))
        if d["overall"]["convention_rules"]:
            lines.append("  conventions applied: " + ", ".join(
                f"{k} {v}" for k, v in d["overall"]["convention_rules"].items()))
        if d["overall"]["uncertain_reasons"]:
            lines.append("  uncertain: " + ", ".join(
                f"{k} {v}" for k, v in d["overall"]["uncertain_reasons"].items()))
        for field, conf in d["overall"]["confusion"].items():
            off = [(k, v) for k, v in conf.items() if k.split(" / ")[0][5:] != k.split(" / ")[1][7:]]
            if off:
                lines.append(f"  {field} confusions: " + ", ".join(f"{k} {v}" for k, v in off[:8]))
        if d.get("samples"):
            lines.append("  samples (our tag; category of difference only):")
            for cat, items in d["samples"].items():
                lines.append(f"    {cat}:")
                for it in items:
                    lines.append(f"      {it['book']} {it['ref']} {it['surface']} {it['strongs']} "
                                 f"{it['lemma']} {it['morph']} [{it['tier']}] -> {it['difference']}")
    if "test" in report:
        t = report["test"]
        block = ["", "TEST (aggregate only)", _core_line("overall", t["overall"])]
        block.append("  by tier:")
        for k, s in t["by_tier"].items():
            block.append(_core_line(k, s))
        for key, title in (("by_confidence", "by confidence"), ("by_tier_and_confidence", "by tier x confidence"),
                           ("by_method", "by method")):
            if t.get(key):
                block.append(f"  {title}:")
                for k, s in t[key].items():
                    block.append(_core_line(k, s))
        assert_aggregate_only("\n".join(block), "test summary")
        lines += block
    lines += ["", "Conventions are documented in ccat_codes.CONVENTIONS and in the JSON report."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ccat-dir", default=str(DEFAULT_CCAT))
    ap.add_argument("--lxx", default=str(DEFAULT_LXX))
    ap.add_argument("--tagged", default=str(DEFAULT_TAGGED))
    ap.add_argument("--decisions", default=str(DEFAULT_DECISIONS))
    ap.add_argument("--reports", default=str(DEFAULT_REPORTS))
    ap.add_argument("--name", default="", help="report file stem (default ccat_validation)")
    ap.add_argument("--split", choices=("all", "dev", "test"), default="all")
    ap.add_argument("--books", default="", help="comma-separated Brenton book ids")
    ap.add_argument("--samples", type=int, default=0, help=f"dev only: up to N (max {MAX_SAMPLES}) "
                    "sampled disagreements per category")
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--fetch", action="store_true", help="download the CCAT files and exit")
    args = ap.parse_args(argv)
    if args.fetch:
        got = fetch_ccat(Path(args.ccat_dir), args.books.split(",") if args.books else None)
        print(f"fetched {len(got)} files into {args.ccat_dir}")
        return 0
    if not Path(args.ccat_dir).exists():
        print(f"no CCAT files in {args.ccat_dir}; run with --fetch first", file=sys.stderr)
        return 2
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
