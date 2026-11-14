"""Private, scoring-only accuracy check against several reference taggings.

Extends ``validate_ccat.py`` (whose firewall, split and report rules apply
unchanged) to every public, even if licence-restricted, high-quality tagging
of the Septuagint the project can measure against:

  ccat   CCAT/CATSS morphology (Rahlfs)              lemma, POS, full parse
  abp    Apostolic Bible Polyglot (ref_abp.py)       Strong's numbers
  ptnk   UD_Ancient_Greek-PTNK (ref_ud.py)           hand-checked lemma, UPOS,
                                                     features; Genesis, Ruth
  glaux  GLAUx (ref_glaux.py)                        machine lemma + morphology

FIREWALL (as for CCAT)
  * reference files only under ``.cache/lxx_validation/<name>/``; reports
    only under ``.cache/lxx_validation/reports/``;
  * nothing in the tagger may import this module or any ``ref_*``/``ccat_*``
    helper (tests/test_lxx_validation*.py check this);
  * test-split books: aggregate numbers only, checked at run time;
  * dev books: category breakdowns, and with ``--samples N`` (max 25) our tag
    plus the category of the difference only -- never a reference's own
    lemma, parse, feature string or Strong's number for a word.

Each reference is aligned to Brenton by content (``ccat_align``), scored on
the words whose forms match, and reported separately.  A combined summary
then takes, for every word aligned and decided in two or more references,
whether every reference agrees with us or every reference disagrees (the
strongest error signal), plus GLAUx's three-way comparison with CCAT.

Usage::

    python3 scripts/lxx_tagging/validate_refs.py --fetch abp,ptnk,glaux
    python3 scripts/lxx_tagging/validate_refs.py [--refs ccat,abp,ptnk,glaux]
        [--split dev|test] [--books Gen,Ruth] [--samples N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validate_ccat as vc  # noqa: E402
from ccat_align import BrentonVerse, align_book  # noqa: E402
from ccat_codes import CONVENTIONS, worst  # noqa: E402
from ccat_reader import CCAT_TEXTS, read_ccat_text  # noqa: E402
from ref_abp import (  # noqa: E402
    ABP_CONF_URL, ABP_ZIP_URL, compare_numbers, load_tbesg_lemmas, read_abp,
)
from ref_common import fetch  # noqa: E402
from ref_glaux import GLAUX_BASE, GLAUX_FILES, GlauxAdapter, read_glaux  # noqa: E402
from ref_ud import PTNK_BASE, PTNK_FILES, UdAdapter, read_ptnk  # noqa: E402

CACHE = vc.REPO / ".cache" / "lxx_validation"
TBESG = vc.REPO / "sources" / "stepbible" / "TBESG.txt"
ALL_REFS = ("ccat", "abp", "ptnk", "glaux")
FEATURE_REFS = {"ccat", "ptnk", "glaux"}

NUMBER_CONVENTIONS = {
    "form-numbered Strong's": "Strong's numbered some forms separately (mou 3450, en 2258, "
    "eipon 2036 ...); STEPBible files them under the headword: ABP number and ours in one "
    "headword set.",
    "same headword, other number": "the TBESG headwords of the two numbers are the same "
    "under the lemma conventions (ccat_codes).",
    "compound number": "ABP codes a crasis or a compound with two numbers; ours is one of them.",
    "compound tag": "our STEPBible compound code (main number + G-number=part) contains ABP's "
    "number, or ABP's headword begins with our main lemma (oukheti for ou + eti).",
    "ABP personal pronoun 1473": "ABP numbers every personal pronoun (mou, sou, autou, hymon) 1473; "
    "ours is a personal-pronoun number (ego/sy forms, autos 846).",
    "name (ABP gives no number)": "ABP marks proper names G* without a number: not checked.",
    "both LXX-only (no open number mapping)": "ABP's decimal additions and STEPBible's "
    "G6100-G9979 both mark words absent from Strong's, but no open crosswalk links the two "
    "numberings, so these are reported as a bucket and not number-checked.",
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def fetch_refs(names: list[str]) -> dict[str, int]:
    got = Counter()
    if "ccat" in names:
        got["ccat"] = len(vc.fetch_ccat(CACHE / "ccat"))
    if "abp" in names:
        got["abp"] += fetch(ABP_CONF_URL, CACHE / "abp" / "abpgrk.conf")
        got["abp"] += fetch(ABP_ZIP_URL, CACHE / "abp" / "ABPGRK.zip")
    if "ptnk" in names:
        for f in PTNK_FILES:
            got["ptnk"] += fetch(PTNK_BASE + f, CACHE / "ptnk" / f)
    if "glaux" in names:
        for f in ("README.md", "metadata.txt"):
            got["glaux"] += fetch(GLAUX_BASE + f, CACHE / "glaux" / f)
        for texts in GLAUX_FILES.values():
            for fid in texts.values():
                got["glaux"] += fetch(GLAUX_BASE + f"xml/{fid}.xml", CACHE / "glaux" / f"{fid}.xml")
    return dict(got)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


class Reference:
    def __init__(self, name: str, texts_for, score):
        self.name = name
        self.texts_for = texts_for   # book -> {text name: verses} | None
        self.score = score           # pair -> result dict | None


def load_references(names, ccat_dir: Path, cache: Path) -> tuple[list[Reference], dict]:
    refs, missing = [], {}
    if "ccat" in names:
        if Path(ccat_dir).exists():
            def ccat_texts(book, d=Path(ccat_dir)):
                out = {}
                for name, files in CCAT_TEXTS.get(book, {}).items():
                    v = read_ccat_text(d, files)
                    if v is not None:
                        out[name] = v
                return out or None
            refs.append(Reference("ccat", ccat_texts, lambda p: vc.score_against(p, vc.CcatAdapter)))
        else:
            missing["ccat"] = "not fetched"
    if "abp" in names:
        abp = read_abp(cache / "abp" / "ABPGRK.zip")
        if abp is None:
            missing["abp"] = "not fetched"
        else:
            lemmas = load_tbesg_lemmas(TBESG)
            refs.append(Reference("abp", lambda b, a=abp: {"ABP": a[b]} if b in a else None,
                                  lambda p, lem=lemmas: score_numbers(p, lem)))
    if "ptnk" in names:
        ptnk = read_ptnk(cache / "ptnk")
        if ptnk is None:
            missing["ptnk"] = "not fetched"
        else:
            refs.append(Reference("ptnk", lambda b, t=ptnk: {"PTNK": t[b]} if b in t else None,
                                  lambda p: vc.score_against(p, UdAdapter)))
    if "glaux" in names:
        gdir = cache / "glaux"
        if gdir.exists():
            refs.append(Reference("glaux", lambda b, d=gdir: read_glaux(d, b),
                                  lambda p: vc.score_against(p, GlauxAdapter)))
        else:
            missing["glaux"] = "not fetched"
    return refs, missing


def score_numbers(pair, lemmas) -> dict | None:
    u = pair.unit
    if not u.tagged:
        return None
    res = {"strongs": "uncertain", "rule": "", "range": "none", "has_strongs": bool(u.strongs)}
    if pair.kind in ("split", "joined"):
        res["rule"] = "word split differently" if pair.kind == "split" else "joined word"
        return res
    if not u.strongs:
        res["rule"] = "no number"
        return res
    res["strongs"], res["rule"], res["range"] = compare_numbers(u.strongs, u.morph, pair.ccat[0].numbers,
                                                                 lemmas, u.lemma)
    return res


def primary(name: str, res: dict | None) -> str | None:
    """One outcome per word and reference, for the combined summary."""
    if res is None:
        return None
    if name == "abp":
        return res["strongs"] if res["has_strongs"] else None
    if not res["has_lemma"] and not res["has_morph"]:
        return None
    parts = []
    if res["has_lemma"]:
        parts.append(res["lemma"])
    if res["has_morph"]:
        parts.append(res["full"])
    return worst(parts)


class NumberStats:
    """Aggregates for the Strong's-number reference."""

    def __init__(self) -> None:
        self.c: Counter = Counter()
        self.outcome: Counter = Counter()
        self.rules: Counter = Counter()
        self.range: Counter = Counter()
        self.range_outcome: Counter = Counter()

    def add(self, res: dict | None) -> None:
        self.c["aligned"] += 1
        if res is None:
            return
        self.c["tagged"] += 1
        if not res["has_strongs"]:
            return
        self.c["with_strongs"] += 1
        self.outcome[res["strongs"]] += 1
        if res["rule"]:
            self.rules[f"{res['strongs']}: {res['rule']}"] += 1
        self.range[res["range"]] += 1
        self.range_outcome[f"{res['range']} / {res['strongs']}"] += 1

    def core(self) -> dict:
        n = sum(self.outcome.values())
        decided = n - self.outcome.get("uncertain", 0)
        out = {
            "aligned_tokens": self.c["aligned"], "tagged": self.c["tagged"],
            "recall": round(self.c["tagged"] / self.c["aligned"], 4) if self.c["aligned"] else None,
            "tagged_with_strongs": self.c["with_strongs"],
            "strongs": {k: self.outcome.get(k, 0) for k in vc.OUTCOMES},
        }
        out["strongs"]["scored"] = n
        if decided:
            out["strongs"]["strict_rate"] = round(self.outcome.get("agree", 0) / decided, 4)
            out["strongs"]["with_conventions_rate"] = round(
                (self.outcome.get("agree", 0) + self.outcome.get("convention", 0)) / decided, 4)
        out["by_range"] = dict(sorted(self.range_outcome.items()))
        return out

    def detailed(self) -> dict:
        out = self.core()
        out["rules"] = dict(self.rules.most_common())
        return out


def _new_stats(name):
    return NumberStats() if name == "abp" else vc.Stats()


def _add(stats, name, pair, res, detailed):
    if name == "abp":
        stats.add(res)
    else:
        vc.add(stats, pair, res, detailed=detailed)


def _number_difference(res: dict) -> list[tuple[str, str]]:
    if res and res["strongs"] == "disagree":
        return [("Strong's number differs", f"Strong's number differs ({res['range']})")]
    return []


# ---------------------------------------------------------------------------
# Reference against reference
# ---------------------------------------------------------------------------

ADAPTERS = {"ccat": vc.CcatAdapter, "ptnk": UdAdapter, "glaux": GlauxAdapter}
REF_PAIRS = (("ccat", "glaux"), ("ccat", "ptnk"), ("glaux", "ptnk"))
METRICS = (("lemma", "has_lemma"), ("pos", "has_morph"), ("full", "has_morph"))
SINGLE = {"exact", "variant", "elision"}


def ref_vs_ref(pair_a, name_a: str, pair_b, name_b: str) -> dict | None:
    """Compare two references' analyses of the same Brenton word with the
    same scorer and conventions used for us (reference A in the "ours" seat:
    its headword(s) and feature bundle against reference B's token)."""
    if pair_a.kind not in SINGLE or pair_b.kind not in SINGLE:
        return None
    A, B = ADAPTERS[name_a], ADAPTERS[name_b]
    ta, tb = pair_a.ccat[0], pair_b.ccat[0]
    res = {"lemma": "uncertain", "lemma_rule": "", "pos": "uncertain", "pos_rule": "",
           "full": "uncertain", "fields": {}, "second": None, "ccat_pos": "", "reason": "",
           "has_lemma": True, "has_morph": True, "unmapped_ours": False}
    try:
        fa = A.feats(ta)
    except vc.Unmapped:
        fa = None
    lemma_a = ", ".join(sorted(A.lemma_set(ta)))
    return vc.score_core(res, lemma_a, fa, tb, B)


class PairStats:
    """Per metric, on the words where us~A, us~B and A~B are all decided:
    how often each of the three agrees (strict, and with conventions)."""

    def __init__(self) -> None:
        self.n: Counter = Counter()
        self.agree: Counter = Counter()      # (metric, who) -> strict agreements
        self.conv: Counter = Counter()       # (metric, who) -> agreements incl. conventions
        self.skipped: Counter = Counter()
        self.ref_fields: Counter = Counter()  # field on which A and B differ (dev)

    def add(self, ra: dict, rb: dict, rr: dict, *, fields: bool = False) -> None:
        for metric, has in METRICS:
            if not (ra[has] and rb[has]):
                continue
            outs = {"us~A": ra[metric], "us~B": rb[metric], "A~B": rr[metric]}
            if "uncertain" in outs.values():
                self.skipped[metric] += 1
                continue
            self.n[metric] += 1
            for who, o in outs.items():
                self.agree[(metric, who)] += o == "agree"
                self.conv[(metric, who)] += o in ("agree", "convention")
        if fields:
            for name, (o, _rule, _a, _b) in rr["fields"].items():
                if o == "disagree":
                    self.ref_fields[name] += 1
            if rr["pos"] == "disagree":
                self.ref_fields["pos"] += 1

    def table(self) -> dict:
        out = {}
        for metric, _ in METRICS:
            n = self.n[metric]
            row = {"words": n, "uncertain_excluded": self.skipped[metric]}
            for who in ("us~A", "us~B", "A~B"):
                row[who] = ({"strict": round(self.agree[(metric, who)] / n, 4),
                             "with_conventions": round(self.conv[(metric, who)] / n, 4)} if n else None)
            out[{"full": "full_parse"}.get(metric, metric)] = row
        return out


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run(args) -> dict:
    reports = vc._check_reports_dir(Path(args.reports))
    module, tagged = vc.load_brenton(Path(args.lxx), Path(args.tagged) if args.tagged else None)
    decisions = vc.read_decisions(Path(args.decisions)) if (tagged and args.decisions) else {}
    tokenize = vc._tokenizer()
    names = [n for n in (args.refs.split(",") if args.refs else ALL_REFS) if n]
    refs, missing = load_references(names, Path(args.ccat_dir), Path(args.cache))
    wanted = set(args.books.split(",")) if args.books else None
    splits_wanted = {"dev", "test"} if args.split == "all" else {args.split}

    per = {r.name: {
        "coverage": {}, "texts": {},
        "split": {"dev": _new_stats(r.name), "test": _new_stats(r.name)},
        "tier": {"dev": defaultdict(lambda n=r.name: _new_stats(n)), "test": defaultdict(lambda n=r.name: _new_stats(n))},
        "conf": {"dev": defaultdict(lambda n=r.name: _new_stats(n)), "test": defaultdict(lambda n=r.name: _new_stats(n))},
        "tier_conf": {"dev": defaultdict(lambda n=r.name: _new_stats(n)),
                      "test": defaultdict(lambda n=r.name: _new_stats(n))},
        "dev_book": defaultdict(lambda n=r.name: _new_stats(n)),
        "dev_pos": defaultdict(lambda n=r.name: _new_stats(n)),
        "samples": defaultdict(list),
    } for r in refs}
    combined = {"dev": Counter(), "test": Counter()}
    three_way = {"dev": Counter(), "test": Counter()}
    pair_stats = {p: {"dev": PairStats(), "test": PairStats()} for p in REF_PAIRS}
    pair_book = {p: defaultdict(PairStats) for p in REF_PAIRS}
    pair_pos = {p: defaultdict(PairStats) for p in REF_PAIRS}
    neither = {p: {"dev": Counter(), "test": Counter()} for p in REF_PAIRS}
    all_disagree_dev = {"by_our_pos": Counter(), "by_tier": Counter(), "by_ccat_difference": Counter(),
                        "by_references": Counter()}
    present: set[str] = set()

    for book in module["books"]:
        bid = book["id"]
        if bid not in CCAT_TEXTS or (wanted and bid not in wanted):
            continue
        book_splits = {vc.split_of(bid, ch["chapter"]) for ch in book["chapters"]}
        if not (book_splits & splits_wanted):
            continue
        bverses = [BrentonVerse(v["ref"], ch["chapter"], v["v"], vc.build_units(v, tokenize, decisions))
                   for ch in book["chapters"] for v in ch["verses"]]
        per_unit: dict[int, dict] = defaultdict(dict)
        unit_pairs: dict[int, dict] = defaultdict(dict)
        unit_of: dict[int, object] = {}
        for ref in refs:
            texts = ref.texts_for(bid)
            if not texts:
                continue
            present |= book_splits & splits_wanted
            best, choice = None, {}
            for tname, verses in texts.items():
                res = align_book(bid, tname, bverses, verses)
                m = sum(res.kinds.values())
                choice[tname] = m
                if best is None or m > sum(best.kinds.values()):
                    best = res
            P = per[ref.name]
            P["coverage"][bid] = dict(best.summary(), split="/".join(sorted(book_splits)))
            if len(choice) > 1:
                P["texts"][bid] = {"chosen": best.text, "tokens_form_matched": choice}
            for pair in best.pairs:
                u = pair.unit
                sp = vc.split_of(bid, (u.ref // 1000) % 1000)
                if sp not in splits_wanted:
                    continue
                res = ref.score(pair)
                _add(P["split"][sp], ref.name, pair, res, sp == "dev")
                _add(P["tier"][sp][u.tier or "-"], ref.name, pair, res, False)
                if u.confidence is not None:
                    band = vc.conf_bucket(u.confidence)
                    _add(P["conf"][sp][band], ref.name, pair, res, False)
                    _add(P["tier_conf"][sp][f"{u.tier or '-'} x {band}"], ref.name, pair, res, False)
                if sp == "dev":
                    _add(P["dev_book"][bid], ref.name, pair, res, True)
                    if res is not None and ref.name != "abp":
                        _add(P["dev_pos"][res["ccat_pos"] or "n/a"], ref.name, pair, res, True)
                    if res is not None:
                        cats = _number_difference(res) if ref.name == "abp" else vc.difference_categories(res)
                        for cat, desc in cats[:1]:
                            P["samples"][cat].append((u, bid, desc))
                if res is not None:
                    per_unit[id(u)][ref.name] = res
                    unit_pairs[id(u)][ref.name] = pair
                    unit_of[id(u)] = u
        # combined summary over this book's words
        for uid, by_ref in per_unit.items():
            u = unit_of[uid]
            sp = vc.split_of(bid, (u.ref // 1000) % 1000)
            for a, b in REF_PAIRS:
                if a not in by_ref or b not in by_ref:
                    continue
                rr = ref_vs_ref(unit_pairs[uid][a], a, unit_pairs[uid][b], b)
                if rr is None:
                    continue
                ra, rb = by_ref[a], by_ref[b]
                pair_stats[(a, b)][sp].add(ra, rb, rr, fields=sp == "dev")
                if sp == "dev":
                    pair_book[(a, b)][bid].add(ra, rb, rr)
                    pair_pos[(a, b)][rr.get("ours_pos") or "n/a"].add(ra, rb, rr)
                pa, pb, pr = primary(a, ra), primary(b, rb), primary("ref", rr)
                if pa == "disagree" and pb == "disagree":
                    neither[(a, b)][sp][{"agree": "references agree with each other",
                                         "convention": "references agree with each other",
                                         "disagree": "references differ from each other",
                                         "uncertain": "reference comparison uncertain"}[pr]] += 1
            decided = {n: o for n, o in ((n, primary(n, r)) for n, r in by_ref.items()) if o and o != "uncertain"}
            if "ccat" in decided and "glaux" in decided:
                c_ok = decided["ccat"] != "disagree"
                g_ok = decided["glaux"] != "disagree"
                three_way[sp][{(True, True): "agree with both", (True, False): "agree with CCAT only",
                               (False, True): "agree with GLAUx only",
                               (False, False): "agree with neither"}[(c_ok, g_ok)]] += 1
            if len(decided) < 2:
                continue
            combined[sp]["words in 2+ references"] += 1
            combined[sp][f"words in {len(decided)} references"] += 1
            oks = [o != "disagree" for o in decided.values()]
            if all(oks):
                combined[sp]["every reference agrees"] += 1
            elif not any(oks):
                combined[sp]["every reference disagrees"] += 1
                if sp == "dev":
                    any_res = next((r for n, r in by_ref.items() if n != "abp"), {})
                    all_disagree_dev["by_our_pos"][any_res.get("ours_pos", "?")] += 1
                    all_disagree_dev["by_tier"][u.tier or "-"] += 1
                    all_disagree_dev["by_references"]["+".join(sorted(decided))] += 1
                    if "ccat" in by_ref:
                        cats = vc.difference_categories(by_ref["ccat"])
                        all_disagree_dev["by_ccat_difference"][cats[0][0] if cats else "none"] += 1
            else:
                combined[sp]["references split"] += 1

    report = {
        "meta": {"harness": "validate_refs.py", "tagged_input": bool(tagged),
                 "decisions_input": bool(decisions), "references": [r.name for r in refs],
                 "missing_references": missing, "splits": sorted(present),
                 "note": "References are used only to measure; nothing here may feed the tagger."},
        "conventions": CONVENTIONS, "number_conventions": NUMBER_CONVENTIONS,
        "references": {},
    }
    for ref in refs:
        P = per[ref.name]
        entry = {"alignment": {"per_book": P["coverage"], "overall": vc._coverage_total(P["coverage"])},
                 "double_texts": P["texts"]}
        if "dev" in present and P["dev_book"]:
            entry["dev"] = {
                "overall": P["split"]["dev"].detailed(),
                "by_tier": {k: v.core() for k, v in sorted(P["tier"]["dev"].items())},
                "by_confidence": {k: v.core() for k, v in sorted(P["conf"]["dev"].items())},
                "by_tier_and_confidence": {k: v.core() for k, v in sorted(P["tier_conf"]["dev"].items())},
                "per_book": {b: s.detailed() for b, s in P["dev_book"].items()},
            }
            if ref.name != "abp":
                entry["dev"]["by_ref_pos"] = {k: v.detailed() for k, v in sorted(P["dev_pos"].items())}
            if args.samples:
                entry["dev"]["samples"] = vc.draw_samples(P["samples"], args.samples, args.seed)
        if "test" in present and P["split"]["test"].c["aligned"]:
            test = {
                "overall": P["split"]["test"].core(),
                "by_tier": {k: v.core() for k, v in sorted(P["tier"]["test"].items())},
                "by_confidence": {k: v.core() for k, v in sorted(P["conf"]["test"].items())},
                "by_tier_and_confidence": {k: v.core() for k, v in sorted(P["tier_conf"]["test"].items())},
            }
            vc.assert_aggregate_only(test, f"{ref.name} test split")
            entry["test"] = test
        report["references"][ref.name] = entry
    report["reference_agreement"] = {}
    for (a, b) in REF_PAIRS:
        key = f"{a}~{b}"
        entry = {"note": f"A = {a}, B = {b}; same words for all three columns"}
        for sp in ("dev", "test"):
            st = pair_stats[(a, b)][sp]
            if sp not in present or not sum(st.n.values()):
                continue
            block = {"table": st.table(), "we_agree_with_neither": dict(neither[(a, b)][sp])}
            if sp == "dev":
                block["per_book"] = {bk: v.table() for bk, v in sorted(pair_book[(a, b)].items())}
                block["by_A_pos"] = {k: v.table() for k, v in sorted(pair_pos[(a, b)].items())}
                block["A_B_differ_on"] = dict(st.ref_fields.most_common())
            else:
                vc.assert_aggregate_only(block, f"{key} test split")
            entry[sp] = block
        if len(entry) > 1:
            report["reference_agreement"][key] = entry
    report["combined"] = {}
    for sp in ("dev", "test"):
        if sp not in present:
            continue
        block = {"counts": dict(combined[sp]), "glaux_three_way": dict(three_way[sp])}
        n = combined[sp]["words in 2+ references"]
        if n:
            block["share_every_reference_agrees"] = round(combined[sp]["every reference agrees"] / n, 4)
            block["share_every_reference_disagrees"] = round(combined[sp]["every reference disagrees"] / n, 4)
        if sp == "dev":
            block["every_reference_disagrees"] = {k: dict(v.most_common()) for k, v in all_disagree_dev.items()}
        else:
            vc.assert_aggregate_only(block, "combined test split")
        report["combined"][sp] = block

    text = render(report)
    reports.mkdir(parents=True, exist_ok=True)
    stem = args.name or "refs_validation"
    (reports / f"{stem}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    (reports / f"{stem}.txt").write_text(text, encoding="utf-8")
    print(text)
    return report


def _number_line(name: str, core: dict) -> str:
    s = core["strongs"]
    return (f"  {name:<22} aligned {core['aligned_tokens']:>7}  recall {vc._pct(core['recall'])}  "
            f"Strong's {vc._pct(s.get('strict_rate'))} / {vc._pct(s.get('with_conventions_rate'))}  "
            f"agree {s['agree']} conv {s['convention']} unc {s['uncertain']} dis {s['disagree']}")


def render(report: dict) -> str:
    lines = ["Reference validation (private; scoring only)", ""]
    m = report["meta"]
    lines.append(f"tagged input: {m['tagged_input']}   decisions: {m['decisions_input']}   "
                 f"references: {', '.join(m['references'])}   splits: {', '.join(m['splits'])}")
    if m["missing_references"]:
        lines.append(f"missing: {m['missing_references']} (run --fetch)")
    for name, e in report["references"].items():
        line = _number_line if name == "abp" else vc._core_line
        tot = e["alignment"]["overall"]
        lines += ["", f"== {name.upper()} =="]
        if tot.get("brenton_tokens"):
            lines.append(f"  alignment: verses {tot['aligned_verses']}/{tot['brenton_verses']} "
                         f"({vc._pct(tot.get('verse_coverage'))}), tokens {tot['tokens_form_matched']}/"
                         f"{tot['brenton_tokens']} ({vc._pct(tot.get('token_coverage'))})")
        weakest = sorted(((c["tokens_form_matched"] / c["brenton_tokens"], b)
                          for b, c in e["alignment"]["per_book"].items() if c["brenton_tokens"]))[:6]
        if weakest:
            lines.append("  weakest books: " + ", ".join(f"{b} {vc._pct(x).strip()}" for x, b in weakest))
        for b, c in e["double_texts"].items():
            lines.append(f"  {b}: Brenton matches {c['chosen']} {c['tokens_form_matched']}")
        for sp in ("dev", "test"):
            if sp not in e:
                continue
            d = e[sp]
            block = [f"  {sp.upper()}{' (aggregate only)' if sp == 'test' else ''}",
                     line("overall", d["overall"])]
            if sp == "dev":
                for b, s in d["per_book"].items():
                    block.append(line(b, s))
            for key, title in (("by_tier", "by tier"), ("by_confidence", "by confidence"),
                               ("by_tier_and_confidence", "by tier x confidence")):
                if d.get(key):
                    block.append(f"   {title}:")
                    block += [line(k, s) for k, s in d[key].items()]
            if sp == "dev" and name == "abp":
                block.append("   by range: " + ", ".join(f"{k} {v}" for k, v in d["overall"]["by_range"].items()))
                block.append("   rules: " + ", ".join(f"{k} {v}" for k, v in d["overall"]["rules"].items()))
            if sp == "dev" and name != "abp":
                block.append("   by reference POS:")
                block += [line(k, s) for k, s in d.get("by_ref_pos", {}).items()]
                o = d["overall"]
                if o.get("convention_rules"):
                    block.append("   conventions: " + ", ".join(f"{k} {v}" for k, v in o["convention_rules"].items()))
                if o.get("uncertain_reasons"):
                    block.append("   uncertain: " + ", ".join(f"{k} {v}" for k, v in o["uncertain_reasons"].items()))
                for field, conf in o.get("confusion", {}).items():
                    off = [(k, v) for k, v in conf.items() if k.split(" / ")[0][5:] != k.split(" / ")[1][7:]]
                    if off:
                        block.append(f"   {field} confusions: " + ", ".join(f"{k} {v}" for k, v in off[:6]))
            if sp == "dev" and d.get("samples"):
                block.append("   samples (our tag; category of difference only):")
                for cat, items in d["samples"].items():
                    for it in items:
                        block.append(f"     {cat}: {it['book']} {it['ref']} {it['surface']} {it['strongs']} "
                                     f"{it['lemma']} {it['morph']} [{it['tier']}] -> {it['difference']}")
            if sp == "test":
                vc.assert_aggregate_only("\n".join(block), f"{name} test summary")
            lines += block
    for key, entry in report.get("reference_agreement", {}).items():
        for sp in ("dev", "test"):
            if sp not in entry:
                continue
            blk = entry[sp]
            out = ["", f"== REFERENCE AGREEMENT {key} ({sp}{', aggregate only' if sp == 'test' else ''}) "
                       f"[{entry['note']}] =="]
            out.append(f"  {'metric':<11}{'words':>8}   {'us~A':>15}   {'us~B':>15}   {'A~B':>15}   (strict / conv)")
            for metric, row in blk["table"].items():
                if not row["words"]:
                    continue
                cells = [f"{vc._pct(row[w]['strict']).strip():>6} / {vc._pct(row[w]['with_conventions']).strip():>6}"
                         for w in ("us~A", "us~B", "A~B")]
                out.append(f"  {metric:<11}{row['words']:>8}   " + "   ".join(cells)
                           + f"   (uncertain excluded {row['uncertain_excluded']})")
            if blk["we_agree_with_neither"]:
                out.append("  we agree with neither: " + ", ".join(
                    f"{k} {v}" for k, v in sorted(blk["we_agree_with_neither"].items())))
            if sp == "dev":
                if blk.get("A_B_differ_on"):
                    out.append("  A and B differ on: " + ", ".join(f"{k} {v}" for k, v in blk["A_B_differ_on"].items()))
                for title, sub in (("per book", blk.get("per_book", {})), ("by A's POS", blk.get("by_A_pos", {}))):
                    rows = []
                    for k, t in sub.items():
                        fp = t["full_parse"]
                        if fp["words"]:
                            rows.append(f"{k} {fp['words']}: " + "/".join(
                                vc._pct(fp[w]["with_conventions"]).strip() for w in ("us~A", "us~B", "A~B")))
                    if rows:
                        out.append(f"  full parse with conventions, us~A/us~B/A~B, {title}: " + "; ".join(rows))
            else:
                vc.assert_aggregate_only("\n".join(out), f"{key} test summary")
            lines += out
    for sp, block in report["combined"].items():
        out = ["", f"== COMBINED ({sp}{', aggregate only' if sp == 'test' else ''}) =="]
        out.append("  " + ", ".join(f"{k} {v}" for k, v in sorted(block["counts"].items())))
        if "share_every_reference_agrees" in block:
            out.append(f"  every reference agrees with us: {vc._pct(block['share_every_reference_agrees'])}; "
                       f"every reference disagrees: {vc._pct(block['share_every_reference_disagrees'])}")
        if block["glaux_three_way"]:
            out.append("  CCAT x GLAUx: " + ", ".join(f"{k} {v}" for k, v in sorted(block["glaux_three_way"].items())))
        if sp == "dev":
            for k, v in block.get("every_reference_disagrees", {}).items():
                out.append(f"  all-disagree {k}: " + ", ".join(f"{a} {b}" for a, b in list(v.items())[:10]))
        else:
            vc.assert_aggregate_only("\n".join(out), "combined test summary")
        lines += out
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--refs", default=",".join(ALL_REFS))
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--ccat-dir", default=str(vc.DEFAULT_CCAT))
    ap.add_argument("--lxx", default=str(vc.DEFAULT_LXX))
    ap.add_argument("--tagged", default=str(vc.DEFAULT_TAGGED))
    ap.add_argument("--decisions", default=str(vc.DEFAULT_DECISIONS))
    ap.add_argument("--reports", default=str(vc.DEFAULT_REPORTS))
    ap.add_argument("--name", default="")
    ap.add_argument("--split", choices=("all", "dev", "test"), default="all")
    ap.add_argument("--books", default="")
    ap.add_argument("--samples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--fetch", default="", help="comma-separated references to download, then exit")
    args = ap.parse_args(argv)
    if args.fetch:
        print(fetch_refs([x for x in args.fetch.split(",") if x]))
        return 0
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
