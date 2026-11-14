"""Tests for the private CCAT validation harness (scripts/lxx_tagging/validate_ccat.py).

Every fixture here is synthetic: a few very common Greek words in CCAT's
*file format*, tagged by hand from ordinary grammar.  No CCAT data is used and
the suite runs with ``.cache/lxx_validation/`` absent.
"""

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "scripts" / "lxx_tagging"
sys.path.insert(0, str(PKG))

from ccat_align import BrentonVerse, Unit, align_book, align_words, link_verses  # noqa: E402
from ccat_codes import (  # noqa: E402
    Unmapped,
    code_shape,
    compare_feats,
    compare_lemma,
    decompose_ccat,
    decompose_robinson,
)
from ccat_greek import beta_to_unicode, contains_greek, form_key, strip_greek  # noqa: E402
from ccat_reader import CCAT_TEXTS, parse_mlxx  # noqa: E402
import validate_ccat  # noqa: E402

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
REF = re.compile(r"\b\d{1,3}:\d{1,3}\b|\b\d{7,8}\b")


def line(form, type_code, parse, lemma, *prefixes):
    """One word line in CCAT's fixed columns."""
    out = f"{form:<25}{type_code:<4}{parse:<7}{lemma:<17}"
    out += "".join(f"{p:<6}" for p in prefixes)
    return out.rstrip()


def mlxx(book, verses):
    """verses: list of (label, [line args])."""
    chunks = []
    for label, words in verses:
        chunks.append(f"{book} {label}")
        chunks.extend(line(*w) for w in words)
        chunks.append("")
    return "\n".join(chunks) + "\n"


# A tiny invented "book": the same three verses on both sides, with CCAT
# numbering one verse ahead and one Brenton verse covering two CCAT verses.
CCAT_VERSES = [
    ("1:1", [("KAI\\", "C", "", "KAI/"), ("E)/LUSEN", "VAI", "AAI3S", "LU/W"),
             ("O(", "RA", "NSM", "O("), ("A)/NQRWPOS", "N2", "NSM", "A)/NQRWPOS"),
             ("TO\\N", "RA", "ASM", "O("), ("DOU=LON", "N2", "ASM", "DOU=LOS")]),
    ("1:2", [("KAI\\", "C", "", "KAI/"), ("A)PE/LUSEN", "VAI", "AAI3S", "LU/W", "A)PO"),
             ("AU)TO\\N", "RD", "ASM", "AU)TO/S"), ("E)PI\\", "P", "", "E)PI/"),
             ("TH\\N", "RA", "ASF", "O("), ("GH=N", "N1", "ASF", "GH=")]),
    ("1:3", [("KAI\\", "C", "", "KAI/"), ("OU)K", "D", "", "OU)"),
             ("H)=N", "V9", "IAI3S", "EI)MI/"), ("LO/GOS", "N2", "NSM", "LO/GOS")]),
    ("1:3a", [("E)GE/NETO", "VBI", "AMI3S", "GI/GNOMAI"), ("DE\\", "X", "", "DE/"),
              ("FWNH\\", "N1", "NSF", "FWNH/")]),
]
BRENTON = [
    ("καὶ ἔλυσεν ὁ ἄνθρωπος τὸν δοῦλον.",
     [("G2532", "καί", "CONJ"), ("G3089", "λύω", "V-AAI-3S"), ("G3588", "ὁ", "T-NSM"),
      ("G444", "ἄνθρωπος", "N-NSM"), ("G3588", "ὁ", "T-ASM"), ("G1401", "δοῦλος", "N-ASM")]),
    ("καὶ ἀπέλυσεν αὐτὸν ἐπὶ τὴν γῆν.",
     [("G2532", "καί", "CONJ"), ("G630", "ἀπολύω", "V-AAI-3S"), ("G846", "αὐτός", "P-ASM"),
      ("G1909", "ἐπί", "PREP"), ("G3588", "ὁ", "T-ASF"), ("G1093", "γῆ", "N-GSF")]),
    ("καὶ οὐκ ἦν λόγος· ἐγένετο δὲ φωνή.",
     [("G2532", "καί", "CONJ"), ("G3756", "οὐ", "PRT-N"), ("G1510", "εἰμί", "V-IAI-3S"),
      ("G3056", "λόγος", "N-NSM"), ("G1096", "γίνομαι", "V-2ADI-3S"), ("G1161", "δέ", "CONJ"),
      ("G5456", "φωνή", "N-NSF")]),
]


def tagged_module(book_id, number, verses=BRENTON):
    from ccat_greek import tokenize_brenton

    out = []
    for v, (text, tags) in enumerate(verses, start=1):
        spans = tokenize_brenton(text)
        anns = [
            {"type": "strongs", "start": s, "end": e, "text": w,
             "data": {"strongs": st, "lemma": lem, "morphology": morph}}
            for (s, e, w), (st, lem, morph) in zip(spans, tags)
        ]
        out.append({"v": v, "ref": number * 1000000 + 1000 + v,
                    "content": {"text": text, "annotations": anns}})
    return {"meta": {"id": "LXX"}, "books": [
        {"id": book_id, "number": number, "chapters": [{"chapter": 1, "verses": out}]}]}


class BetaCodeTests(unittest.TestCase):
    def test_letters_breathings_accents_and_subscript(self):
        self.assertEqual(beta_to_unicode("A)RXH=|"), "ἀρχῇ")
        self.assertEqual(beta_to_unicode("QEO\\S"), "θεὸς")
        self.assertEqual(beta_to_unicode("LO/GOS"), "λόγος")

    def test_capitals_take_marks_before_or_after_the_letter(self):
        self.assertEqual(beta_to_unicode("*)ABRAA/M"), "Ἀβραάμ")
        self.assertEqual(beta_to_unicode("*A)BRAA/M"), "Ἀβραάμ")
        self.assertEqual(beta_to_unicode("*(/ARMA"), "Ἅρμα")

    def test_diaeresis_elision_and_final_sigma(self):
        self.assertEqual(beta_to_unicode("PRWI/+"), "πρωΐ")
        self.assertEqual(beta_to_unicode("E)P'"), "ἐπ’")
        self.assertEqual(beta_to_unicode("PA=S KAI\\"), "πᾶς καὶ")

    def test_strip_and_form_keys(self):
        self.assertEqual(strip_greek("Ἐφʼ"), "εφ")
        self.assertEqual(strip_greek("ᾠδῇ"), "ωδη")
        self.assertEqual(form_key("ἐποίησεν"), form_key("ἐποίησε"))
        self.assertEqual(form_key("ἐφʼ"), form_key("ἐπ’"))
        self.assertEqual(form_key("οὐχ"), form_key("οὐκ"))
        self.assertEqual(form_key("ἐξ"), form_key("ἐκ"))
        self.assertNotEqual(form_key("ἐν"), form_key("ἕν") + "x")


class ReaderTests(unittest.TestCase):
    def test_verses_columns_preverbs_and_positional_parse(self):
        text = mlxx("Exod", CCAT_VERSES[:2]) + "Exod 2:1\n" + line("*ABEL", "N", " S", "*ABEL") + "\n"
        verses = parse_mlxx(text)
        self.assertEqual([v.label for v in verses], ["1:1", "1:2", "2:1"])
        self.assertEqual((verses[0].chapter, verses[0].verse), (1, 1))
        t = verses[1].tokens[1]
        self.assertEqual((t.type_code, t.parse_code, t.lemma_beta, t.prefixes),
                         ("VAI", "AAI3S", "LU/W", ("A)PO",)))
        name = verses[2].tokens[0]
        self.assertEqual(name.parse_code, " S")  # number only, case unknown
        f = decompose_ccat(name.type_code, name.parse_code, name.lemma_beta)
        self.assertEqual((f.pos, f.case, f.number, f.proper), ("noun", "", "S", True))

    def test_lettered_verse_crasis_and_one_chapter_books(self):
        text = ("Obad 1\n" + line("KA)GW\\", "C+RPNS", "", "KAI/+E)GW/") + "\n\n"
                + "Obad 1a\n" + line("KAI\\", "C", "", "KAI/") + "\n")
        verses = parse_mlxx(text)
        self.assertEqual([(v.chapter, v.verse, v.letter) for v in verses], [(1, 1, ""), (1, 1, "a")])
        tok = verses[0].tokens[0]
        self.assertEqual((tok.type_code, tok.parse_code, tok.lemma_beta), ("C+RP", "NS", "KAI/+E)GW/"))
        f = decompose_ccat(tok.type_code, tok.parse_code, tok.lemma_beta)
        self.assertEqual([p.pos for p in f.parts], ["conj", "pron"])
        self.assertEqual((f.parts[1].case, f.parts[1].person), ("N", "1"))


class CodeTests(unittest.TestCase):
    def test_ccat_verbs(self):
        f = decompose_ccat("VAI", "AAI3S")
        self.assertEqual((f.pos, f.tense, f.voice, f.mood, f.person, f.number, f.second),
                         ("verb", "A", "A", "I", "3", "S", False))
        f = decompose_ccat("VBI", "AMI3S")
        self.assertTrue(f.second)
        f = decompose_ccat("V1", "PAPNSM")
        self.assertEqual((f.mood, f.case, f.number, f.gender), ("P", "N", "S", "M"))
        f = decompose_ccat("VA", "AAN")
        self.assertEqual((f.mood, f.person), ("N", ""))
        f = decompose_ccat("VX", "XAI1S")
        self.assertEqual(f.tense, "R")
        f = decompose_ccat("V1", "PAD2P")
        self.assertEqual(f.mood, "M")

    def test_ccat_nominals_and_pronouns(self):
        f = decompose_ccat("N2", "DSM", "LO/GOS")
        self.assertEqual((f.pos, f.case, f.number, f.gender), ("noun", "D", "S", "M"))
        f = decompose_ccat("RP", "GS", "E)GW/")
        self.assertEqual((f.pos, f.sub, f.person, f.gender), ("pron", "P", "1", ""))
        f = decompose_ccat("RD", "GSM", "AU)TO/S")
        self.assertEqual(f.sub, "P")
        f = decompose_ccat("A1", "ASNC")
        self.assertEqual(f.degree, "C")
        self.assertEqual(decompose_ccat("D", "").pos, "adv")
        self.assertEqual(decompose_ccat("M", "").pos, "num")
        with self.assertRaises(Unmapped):
            decompose_ccat("V1", "PQI3S")
        with self.assertRaises(Unmapped):
            decompose_ccat("Q", "")

    def test_code_shape_hides_letters(self):
        self.assertEqual(code_shape("VAI", "AAI3S"), "Vxx|xxx9x")
        self.assertEqual(code_shape("N", " S"), "N|_x")

    def test_robinson_codes(self):
        f = decompose_robinson("V-2AAP-NSM")
        self.assertEqual((f.tense, f.voice, f.mood, f.case, f.second), ("A", "A", "P", "N", True))
        f = decompose_robinson("V-PNI-3S")
        self.assertEqual((f.voice, f.sub, f.person), ("MP", "deponent", "3"))
        f = decompose_robinson("V-ADI-3S")
        self.assertEqual(f.voice, "M")
        f = decompose_robinson("N-GSM-P")
        self.assertEqual((f.pos, f.case, f.proper), ("noun", "G", True))
        f = decompose_robinson("P-1GS")
        self.assertEqual((f.pos, f.sub, f.person, f.case, f.number), ("pron", "P", "1", "G", "S"))
        f = decompose_robinson("S-1SGSN")
        self.assertEqual((f.sub, f.person, f.case, f.gender), ("S", "1", "G", "N"))
        f = decompose_robinson("F-3ASM")
        self.assertEqual((f.sub, f.person, f.case), ("F", "3", "A"))
        self.assertEqual(decompose_robinson("A-NUI").pos, "num")
        self.assertEqual(decompose_robinson("A-ASN-C").degree, "C")
        self.assertEqual(decompose_robinson("PRT-N").pos, "ptcl")
        self.assertEqual(decompose_robinson("COND").pos, "conj")
        f = decompose_robinson("P-1NS + G2532=CONJ")
        self.assertEqual([p.pos for p in f.parts], ["pron", "conj"])
        with self.assertRaises(Unmapped):
            decompose_robinson("V-XYZ")

    def test_field_comparison_and_conventions(self):
        # M/P in the present: forms identical -> convention
        cmp = compare_feats(decompose_robinson("V-PPI-3S"), decompose_ccat("V1", "PMI3S"), "agree")
        self.assertEqual(cmp.fields["voice"][0], "convention")
        self.assertEqual(cmp.full, "convention")
        # M/P in the aorist: different forms -> disagree
        cmp = compare_feats(decompose_robinson("V-API-3S"), decompose_ccat("VAI", "AMI3S"), "agree")
        self.assertEqual(cmp.fields["voice"][0], "disagree")
        # deponent labels are notational
        cmp = compare_feats(decompose_robinson("V-2ADI-3S"), decompose_ccat("VBI", "AMI3S"), "agree")
        self.assertEqual(cmp.full, "agree")
        # case difference is a disagreement, reported field by field
        cmp = compare_feats(decompose_robinson("N-GSF"), decompose_ccat("N1", "ASF"), "agree")
        self.assertEqual(cmp.fields["case"], ("disagree", "", "G", "A"))
        # particle vs adverb for the same word: indeclinable class convention
        cmp = compare_feats(decompose_robinson("PRT-N"), decompose_ccat("D", ""), "agree")
        self.assertEqual((cmp.pos, cmp.pos_rule), ("convention", "indeclinable class"))
        # ...but not when the lemma differs
        cmp = compare_feats(decompose_robinson("PRT-N"), decompose_ccat("D", ""), "disagree")
        self.assertEqual(cmp.pos, "disagree")
        # dual number cannot be compared
        cmp = compare_feats(decompose_robinson("N-NPM"), decompose_ccat("N2", "NDM"), "agree")
        self.assertEqual(cmp.full, "uncertain")

    def test_lemmas(self):
        self.assertEqual(compare_lemma("λόγος", "LO/GOS")[0], "agree")
        self.assertEqual(compare_lemma("ἀπολύω", "LU/W", ("A)PO",))[0], "agree")
        self.assertEqual(compare_lemma("ἀφίστημι", "I(/STHMI", ("A)PO",))[0], "agree")
        self.assertEqual(compare_lemma("καθίζω", "I(/ZW", ("KATA",))[0], "agree")
        self.assertEqual(compare_lemma("ἐξανατέλλω", "TE/LLW", ("E)K", "A)NA"))[0], "agree")
        self.assertEqual(compare_lemma("συμβαίνω", "BAI/NW", ("SUN",))[0], "agree")
        self.assertEqual(compare_lemma("γίνομαι", "GI/GNOMAI"), ("convention", "Attic/Koine spelling"))
        self.assertEqual(compare_lemma("ἡμεῖς", "E)GW/"), ("convention", "pronoun headword"))
        self.assertEqual(compare_lemma("ἔπω, ἐρῶ, εἶπον", "EI)=PON")[0], "agree")
        self.assertEqual(compare_lemma("θάλασσα", "QA/LATTA"), ("convention", "Attic/Koine spelling"))
        self.assertEqual(compare_lemma("κἀγώ", "KAI/+E)GW/"), ("convention", "crasis headword"))
        self.assertEqual(compare_lemma("λόγος", "LU/W")[0], "disagree")


class AlignmentTests(unittest.TestCase):
    def _brenton(self):
        from ccat_greek import tokenize_brenton

        out = []
        for v, (text, _tags) in enumerate(BRENTON, start=1):
            units = [Unit(2001000 + v, i, s, e, w) for i, (s, e, w) in enumerate(tokenize_brenton(text))]
            out.append(BrentonVerse(2001000 + v, 1, v, units))
        return out

    def test_verse_links_follow_content_not_numbers(self):
        cverses = parse_mlxx(mlxx("Exod", CCAT_VERSES))
        # renumber CCAT one ahead: Brenton 1 -> CCAT "1:2" etc.
        for v in cverses:
            v.verse += 1
        links = link_verses(self._brenton(), cverses)
        self.assertEqual([sorted(l) for _, l, _ in links], [[0], [1], [2, 3]])

    def test_words_movable_nu_elision_and_joined(self):
        a = [form_key(w) for w in ["καὶ", "ἐποίησε", "ἐπʼ", "αὐτῷ", "διατοῦτο"]]
        b = [form_key(beta_to_unicode(w)) for w in ["KAI\\", "E)POI/HSEN", "E)P'", "AU)TW=|", "DIA\\", "TOU=TO"]]
        groups = align_words(a, b)
        self.assertEqual(groups[:4], [([0], [0], "exact"), ([1], [1], "exact"), ([2], [2], "exact"),
                                      ([3], [3], "exact")])
        self.assertEqual(groups[4], ([4], [4, 5], "joined"))

    def test_only_matching_forms_are_paired(self):
        a = [form_key(w) for w in ["καὶ", "εἶδε", "τὸν", "οἶκον"]]
        b = [form_key(beta_to_unicode(w)) for w in ["KAI\\", "E)POI/HSE", "TO\\N", "OI)=KON"]]
        groups = align_words(a, b)
        self.assertEqual([g[0][0] for g in groups], [0, 2, 3])

    def test_book_alignment_counts(self):
        cverses = parse_mlxx(mlxx("Exod", CCAT_VERSES))
        res = align_book("Exod", "Exod", self._brenton(), cverses)
        self.assertEqual(res.aligned_verses, 3)
        self.assertEqual(sum(res.kinds.values()), 19)  # every word pairs; οὐκ/οὐ by key


class FirewallTests(unittest.TestCase):
    def _run(self, book_id, number, ccat_file, label_book, samples=5, verses=BRENTON):
        tmp = Path(tempfile.mkdtemp())
        ccat = tmp / "ccat"
        ccat.mkdir()
        (ccat / ccat_file).write_text(mlxx(label_book, CCAT_VERSES), encoding="latin-1")
        lxx = tmp / "LXX_tagged.json"
        lxx.write_text(json.dumps(tagged_module(book_id, number, verses), ensure_ascii=False), encoding="utf-8")
        reports = tmp / "reports"
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = validate_ccat.main([
                "--ccat-dir", str(ccat), "--lxx", str(lxx), "--tagged", str(lxx),
                "--decisions", str(tmp / "none.tsv"), "--reports", str(reports),
                "--samples", str(samples)])
        self.assertEqual(code, 0)
        files = {p.name: p.read_text(encoding="utf-8") for p in reports.iterdir()}
        return out.getvalue(), files

    def test_test_split_prints_no_forms_lemmas_or_refs(self):
        self.assertEqual(validate_ccat.split_of("Exod", 1), "test")
        stdout, files = self._run("Exod", 2, "03.Exod.mlxx", "Exod")
        for text in [stdout] + list(files.values()):
            self.assertIsNone(GREEK.search(text), "Greek text in a test-split report")
            self.assertIsNone(REF.search(text), "verse reference in a test-split report")
        report = json.loads(files["ccat_validation.json"])
        self.assertNotIn("dev", report)
        self.assertNotIn("samples", json.dumps(report))
        self.assertEqual(report["test"]["overall"]["aligned_tokens"], 19)
        # the deliberate case error (γῆν tagged genitive) is counted, not shown
        self.assertEqual(report["test"]["overall"]["full_parse"]["disagree"], 1)

    def test_dev_samples_show_our_tag_and_the_category_only(self):
        self.assertEqual(validate_ccat.split_of("Lev", 1), "dev")
        stdout, files = self._run("Lev", 3, "04.Lev.mlxx", "Lev")
        self.assertIn("case differs: ours G, theirs A", stdout)
        report = json.loads(files["ccat_validation.json"])
        samples = report["dev"]["samples"]
        self.assertEqual(len(samples["case differs"]), 1)
        # CCAT's own codes never appear, only categories
        blob = json.dumps(samples)
        self.assertNotIn("N1", blob)
        self.assertNotIn("ASF", blob)

    def test_dev_lemma_samples_never_show_ccat_lemma(self):
        verses = [list(v) for v in BRENTON]
        tags = list(verses[0][1])
        tags[1] = ("G3068", "λούω", "V-AAI-3S")  # wrong lemma for ἔλυσεν
        verses[0] = (verses[0][0], tags)
        stdout, files = self._run("Lev", 3, "04.Lev.mlxx", "Lev", verses=verses)
        self.assertIn("lemma differs", stdout)
        self.assertNotIn("λύω", stdout)  # CCAT's headword
        sample_text = json.dumps(json.loads(files["ccat_validation.json"])["dev"]["samples"], ensure_ascii=False)
        self.assertNotIn("λύω", sample_text)
        self.assertIn("λούω", sample_text)      # ours

    def test_aggregate_guard_rejects_forms_and_refs(self):
        with self.assertRaises(validate_ccat.FirewallError):
            validate_ccat.assert_aggregate_only({"x": "λόγος"}, "t")
        with self.assertRaises(validate_ccat.FirewallError):
            validate_ccat.assert_aggregate_only("seen at 3:14", "t")
        with self.assertRaises(validate_ccat.FirewallError):
            validate_ccat.assert_aggregate_only({"ref": 2003014}, "t")
        validate_ccat.assert_aggregate_only({"lemma": {"agree": 3}, "rate": 0.9731}, "t")

    def test_reports_never_go_where_the_tagger_reads(self):
        for name in ("outputs", "scripts", "sources"):
            with self.assertRaises(validate_ccat.FirewallError):
                validate_ccat._check_reports_dir(ROOT / name / "lxx_validation")

    def test_nothing_in_the_tagger_imports_the_harness(self):
        harness = {"validate_ccat", "ccat_greek", "ccat_reader", "ccat_align", "ccat_codes",
                   "validate_refs", "ref_common", "ref_abp", "ref_ud", "ref_glaux"}
        pattern = re.compile(r"^\s*(?:from|import)\s+(?:\.|lxx_tagging\.)?(\w+)", re.M)
        for path in PKG.glob("*.py"):
            if path.stem in harness:
                continue
            source = path.read_text(encoding="utf-8")
            imported = set(pattern.findall(source))
            self.assertFalse(imported & harness, f"{path.name} imports the CCAT harness")
            for name in ("ccat", "abp", "ptnk", "glaux"):
                self.assertNotIn(f"lxx_validation/{name}", source, f"{path.name} refers to reference files")

    def test_split_is_a_partition(self):
        self.assertFalse(validate_ccat.DEV & validate_ccat.TEST)
        books = set(CCAT_TEXTS)
        self.assertEqual(validate_ccat.DEV | validate_ccat.TEST | {"Ps"}, books)
        self.assertEqual(validate_ccat.split_of("Ps", 75), "dev")
        self.assertEqual(validate_ccat.split_of("Ps", 76), "test")

    def test_no_greek_check_helper(self):
        self.assertTrue(contains_greek("ἀρχή"))
        self.assertFalse(contains_greek("N1 ASF 12.5%"))


if __name__ == "__main__":
    unittest.main()
