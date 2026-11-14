"""Tests for the private reference-set scorers (validate_refs.py, ref_*.py).

All fixtures are invented: a few common Greek words in each reference's
*file format* (a SWORD zText module, CoNLL-U, GLAUx XML), annotated by hand
from ordinary grammar.  Nothing is read from ``.cache/lxx_validation/``.
"""

import contextlib
import io
import json
import re
import struct
import sys
import tempfile
import unittest
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "lxx_tagging"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ccat_align import Pair, Unit  # noqa: E402
from ccat_codes import Feats, Unmapped, compare_feats, decompose_robinson  # noqa: E402
from ccat_greek import tokenize_brenton  # noqa: E402
from ref_abp import MODULE_PATH, Tbesg, classify_abp, compare_numbers, read_abp  # noqa: E402
from ref_common import RefToken  # noqa: E402
from ref_glaux import decompose_agdt, read_glaux_file  # noqa: E402
from ref_ud import UdAdapter, decompose_ud, parse_conllu, parse_feats  # noqa: E402
import validate_ccat  # noqa: E402
import validate_refs  # noqa: E402

GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")
REF = re.compile(r"\b\d{1,3}:\d{1,3}\b|\b\d{7,8}\b")

# One invented verse per book, tagged by hand.
VERSE = "καὶ εἶπεν αὐτῷ ὁ ἄνθρωπος λόγον."
TAGS = [("G2532", "καί", "CONJ"), ("G2036", "ἔπω, ἐρῶ, εἶπον", "V-2AAI-3S"), ("G846", "αὐτός", "P-DSM"),
        ("G3588", "ὁ", "T-NSM"), ("G444", "ἄνθρωπος", "N-NSM"), ("G3056", "λόγος", "N-GSM")]  # λόγον: case wrong
ABP_WORDS = [("2532", "και"), ("2036", "είπεν"), ("1473", "αυτώ"), ("3588", "ο"), ("444", "άνθρωπος"),
             ("3004", "λόγον")]  # 3004 against our 3056: a number disagreement
UD_WORDS = [("καὶ", "καί", "CCONJ", "_"), ("εἶπεν", "λέγω", "VERB",
             "Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin|Voice=Act"),
            ("αὐτῷ", "αὐτός", "PRON", "Case=Dat|Gender=Masc|Number=Sing|Person=3|PronType=Prs"),
            ("ὁ", "ὁ", "DET", "Case=Nom|Definite=Def|Gender=Masc|Number=Sing|PronType=Art"),
            ("ἄνθρωπος", "ἄνθρωπος", "NOUN", "Case=Nom|Gender=Masc|Number=Sing"),
            ("λόγον", "λόγος", "NOUN", "Case=Acc|Gender=Masc|Number=Sing")]


GLAUX_WORDS = [("καὶ", "καί", "c--------"), ("εἶπεν", "λέγω", "v3saia---"), ("αὐτῷ", "αὐτός", "p-s---md-"),
               ("ὁ", "ὁ", "l-s---mn-"), ("ἄνθρωπος", "ἄνθρωπος", "n-s---mn-"),
               ("λόγον", "λόγος", "n-s---ma-")]


def glaux_xml():
    words = "".join(f'<word id="{i}" form="{f}" div_section="1.1" lemma="{lem}" postag="{tag}"/>'
                    for i, (f, lem, tag) in enumerate(GLAUX_WORDS, 1))
    return f'<treebank><sentence id="1">{words}</sentence></treebank>'


def module(book, number):
    spans = tokenize_brenton(VERSE)
    anns = [{"type": "strongs", "start": s, "end": e, "text": w,
             "data": {"strongs": st, "lemma": lem, "morphology": m}}
            for (s, e, w), (st, lem, m) in zip(spans, TAGS)]
    v = {"v": 1, "ref": number * 1000000 + 1001, "content": {"text": VERSE, "annotations": anns}}
    return {"meta": {}, "books": [{"id": book, "number": number, "chapters": [{"chapter": 1, "verses": [v]}]}]}


def sword_zip(path: Path, entries: list[str]) -> None:
    raw = b""
    index = b""
    for e in entries:
        data = e.encode("utf-8")
        index += struct.pack("<IIH", 0, len(raw), len(data))
        raw += data
    comp = zlib.compress(raw)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(MODULE_PATH + "ot.bzs", struct.pack("<III", 0, len(comp), len(raw)))
        z.writestr(MODULE_PATH + "ot.bzv", index)
        z.writestr(MODULE_PATH + "ot.bzz", comp)


def abp_entries(book_osis):
    words = " ".join(f'<w lemma="strong:G{n}" src="{i}">{f}</w>' for i, (n, f) in enumerate(ABP_WORDS, 1))
    return ["", '<milestone type="x-importer"/>',
            f'<div osisID="{book_osis}" sID="b1" type="book"/> <title>T</title>',
            f'<chapter osisID="{book_osis}.1" sID="c1"/>', words]


def conllu(book_code):
    lines = ["# sent_id = s1", f"# text = {VERSE}"]
    for i, (form, lemma, upos, feats) in enumerate(UD_WORDS, 1):
        lines.append(f"{i}\t{form}\t{lemma}\t{upos}\t_\t{feats}\t0\tdep\t_\tRef={book_code}_1.1")
    lines.append(f"7\t.\t.\tPUNCT\t_\t_\t0\tpunct\t_\tRef={book_code}_1.1")
    return "\n".join(lines) + "\n\n"


class AbpTests(unittest.TestCase):
    def test_sword_module_walk_and_nehemiah_folded_into_ezra(self):
        tmp = Path(tempfile.mkdtemp())
        entries = abp_entries("Gen") + ['<chapter osisID="Gen.2" sID="c2"/>', "",
                                        '<w lemma="strong:G2532 strong:G1473">κἀγώ</w>']
        entries += ['<div osisID="Neh" sID="b2" type="book"/>', '<chapter osisID="Neh.1" sID="c3"/>',
                    '<w lemma="strong:G*">Νεεμιας</w>']
        sword_zip(tmp / "m.zip", entries)
        books = read_abp(tmp / "m.zip")
        gen = books["Gen"]
        self.assertEqual([(v.chapter, v.verse) for v in gen], [(1, 1), (2, 2)])  # 2:1 is empty
        self.assertEqual(gen[0].tokens[1].numbers, ("2036",))
        self.assertEqual(gen[1].tokens[0].numbers, ("2532", "1473"))
        self.assertEqual([(v.chapter, v.verse) for v in books["Ezra"]], [(11, 1)])

    def test_number_classes(self):
        self.assertEqual(classify_abp("1510.7.3"), ("classic", 1510))
        self.assertEqual(classify_abp("180.2"), ("extra", 180))
        self.assertEqual(classify_abp("*"), ("name", None))

    def test_number_comparison_and_conventions(self):
        tb = Tbesg({1401: "δοῦλος", 1399: "δούλη", 3056: "λόγος", 3004: "λέγω"}, {11})
        self.assertEqual(compare_numbers("G3056", "N-NSM", ("3056",), tb)[0], "agree")
        self.assertEqual(compare_numbers("G1510", "V-IAI-3S", ("1510.7.3",), tb)[0], "agree")
        self.assertEqual(compare_numbers("G1473", "P-1GS", ("3450",), tb)[:2],
                         ("convention", "form-numbered Strong's"))
        self.assertEqual(compare_numbers("G4771", "P-2GS", ("1473",), tb)[:2],
                         ("convention", "ABP personal pronoun 1473"))
        self.assertEqual(compare_numbers("G1401", "N-NSF", ("1399",), tb, "δούλη")[:2],
                         ("convention", "same headword, other number"))
        self.assertEqual(compare_numbers("G1473", "P-1NS + G2532=CONJ", ("2532", "1473"), tb)[0], "convention")
        self.assertEqual(compare_numbers("G3756", "PRT-N + G2089=ADV", ("2089",), tb)[:2],
                         ("convention", "compound tag"))
        self.assertEqual(compare_numbers("G0011", "", ("*",), tb)[0], "uncertain")  # TBESG name
        self.assertEqual(compare_numbers("G3056", "N-NSM", ("*",), tb)[0], "disagree")
        self.assertEqual(compare_numbers("G6100", "A-NSM", ("4.1",), tb)[2], "both-extra")
        self.assertEqual(compare_numbers("G3056", "N-NSM", ("3004",), tb)[0], "disagree")
        # a name keyed by its Hebrew number is never compared as a Greek number
        self.assertEqual(compare_numbers("H3056", "", ("*",), tb)[0], "uncertain")
        self.assertEqual(compare_numbers("H3056", "", ("11",), tb)[0], "uncertain")
        self.assertEqual(compare_numbers("H3056", "", ("4.1",), tb)[0], "uncertain")
        self.assertEqual(compare_numbers("H3056", "", ("3056",), tb)[:2],
                         ("disagree", "ours a name, ABP a common word"))


class UdTests(unittest.TestCase):
    def test_tense_aspect_mood_voice(self):
        f = decompose_ud("VERB", parse_feats("Aspect=Perf|Mood=Ind|Number=Sing|Person=3|Tense=Past|"
                                             "VerbForm=Fin|Voice=Act"))
        self.assertEqual((f.tense, f.voice, f.mood, f.person, f.number), ("A", "A", "I", "3", "S"))
        self.assertEqual(decompose_ud("VERB", parse_feats("Aspect=Imp|Tense=Past|Mood=Ind|VerbForm=Fin")).tense, "I")
        self.assertEqual(decompose_ud("VERB", parse_feats("Aspect=Perf|Tense=Pres|Mood=Ind|VerbForm=Fin")).tense, "R")
        f = decompose_ud("VERB", parse_feats("Aspect=Perf|Case=Nom|Gender=Masc|Number=Sing|VerbForm=Part|"
                                             "Voice=Mid,Pass"))
        self.assertEqual((f.tense, f.mood, f.voice, f.case), ("A", "P", "MP", "N"))
        self.assertEqual(decompose_ud("VERB", parse_feats("Aspect=Imp|VerbForm=Inf")).mood, "N")

    def test_nominals_pronouns_and_underspecified_values(self):
        f = decompose_ud("NOUN", parse_feats("Case=Acc,Nom|Gender=Neut|Number=Sing"))
        self.assertEqual(f.case, "A|N")
        f = decompose_ud("PRON", parse_feats("Case=Gen|Number=Sing|Person=1|PronType=Prs"))
        self.assertEqual((f.pos, f.sub, f.person), ("pron", "P", "1"))
        self.assertEqual(decompose_ud("DET", parse_feats("PronType=Art|Case=Nom")).pos, "art")
        self.assertEqual(decompose_ud("PROPN", {}).proper, True)
        with self.assertRaises(Unmapped):
            decompose_ud("SYM", {})
        cmp = compare_feats(decompose_robinson("N-ASN"), f_noun := decompose_ud(
            "NOUN", parse_feats("Case=Acc,Nom|Gender=Neut|Number=Sing")), "agree")
        self.assertEqual(cmp.fields["case"][0], "convention")
        cmp = compare_feats(decompose_robinson("N-GSN"), f_noun, "agree")
        self.assertEqual(cmp.fields["case"][0], "disagree")

    def test_conllu_reading(self):
        text = conllu("GEN") + ("# sent_id = s2\n1-2\tκἀγώ\t_\t_\t_\t_\t_\t_\t_\tRef=GEN_1.2\n"
                                "1\tκαί\tκαί\tCCONJ\t_\t_\t0\tcc\t_\tRef=GEN_1.2\n"
                                "2\tἐγώ\tἐγώ\tPRON\t_\tCase=Nom|Number=Sing|Person=1|PronType=Prs\t0\tnsubj\t_"
                                "\tRef=GEN_1.2\n\n")
        rows = list(parse_conllu(text))
        self.assertEqual(len(rows), 7)  # punctuation skipped, the multiword token is one word
        book, ch, v, tok = rows[-1]
        self.assertEqual((book, ch, v, tok.form, tok.lemma), ("GEN", 1, 2, "κἀγώ", "καί+ἐγώ"))
        self.assertEqual([p.pos for p in UdAdapter.feats(tok).parts], ["conj", "pron"])


class GlauxTests(unittest.TestCase):
    def test_postags(self):
        f = decompose_agdt("v3saim---")
        self.assertEqual((f.pos, f.person, f.number, f.tense, f.mood, f.voice), ("verb", "3", "S", "A", "I", "M"))
        f = decompose_agdt("v-sappmn-")
        self.assertEqual((f.mood, f.case, f.gender), ("P", "N", "M"))
        f = decompose_agdt("p1s---cg-")
        self.assertEqual((f.pos, f.person, f.gender, f.case), ("pron", "1", "", "G"))
        self.assertEqual(decompose_agdt("b--------").pos, "conj")
        self.assertTrue(decompose_agdt("n-s---mn-", "Ἀβραάμ").proper)
        with self.assertRaises(Unmapped):
            decompose_agdt("z--------")

    def test_xml_reading(self):
        tmp = Path(tempfile.mkdtemp()) / "x.xml"
        tmp.write_text('<treebank><sentence id="1">'
                       '<word id="1" form="καὶ" div_section="2.3" lemma="καί" postag="c--------"/>'
                       '<word id="2" form="," div_section="2.3" lemma="," postag="u--------"/>'
                       '<word id="3" form="λόγος" div_section="2.4" lemma="λόγος" postag="n-s---mn-"/>'
                       '</sentence></treebank>', encoding="utf-8")
        rows = list(read_glaux_file(tmp))
        self.assertEqual([(c, v, t.form) for c, v, t in rows], [(2, 3, "καὶ"), (2, 4, "λόγος")])


class RefsFirewallTests(unittest.TestCase):
    def _run(self, book, number, osis, ud_code, samples=5):
        tmp = Path(tempfile.mkdtemp())
        cache = tmp / "cache"
        (cache / "abp").mkdir(parents=True)
        sword_zip(cache / "abp" / "ABPGRK.zip", abp_entries(osis))
        (cache / "ptnk").mkdir()
        for name in ("grc_ptnk-ud-train.conllu", "grc_ptnk-ud-dev.conllu", "grc_ptnk-ud-test.conllu"):
            (cache / "ptnk" / name).write_text(conllu(ud_code) if name.endswith("train.conllu") else "",
                                               encoding="utf-8")
        (cache / "glaux").mkdir()
        (cache / "glaux" / {"Ruth": "0527-010.xml", "Gen": "0527-001.xml"}[book]).write_text(glaux_xml(),
                                                                                         encoding="utf-8")
        lxx = tmp / "tagged.json"
        lxx.write_text(json.dumps(module(book, number), ensure_ascii=False), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = validate_refs.main(["--refs", "abp,ptnk,glaux", "--cache", str(cache), "--ccat-dir",
                                       str(tmp / "no-ccat"), "--lxx", str(lxx), "--tagged", str(lxx),
                                       "--decisions", str(tmp / "none.tsv"), "--reports", str(tmp / "r"),
                                       "--samples", str(samples)])
        self.assertEqual(code, 0)
        files = {p.name: p.read_text(encoding="utf-8") for p in (tmp / "r").iterdir()}
        return out.getvalue(), files

    def test_test_split_book_prints_no_forms_lemmas_numbers_or_refs(self):
        self.assertEqual(validate_ccat.split_of("Ruth", 1), "test")
        stdout, files = self._run("Ruth", 8, "Ruth", "RUT")
        for text in [stdout] + list(files.values()):
            self.assertIsNone(GREEK.search(text))
            self.assertIsNone(REF.search(text))
        report = json.loads(files["refs_validation.json"])
        abp = report["references"]["abp"]["test"]["overall"]
        self.assertEqual(abp["strongs"]["disagree"], 1)
        self.assertEqual(abp["strongs"]["convention"], 1)  # αὐτῷ: ABP's 1473 for personal pronouns
        ptnk = report["references"]["ptnk"]["test"]["overall"]
        self.assertEqual(ptnk["full_parse"]["disagree"], 1)  # λόγον tagged genitive
        self.assertNotIn("dev", report["references"]["abp"])
        self.assertEqual(report["combined"]["test"]["counts"]["every reference disagrees"], 1)
        # reference against reference, on the same words as us against each
        agree = report["reference_agreement"]["glaux~ptnk"]["test"]
        full = agree["table"]["full_parse"]
        self.assertEqual(full["words"], 6)
        self.assertEqual(full["A~B"]["with_conventions"], 1.0)    # the two references agree
        self.assertAlmostEqual(full["us~A"]["with_conventions"], 5 / 6, places=3)
        self.assertEqual(agree["we_agree_with_neither"], {"references agree with each other": 1})

    def test_dev_samples_show_categories_never_reference_answers(self):
        stdout, files = self._run("Gen", 1, "Gen", "GEN")
        report = json.loads(files["refs_validation.json"])
        abp_samples = report["references"]["abp"]["dev"]["samples"]
        self.assertEqual(list(abp_samples), ["Strong's number differs"])
        blob = json.dumps(report["references"], ensure_ascii=False)
        self.assertNotIn("3004", blob)          # ABP's number for the word
        self.assertIn("case differs: ours G, theirs A", stdout)
        self.assertNotIn("λέγω", blob)          # PTNK's lemma for εἶπεν (ours: ἔπω ...)
        disagree = report["combined"]["dev"]["every_reference_disagrees"]
        self.assertEqual(disagree["by_references"], {"abp+glaux+ptnk": 1})
        self.assertIn("REFERENCE AGREEMENT glaux~ptnk (dev)", stdout)
        self.assertEqual(report["reference_agreement"]["glaux~ptnk"]["dev"]["A_B_differ_on"], {})

    def test_reference_against_reference(self):
        from ref_glaux import GlauxAdapter  # noqa: F401
        a = Pair(None, [RefToken("εἶπεν", lemma="λέγω", tag="v3saia---")], "exact")
        b = Pair(None, [RefToken("εἶπεν", lemma="λέγω", tag="VERB|Aspect=Perf|Mood=Ind|Number=Sing|Person=3|"
                                                          "Tense=Past|VerbForm=Fin|Voice=Act")], "exact")
        rr = validate_refs.ref_vs_ref(a, "glaux", b, "ptnk")
        self.assertEqual((rr["lemma"], rr["pos"], rr["full"]), ("agree", "agree", "agree"))
        c = Pair(None, [RefToken("εἶπεν", lemma="λέγω", tag="v3sfia---")], "exact")  # future: differs
        rr = validate_refs.ref_vs_ref(c, "glaux", b, "ptnk")
        self.assertEqual(rr["fields"]["tense"][0], "disagree")
        self.assertIsNone(validate_refs.ref_vs_ref(Pair(None, a.ccat, "joined"), "glaux", b, "ptnk"))

    def test_unit_scoring_against_an_adapter(self):
        u = Unit(1, 0, 0, 5, "λόγον", strongs="G3056", lemma="λόγος", morph="N-ASM", tagged=True)
        tok = RefToken("λόγον", lemma="λόγος", tag="NOUN|Case=Acc|Gender=Masc|Number=Sing")
        res = validate_ccat.score_against(Pair(u, [tok], "exact"), UdAdapter)
        self.assertEqual((res["lemma"], res["pos"], res["full"]), ("agree", "agree", "agree"))
        self.assertIsInstance(UdAdapter.feats(tok), Feats)


if __name__ == "__main__":
    unittest.main()
