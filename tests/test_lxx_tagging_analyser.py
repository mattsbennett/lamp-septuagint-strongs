"""The morphological analyser: phonology, paradigms, and analysis on fixtures.

The paradigm tests spot-check one class each (the forms are the grammar
books'); the analyser tests build it over small synthetic TBESG and TAGNT
files, so they run in a moment and say exactly what each rule does.
"""

import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import analyser as analyser_module  # noqa: E402
from lxx_tagging import lexicon, ntforms  # noqa: E402
from lxx_tagging import morph_irregular as irregular  # noqa: E402
from lxx_tagging import morph_nominal as nominal  # noqa: E402
from lxx_tagging import morph_phon as ph  # noqa: E402
from lxx_tagging import morph_verbs as verbs  # noqa: E402
from lxx_tagging.candidates import Tagger, default_tagger  # noqa: E402
from lxx_tagging.text import normalize, tokenize  # noqa: E402
from lxx_tagging_fixtures import tagnt_row, tbesg_row, TBESG_HEADER  # noqa: E402

P = ph.PREVERB_BY_NAME


def keys(words: str) -> set[str]:
    return {normalize(word) for word in words.split()}


def verb_forms(lemma, chain, systems, **options) -> dict[str, set[str]]:
    lexeme = verbs.VerbLexeme("G1", lemma, tuple(P[name] for name in chain), systems, **options)
    found: dict[str, set[str]] = {}
    for form in verbs.Generator().forms(lexeme):
        found.setdefault(form.key, set()).add(form.code)
    return found


def noun_forms(lemma, gender, genitive=None) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for declension in nominal.noun_hypotheses(normalize(lemma), gender, genitive):
        for form in nominal.decline(declension):
            found.setdefault(form.key, set()).add(form.case_number + form.gender)
    return found


def adjective_forms(lemma) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for declensions in nominal.adjective_hypotheses(normalize(lemma)):
        for declension in declensions:
            for form in nominal.decline(declension):
                found.setdefault(form.key, set()).add(form.case_number + form.gender)
    return found


class PhonologyTests(unittest.TestCase):
    def test_recessive_accent_and_the_compound_verbs_limit(self):
        self.assertEqual(ph.accentuate("λυομεν"), ("λύομεν",))
        # Not before the augment, nor past the preverb's last syllable.
        self.assertEqual(ph.accentuate("ἀπηλθον", floor=1), ("ἀπῆλθον",))
        self.assertEqual(ph.accentuate("ἀποδοσ", floor=1), ("ἀπόδοσ",))
        # An unknown quantity leaves acute and circumflex open (λῦσαι, κρῖναι).
        self.assertEqual(set(ph.accentuate("λυσαι", "syllable", syllable=-2)), {"λύσαι", "λῦσαι"})

    def test_persistent_accent_moves_only_as_the_ultima_requires(self):
        self.assertEqual(ph.accentuate("ἀνθρωπου", "syllable", syllable=0), ("ἀνθρώπου",))
        self.assertEqual(ph.accentuate("ὀνοματοσ", "syllable", syllable=0), ("ὀνόματοσ",))

    def test_contraction_with_its_accent(self):
        self.assertEqual(ph.contract("ποιέομεν", 4), "ποιοῦμεν")
        self.assertEqual(ph.contract("τιμάει", 4), "τιμᾷ")
        self.assertEqual(ph.contract("ἐποίεον", 5), "ἐποίουν")
        self.assertEqual(ph.contract("δηλόει", 4), "δηλοῖ")

    def test_augment_reduplication_and_sandhi(self):
        self.assertEqual(ph.augment("ἀκου")[0][0], "ἠκου")
        self.assertEqual(ph.augment("οἰκοδομ")[0][0], "ᾠκοδομ")
        self.assertEqual(ph.reduplicate("γραφ")[0][0], "γεγραφ")
        self.assertEqual(ph.reduplicate("σταλκ")[0][0], "ἐσταλκ")
        self.assertEqual(ph.perfect_middle("γεγραφ", "ται"), "γεγραπται")
        self.assertEqual(ph.perfect_middle("ἡγνισ", "σθαι"), "ἡγνισθαι")

    def test_preverbs_join_and_split(self):
        self.assertEqual(ph.attach((P["ἀπό"],), "ἐστειλα")[0][0], "ἀπεστειλα")
        self.assertEqual(ph.attach((P["σύν"],), "βαλλω")[0][0], "συμβαλλω")
        self.assertEqual(ph.attach((P["κατά"],), "ἱστημι")[0][0], "καθιστημι")
        splits = {(tuple(p.name for p in chain), base) for chain, base in ph.split_preverbs("ἐπισυναγω")}
        self.assertIn((("ἐπί", "σύν"), "ἀγω"), splits)
        self.assertIn((("ἀπό",), "αἱρεω"), {(tuple(p.name for p in c), b) for c, b in ph.split_preverbs("ἀφαιρεω")})


class VerbParadigmTests(unittest.TestCase):
    def test_thematic_verb(self):
        forms = verb_forms("λύω", [], {
            "pres": [verbs.Stem("pres", "λῡ")], "fut": [verbs.Stem("fut", "λῡσ")], "aor1": [verbs.Stem("aor1", "λῡσ")],
            "aorp": [verbs.Stem("aorp", "λυθ")], "perf": [verbs.Stem("perf", "λελυκ")],
            "perfm": [verbs.Stem("perfm", "λελυ")]})
        self.assertLessEqual(keys("λύω ἔλυον λυόμενος λύσω ἔλυσα λῦσαι λύσας ἐλύθην λυθῶ λυθείς λυθεῖσα "
                                  "λυθήσομαι λέλυκα λελυκώς λελυκυῖα λέλυμαι λελυμένος ἐλελύκειν λύθητι"), set(forms))
        self.assertIn("V-AAI-1S", forms[normalize("ἔλυσα")])
        self.assertIn("V-APP-NSM", forms[normalize("λυθείς")])
        self.assertIn("V-RPP-NSM", forms[normalize("λελυμένος")])

    def test_contract_present_in_a_compound_with_a_second_aorist(self):
        forms = verb_forms("ἀποποιέω", ["ἀπό"], {"pres": [verbs.Stem("pres", "ποιε", contract="ε")],
                                                 "aor2": [verbs.Stem("aor2", "λαβ", second=True)]})
        self.assertLessEqual(keys("ἀποποιῶ ἀποποιεῖς ἀποποιοῦσιν ἀπεποίουν ἀποποιεῖν ἀποποιοῦντος ἀποποιουσῶν "
                                  "ἀπέλαβον ἀπολαβεῖν ἀπολαβών ἀπόλαβε ἀπέλαβαν"), set(forms))
        self.assertIn("V-2AAI-3P", forms[normalize("ἀπέλαβαν")])
        self.assertIn("V-PAN", forms[normalize("ἀποποιεῖν")])

    def test_athematic_tables_compound(self):
        forms = verb_forms("ἀποδίδωμι", ["ἀπό"], {"t": [irregular.table_stem("δίδωμι")]})
        self.assertLessEqual(keys("ἀποδίδωσιν ἀπεδίδου ἀποδοῦναι ἀποδούς ἀπόδος ἀπέδωκα ἀποδῷ"), set(forms))
        self.assertIn("V-AAI-1S", forms[normalize("ἀπέδωκα")])
        self.assertIn("V-2AAM-2S", forms[normalize("ἀπόδος")])
        forms = verb_forms("ἀνίστημι", ["ἀνά"], {"t": [irregular.table_stem("ἵστημι")]})
        self.assertLessEqual(keys("ἀνέστη ἀνάστηθι ἀναστάς ἀναστᾶσα ἀνέστηκεν"), set(forms))
        forms = verb_forms("πάρειμι", ["παρά"], {"t": [irregular.table_stem("εἰμί")]})
        self.assertLessEqual(keys("πάρεστιν παρῆν παρών παρόντος παρέσται"), set(forms))

    def test_suppletive_stems_transfer_to_a_compound(self):
        systems = {}
        for stem in irregular.principal_parts("ἔρχομαι"):
            systems.setdefault(stem.system, []).append(stem)
        forms = verb_forms("κατέρχομαι", ["κατά"], systems)
        self.assertLessEqual(keys("κατῆλθεν κατελθεῖν κατελθών κατελεύσεται κατελήλυθα κάτελθε"), set(forms))

    def test_nymi_present(self):
        found = {entry[4] for entry in irregular.nymi_forms("δεικνυ")}
        self.assertLessEqual(keys("δείκνυμι δεικνύασι ἐδείκνυ δεικνύναι δεικνύς δεικνῦσα δεικνύμενος"), found)


class NominalParadigmTests(unittest.TestCase):
    def test_first_and_second_declension(self):
        self.assertLessEqual(keys("θεοῦ θεῷ θεούς θεῶν"), set(noun_forms("θεός", "M")))
        self.assertLessEqual(keys("ἡμέρας ἡμέρᾳ ἡμερῶν"), set(noun_forms("ἡμέρα", "F")))
        self.assertLessEqual(keys("ἀληθείας ἀλήθειαν"), set(noun_forms("ἀλήθεια", "F")))
        # Brenton's -ρας beside the New Testament's -ρης.
        self.assertLessEqual(keys("μαχαίρας μαχαίρης"), set(noun_forms("μάχαιρα", "F")))
        self.assertLessEqual(keys("δόξης δόξῃ"), set(noun_forms("δόξα", "F")))
        self.assertLessEqual(keys("ὀστοῦ ὀστᾶ ὀστέα ὀστέων"), set(noun_forms("ὀστέον", "N")))

    def test_third_declension(self):
        self.assertLessEqual(keys("ὀνόματος ὀνόμασιν"), set(noun_forms("ὄνομα", "N")))
        self.assertLessEqual(keys("γένους γένη γενῶν"), set(noun_forms("γένος", "N")))
        self.assertLessEqual(keys("πόλεως πόλεις πόλεσι"), set(noun_forms("πόλις", "F")))
        self.assertLessEqual(keys("ἐλπίδος ἐλπίσι"), set(noun_forms("ἐλπίς", "F")))
        self.assertLessEqual(keys("βασιλέως βασιλεῖ βασιλεῖς βασιλεῦσι"), set(noun_forms("βασιλεύς", "M")))
        self.assertLessEqual(keys("σαρκός σαρξί"), set(noun_forms("σάρξ", "F")))
        self.assertLessEqual(keys("ἄρχοντος ἄρχουσι αἰῶνος"), set(noun_forms("ἄρχων", "M")) | set(noun_forms("αἰών", "M")))
        self.assertLessEqual(keys("πήχεως πήχεις πηχῶν"), set(noun_forms("πῆχυς", "M")))

    def test_adjectives(self):
        self.assertLessEqual(keys("ἁγία ἁγίας ἁγίων"), set(adjective_forms("ἅγιος")))
        self.assertLessEqual(keys("ἀληθοῦς ἀληθεῖ ἀληθῆ ἀληθεῖς ἀληθές"), set(adjective_forms("ἀληθής")))
        self.assertLessEqual(keys("ταχεῖα ταχέα"), set(adjective_forms("ταχύς")))
        self.assertLessEqual(keys("μείζονος μείζω μείζους μεῖζον"), set(adjective_forms("μείζων")))
        self.assertLessEqual(keys("χρυσοῦ χρυσῆ χρυσᾶ"), set(adjective_forms("χρυσοῦς")))
        self.assertLessEqual(keys("χιλίους χίλιαι"), set(adjective_forms("χίλιοι")))
        # A two-termination adjective's feminine is its masculine form.
        self.assertIn("NSF", adjective_forms("ἀλλόφυλος")[normalize("ἀλλόφυλος")])

    def test_irregular_nominals(self):
        gender, forms = irregular.irregular_nominal(normalize("ἅπας"))
        self.assertIn(("GPM", normalize("ἁπάντων")), forms)
        gender, forms = irregular.irregular_nominal(normalize("οὐδείς"))
        self.assertIn(("ASF", normalize("οὐδεμίαν")), forms)
        self.assertIn(("NSM", normalize("οὐδείς")), forms)


TBESG_ROWS = [
    tbesg_row("G3004", "G3004G =", "G3004G", "λέγω", "G:V", "to say"),
    tbesg_row("G0649", "G0649 =", "G0649", "ἀποστέλλω", "G:V", "to send"),
    tbesg_row("G6400", "G6400 =", "G6400", "ἐξαποστέλλω", "G:V", "to send out"),
    tbesg_row("G4160", "G4160 =", "G4160", "ποιέω", "G:V", "to do"),
    tbesg_row("G1325", "G1325 =", "G1325", "δίδωμι", "G:V", "to give"),
    tbesg_row("G0591", "G0591 =", "G0591", "ἀποδίδωμι", "G:V", "to give back"),
    tbesg_row("G3056", "G3056 =", "G3056", "λόγος", "G:N-M", "word"),
    tbesg_row("G7000", "G7000 =", "G7000", "ἐγγύη", "G:N-F", "surety"),
    tbesg_row("G0018", "G0018 =", "G0018", "ἀγαθός", "G:A", "good"),
    tbesg_row("G0430", "G0430 =", "G0430", "ἀντί", "G:PREP", "instead of"),
]
TAGNT_ROWS = [
    tagnt_row("Mat.1.1#01", "ἀπέστειλεν", "G0649=V-AAI-3S", "ἀποστέλλω=to send"),
    tagnt_row("Mat.1.1#02", "ἀποστελῶ", "G0649=V-FAI-1S", "ἀποστέλλω=to send"),
    tagnt_row("Mat.1.1#03", "ἀπεστάλη", "G0649=V-2API-3S", "ἀποστέλλω=to send"),
    tagnt_row("Mat.1.1#04", "ἀπέσταλκα", "G0649=V-RAI-1S", "ἀποστέλλω=to send"),
    tagnt_row("Mat.1.2#01", "ἐποίησεν", "G4160=V-AAI-3S", "ποιέω=to do"),
    tagnt_row("Mat.1.2#02", "λόγου", "G3056=N-GSM", "λόγος=word"),
    tagnt_row("Mat.1.2#03", "λόγον", "G3056=N-ASM", "λόγος=word"),
    tagnt_row("Mat.1.2#04", "ἀγαθοῦ", "G0018=A-GSM", "ἀγαθός=good"),
    tagnt_row("Mat.1.2#05", "ἀγαθήν", "G0018=A-ASF", "ἀγαθός=good"),
    tagnt_row("Mat.1.3#01", "ἀποδώσει", "G0591=V-FAI-3S", "ἀποδίδωμι=to give back"),
    tagnt_row("Mat.1.3#02", "ἀποδῷ", "G0591=V-2AAS-3S", "ἀποδίδωμι=to give back"),
]


class AnalyserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        (root / "TBESG.txt").write_text("﻿" + "\n".join(TBESG_HEADER + TBESG_ROWS) + "\n", encoding="utf-8")
        (root / "TAGNT_Mat-Jhn.txt").write_text("# fixture\n" + "\n".join(TAGNT_ROWS) + "\n", encoding="utf-8")
        (root / "TAGNT_Act-Rev.txt").write_text("# fixture\n", encoding="utf-8")
        cls.lexicon = lexicon.build(root / "TBESG.txt")
        cls.forms = ntforms.build(ntforms.tagnt_paths(root))
        # The fixture TAGNT uses few codes: accept any, to test the parses.
        cls.analyser = analyser_module.Analyser(cls.lexicon, cls.forms, codes=None)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def readings(self, word, analyser=None):
        return {(c.strongs, c.morph) for c in (analyser or self.analyser).candidates(normalize(word))}

    def test_an_lxx_only_compound_inherits_its_simplexs_attested_stems(self):
        # ἐξαποστέλλω is not in the fixture TAGNT: its aorist, future, aorist
        # passive and perfect come from ἀποστέλλω's forms and the rules.
        self.assertIn(("G6400", "V-AAI-3S"), self.readings("ἐξαπέστειλεν"))
        self.assertIn(("G6400", "V-FAI-3S"), self.readings("ἐξαποστελεῖ"))
        self.assertIn(("G6400", "V-2API-3P"), self.readings("ἐξαπεστάλησαν"))
        self.assertIn(("G6400", "V-AAP-NSM"), self.readings("ἐξαποστείλας"))

    def test_unattested_forms_of_attested_words(self):
        self.assertIn(("G3056", "N-DPM"), self.readings("λόγοις"))
        self.assertIn(("G4160", "V-FAI-3P"), self.readings("ποιήσουσιν"))
        self.assertIn(("G4160", "V-PAI-3S"), self.readings("ποιεῖ"))
        self.assertIn(("G18", "A-DPF"), self.readings("ἀγαθαῖς"))
        self.assertIn(("G7000", "N-GSF"), self.readings("ἐγγύης"))

    def test_athematic_compound(self):
        self.assertIn(("G591", "V-2AAN"), self.readings("ἀποδοῦναι"))
        self.assertIn(("G591", "V-AAI-3P"), self.readings("ἀπέδωκαν"))

    def test_suppletive_forms_by_rule_of_the_lexeme(self):
        self.assertIn(("G3004", "V-2AAI-3S"), self.readings("εἶπεν"))

    def test_accent_decides_and_a_mismatch_is_kept_only_as_a_fallback(self):
        exact = self.analyser.candidates(normalize("λόγοις"))
        self.assertTrue(all("accent differs" not in c.notes for c in exact))
        fallback = self.analyser.candidates(normalize("λογοῖς"))
        self.assertTrue(fallback and all("accent differs" in c.notes for c in fallback))

    def test_weights_are_relative_to_the_best_and_codes_are_tagnts(self):
        candidates = self.analyser.candidates(normalize("ἐξαπέστειλεν"))
        self.assertEqual(max(c.weight for c in candidates), 1.0)
        self.assertTrue(all(c.source == "analyser" for c in candidates))
        # With TAGNT's inventory, a parse TAGNT has no code for is offered
        # with morphology None (and its parse in the notes), at half weight.
        strict = analyser_module.Analyser(self.lexicon, self.forms)
        for candidate in strict.candidates(normalize("ἐξαπέστειλεν")):
            self.assertTrue(candidate.morph is None or candidate.morph in self.forms.morph_codes)
        uncoded = [c for c in strict.candidates(normalize("λόγοις")) if c.morph is None]
        self.assertTrue(uncoded and "N-DPM" in uncoded[0].notes)

    def test_elision_restores_the_full_word(self):
        self.assertIn(("G430", "PREP"), self.readings("ἀντ’"))
        self.assertIn(("G430", "PREP"), self.readings("ἀνθ’"))

    def test_held_out_forms_are_not_learned(self):
        # Withheld, ἐποίησεν is no longer evidence: it is read by the rules.
        held = analyser_module.Analyser(self.lexicon, self.forms, codes=None, exclude_keys={normalize("ἐποίησεν")})
        self.assertEqual(held.evidence.rows("G4160"), [])
        self.assertTrue(self.analyser.candidates(normalize("ἐποίησεν"))[0].notes.startswith("nt:"))
        top = held.candidates(normalize("ἐποίησεν"))[0]
        self.assertEqual((top.strongs, top.morph), ("G4160", "V-AAI-3S"))
        self.assertTrue(top.notes.startswith("rule:"))

    def test_vocabulary_limits_the_index(self):
        small = analyser_module.Analyser(self.lexicon, self.forms, vocabulary={"λογοισ"}, codes=None)
        self.assertEqual(set(small.index), {"λογοισ"})

    def test_registered_before_the_tolerant_lookup(self):
        tagger = default_tagger(self.forms, self.lexicon)
        tagger.register("analyser", self.analyser, before="nt-tolerant")
        self.assertEqual(tagger.provider_names(), ["nt-forms", "tbesg", "analyser", "nt-tolerant", "tbesg-tolerant"])
        item = tagger.tag_tokens(tokenize("καὶ ἐξαπέστειλεν"))[1][0]
        self.assertEqual(item.tier, "a")
        self.assertEqual(item.candidates[0].strongs, "G6400")


if __name__ == "__main__":
    unittest.main()
