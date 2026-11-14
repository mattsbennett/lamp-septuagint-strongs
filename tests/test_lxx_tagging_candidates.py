"""Candidate readings, the case filter, tiers, and the plug-in points."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import lexicon, ntforms  # noqa: E402
from lxx_tagging.candidates import (  # noqa: E402
    CaseProfile, Candidate, Choice, annotation_data, default_tagger, paradigm_codes,
)
from lxx_tagging.text import normalize, tokenize  # noqa: E402
from lxx_tagging_fixtures import write_sources  # noqa: E402


class CandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        sources = write_sources(Path(cls.directory.name))
        cls.forms = ntforms.build(ntforms.tagnt_paths(sources))
        cls.lexicon = lexicon.build(sources / "TBESG.txt")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def tagger(self, profile=None, **options):
        return default_tagger(self.forms, self.lexicon, profile, **options)

    def outcome(self, text, surface, tagger=None):
        for item, choice in (tagger or self.tagger()).tag_text(text):
            if item.token.surface == surface:
                return item, choice
        raise AssertionError(f"{surface!r} not in {text!r}")

    def chosen(self, text, surface, tagger=None):
        _, choice = self.outcome(text, surface, tagger)
        return annotation_data(choice.candidate) if choice else None

    def test_one_reading_is_tagged_in_tagnts_shape(self):
        self.assertEqual(self.chosen("ἡ βίβλος", "βίβλος"),
                         {"strongs": "G976", "morphology": "N-NSF", "lemma": "βίβλος"})
        # Capitalised mid-sentence, the same word would have to be a name.
        self.assertIsNone(self.chosen("ἡ Βίβλος", "Βίβλος"))

    def test_a_capital_mid_sentence_is_a_title_and_lower_case_is_not(self):
        self.assertEqual(self.chosen("ὁ οἶκος Κυρίου", "Κυρίου")["morphology"], "N-GSM-T")
        self.assertEqual(self.chosen("ὁ οἶκος κυρίου", "κυρίου")["morphology"], "N-GSM")

    def test_a_capital_mid_sentence_is_never_tagged_as_a_common_word(self):
        item, choice = self.outcome("ἐν τῷ Λίβανον", "Λίβανον")
        self.assertIsNone(choice)
        self.assertEqual((item.tier, item.reason), ("d", "name-like: only common readings"))
        self.assertEqual([c.strongs for c, _ in item.set_aside][:1], ["G3030"])
        self.assertEqual(self.chosen("καὶ λίβανον", "λίβανον")["strongs"], "G3030")

    def test_brentons_own_capitals_decide_at_the_start_of_a_sentence(self):
        texts = ["ἐν τῷ Λίβανον", "καὶ λίβανον", "εἰς Λίβανον"]
        profile = CaseProfile.from_texts(texts)
        self.assertEqual(profile.leaning("λίβανον"), "mixed")
        item, choice = self.outcome("Λίβανον.", "Λίβανον", self.tagger(profile))
        self.assertIsNone(choice)
        self.assertEqual(item.tier, "d")
        # The odd capital in many lower-case uses is a slip, not a name.
        profile = CaseProfile.from_texts(["ἐν καὶ"] * 40 + ["ἐν Καὶ"])
        self.assertEqual(profile.leaning("καί"), "lower")

    def test_a_name_whose_form_does_not_show_its_case_is_tagged_without_morphology(self):
        self.assertEqual(self.chosen("καὶ Ἀβραὰμ", "Ἀβραὰμ"), {"strongs": "G11", "lemma": "Ἀβραάμ"})
        self.assertEqual(self.chosen("καὶ Μωυσῆς", "Μωυσῆς"),
                         {"strongs": "G3475", "morphology": "N-NSM-P", "lemma": "Μωϋσῆς, Μωσῆς"})
        # Ἁβραάμ is the same name with the other breathing; its case is not taken.
        self.assertEqual(self.chosen("καὶ Ἁβραὰμ", "Ἁβραὰμ"), {"strongs": "G11", "lemma": "Ἀβραάμ"})
        self.assertEqual(self.chosen("καὶ Ἄννας", "Ἄννας"), {"strongs": "G452", "lemma": "Ἅννας"})
        # Only the dictionary form: Brenton's Ἄνναν is Hannah, the NT's Ἅνναν Annas.
        self.assertIsNone(self.chosen("καὶ Ἄνναν", "Ἄνναν"))

    def test_lexicon_names_and_indeclinables(self):
        self.assertEqual(self.chosen("καὶ Ἰωδὰ", "Ἰωδὰ"), {"strongs": "G6043", "lemma": "Ἰωδά"})
        self.assertEqual(self.chosen("ἔτη τεσσαράκοντα", "τεσσαράκοντα"),
                         {"strongs": "G5062", "morphology": "A-NUI", "lemma": "τεσσαράκοντα"})
        # G1 is Ἀλφα in TAGNT and also ἆ: the interjection keeps its own lemma.
        self.assertEqual(self.chosen("καὶ ἆ", "ἆ"), {"strongs": "G1", "morphology": "INJ", "lemma": "ἆ"})
        # A name for a lower-case word is not trusted.
        item, choice = self.outcome("καὶ ἰωδὰ", "ἰωδὰ")
        self.assertEqual((item.tier, choice), ("u", None))
        # A declining word's lemma is only one of its forms: left to an analyser.
        self.assertEqual(self.outcome("καὶ ἀγγέλλω", "ἀγγέλλω")[0].tier, "e")

    def test_licensed_alternates_only(self):
        item, choice = self.outcome("καὶ ἔλεγε", "ἔλεγε")
        self.assertEqual(annotation_data(choice.candidate)["morphology"], "V-IAI-3S")
        self.assertEqual(item.candidates[0].source, "nt-form:movable-nu")
        self.assertEqual(self.outcome("τοῦ φαίνειν", "φαίνειν")[0].tier, "e")

    def test_joined_and_crasis_words(self):
        tagged = [(item.token.surface, annotation_data(choice.candidate) if choice else None)
                  for item, choice in self.tagger().tag_text("ἱνατί κᾀγὼ")]
        self.assertEqual(tagged, [
            ("ἱνα", {"strongs": "G2443", "morphology": "CONJ", "lemma": "ἵνα"}),
            ("τί", {"strongs": "G5101", "morphology": "I-NSN", "lemma": "τίς"}),
            ("κᾀγὼ", {"strongs": "G1473", "morphology": "P-1NS + G2532=CONJ", "lemma": "κἀγώ"}),
        ])

    def test_tiers_for_several_parses_and_several_words(self):
        item, _ = self.outcome("τὸ ὄνομα", "ὄνομα")
        self.assertEqual(item.tier, "b")
        self.assertEqual({c.morph for c in item.candidates}, {"N-ASN", "N-NSN", "N-VSN"})
        self.assertEqual(self.outcome("τίς ἐστιν", "τίς")[0].tier, "c")
        self.assertEqual(self.outcome("καὶ ἀνομίας", "ἀνομίας")[0].tier, "b")
        self.assertEqual(self.outcome("καὶ ἀόρατος", "ἀόρατος")[0].tier, "e")
        self.assertEqual(self.outcome("καὶ Μωάβ", "Μωάβ")[0].tier, "d")

    def test_paradigm_codes_are_identities_of_every_paradigm(self):
        self.assertEqual(paradigm_codes("N-ASN"), ["N-NSN", "N-VSN"])
        self.assertEqual(paradigm_codes("A-GSM"), ["A-GSN"])
        self.assertEqual(paradigm_codes("N-GSM"), [])             # a noun has one gender
        self.assertEqual(paradigm_codes("V-PAI-2P"), ["V-PAM-2P"])
        self.assertEqual(paradigm_codes("N-GSF", normalize("ἀνομίας")), ["N-APF"])
        self.assertEqual(paradigm_codes("N-GSF", normalize("χαρᾶς")), [])
        self.assertEqual(paradigm_codes("N-NPF", normalize("πόλεις")), ["N-VPF", "N-APF"])
        self.assertEqual(paradigm_codes("V-PAI-1S", normalize("λέγω")), ["V-PAS-1S"])
        self.assertEqual(paradigm_codes("V-FAI-1S", normalize("ποιήσω")), ["V-AAS-1S"])
        self.assertEqual(paradigm_codes("V-FAI-1S", normalize("μενῶ")), [])
        self.assertEqual(paradigm_codes("V-FAI-1S", normalize("δώσω"), "δίδωμι"), [])
        self.assertEqual(paradigm_codes("V-FAI-1S", normalize("ἀποδώσω"), "ἀποδίδωμι"), [])
        # Without the paradigm rules the attested reading alone is tagged.
        _, choice = self.outcome("τὸ ὄνομα", "ὄνομα", self.tagger(paradigm=False))
        self.assertEqual(choice.candidate.morph, "N-ASN")

    def test_an_analyser_provider_and_a_chooser_plug_in(self):
        seen = []

        def analyser(key, token):
            seen.append((key, token.surface))
            if key == normalize("ἀόρατος"):
                return [Candidate("G517", "ἀόρατος", "N-NSF", "analyser", 0.9)]
            return []

        tagger = self.tagger()
        tagger.register("analyser", analyser, before="nt-tolerant")
        self.assertEqual(tagger.provider_names(), ["nt-forms", "tbesg", "analyser", "nt-tolerant", "tbesg-tolerant"])
        self.assertEqual(self.chosen("καὶ ἀόρατος", "ἀόρατος", tagger)["strongs"], "G517")
        # Consulted only where nothing earlier answered.
        self.assertEqual(seen, [(normalize("ἀόρατος"), "ἀόρατος")])

        def first_reading(verse):
            return [Choice(item.candidates[0], 0.5, 0.1, "first") if item.tier in ("a", "b") else None
                    for item in verse]

        tagger.chooser = first_reading
        outcome = tagger.tag_text("τὸ ὄνομα")
        self.assertEqual(outcome[1][1].candidate.morph, "N-ASN")
        self.assertEqual(outcome[1][1].method, "first")

    def test_a_chooser_must_answer_every_token_with_tagnt_codes(self):
        tagger = self.tagger(chooser=lambda verse: [])
        with self.assertRaises(ValueError):
            tagger.tag_tokens(tokenize("καὶ ἐν"))
        invented = Candidate("G2532", "καί", "CONJ-X", "chooser")
        tagger.chooser = lambda verse: [Choice(invented) for _ in verse]
        with self.assertRaises(ValueError):
            tagger.tag_tokens(tokenize("καὶ ἐν"))


if __name__ == "__main__":
    unittest.main()
