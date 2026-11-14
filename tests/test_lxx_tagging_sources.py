"""The TAGNT form table and the TBESG index the LXX tagger reads."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import lexicon, ntforms  # noqa: E402
from lxx_tagging.ntforms import Reading, case_of, reading_kind  # noqa: E402
from lxx_tagging.text import normalize  # noqa: E402
from lxx_tagging_fixtures import write_sources  # noqa: E402


class FormTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        sources = write_sources(Path(cls.directory.name))
        cls.forms = ntforms.build(ntforms.tagnt_paths(sources))
        cls.lexicon = lexicon.build(sources / "TBESG.txt")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_readings_use_the_shipped_key_and_lemma_conventions(self):
        self.assertEqual(self.forms.readings(normalize("Βίβλος")),
                         {Reading("G976", "βίβλος", "N-NSF"): 1})
        # G2962G and G2962H are both G2962, as TAGNT.json writes them.
        self.assertEqual(self.forms.readings(normalize("κυρίου")), {
            Reading("G2962", "κύριος", "N-GSM-T"): 1, Reading("G2962", "κύριος", "N-GSM"): 1})
        self.assertEqual(self.forms.lemma_for("G3475"), "Μωϋσῆς, Μωσῆς")

    def test_every_row_counts_including_variants_and_decomposed_text(self):
        # ἀμήν is a Textus Receptus row, decomposed and followed by ".¶".
        self.assertEqual(self.forms.readings("ἀμήν"), {Reading("G281", "ἀμήν", "INJ-HEB"): 1})
        self.assertEqual(self.forms.surfaces("ἀμήν"), {"ἀμήν": 1})
        self.assertEqual(self.forms.rows, 31)

    def test_compound_rows_keep_the_shipped_shape_and_their_parts(self):
        self.assertEqual(self.forms.readings("ἱνατί"),
                         {Reading("G2443", "ἵνα", "CONJ + G5101=I-NSN"): 1})
        self.assertEqual(self.forms.parts_of("ἱνατί"),
                         {(Reading("G2443", "ἵνα", "CONJ"), Reading("G5101", "τίς", "I-NSN")): 1})
        # κἀγώ's lemma column names one word, so it has no parts to give.
        self.assertEqual(self.forms.parts_of(normalize("κἀγὼ")), {})
        self.assertIn("CONJ + G5101=I-NSN", self.forms.morph_codes)
        self.assertIn("I-NSN", self.forms.morph_codes)

    def test_capitals_and_letters_indexes(self):
        self.assertEqual(set(self.forms.caps_readings("τισ")),
                         {Reading("G5101", "τίς", "I-NSM"), Reading("G5100", "τις", "X-NSM")})
        self.assertEqual(set(self.forms.bare_readings("αβρααμ")),
                         {Reading("G11", "Ἀβραάμ", "N-GSM-P"), Reading("G11", "Ἀβραάμ", "N-NSM-P")})

    def test_case_shown_by_form(self):
        self.assertFalse(self.forms.case_shown_by_form("G11"))     # one spelling
        self.assertTrue(self.forms.case_shown_by_form("G3475"))    # Μωϋσῆς, Μωϋσέως
        self.assertEqual(case_of("N-GSM-P"), "G")
        self.assertIsNone(case_of("A-NUI"))
        self.assertEqual(reading_kind("N-GSM-T"), "title")
        self.assertEqual(reading_kind("N-NSM-P"), "name")
        self.assertEqual(reading_kind("N-NPM-PG"), "name")
        self.assertEqual(reading_kind("N-GSM"), "common")

    def test_lexicon_entries_and_lookups(self):
        david = self.lexicon.lookup("Δαυίδ")
        self.assertTrue(david.exact)
        self.assertEqual([(e.strongs, e.lemma, e.pos_class, e.name_kind) for e in david.entries],
                         [("G1138", "Δαυείδ, Δαυίδ, Δαβίδ", "proper", "P")])
        tolerant = self.lexicon.lookup("δαυιδ")
        self.assertEqual(tolerant.tolerance, "accents")
        self.assertIsNone(self.lexicon.lookup("δαυιδ", tolerant=False))
        self.assertEqual([e.extended for e in self.lexicon.by_strongs("G0001")], ["G0001G", "G0001H"])
        porch = self.lexicon.lookup("αιλάμ").entries[0]
        self.assertEqual(porch.pos_class, "transliteration")
        self.assertTrue(porch.indeclinable)
        self.assertEqual([e.lemma for e in self.lexicon.of_class("verb")], ["ἀγγέλλω"])
        self.assertIn(normalize("χθές"), self.lexicon.lemma_keys("adverb"))
        self.assertEqual(self.lexicon.lookup("τεσσαράκοντα").entries[0].grammar, "A-NUI")
        # The letter's sign is not indexed as a word; the interjection is.
        self.assertIsNone(self.lexicon.lookup("α", tolerant=False))
        self.assertEqual([e.extended for e in self.lexicon.lookup("ἆ").entries], ["G0001H"])


if __name__ == "__main__":
    unittest.main()
