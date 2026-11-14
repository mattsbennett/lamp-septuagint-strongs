import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from lxx_tagging.text import (  # noqa: E402
    alternates, bare, caps_pattern, licensed, normalize, normalize_with_rules, other_breathing,
    split_joined, tokenize,
)


class NormalisationTests(unittest.TestCase):
    """Each key rule folds what never distinguishes words, and nothing else."""

    def test_case_grave_and_final_sigma_fold(self):
        self.assertEqual(normalize("Θεὸς"), normalize("θεός"))
        self.assertEqual(normalize("θεός"), "θεόσ")

    def test_a_second_accent_from_an_enclitic_is_dropped(self):
        key, rules = normalize_with_rules("ὄνομά")
        self.assertEqual(key, "ὄνομα")
        self.assertIn("enclitic-accent", rules)
        self.assertEqual(normalize("δίδωσί"), normalize("δίδωσι"))

    def test_elision_marks_are_one_mark(self):
        # Brenton's U+02BC, TAGNT's koronis U+1FBD and the typographic apostrophe.
        self.assertEqual(normalize("καθʼ"), normalize("καθ᾽"))
        self.assertEqual(normalize("καθʼ"), normalize("καθ’"))

    def test_rho_breathing_diaeresis_and_crasis_subscript(self):
        self.assertEqual(normalize("ἐῤῥέθη"), normalize("ἐρρέθη"))
        self.assertEqual(normalize("Μωυσῆς"), normalize("Μωϋσῆς"))
        key, rules = normalize_with_rules("κᾀγὼ")
        self.assertEqual(key, normalize("κἀγώ"))
        self.assertIn("crasis-subscript", rules)

    def test_grave_on_monosyllabic_tis_is_the_indefinite(self):
        self.assertEqual(normalize("τὶς"), normalize("τις"))
        self.assertNotEqual(normalize("τὶς"), normalize("τίς"))
        self.assertEqual(normalize("τὶ"), "τι")

    def test_accent_breathing_and_subscript_distinctions_survive(self):
        for pair in (("ἤ", "ἦ"), ("ἤ", "ᾗ"), ("εἰς", "εἷς"), ("ἐν", "ἕν"), ("αὐτή", "αὕτη"),
                     ("τίς", "τις"), ("ὁ", "ὅ"), ("ἄρα", "ἆρα"), ("ἀλλά", "ἄλλα"), ("ἡ", "ᾗ")):
            with self.subTest(pair=pair):
                self.assertNotEqual(normalize(pair[0]), normalize(pair[1]))
        # A grave is an acute before another word, so ὃ is still the relative.
        self.assertEqual(normalize("ὃ"), normalize("ὅ"))
        self.assertNotEqual(normalize("ὃ"), normalize("ὁ"))

    def test_editorial_dot_and_leading_apostrophe(self):
        self.assertEqual(normalize("Ὠ̣ΔΗΣ"), normalize("ὨΔΗΣ"))
        key, rules = normalize_with_rules("ʼΑγὰγ")
        self.assertEqual(key, normalize("Ἀγάγ"))
        self.assertIn("apostrophe-breathing", rules)

    def test_bare_and_capitals_pattern(self):
        self.assertEqual(bare("ἀρχῇ"), "αρχη")
        # ἘΝ shows its breathing; ΤΟΥ shows nothing, so only letters count.
        self.assertEqual(caps_pattern(normalize("ἘΝ"), True), caps_pattern(normalize("ἐν"), True))
        self.assertNotEqual(caps_pattern(normalize("ἐν"), True), caps_pattern(normalize("ἕν"), True))
        self.assertEqual(caps_pattern(normalize("ΤΟΥ"), False), "του")

    def test_other_breathing_is_on_the_initial_vowel_or_diphthong(self):
        self.assertEqual(other_breathing(normalize("Ἁβραάμ")), normalize("Ἀβραάμ"))
        self.assertEqual(other_breathing(normalize("Αὐσίτιδι")), normalize("Αὑσίτιδι"))
        self.assertIsNone(other_breathing(normalize("Κύριος")))


class AlternateTests(unittest.TestCase):
    def keys(self, word):
        return {(alternate.key, alternate.rule) for alternate in alternates(normalize(word))}

    def test_movable_nu_after_e_and_si_but_not_ei(self):
        self.assertIn((normalize("εἶπεν"), "movable-nu"), self.keys("εἶπε"))
        self.assertIn((normalize("πᾶσιν"), "movable-nu"), self.keys("πᾶσι"))
        self.assertIn((normalize("ἐστίν"), "movable-nu"), self.keys("ἐστί"))
        # λέγει is not λέγειν, nor φαίνειν φαίνει.
        self.assertFalse(any(rule == "movable-nu" for _, rule in self.keys("λέγει")))
        self.assertFalse(any(rule == "movable-nu" for _, rule in self.keys("φαίνειν")))

    def test_movable_nu_is_licensed_only_where_greek_has_it(self):
        self.assertTrue(licensed("movable-nu", "εἶπεν", "εἶπον", "V-2AAI-3S"))
        self.assertTrue(licensed("movable-nu", "πᾶσιν", "πᾶς", "A-DPM"))
        self.assertTrue(licensed("movable-nu", normalize("ἔμπροσθεν"), "ἔμπροσθεν", "PREP"))
        self.assertFalse(licensed("movable-nu", "λέγειν", "λέγω", "V-PAN"))
        self.assertFalse(licensed("movable-nu", "μέν", "μέν", "PRT"))

    def test_word_specific_alternates(self):
        self.assertIn((normalize("οὕτως"), "movable-sigma"), self.keys("οὕτω"))
        self.assertIn((normalize("ἐξ"), "ek-ex"), self.keys("ἐκ"))
        self.assertIn((normalize("οὐχ"), "ou-ouk-ouch"), self.keys("οὐκ"))
        self.assertIn((normalize("λήμψεται"), "lemps"), self.keys("λήψεται"))
        self.assertIn((normalize("ἐλήμφθη"), "lemps"), self.keys("ἐλήφθη"))
        self.assertTrue(licensed("lemps", "λήμψεται", "λαμβάνω", "V-FDI-3S"))
        self.assertFalse(licensed("lemps", "λήμψεται", "λείπω", "V-FDI-3S"))
        self.assertFalse(licensed("ou-ouk-ouch", "οὐχ", "οὐχί", "PRT-I"))


class TokenTests(unittest.TestCase):
    def test_offsets_are_exact_and_punctuation_is_outside(self):
        text = "Καὶ εἶπεν ὁ Θεὸς, γενηθήτω φῶς· καὶ ἐγένετο φῶς."
        tokens = tokenize(text)
        self.assertEqual([t.surface for t in tokens],
                         ["Καὶ", "εἶπεν", "ὁ", "Θεὸς", "γενηθήτω", "φῶς", "καὶ", "ἐγένετο", "φῶς"])
        for token in tokens:
            self.assertEqual(text[token.start:token.end], token.surface)
        self.assertEqual([t.index for t in tokens], list(range(len(tokens))))

    def test_elided_word_keeps_its_apostrophe_in_the_span(self):
        text = "καὶ καθʼ ὁμοίωσιν"
        token = tokenize(text)[1]
        self.assertEqual(token.surface, "καθʼ")
        self.assertTrue(token.elided)
        self.assertEqual(token.key, "καθ’")

    def test_position_and_case(self):
        tokens = tokenize("Ἡ δὲ γῆ, Κύριε, ἐν Σιὼν. Ὁ λαὸς ΚΑΙ")
        by_surface = {t.surface: t for t in tokens}
        self.assertEqual((by_surface["Ἡ"].position, by_surface["Ἡ"].case), ("initial", "capital"))
        self.assertEqual(by_surface["Κύριε"].position, "pause")
        self.assertTrue(by_surface["Σιὼν"].name_like)
        self.assertEqual(by_surface["Ὁ"].position, "initial")
        self.assertFalse(by_surface["Ὁ"].name_like)
        self.assertEqual(by_surface["ΚΑΙ"].case, "caps")
        self.assertFalse(by_surface["ΚΑΙ"].first_marks_known)
        self.assertTrue(tokenize("ἘΝ ἀρχῇ")[0].first_marks_known)

    def test_a_joined_word_is_split_into_spans_of_its_own(self):
        text = "Διατοῦτο οὕτως ἐπιτοαυτὸ"
        tokens = tokenize(text)
        self.assertEqual([(t.surface, t.key, t.part, t.parts) for t in tokens], [
            ("Δια", "διά", 0, 2), ("τοῦτο", "τοῦτο", 1, 2), ("οὕτως", "οὕτωσ", 0, 1),
            ("ἐπι", "ἐπί", 0, 3), ("το", "τό", 1, 3), ("αυτὸ", "αὐτό", 2, 3),
        ])
        for token in tokens:
            self.assertEqual(text[token.start:token.end], token.surface)
        first, second = tokens[0], tokens[1]
        self.assertEqual((first.word_start, first.word_end), (0, 8))
        self.assertEqual((second.word_start, second.word_end), (0, 8))
        self.assertEqual(first.case, "capital")
        self.assertEqual((second.case, second.position), ("lower", "medial"))
        self.assertEqual(second.word_key, normalize("Διατοῦτο"))

    def test_joined_words_match_on_letters_alone(self):
        self.assertEqual([key for _, _, key in split_joined("ἱνατί")], ["ἵνα", "τί"])
        self.assertEqual([key for _, _, key in split_joined("ἀναμέσον")], ["ἀνά", "μέσον"])
        self.assertIsNone(split_joined("διαφθορά"))


if __name__ == "__main__":
    unittest.main()
