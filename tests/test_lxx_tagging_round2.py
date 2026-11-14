"""Round 2 of the Septuagint tagger: the parse of loosely matched names,
capitalised common words, whether a name transliterates the Hebrew or stands
for it, association across verses, misprints, TBESG lemmas without a
breathing, and the chooser's argument and coordination features.

Every word here is invented or taken from the public-domain Brenton text;
nothing comes from a reference tagging."""

import dataclasses
import sys
import tempfile
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import capitals, chooser, lexicon, name_parse, names, ntforms, tag_septuagint  # noqa: E402
from lxx_tagging.candidates import Candidate, Choice, TokenCandidates, default_tagger  # noqa: E402
from lxx_tagging.names import GreekUnit, HebrewWord, Match, NameRun, Settings, TokenView, VerseView  # noqa: E402
from lxx_tagging.text import MISPRINTS, normalize, tokenize  # noqa: E402
from lxx_tagging_fixtures import write_sources  # noqa: E402


class NameParseTests(unittest.TestCase):
    def case(self, surface, lemma, previous=None, parses=()):
        return name_parse.name_case(surface, normalize(surface), lemma, parses, previous)[::3]

    def test_rules_read_the_case_from_the_word_and_its_context(self):
        self.assertEqual(self.case("Ῥεβέκκᾳ", "Ῥεβέκκα"), ("D", "subscript"))
        self.assertEqual(self.case("Ἰσαακ", "Ἰσαάκ", name_parse.Neighbour("τῷ", "ὁ", "T-DSM")), ("D", "article"))
        self.assertEqual(self.case("Μαδίαμ", "Μαδιάν", name_parse.Neighbour("ἐν", "ἐν", "PREP")),
                         ("D", "preposition"))
        self.assertEqual(self.case("Ἡσαΐαν", "Ἡσαΐας"), ("A", "ending"))
        self.assertEqual(self.case("Νάθαν", "Ναθάν", name_parse.Neighbour("υἱὸς", normalize("υἱός"), "N-NSM")),
                         ("G", "kinship"))
        self.assertEqual(self.case("Ἄχαζ", "Ἀχάζ", name_parse.Neighbour("βασιλεὺς", normalize("βασιλεύς"), "N-NSM")),
                         ("N", "apposition"))

    def test_an_undeclined_spelling_is_not_read_as_the_nt_forms_case(self):
        # Ἠλιού is not Ἡλίου, the genitive of Ἡλίας: its accent has moved.
        self.assertEqual(self.case("Ἠλιοὺ", "Ἡλίας", None, ["N-GSM-P"]), (None, ""))
        # Μωυσῆ has no subscript, so it is no dative.
        self.assertTrue(name_parse.no_dative("Μωυσῆ"))
        self.assertFalse(name_parse.no_dative("Μωυσῇ"))

    def test_apply_rewrites_the_parse_and_caps_an_unknown_ones_confidence(self):
        tokens = tokenize("τῷ Ἰσαὰκ καὶ Ἄχαζ")
        article = Candidate("G3588", "ὁ", "T-DSM", "nt-form", 10)
        isaac = Candidate("G2464", "Ἰσαάκ", "N-ASM-P", "nt-tolerant", 6, "name", False)
        conj = Candidate("G2532", "καί", "CONJ", "nt-form", 10)
        ahaz = Candidate("G881", "Ἀχάζ", "N-ASM-P", "nt-tolerant", 1, "name", False)
        outcome = [
            (TokenCandidates(tokens[0], [article], [], "a"), Choice(article, 0.99, 9, "unique")),
            (TokenCandidates(tokens[1], [isaac], [], "u"), Choice(isaac, 0.93, 5, "tolerant")),
            (TokenCandidates(tokens[2], [conj], [], "a"), Choice(conj, 0.99, 9, "unique")),
            (TokenCandidates(tokens[3], [ahaz], [], "u"), Choice(ahaz, 0.93, 5, "tolerant")),
        ]
        results = [("Gen", 1, {"ref": 1001001}, outcome)]
        done = name_parse.apply(results, {"N-DSM-P", "N-ASM-P", "N-NSM-P"})
        self.assertEqual(done, {(0, 1): "article", (0, 3): "unknown"})
        self.assertEqual(outcome[1][1].candidate.morph, "N-DSM-P")
        self.assertEqual(outcome[1][1].confidence, name_parse.RULE_CONFIDENCE["article"])
        self.assertEqual(outcome[3][1].candidate.morph, "N-ASM-P")
        self.assertLessEqual(outcome[3][1].confidence, name_parse.UNKNOWN_CONFIDENCE)

    def test_a_code_tagnt_lacks_is_not_written(self):
        isaac = Candidate("G2464", "Ἰσαάκ", "N-ASM-P", "nt-tolerant", 6, "name", False)
        candidate, rule = name_parse.reparse(isaac, "Ἰσαακ", "ἰσαακ", [], name_parse.Neighbour("τῷ", "ὁ", "T-DSM"),
                                             {"N-ASM-P"})
        self.assertEqual((candidate.morph, rule), ("N-ASM-P", ""))


class CapitalsTests(unittest.TestCase):
    class Tagger:
        """A stand-in tagger: the lower-case readings of each key."""

        def __init__(self, readings):
            self.readings = readings

        def candidates_for(self, token):
            found = self.readings.get(token.key, [])
            return TokenCandidates(token, found, [], "a" if found else "e")

    def item(self, text, index):
        return TokenCandidates(tokenize(text)[index], [], [], "d")

    def test_a_compass_point_with_no_hebrew_name_is_offered_as_a_common_word(self):
        north = Candidate("G1005", "βορρᾶς", "N-ASM", "nt-form", 2)
        tagger = self.Tagger({normalize("Βοῤῥᾶν"): [north]})
        found = capitals.lower_case_candidates(tagger, self.item("πρὸς Βοῤῥᾶν", 1),
                                               "no Hebrew name in the verse", {"G1005"})
        self.assertEqual((found.tier, found.reason, found.candidates), ("a", capitals.REASON, [north]))

    def test_names_and_guesses_are_not(self):
        north = Candidate("G1005", "βορρᾶς", "N-ASM", "nt-form", 2)
        tagger = self.Tagger({normalize("Βοῤῥᾶν"): [north]})
        item = self.item("πρὸς Βοῤῥᾶν", 1)
        # A Hebrew name of the verse may be its counterpart: not offered.
        self.assertIsNone(capitals.lower_case_candidates(
            tagger, item, "no matching Hebrew name; a Hebrew name of the verse may be its counterpart", {"G1005"}))
        # Brenton never prints the word in lower case: not offered.
        self.assertIsNone(capitals.lower_case_candidates(tagger, item, "no Hebrew name in the verse", set()))
        # An accent-mismatched analyser reading is a guess.
        guess = Candidate("G4476", "ῥαφίς", "N-NSF", "analyser", 0.3)
        self.assertFalse(capitals.acceptable(guess, False))
        # A capitalised verb is likelier a name, unless an article agrees (ὁ Ὤν).
        being = Candidate("G1510", "εἰμί", "V-PAP-NSM", "nt-form", 47)
        tagger = self.Tagger({normalize("Ὢν"): [being]})
        self.assertIsNone(capitals.lower_case_candidates(tagger, self.item("καὶ Ὢν", 1),
                                                         "no Hebrew name in the verse", {"G1510"}, "CONJ"))
        self.assertIsNotNone(capitals.lower_case_candidates(tagger, self.item("ὁ Ὢν", 1),
                                                            "no Hebrew name in the verse", {"G1510"}, "T-NSM"))


def word(key, gloss, ref=1, index=0):
    return HebrewWord(ref, index, "x", "x", key, "Npm", "person", "שָׁם", gloss)


class StubStage:
    """What `name_relations` and `associate` read of a stage."""

    settings = Settings()

    def __init__(self, scores=None, hebrew=None):
        self.scores = scores or {}
        self.hebrew_by_ref = hebrew or {}

    def unified(self, w):
        return {w.key}

    def confident(self, match):
        return match.method == "hebrew-verse" and match.score >= self.settings.confident

    def score(self, letters, w):
        return self.scores.get((letters, w.key), 0.0)

    def role(self, view, gold=None):
        return "candidate" if view.tier == "d" and not view.strongs else None

    def target(self, ref):
        return ref if ref in self.hebrew_by_ref else None

    def names_of(self, ref):
        return self.hebrew_by_ref.get(ref, [])

    def anchored(self, verse, roles):
        return {}

    def free(self, words, anchored):
        return list(words)


class RelationTests(unittest.TestCase):
    def run_of(self, found, stage):
        return NameRun({p: None for p in found}, {}, {}, found, None, {}, 0, stage)

    def test_another_name_printed_there_stands_for_the_hebrew_one(self):
        azariah, zechariah = word("H5838", "Azariah"), word("H2148", "Zechariah")
        unit = GreekUnit("αζαριαν", [0], ["Ἀζαρίαν"], "candidate")
        stage = StubStage({("αζαριαν", "H5838"): 14.0, ("αζαριαν", "H2148"): 4.0})
        found = {(v, 0): (Match(azariah, 14.0, 10.0, "hebrew-verse"), unit) for v in range(3)}
        found[(3, 0)] = (Match(zechariah, 4.0, 3.0, "hebrew-verse"), unit)        # verse-level, but another name
        found[(4, 0)] = (Match(zechariah, 0.5, 1.0, "hebrew-order"), unit)        # placed by order, a poor reading
        relations = names.name_relations(self.run_of(found, stage))
        self.assertEqual([relations[(v, 0)] for v in range(5)],
                         ["transliterates"] * 3 + ["stands-for", "stands-for"])

    def test_a_regular_spelling_the_model_reads_poorly_transliterates(self):
        ezra = word("H5830", "Ezra")
        unit = GreekUnit("εσδρασ", [0], ["Ἔσδρας"], "candidate")
        found = {(v, 0): (Match(ezra, 0.4, 1.0, "hebrew-order"), unit) for v in range(3)}
        relations = names.name_relations(self.run_of(found, StubStage()))
        self.assertEqual(set(relations.values()), {"transliterates"})


class AssociationTests(unittest.TestCase):
    def test_a_form_whose_verses_always_have_one_unaccounted_name_takes_it(self):
        cush, egypt = word("H3568", "Cush"), word("H4714", "Egypt")
        verses, hebrew = [], {}
        for number in range(4):
            ref = 1010000 + number
            tokens = tokenize("καὶ Αἰθιοπίας καὶ Αἰγύπτου")
            views = [TokenView(tokens[0], "a", strongs="G2532"), TokenView(tokens[1], "d"),
                     TokenView(tokens[2], "a", strongs="G2532"), TokenView(tokens[3], "a", strongs="G125")]
            verses.append(VerseView(ref, views))
            hebrew[ref] = [cush, egypt] if number < 3 else [egypt]
        stage = StubStage(hebrew=hebrew)
        # Egypt is accounted for (matched) in every verse; Cush is left.
        everything = {(n, 3): (Match(egypt, 12.0, 9.0, "hebrew-verse"), GreekUnit("αιγυπτου", [3], [], "anchor"))
                      for n in range(4)}
        added = names.associate(stage, verses, everything)
        # Three of four verses have Cush: 75%, under the 90% share.
        self.assertEqual(added, {})
        hebrew[1010003] = [cush, egypt]
        added = names.associate(stage, verses, everything)
        self.assertEqual({p: m.hebrew.key for p, (m, _) in added.items()},
                         {(n, 1): "H3568" for n in range(4)})
        self.assertEqual({m.method for m, _ in added.values()}, {"hebrew-association"})


class TextAndLexiconTests(unittest.TestCase):
    def test_misprints_are_listed_and_marked(self):
        token = tokenize("γἱοὶ δὲ Ῥουβήν")[0]
        self.assertEqual((token.key, "misprint" in token.rules, token.surface), ("υἱοί", True, "γἱοὶ"))
        self.assertNotIn("misprint", tokenize("υἱοὶ δὲ")[0].rules)
        self.assertEqual(MISPRINTS[normalize("Δέγει")], normalize("Λέγει"))

    def test_a_tbesg_lemma_printed_without_a_breathing_matches(self):
        with tempfile.TemporaryDirectory() as name:
            sources = write_sources(Path(name))
            tagger = default_tagger(ntforms.build(ntforms.tagnt_paths(sources)),
                                    lexicon.build(sources / "TBESG.txt"))
            found = tagger.candidates_for(tokenize("τὸ αἰλὰμ")[1])
        self.assertEqual([(c.strongs, c.morph) for c in found.candidates], [("G6161", None)])


class ChooserFeatureTests(unittest.TestCase):
    def test_verb_objects_counts_certain_accusatives_after_a_verb(self):
        def item(text, morph, lemma, index=0, position="medial"):
            token = tokenize(text)[index]
            token = dataclasses.replace(token, position=position)
            return TokenCandidates(token, [Candidate("G1", lemma, morph, "nt-form", 1)], [], "a")
        items = [item("ἐποίησε", "V-AAI-3S", "ποιέω", position="initial"), item("τὸν", "T-ASM", "ὁ"),
                 item("ἦλθε", "V-2AAI-3S", "ἔρχομαι", position="initial"), item("καὶ", "CONJ", "καί")]
        counts = chooser.verb_objects(items)
        self.assertEqual(counts[chooser.lemma_key("ποιέω")], (1, 1))
        self.assertEqual(counts[chooser.lemma_key("ἔρχομαι")], (1, 0))
        self.assertEqual(chooser._transitivity(counts, chooser.lemma_key("ποιέω")), "u")

    def test_coordination_and_a_subordinate_clause_reset_the_state(self):
        readings = chooser.Readings()

        def rid(morph, lemma):
            option = chooser.Option(Candidate("G1", lemma, morph, "nt-form", 1), ("G1", lemma, morph, "common"),
                                    chooser.parse_morph(morph), chooser.lemma_key(lemma), 1.0, (), True)
            return readings.get(option)
        state = chooser.INITIAL_STATE
        for morph, lemma in (("N-APN", "κτῆνος"), ("P-GPM", "αὐτός"), ("CONJ", "καί")):
            state = chooser.next_state(readings, state, rid(morph, lemma))
        self.assertEqual(state[11], "AP")         # the possessive leaves the noun the first member
        feats = chooser.state_features(readings, state, rid("A-APN", "πᾶς"))
        self.assertIn("CO:A|AA", feats)
        state = chooser.next_state(readings, state, rid("V-AAI-3P", "ποιέω"))
        state = chooser.next_state(readings, state, rid("CONJ", "ὅτι"))
        self.assertEqual(state[4], "")            # ὅτι's clause has its own verb


if __name__ == "__main__":
    unittest.main()
