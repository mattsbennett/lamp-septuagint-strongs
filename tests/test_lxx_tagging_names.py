"""The name stage: Greek names matched to the Hebrew names of the same verse."""

import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import hebrew_names, lexicon, names, ntforms, tag_septuagint, translit  # noqa: E402
from lxx_tagging.candidates import Candidate, Choice, Tagger, annotation_data  # noqa: E402
from lxx_tagging.names import GreekUnit, Match, TokenView, VerseView  # noqa: E402
from lxx_tagging.text import tokenize  # noqa: E402
from lxx_tagging_fixtures import module, write_hebrew_sources, write_sources  # noqa: E402


def verse(ref, text, tags=None, names_=()):
    """A VerseView: the words in `names_` are tier-d candidates, those in
    `tags` (surface -> (strongs, lemma, morph)) tagged, the rest untagged."""
    tags = tags or {}
    views = []
    for token in tokenize(text):
        if token.surface in names_:
            views.append(TokenView(token, "d", "name-like: no candidate"))
        elif token.surface in tags:
            strongs, lemma, morph = tags[token.surface]
            views.append(TokenView(token, "a", "one reading", False, strongs, lemma, morph, "unique"))
        else:
            views.append(TokenView(token, "e", "no candidate"))
    return VerseView(ref, views)


class Sources:
    """The fixture TAGNT, TBESG, TAHOT and TBESH, and a stage over them."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.sources = write_hebrew_sources(write_sources(Path(cls.directory.name)))
        cls.lexicon = lexicon.build(cls.sources / "TBESG.txt")
        cls.forms = ntforms.build(ntforms.tagnt_paths(cls.sources))
        cls.hebrew = hebrew_names.build(cls.sources)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def stage(self, key_mode="hebrew", codes=None, refs=(1010001, 1010016, 1020002, 1020003)):
        return names.build_stage(self.sources, self.lexicon, hebrew_names.VerseMap.identity(refs), key_mode,
                                 codes, hebrew=self.hebrew, forms=self.forms)


class HebrewSideTests(Sources, unittest.TestCase):
    def test_sounds_read_vowel_letters_and_the_furtive_patah(self):
        self.assertEqual(hebrew_names.sounds("אֲבִימֶ֫לֶךְ"), tuple("ʔabimelek"))
        self.assertEqual(hebrew_names.sounds("מוֹאָב"), tuple("moʔab"))      # וֹ is a vowel
        self.assertEqual(hebrew_names.sounds("נֹחַ"), tuple("noaḥ"))         # the patah before its ח
        self.assertEqual(hebrew_names.sounds("שָׂרָה"), tuple("śara"))        # silent final ה, sin
        self.assertEqual(hebrew_names.sounds("דָּוִד"), tuple("dawid"))

    def test_variants_the_septuagint_follows(self):
        variants = dict((name, sounds) for sounds, name in hebrew_names.sound_variants("עֲזַרְיָ֫הוּ"))
        self.assertEqual(variants["yahu"][-2:], ("y", "a"))                  # Ἀζαρίας, not -ιαου
        gentilic = dict((name, s) for s, name in hebrew_names.sound_variants("כְּנַעֲנִי", gentilic=True))
        self.assertEqual(gentilic["gentilic"][-1], "n")                       # Χανανα-ῖος
        two = hebrew_names.sound_variants("נְבוּכַדְנֶאצַּר, נְבוּכַדְרֶאצַּר")
        self.assertEqual(len([v for v in two if v[1] == ""]), 2)               # each spelling its own

    def test_tahot_words_keys_kinds_and_joined_names(self):
        words = {w.gloss: w for w in self.hebrew.words(1020002)}
        self.assertEqual(words["Abimelech"].dstrongs, "H0040G")
        self.assertEqual(words["Abimelech"].key, "H40")                       # the shipped module's key
        self.assertEqual(words["Gerar"].kind, "place")
        self.assertEqual(words["king"].kind, "common")
        gentilics = {w.gloss: w for w in self.hebrew.words(1010016)}
        self.assertEqual((gentilics["Jebusite"].kind, gentilics["Jebusite"].key), ("gentilic", "H2983"))
        # Beth-shemesh over two joined words is one name; the X word is not text.
        self.assertEqual([(w.gloss, w.stem) for w in self.hebrew.words(1020003)],
                         [("Beth-shemesh", "בֵּֽית שֶׁ֖מֶשׁ")])
        self.assertEqual(hebrew_names.parts("בֵּית שֶׁ֫מֶשׁ"), ["בֵּית", "שֶׁ֫מֶשׁ"])

    def test_identity_verse_map(self):
        mapping = hebrew_names.VerseMap.identity([1020002])
        self.assertEqual(mapping[1020002], 1020002)
        self.assertIsNone(mapping[1020005])
        self.assertEqual(hebrew_names.VerseMap.from_json(mapping.to_json()).mapping, mapping.mapping)


class TransliterationTests(Sources, unittest.TestCase):
    def test_a_name_scores_well_against_its_hebrew_and_badly_against_another(self):
        stage = self.stage()
        words = {w.gloss: w for w in self.hebrew.words(1020002)}
        right = stage.score("αβιμελεχ", words["Abimelech"])
        self.assertGreater(right, 10)
        self.assertGreater(right - stage.score("αβιμελεχ", words["Abraham"]), 10)
        self.assertLess(stage.score("αβιμελεχ", words["Gerar"]), 0)

    def test_greek_endings_are_not_evidence_and_gentilics_pair_with_gentilics(self):
        stage = self.stage()
        gerar = {w.gloss: w for w in self.hebrew.words(1020002)}["Gerar"]
        # Γεράρων: a case ending on the transliteration.
        self.assertGreater(stage.score("γεραρων", gerar), 5)
        jebusite = {w.gloss: w for w in self.hebrew.words(1010016)}["Jebusite"]
        self.assertGreater(stage.score("ιεβουσαιον", jebusite), 5)

    def test_declension_stripping(self):
        splits = {(s.ending, s.gentilic) for s, _ in translit.decompositions("αζαριου")}
        self.assertIn(("ου", False), splits)
        self.assertIn(("", False), splits)            # or not declined at all
        self.assertIn(("ιου", True), splits)          # or a gentilic
        # -ου is a genitive in three paradigms; the name's other forms decide.
        self.assertEqual({p for p, _, _, _, _ in names.paradigm_readings("Βαναίου")}, {"os", "as", "es"})
        self.assertIn(("is", "γαλααδιτ", "G", "S", "ιδος"), names.paradigm_readings("Γαλααδίτιδος"))
        self.assertEqual(names.nominative("Βαναίου", "ου", "as"), "Βαναίας")
        self.assertEqual(names.nominative("Μαρδοχαῖον", "ον", "os"), "Μαρδοχαῖος")
        # A long ending may have moved the accent: Ἀσσυρίων is not *Ἀσσυρίος.
        self.assertIsNone(names.nominative("Ἀσσυρίων", "ων", "os"))
        self.assertEqual(names.display_form("Ἀβιμέλεχ"), "Ἀβιμέλεχ")
        self.assertEqual(names.display_form("Ἰωὰς"), "Ἰωάς")


class LemmaTests(Sources, unittest.TestCase):
    def test_one_lemma_per_name_the_nominative_where_it_declines(self):
        def word(key, gloss, kind="person"):
            return hebrew_names.HebrewWord(1, 1, "x", "x", key, "Npl" if kind == "place" else "Npf", kind, "x", gloss)
        leah, gilead, shechem = word("H3812", "Leah"), word("H1568", "Gilead", "place"), word("H7927", "Shechem", "place")
        texts = ["Λεία καὶ Λείας", "Γαλαὰδ τῆς Γαλααδίτιδος", "Σικίμων καὶ Σίκιμα", "ΛΕΙΑ"]
        verses = [verse(1001001 + n, text) for n, text in enumerate(texts)]
        targets = {"λεια": leah, "λειασ": leah, "γαλααδ": gilead, "γαλααδιτιδοσ": gilead, "σικιμων": shechem,
                   "σικιμα": shechem}
        everything = {}
        for number, item in enumerate(verses):
            for position, view in enumerate(item.tokens):
                letters = names.letters_of(view.token.key)
                if letters in targets:
                    everything[(number, position)] = (Match(targets[letters], 10.0, 10.0, "hebrew-verse"),
                                                      GreekUnit(letters, [position], [view.token.surface], "candidate"))
        lemmas = names.lemma_groups(self.stage(), everything, verses)
        by_surface = {verses[v].tokens[p].token.surface: found for (v, p), found in lemmas.items()}
        self.assertEqual(by_surface["Λεία"], ("Λεία", ("N", "S"), "a"))
        self.assertEqual(by_surface["Λείας"], ("Λεία", ("G", "S"), "a"))
        self.assertEqual(by_surface["ΛΕΙΑ"][0], "Λεία")             # capitals take the printed name's accents
        self.assertEqual(by_surface["Γαλαὰδ"], ("Γαλαάδ", None, None))   # undeclined, and not Γαλααδῖτις
        self.assertEqual(by_surface["Σικίμων"], ("Σίκιμα", ("G", "P"), "ta"))


class AssignmentTests(Sources, unittest.TestCase):
    def test_assignment_is_one_to_one_and_maximises_the_total(self):
        # Greedy would give row 0 its best (5) and row 1 the leftover (1).
        self.assertEqual(names.assign([[5.0, 4.0], [4.5, 1.0]]), [1, 0])
        self.assertEqual(names.assign([[0.0, 0.0], [0.0, 3.0]]), [None, 1])
        margins = names.assign_with_margins([[5.0, 4.0], [4.5, 1.0]])
        self.assertEqual([column for column, _, _ in margins], [1, 0])
        self.assertAlmostEqual(margins[0][1], 8.5 - 6.0)

    def test_a_list_of_names_lines_up(self):
        stage = self.stage()
        text = "Αὗται δὲ αἱ γενέσεις τῶν υἱῶν Νῶε, Σὴμ, Χὰμ, Ἰάφεθ"
        found, reasons, _ = stage.run([verse(1010001, text, names_=("Νῶε", "Σὴμ", "Χὰμ", "Ἰάφεθ"))])
        matched = {verse(1010001, text).tokens[p].token.surface: m.hebrew.gloss for (_, p), (m, _) in found.items()}
        self.assertEqual(matched, {"Νῶε": "Noah", "Σὴμ": "Shem", "Χὰμ": "Ham", "Ἰάφεθ": "Japheth"})

    def test_anchored_names_and_names_with_no_hebrew_verse(self):
        stage = self.stage()
        text = "καὶ ἀπέστειλεν Ἀβιμέλεχ βασιλεὺς Γεράρων καὶ Ἀβραὰμ"
        tags = {"Ἀβραὰμ": ("G11", "Ἀβραάμ", "")}
        found, _, _ = stage.run([verse(1020002, text, tags, ("Ἀβιμέλεχ", "Γεράρων"))])
        self.assertEqual(sorted(m.hebrew.key for m, _ in found.values()), ["H1642", "H40"])
        found, reasons, _ = stage.run([verse(1030001, text, tags, ("Ἀβιμέλεχ",))])
        self.assertEqual(found, {})
        self.assertEqual(set(reasons.values()), {"no Hebrew verse"})


class ConsistencyTests(Sources, unittest.TestCase):
    def word(self, key, gloss):
        return hebrew_names.HebrewWord(1, 1, "x", "x", key, "Npm", "person", "שָׁם", gloss)

    def test_the_consistency_pass_tags_an_agreed_form_and_refuses_a_conflicted_one(self):
        stage = self.stage()
        joash, jehoash, abimelech = self.word("H3101", "Joash"), self.word("H3060", "Jehoash"), self.word("H0040G", "Abimelech")

        def unit(letters):
            return GreekUnit(letters, [0], [letters], "candidate")

        found = {
            (0, 0): (Match(abimelech, 12.0, 12.0, "hebrew-verse"), unit("αβιμελεχ")),
            (1, 0): (Match(abimelech, 12.0, 12.0, "hebrew-verse"), unit("αβιμελεχ")),
            (2, 0): (Match(joash, 12.0, 12.0, "hebrew-verse"), unit("ιωασ")),
            (3, 0): (Match(joash, 12.0, 12.0, "hebrew-verse"), unit("ιωασ")),
            (4, 0): (Match(jehoash, 12.0, 12.0, "hebrew-verse"), unit("ιωασ")),
        }
        reasons = {(5, 0): "no matching Hebrew name", (6, 0): "no matching Hebrew name"}
        verse_units = {5: [(unit("αβιμελεχ"), "no matching Hebrew name")],
                       6: [(unit("ιωασ"), "no matching Hebrew name")]}
        added = stage.consistency([], found, reasons, verse_units)
        self.assertEqual({k: (m.hebrew.key, m.method) for k, (m, _) in added.items()},
                         {(5, 0): ("H40", "hebrew-consistency")})
        self.assertIn("several Hebrew numbers", reasons[(6, 0)])

    def test_a_weak_match_the_forms_other_matches_contradict_is_withdrawn(self):
        stage = self.stage()
        elisha, philistine = self.word("H0477", "Elisha"), self.word("H6430", "Philistine")
        unit = GreekUnit("ελισαιε", [0], ["Ἐλισαιέ"], "candidate")
        # Ten times Elisha, confidently; once, weakly, "Philistine".
        found = {(number, 0): (Match(elisha, 10.0, 10.0, "hebrew-verse"), unit) for number in range(10)}
        found[(10, 0)] = (Match(philistine, 3.7, 3.7, "hebrew-verse"), unit)
        added = stage.consistency([], found, {}, {10: [(unit, "")]})
        self.assertNotIn((10, 0), found)
        self.assertEqual(added[(10, 0)][0].hebrew.key, "H477")


class KeyTests(Sources, unittest.TestCase):
    TEXT = "καὶ ἀπέστειλεν ὁ Ἀβιμέλεχ βασιλεὺς Γεράρων"
    TAGS = {"ὁ": ("G3588", "ὁ", "T-NSM")}

    def run_stage(self, key_mode):
        def factory(exclude):
            return names.build_stage(self.sources, self.lexicon, hebrew_names.VerseMap.identity([1020002]), key_mode,
                                     self.forms.morph_codes | {"N-NSM-P"}, hebrew=self.hebrew, exclude=exclude,
                                     forms=self.forms)
        views = [verse(1020002, self.TEXT, self.TAGS, ("Ἀβιμέλεχ", "Γεράρων"))]
        return names.tag_names(views, factory), views

    def test_hebrew_keys_carry_no_morphology(self):
        run, views = self.run_stage("hebrew")
        decision = next(d for (_, p), d in run.decisions.items() if views[0].tokens[p].token.surface == "Ἀβιμέλεχ")
        self.assertEqual((decision.strongs, decision.lemma, decision.morph), ("H40", "Ἀβιμέλεχ", None))
        self.assertEqual(decision.grammar, "N-NSM-P")       # what the article shows, for the decisions file
        self.assertEqual(annotation_data(names.candidate_for(decision)), {"strongs": "H40", "lemma": "Ἀβιμέλεχ"})

    def test_greek_extended_keys_carry_the_code(self):
        run, views = self.run_stage("greek-extended")
        decision = next(d for (_, p), d in run.decisions.items() if views[0].tokens[p].token.surface == "Ἀβιμέλεχ")
        self.assertEqual((decision.strongs, decision.morph), ("G10040", "N-NSM-P"))
        self.assertEqual(names.greek_extended("H1642"), "G11642")

    def test_no_greek_code_on_a_hebrew_key_and_the_guard_accepts_no_code(self):
        run, _ = self.run_stage("hebrew")
        position, decision = next(iter(run.decisions.items()))
        run.decisions = {position: names.NameDecision(**{**vars(decision), "morph": "N-NSM-P"})}
        results = tag_septuagint.Results()
        token = tokenize(self.TEXT)[position[1]]
        from lxx_tagging.candidates import TokenCandidates
        results.append(("Gen", 20, {"ref": 1020002}, [(TokenCandidates(token, []), None)] * (position[1] + 1)))
        with self.assertRaises(ValueError):
            tag_septuagint.apply_names(results, run, None)
        # A choice whose candidate has a Strong's number and lemma but no
        # code passes the Tagger's code guard.
        named = Candidate("H40", "Ἀβιμέλεχ", None, names.SOURCE, 12.0, "name")
        tagger = Tagger([], codes={"N-NSM-P"},
                        chooser=lambda items: [Choice(named, 0.9, 5.0, "hebrew-verse") for _ in items])
        outcome = tagger.tag_text("Ἀβιμέλεχ")
        self.assertEqual(outcome[0][1].candidate.morph, None)


class CommandTests(unittest.TestCase):
    VERSES = {
        ("Gen", 20, 2): "Εἶπε δὲ Ἀβραὰμ περὶ Σάῤῥας καὶ ἀπέστειλεν ὁ Ἀβιμέλεχ βασιλεὺς Γεράρων",
        ("Gen", 1, 1): "ἘΝ ἀρχῇ ἡ βίβλος καὶ Κυρίου.",
    }

    def run_command(self, directory: Path, key_mode: str) -> Path:
        sources = write_hebrew_sources(write_sources(directory / "sources"))
        source = directory / "LXX.json"
        source.write_text(json.dumps(module(self.VERSES), ensure_ascii=False) + "\n", encoding="utf-8")
        output = directory / "out"
        with redirect_stdout(StringIO()):
            tag_septuagint.main(["--input", str(source), "--output-dir", str(output), "--sources", str(sources),
                                 "--chooser-cache", str(directory / "chooser-cache"), "--verse-map", "identity",
                                 "--name-keys", key_mode])
        return output

    def test_names_are_tagged_recorded_and_reported(self):
        with tempfile.TemporaryDirectory() as name:
            output = self.run_command(Path(name), "hebrew")
            tagged = json.loads((output / "LXX_tagged.json").read_text(encoding="utf-8"))
            report = json.loads((output / "lxx_tagging_report.json").read_text(encoding="utf-8"))
            with open(output / "lxx_tagging_decisions.tsv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
        verse = next(v for b in tagged["books"] for c in b["chapters"] for v in c["verses"] if v["ref"] == 1020002)
        data = {a["text"]: a["data"] for a in verse["content"]["annotations"]}
        self.assertEqual(data["Ἀβιμέλεχ"], {"strongs": "H40", "lemma": "Ἀβιμέλεχ"})
        self.assertEqual(data["Ἀβραὰμ"]["strongs"], "G11")          # the NT's name keeps its Greek number
        row = next(r for r in rows if r["surface"] == "Ἀβιμέλεχ")
        self.assertEqual((row["method"], row["morph"], row["name_key"]), ("hebrew-verse", "", "tahot"))
        self.assertIn("H0040G", row["name_hebrew"])
        self.assertEqual(row["name_grammar"], "N-NSM-P")
        self.assertEqual(list(rows[0]), list(tag_septuagint.DECISION_COLUMNS))
        self.assertEqual(report["names"]["keyMode"], "hebrew")
        self.assertGreaterEqual(report["names"]["tagged"]["byMethod"]["hebrew-verse"], 2)
        self.assertEqual(report["tagged"], sum(1 for r in rows if r["strongs"]))

    @staticmethod
    def rows(output: Path) -> list[dict]:
        with open(output / "lxx_tagging_decisions.tsv", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))

    def test_greek_extended_and_off(self):
        with tempfile.TemporaryDirectory() as name:
            output = self.run_command(Path(name), "greek-extended")
            row = next(r for r in self.rows(output) if r["surface"] == "Ἀβιμέλεχ")
            self.assertEqual((row["strongs"], row["morph"]), ("G10040", "N-NSM-P"))
        with tempfile.TemporaryDirectory() as name:
            output = self.run_command(Path(name), "off")
            rows = self.rows(output)
            row = next(r for r in rows if r["surface"] == "Ἀβιμέλεχ")
            self.assertEqual((row["tier"], row["strongs"]), ("d", ""))
            report = json.loads((output / "lxx_tagging_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("names", report)


class HashOrderTests(unittest.TestCase):
    """The stage gives the same result whatever Python's string hashing.

    A set of strings is iterated in an order that changes from process to
    process (PYTHONHASHSEED), so two runs in one process cannot see it: this
    runs the stage on the fixture in subprocesses under several seeds. The
    fixture gains the two ties that once leaked set order into the model:
    Jerusalem's lamed pointed with both patah and hiriq (which vowel
    `hebrew_names.sounds` read was the set's first), and TBESG's two rows of
    G7 Ἀβιά, whose TBESH lemmas differ (the order `gold_pairs` gave them, and
    so the order the model was fitted in)."""

    SEEDS = ("0", "1", "2", "3")
    SCRIPT = r"""
import json, sys
from pathlib import Path
root, directory = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(root / "scripts"), str(root / "tests")]
from lxx_tagging import hebrew_names, lexicon, names, ntforms
from lxx_tagging.names import TokenView, VerseView
from lxx_tagging.text import tokenize
from lxx_tagging_fixtures import tahot_row, tbesg_row, write_hebrew_sources, write_sources

sources = write_hebrew_sources(write_sources(directory))
JERUSALEM = "יְרוּשָׁלִַם"
ABIJAH, ABIJAHU = "אֲבִיָּה", "אֲבִיָּהוּ"
with open(sources / "TBESG.txt", "a", encoding="utf-8") as handle:
    handle.write(tbesg_row("G0007", "G0007G = the Greek of", "H0029I", "Ἀβιά", "N:N-M-P", "Abijah") + "\n")
    handle.write(tbesg_row("G0007", "G0007H = the Greek of", "H0029L", "Ἀβιά", "N:N-M-P", "Abijah") + "\n")
with open(sources / "TBESH.txt", "a", encoding="utf-8") as handle:
    handle.write(f"H0029\tH0029I =\tH0029I\t{ABIJAHU}, {ABIJAH}\tx\tN:N-M-P\tAbijah\tdef\n")
    handle.write(f"H0029\tH0029L =\tH0029L\t{ABIJAH}\tx\tN:N-M-P\tAbijah\tdef\n")
    handle.write(f"H3389\tH3389 =\tH3389\t{JERUSALEM}\tx\tN:N--L\tJerusalem\tdef\n")
with open(sources / "TAHOT_Gen-Deu.txt", "a", encoding="utf-8") as handle:
    handle.write(tahot_row("Gen.20.4#01=L", JERUSALEM, "{H3389}", "HNpl", JERUSALEM, "Jerusalem") + "\n")
    handle.write(tahot_row("Gen.20.4#02=L", ABIJAH, "{H0029L}", "HNpm", ABIJAH, "Abijah") + "\n")

lexicon_ = lexicon.build(sources / "TBESG.txt")
forms = ntforms.build(ntforms.tagnt_paths(sources))
hebrew = hebrew_names.build(sources)
tbesh = hebrew_names.read_tbesh(sources / "TBESH.txt")
texts = {1020002: ("Ἀβιμέλεχ", "Γεράρων"),
         1020004: ("Ἰερουσαλὴμ", "Ἀβιὰς")}
verses = [VerseView(ref, [TokenView(token, "d", "name-like: no candidate") for token in tokenize(" ".join(words))])
          for ref, words in texts.items()]

def factory(exclude):
    return names.build_stage(sources, lexicon_, hebrew_names.VerseMap.identity(list(texts)), "hebrew", None,
                             hebrew=hebrew, exclude=exclude, forms=forms, tbesh=tbesh)

stage = factory(set())
run = names.tag_names(verses, factory)
print(json.dumps({
    "sounds": hebrew_names.sound_variants(JERUSALEM, False),
    "gold": [repr(pair) for pair in names.gold_pairs(lexicon_, tbesh)],
    "model": stage.model.to_json(),
    "fitted": run.stage.model.to_json(),
    "decisions": sorted(repr((position, vars(decision))) for position, decision in run.decisions.items()),
    "found": sorted(repr((position, match.hebrew.key, match.score, match.margin, match.method))
                    for position, (match, _) in run.found.items()),
    "notes": sorted(map(repr, run.notes.items())),
    "report": names.report(run),
}, sort_keys=True))
"""

    def test_the_same_result_under_every_hash_seed(self):
        import os
        import subprocess
        outputs = {}
        for seed in self.SEEDS:
            with tempfile.TemporaryDirectory() as name:
                env = {**os.environ, "PYTHONHASHSEED": seed}
                done = subprocess.run([sys.executable, "-c", self.SCRIPT, str(REPOSITORY_ROOT), name], env=env,
                                      cwd=name, capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr)
            outputs[seed] = json.loads(done.stdout)
        first = outputs[self.SEEDS[0]]
        self.maxDiff = 2000
        for seed in self.SEEDS[1:]:
            for part in first:
                self.assertEqual(outputs[seed][part], first[part], f"{part} differs under PYTHONHASHSEED={seed}")
        # The fixture does exercise both ties: Jerusalem read with its patah
        # (the lamed's own vowel, `hebrew_names.VOWEL_RANK`), and both of
        # Abijah's gold pairs, which the stage matched.
        self.assertEqual([list(sounds) for sounds, _ in first["sounds"]], [list("yərušalam")])
        self.assertEqual(sum("'G7'" in pair for pair in first["gold"]), 2)
        self.assertEqual(sorted(found.split("'")[1] for found in first["found"]), ["H1642", "H29", "H3389", "H40"])


if __name__ == "__main__":
    unittest.main()
