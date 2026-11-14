"""The contextual chooser: what it learns, determinism, calibration, the cache."""

import gzip
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import chooser, ntcorpus, ntforms  # noqa: E402
from lxx_tagging.candidates import Candidate, TokenCandidates  # noqa: E402
from lxx_tagging.pipeline import TaggerSpec  # noqa: E402
from lxx_tagging.ntforms import Reading  # noqa: E402
from lxx_tagging.text import tokenize  # noqa: E402
from lxx_tagging_fixtures import TAGNT_ROWS, write_sources  # noqa: E402

# Invented nouns, so nothing but context can decide their case.
NOUNS = ("βαλα", "γαλα", "δαλα", "ζαλα", "θαλα", "καλα", "λαλα", "μαλα", "ναλα", "παλα", "ραλα", "σαλα")


def item(token, readings, tier=None):
    """A token's TokenCandidates from (strongs, lemma, morph, weight) tuples."""
    candidates = [Candidate(s, l, m, "nt-form", w) for s, l, m, w in readings]
    if tier is None:
        tier = "a" if len({c.reading for c in candidates}) == 1 else (
            "b" if len({(c.strongs, c.lemma) for c in candidates}) == 1 else "c")
    return TokenCandidates(token, candidates, [], tier, "")


def verse(words):
    """words: [(surface, readings, gold morph or None)] -> (TokenCandidates list, gold verse)."""
    tokens = tokenize(" ".join(surface for surface, _, _ in words))
    items, gold = [], []
    for token, (_, readings, morph) in zip(tokens, words):
        items.append(item(token, readings))
        chosen = next((r for r in readings if r[2] == morph), None) if morph else readings[0]
        gold.append(Reading(chosen[0], chosen[1], chosen[2]))
    return items, SimpleNamespace(gold=gold)


def article(morph):
    return [("G3588", "ὁ", morph, 100.0)]


def noun(index, codes, weights=None):
    weights = weights or [1.0] * len(codes)
    return [(f"G{9000 + index}", NOUNS[index], code, weight) for code, weight in zip(codes, weights)]


def train_on(pairs, epochs=4):
    model = chooser.Model()
    examples = []
    for items, gold in pairs:
        slots = model.lattice(items)
        examples.append(chooser.Example(slots, [chooser.gold_index(s, r) for s, r in zip(slots, gold.gold)]))
    return chooser.train(examples, model, epochs=epochs)


def choose(model, items):
    return chooser.ModelChooser(model, chooser.Calibration({}))(items)


class MorphologyTests(unittest.TestCase):
    def test_codes_decompose_into_fields(self):
        a = chooser.parse_morph("N-GSM-P")
        self.assertEqual((a.pos, a.case, a.num, a.gen, a.suffix), ("N", "G", "S", "M", "P"))
        v = chooser.parse_morph("V-2AAI-3S")
        self.assertEqual((v.cls, v.tense, v.voice, v.mood, v.person, v.num), ("VF", "2A", "A", "I", "3", "S"))
        p = chooser.parse_morph("V-PAP-NSM")
        self.assertEqual((p.cls, p.case, p.num, p.gen), ("VP", "N", "S", "M"))
        self.assertEqual(chooser.parse_morph("V-PAN").cls, "VN")
        self.assertEqual((chooser.parse_morph("P-1NS").person, chooser.parse_morph("P-1NS").case), ("1", "N"))
        self.assertEqual(chooser.parse_morph("S-1SGSN")[2:5], ("G", "S", "N"))
        crasis = chooser.parse_morph("CONJ + G1565=D-NSM")
        self.assertEqual((crasis.pos, crasis.case, crasis.compound), ("CONJ", "N", True))
        self.assertEqual(chooser.parse_morph(None).pos, "NAME")


class LearningTests(unittest.TestCase):
    def test_agreement_with_the_article_decides_a_nouns_case(self):
        training = []
        for index in range(8):
            for art, gold in (("T-GSM", "N-GSM"), ("T-APM", "N-APM")):
                words = [("τοῦ" if art == "T-GSM" else "τοὺς", article(art), art),
                         (NOUNS[index], noun(index, ["N-GSM", "N-APM"], [3.0, 1.0]), gold)]
                training.append(verse(words))
        model = train_on(training)
        for art, expected in (("T-GSM", "N-GSM"), ("T-APM", "N-APM")):
            items, _ = verse([("τοῦ" if art == "T-GSM" else "τοὺς", article(art), art),
                              (NOUNS[10], noun(10, ["N-GSM", "N-APM"], [3.0, 1.0]), None)])
            choices = choose(model, items)
            self.assertEqual(choices[1].candidate.morph, expected)
            self.assertEqual(choices[1].method, "model")
            self.assertEqual(choices[0].method, "unique")

    def test_a_preposition_governs_the_case_after_it(self):
        training = []
        for index in range(8):
            for preposition, lemma, strongs, gold in (("ἐν", "ἐν", "G1722", "N-DSF"), ("εἰς", "εἰς", "G1519", "N-ASF")):
                words = [(preposition, [(strongs, lemma, "PREP", 50.0)], "PREP"),
                         (NOUNS[index], noun(index, ["N-DSF", "N-ASF"]), gold)]
                training.append(verse(words))
        model = train_on(training)
        for preposition, lemma, strongs, expected in (("ἐν", "ἐν", "G1722", "N-DSF"), ("εἰς", "εἰς", "G1519", "N-ASF")):
            items, _ = verse([(preposition, [(strongs, lemma, "PREP", 50.0)], "PREP"),
                              (NOUNS[11], noun(11, ["N-DSF", "N-ASF"]), None)])
            self.assertEqual(choose(model, items)[1].candidate.morph, expected)

    def test_training_and_choosing_are_deterministic(self):
        def run():
            training = [verse([("τοῦ", article("T-GSM"), "T-GSM"),
                               (NOUNS[i], noun(i, ["N-GSM", "N-APM"]), "N-GSM")]) for i in range(5)]
            model = train_on(training)
            items, _ = verse([("τοῦ", article("T-GSM"), "T-GSM"), (NOUNS[9], noun(9, ["N-GSM", "N-APM"]), None)])
            return json.dumps(model.to_json(), sort_keys=True), [(c.candidate.reading, c.margin) for c in choose(model, items)]
        self.assertEqual(run(), run())

    def test_untaggable_tiers_are_left_alone_and_margins_are_reported(self):
        tokens = tokenize("ὁ ἀόρατος καλα")
        verse_items = [item(tokens[0], article("T-NSM")),
                       TokenCandidates(tokens[1], [], [], "e", "no candidate"),
                       item(tokens[2], noun(5, ["N-NSM", "N-VSM"]))]
        model = train_on([verse([("ὁ", article("T-NSM"), "T-NSM"), (NOUNS[1], noun(1, ["N-NSM", "N-VSM"]), "N-NSM")])])
        choices = choose(model, verse_items)
        self.assertIsNone(choices[1])
        self.assertEqual(choices[0].margin, math.inf)
        self.assertTrue(math.isfinite(choices[2].margin))
        self.assertTrue(0.0 <= choices[2].confidence <= 1.0)


class ToleranceTests(unittest.TestCase):
    def test_a_tolerant_match_in_another_case_than_a_subscript_shows_is_set_aside(self):
        token = tokenize("κληρονομίᾳ")[0]
        found = TokenCandidates(token, [Candidate("G2817", "κληρονομία", "N-NSF", "nt-tolerant", 5.0, taggable=False)],
                                [], "u", "tolerant match only")
        options, method = chooser.options_for(found)
        self.assertEqual(method, "tolerant-mismatch")
        dative = TokenCandidates(token, found.candidates + [
            Candidate("G2817", "κληρονομία", "N-DSF", "nt-tolerant", 1.0, taggable=False)], [], "u", "")
        options, method = chooser.options_for(dative)
        self.assertEqual((method, [o.reading[2] for o in options]), ("tolerant", ["N-DSF"]))


class CalibrationTests(unittest.TestCase):
    def test_confidence_never_falls_as_the_margin_grows(self):
        records = [("model", "b", "nt", float(m), (m * 7919) % 100 < 40 + m // 20) for m in range(1200)]
        calibration = chooser.Calibration.fit(records)
        values = [calibration.confidence("model", "b", "nt", m) for m in range(0, 1200, 7)]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])
        # An unseen group falls back to a broader one.
        self.assertEqual(calibration.confidence("model", "c", "analyser", 5.0),
                         calibration.confidence("model", "b", "nt", 5.0))

    def test_bands(self):
        self.assertEqual([chooser.band(x) for x in (0.1, 0.6, 0.85, 0.92, 0.97, 0.995)], list(chooser.BANDS))


class CacheTests(unittest.TestCase):
    def test_the_key_changes_with_the_sources_and_the_code(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            source, code = directory / "TAGNT.txt", directory / "chooser.py"
            source.write_text("a", encoding="utf-8")
            code.write_text("x = 1", encoding="utf-8")
            first = chooser.cache_key([source], [code], {"epochs": 1})
            self.assertEqual(first, chooser.cache_key([source], [code], {"epochs": 1}))
            source.write_text("b", encoding="utf-8")
            second = chooser.cache_key([source], [code], {"epochs": 1})
            code.write_text("x = 2", encoding="utf-8")
            # Code is hashed once per process: the key describes the code loaded.
            self.assertEqual(chooser.cache_key([source], [code], {"epochs": 1}), second)
            chooser._FILE_DIGESTS.pop((str(code.resolve()), True))
            third = chooser.cache_key([source], [code], {"epochs": 1})
            self.assertEqual(len({first, second, third, chooser.cache_key([source], [code], {"epochs": 2})}), 4)

    def test_trains_once_then_reuses_the_cached_model_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            sources = write_sources(directory / "sources")
            factory = TaggerSpec(tbesg_path=sources / "TBESG.txt")
            messages = []
            first = chooser.load_or_train(sources, factory, directory / "cache", log=messages.append, folds=2)
            files = sorted((directory / "cache").glob("chooser-*.json.gz"))
            self.assertEqual(len(files), 1)
            stored = files[0].read_bytes()
            self.assertTrue(any("training" in m for m in messages))
            messages.clear()
            second = chooser.load_or_train(sources, factory, directory / "cache", log=messages.append, folds=2)
            self.assertEqual(messages, [])
            self.assertEqual(first.model.to_json(), second.model.to_json())
            again = chooser.load_or_train(sources, factory, directory / "cache", log=messages.append, folds=2,
                                          retrain=True)
            self.assertEqual(files[0].read_bytes(), stored)
            self.assertEqual(again.model.to_json(), first.model.to_json())
            with gzip.open(files[0], "rt", encoding="utf-8") as handle:
                self.assertIn("calibration", json.load(handle))


class CorpusTests(unittest.TestCase):
    def test_nt_text_is_set_in_brentons_capitals_with_each_tokens_reading(self):
        with tempfile.TemporaryDirectory() as name:
            sources = write_sources(Path(name))
            verses, rows = ntcorpus.read(ntforms.tagnt_paths(sources))
        by_reference = {v.osis: v for v in verses}
        # A title is capitalised, a common word lowered, a verse's first word raised.
        texts = [v.text for v in verses]
        self.assertTrue(any("Θεοῦ" in text for text in texts))
        first = by_reference["Mat.1.1"]
        self.assertTrue(first.text[0].isupper())
        self.assertEqual(len(first.tokens), len(first.gold))
        joined = by_reference["Mat.1.6"]
        self.assertEqual([g.morph for g in joined.gold[:2]], ["CONJ", "I-NSN"])
        self.assertEqual(len(rows), len(TAGNT_ROWS))


if __name__ == "__main__":
    unittest.main()
