"""The tagging command: module in, tagged module and reports out."""

import csv
import json
import sys
import tempfile
import unicodedata
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lxx_tagging import lexicon, ntforms, tag_septuagint  # noqa: E402
from lxx_tagging.candidates import CaseProfile, default_tagger  # noqa: E402
from lxx_tagging_fixtures import module, write_hebrew_sources, write_sources  # noqa: E402

VERSES = {
    ("Gen", 1, 1): "ἘΝ ἀρχῇ ἡ βίβλος καὶ Κυρίου, Διατοῦτο ἔλεγε Μωυσῆς.",
    ("Gen", 1, 2): "Καὶ ἐν τῷ Λίβανον καθʼ ὄνομα Ἀβραὰμ ἀόρατος;",
    ("Exod", 2, 1): "Καὶ εἶπε κᾀγὼ ἱνατί ἆ",
}


class CommandTests(unittest.TestCase):
    def run_command(self, directory: Path) -> Path:
        sources = write_hebrew_sources(write_sources(directory / "sources"))
        source = directory / "LXX.json"
        source.write_text(json.dumps(module(VERSES), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        output = directory / "out"
        with redirect_stdout(StringIO()):
            tag_septuagint.main(["--input", str(source), "--output-dir", str(output), "--sources", str(sources),
                                 "--chooser-cache", str(directory / "chooser-cache"), "--verse-map", "identity"])
        return output

    def test_the_tagged_module_keeps_every_text_and_adds_tagnts_annotations(self):
        with tempfile.TemporaryDirectory() as name:
            output = self.run_command(Path(name))
            original = module(VERSES)
            tagged = json.loads((output / "LXX_tagged.json").read_text(encoding="utf-8"))
        self.assertEqual(tagged["meta"]["features"],
                         {"strongs": True, "morphology": True, "paragraphs": True})
        # The module says where its tags came from and that a program assigned
        # them (STEPBible's credit, and the disclosure of our own analysis);
        # the rest of its metadata is the input's.
        described = ("features", "sourceTexts", "license", "copyright", "description")
        self.assertEqual({k: v for k, v in tagged["meta"].items() if k not in described},
                         {k: v for k, v in original["meta"].items() if k not in described})
        sources = tagged["meta"]["sourceTexts"]["annotations"]
        self.assertEqual([s["part"] for s in sources], ["strongs", "strongs"])
        self.assertIn("STEPBible", sources[0]["source"])
        self.assertIn("not checked by hand", sources[0]["method"])
        self.assertIn("Hebrew", sources[1]["source"])
        for source in sources:   # the app prints "source, method."
            self.assertTrue(source["method"][0].islower() and not source["method"].endswith("."))
        self.assertEqual(tagged["meta"]["license"], tag_septuagint.TAGGED_LICENSE)
        self.assertIn("STEPBible.org", tagged["meta"]["copyright"])
        if original["meta"].get("description"):
            self.assertTrue(tagged["meta"]["description"].startswith(original["meta"]["description"]))
            self.assertIn("not been checked by hand", tagged["meta"]["description"])
        verses = {verse["ref"]: verse for book in tagged["books"]
                  for chapter in book["chapters"] for verse in chapter["verses"]}
        for book in original["books"]:
            for chapter in book["chapters"]:
                for verse in chapter["verses"]:
                    self.assertEqual(verses[verse["ref"]]["content"]["text"], verse["content"]["text"])
        first = verses[1001001]["content"]
        self.assertEqual(first["annotations"][0], {
            "type": "strongs", "start": 0, "end": 2, "text": "ἘΝ",
            "data": {"strongs": "G1722", "morphology": "PREP", "lemma": "ἐν"}})
        for annotation in first["annotations"]:
            self.assertEqual(first["text"][annotation["start"]:annotation["end"]], annotation["text"])
        by_text = {a["text"]: a["data"] for a in first["annotations"]}
        self.assertEqual(by_text["Κυρίου"]["morphology"], "N-GSM-T")
        self.assertEqual(by_text["Δια"], {"strongs": "G1223", "morphology": "PREP", "lemma": "διά"})
        self.assertEqual(by_text["Μωυσῆς"]["strongs"], "G3475")
        second = {a["text"] for a in verses[1001002]["content"].get("annotations", [])}
        self.assertNotIn("Λίβανον", second)

    def test_reports_and_decisions_describe_every_token(self):
        with tempfile.TemporaryDirectory() as name:
            output = self.run_command(Path(name))
            report = json.loads((output / "lxx_tagging_report.json").read_text(encoding="utf-8"))
            with open(output / "lxx_tagging_decisions.tsv", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
        self.assertEqual(list(rows[0]), list(tag_septuagint.DECISION_COLUMNS))
        self.assertEqual(len(rows), report["tokens"])
        self.assertEqual(sum(tier["tokens"] for tier in report["tiers"].values()), report["tokens"])
        self.assertEqual(report["tagged"], sum(1 for row in rows if row["strongs"]))
        self.assertEqual(set(report["byBook"]), {"Gen", "Exod"})
        libanon = next(row for row in rows if row["surface"] == "Λίβανον")
        self.assertEqual((libanon["tier"], libanon["strongs"]), ("d", ""))
        self.assertIn("G3030|λίβανος|N-ASM", libanon["set_aside"])
        dia = next(row for row in rows if row["surface"] == "Δια")
        self.assertEqual((dia["part"], dia["key"], dia["osis"]), ("1/2", "διά", "Gen.1.1"))
        self.assertTrue(any(form["key"] == "ἀόρατοσ" for form in report["untaggedForms"]))

    def test_output_is_deterministic(self):
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as name:
                output = self.run_command(Path(name))
                outputs.append({path.name: path.read_bytes() for path in sorted(output.iterdir())})
        self.assertEqual(outputs[0], outputs[1])


LXX = REPOSITORY_ROOT / "outputs" / "lxx_tagging" / "LXX_source.json"


@unittest.skipUnless(LXX.exists(), "outputs/lxx_tagging/LXX_source.json has not been built")
class SeptuagintTests(unittest.TestCase):
    """Genesis 1 tagged from the real TAGNT and TBESG."""

    def test_genesis_one(self):
        source = json.loads(LXX.read_text(encoding="utf-8"))
        forms = ntforms.build(ntforms.tagnt_paths())
        profile = CaseProfile.from_texts(
            tag_septuagint.verse_text(verse) for _, _, verse in tag_septuagint.verses(source))
        genesis = {"meta": source["meta"], "books": [
            {**source["books"][0], "chapters": source["books"][0]["chapters"][:1]}]}
        tagger = default_tagger(forms, lexicon.build(), profile)
        tagged, results = tag_septuagint.tag_module(genesis, tagger)
        tag_septuagint.check_texts_unchanged(genesis, tagged)
        verse = tagged["books"][0]["chapters"][0]["verses"][0]["content"]
        self.assertEqual(verse["text"], "ἘΝ ἀρχῇ ἐποίησεν ὁ Θεὸς τὸν οὐρανὸν καὶ τὴν γῆν.")
        data = {a["text"]: a["data"] for a in verse["annotations"]}
        # Lemmas are TAGNT's strings verbatim, as TAGNT.json ships them: its
        # accents are the Greek oxia (U+1F7x), so they are not NFC.
        self.assertEqual(data["ἐποίησεν"]["lemma"], forms.lemma_for("G4160"))
        nfc = {a["text"]: {**a["data"], "lemma": unicodedata.normalize("NFC", a["data"]["lemma"])}
               for a in verse["annotations"]}
        self.assertEqual(nfc["ἘΝ"], {"strongs": "G1722", "morphology": "PREP", "lemma": "ἐν"})
        self.assertEqual(nfc["ἐποίησεν"], {"strongs": "G4160", "morphology": "V-AAI-3S", "lemma": "ποιέω"})
        self.assertEqual(data["οὐρανὸν"]["morphology"], "N-ASM")
        for book in tagged["books"]:
            for chapter in book["chapters"]:
                for record in chapter["verses"]:
                    for annotation in record["content"].get("annotations", []):
                        self.assertEqual(record["content"]["text"][annotation["start"]:annotation["end"]],
                                         annotation["text"])
                        # TAGNT's own codes, which the shipped decoder reads.
                        if "morphology" in annotation["data"]:
                            self.assertIn(annotation["data"]["morphology"], forms.morph_codes)
        tiers = [item.tier for _, _, _, outcome in results for item, _ in outcome]
        self.assertGreater(tiers.count("a") / len(tiers), 0.5)


if __name__ == "__main__":
    unittest.main()
