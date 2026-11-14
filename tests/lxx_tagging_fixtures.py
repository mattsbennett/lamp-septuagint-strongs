"""Small synthetic TAGNT and TBESG files for the LXX tagging tests.

The rows are shaped like the real files' rows; the words are chosen to
exercise one rule each, and the comments say which.
"""

import re
import unicodedata
from pathlib import Path


def tagnt_row(reference, greek, tag, lemma, word_type="NKO"):
    """`Mat.1.1#01=NKO	Βίβλος (Biblos)	[The] book	G0976=N-NSF	βίβλος=book	NA28+…`, 17 columns."""
    columns = [f"{reference}={word_type}", f"{greek} (x)", "gloss", tag, lemma, "NA28+NA27"]
    return "\t".join(columns + [""] * (17 - len(columns)))


TAGNT_ROWS = [
    tagnt_row("Mat.1.1#01", "Βίβλος", "G0976=N-NSF", "βίβλος=book"),
    tagnt_row("Mat.1.1#02", "καὶ", "G2532=CONJ", "καί=and"),
    tagnt_row("Mat.1.1#03", "ἐν", "G1722=PREP", "ἐν=in"),
    tagnt_row("Mat.1.1#04", "ὁ", "G3588=T-NSM", "ὁ=the"),
    # A title and a common noun: TAGNT marks the LORD -T and a master not.
    tagnt_row("Mat.1.2#01", "θεοῦ", "G2316=N-GSM-T", "θεός=God"),
    tagnt_row("Mat.1.2#02", "θεοῦ", "G2316=N-GSM-T", "θεός=God"),
    tagnt_row("Mat.1.2#03", "κυρίου", "G2962G=N-GSM-T", "κύριος=lord"),
    tagnt_row("Mat.1.2#04", "κυρίου", "G2962H=N-GSM", "κύριος=lord"),
    tagnt_row("Mat.1.2#05", "κύριος", "G2962G=N-NSM-T", "κύριος=lord"),
    tagnt_row("Mat.1.2#06", "δύο", "G1417=A-NUI", "δύο=two"),
    tagnt_row("Mat.1.2#07", "ἰδοὺ", "G2400=INJ", "ἰδού=behold"),
    tagnt_row("Mat.1.2#08", "διὰ", "G1223=PREP", "διά=through"),
    # Frankincense, lower case in the NT; Brenton's Λίβανον is Lebanon.
    tagnt_row("Mat.1.3#01", "λίβανον", "G3030=N-ASM", "λίβανος=frankincense"),
    # An indeclinable name, tagged by its sentence's case.
    tagnt_row("Mat.1.3#02", "Ἀβραάμ.", "G0011=N-GSM-P", "Ἀβραάμ=Abraham"),
    tagnt_row("Mat.1.3#03", "Ἀβραὰμ", "G0011=N-NSM-P", "Ἀβραάμ=Abraham"),
    # Annas: the accusative is Brenton's Hannah with the other breathing.
    tagnt_row("Mat.1.3#04", "Ἅννας", "G0452=N-NSM-P", "Ἅννας=Annas"),
    tagnt_row("Mat.1.3#05", "Ἅνναν", "G0452=N-ASM-P", "Ἅννας=Annas"),
    # A declining name: each form one case.
    tagnt_row("Mat.1.4#01", "Μωϋσῆς", "G3475=N-NSM-P", "Μωϋσῆς, Μωσῆς=Moses"),
    tagnt_row("Mat.1.4#02", "Μωϋσέως", "G3475=N-GSM-P", "Μωϋσῆς, Μωσῆς=Moses"),
    # Movable nu: the NT has only the form with ν.
    tagnt_row("Mat.1.5#01", "ἔλεγεν", "G3004G=V-IAI-3S", "λέγω=to say"),
    tagnt_row("Mat.1.5#02", "φαίνει", "G5316=V-PAI-3S", "φαίνω=to shine"),
    # Joined words TAGNT tags as two.
    tagnt_row("Mat.1.6#01", "ἱνατί", "G2443=CONJ + G5101=I-NSN", "ἵνα=in order that + τίς=which?"),
    tagnt_row("Mat.1.6#02", "κἀγὼ", "G1473=P-1NS + G2532=CONJ", "κἀγώ=and I"),
    # Two words with one form.
    tagnt_row("Mat.1.7#01", "τίς", "G5101=I-NSM", "τίς=which?"),
    tagnt_row("Mat.1.7#02", "τίς", "G5100=X-NSM", "τις=one"),
    # Paradigm identities: a neuter seen once as accusative, a feminine
    # genitive in -ας; the other codes appear elsewhere, so they exist.
    tagnt_row("Mat.1.8#01", "ὄνομα", "G3686=N-ASN", "ὄνομα=name"),
    tagnt_row("Mat.1.8#02", "πνεῦμα", "G4151G=N-NSN", "πνεῦμα=spirit"),
    tagnt_row("Mat.1.8#03", "τέκνον", "G5043=N-VSN", "τέκνον=child"),
    tagnt_row("Mat.1.8#04", "ἀνομίας", "G0458=N-GSF", "ἀνομία=lawlessness"),
    tagnt_row("Mat.1.8#05", "ἡμέρας", "G2250=N-APF", "ἡμέρα=day"),
    # Only in the Textus Receptus, and decomposed in the file.
    tagnt_row("Mat.1.9#01", unicodedata.normalize("NFD", "ἀμήν.¶"), "G0281=INJ-HEB", "ἀμήν=amen", "K"),
]

TBESG_HEADER = [
    "TBESG - Translators Brief lexicon of Extended Strongs for Greek - STEPBible.org CC BY",
    "",
    "eStrong#\tdStrong# = Relation\tuStrong#\tGreek\tTransliteration\tGram\tGloss\tDefinition",
]


def tbesg_row(number, relation, related, lemma, part, gloss):
    return "\t".join([number, relation, related, lemma, "x", part, gloss, "<b>definition</b>"])


TBESG_ROWS = [
    # One number, two words: the letter and the interjection.
    tbesg_row("G0001", "G0001G =", "G0001G", "α, Ἀλφα", "G:N-LI", "Alpha"),
    tbesg_row("G0001", "G0001H =", "G0001H", "ἆ", "G:INJ", "ah!"),
    tbesg_row("G0011", "G0011 = the Greek of", "H0085", "Ἀβραάμ", "N:N-M-P", "Abraham"),
    tbesg_row("G1138", "G1138 = the Greek of", "H1732", "Δαυείδ, Δαυίδ, Δαβίδ", "N:N-M-P", "David"),
    tbesg_row("G6043", "G6043 =", "G6043", "Ἰωδά", "N:N-M-P", "Joda"),
    tbesg_row("G6161", "G6161 =", "G6161", "αιλάμ", "N:N", "porch"),
    tbesg_row("G6000", "G6000 =", "G6000", "ἀγγέλλω", "G:V", "to report"),
    tbesg_row("G5504", "G5504 =", "G5504", "χθές", "G:ADV", "yesterday"),
    tbesg_row("G5062", "G5062 =", "G5062", "τεσσαράκοντα", "G:A-NUI", "forty"),
]


def write_sources(directory: Path) -> Path:
    """The fixture TAGNT (split over the two files the real data has) and TBESG."""
    directory.mkdir(parents=True, exist_ok=True)
    half = len(TAGNT_ROWS) // 2
    header = "# TAGNT fixture\n"
    (directory / "TAGNT_Mat-Jhn.txt").write_text(header + "\n".join(TAGNT_ROWS[:half]) + "\n", encoding="utf-8")
    (directory / "TAGNT_Act-Rev.txt").write_text(header + "\n".join(TAGNT_ROWS[half:]) + "\n", encoding="utf-8")
    (directory / "TBESG.txt").write_text("\ufeff" + "\n".join(TBESG_HEADER + TBESG_ROWS) + "\n", encoding="utf-8")
    return directory


def module(verses):
    """A translation module in LXX.json's shape from {(book, chapter, verse): text}."""
    books = {}
    numbers = {"Gen": 1, "Exod": 2}
    for (book, chapter, verse), text in sorted(verses.items(), key=lambda item: (numbers[item[0][0]], item[0][1:])):
        chapters = books.setdefault(book, {})
        chapters.setdefault(chapter, []).append({
            "content": {"text": text}, "ref": numbers[book] * 1_000_000 + chapter * 1_000 + verse, "v": verse,
        })
    return {
        "meta": {"id": "LXX", "type": "translation", "versification": "lxx",
                 "features": {"strongs": False, "morphology": False, "paragraphs": True}},
        "books": [
            {"id": book, "name": book, "number": numbers[book],
             "chapters": [{"chapter": c, "verses": v} for c, v in sorted(chapters.items())]}
            for book, chapters in books.items()
        ],
    }


# ---------------------------------------------------------------------------
# The Hebrew side of the name stage: TAHOT and TBESH rows shaped like the
# real files' (12 columns: reference and type, Hebrew with its morpheme
# slashes, transliteration, gloss, dStrongs, grammar, variants, instance,
# alternative, conjoin, expanded tags).

def tahot_row(reference, hebrew, dstrongs, grammar, lemma, gloss):
    stem = re.search(r"\{(H\d+[A-Za-z]?)\}", dstrongs).group(1)
    expanded = dstrongs.replace("{" + stem + "}", "{" + f"{stem}={lemma}={gloss}»{gloss}@x" + "}")
    columns = [reference, hebrew, "x", gloss, dstrongs, grammar, "", "", stem, "", "", expanded]
    return "\t".join(columns + [""] * 4)


TAHOT_ROWS = [
    # Gen 10:1, a list: Noah, Shem, Ham, Japheth.
    tahot_row("Gen.10.1#01=L", "וְ/אֵ֙לֶּה֙", "H9002/{H0428}", "HC/Tm", "אֵלֶּה", "these"),
    tahot_row("Gen.10.1#02=L", "נֹ֔חַ", "{H5146}", "HNpm", "נֹחַ", "Noah"),
    tahot_row("Gen.10.1#03=L", "שֵׁ֖ם", "{H8035}", "HNpm", "שֵׁם", "Shem"),
    tahot_row("Gen.10.1#04=L", "חָ֣ם", "{H2526}", "HNpm", "חָם", "Ham"),
    tahot_row("Gen.10.1#05=L", "וָ/יָ֑פֶת", "H9002/{H3315}", "HC/Npm", "יֶ֫פֶת", "Japheth"),
    # Gen 10:16, gentilics with the article.
    tahot_row("Gen.10.16#01=L", "הַ/יְבוּסִי֙", "H9009/{H2983G}", "HTd/Ngmsa", "יְבוּסִי", "Jebusite"),
    tahot_row("Gen.10.16#02=L", "הָ֣/אֱמֹרִ֔י", "H9009/{H0567}", "HTd/Ngmsa", "אֱמֹרִי", "Amorite"),
    # Gen 20:2: Abraham, Sarah, Abimelech (TAHOT's H0040G), Gerar.
    tahot_row("Gen.20.2#01=L", "אַבְרָהָ֛ם", "{H0085}", "HNpm", "אַבְרָהָם", "Abraham"),
    tahot_row("Gen.20.2#02=L", "שָׂרָ֥ה", "{H8283}", "HNpf", "שָׂרָה", "Sarah"),
    tahot_row("Gen.20.2#03=L", "מֶ֣לֶךְ", "{H4428G}", "HNcmsc", "מֶ֫לֶךְ", "king"),
    tahot_row("Gen.20.2#04=L", "אֲבִימֶ֙לֶךְ֙", "{H0040G}", "HNpm", "אֲבִימֶ֫לֶךְ", "Abimelech"),
    tahot_row("Gen.20.2#05=L", "גְּרָ֔ר", "{H1642}", "HNpl", "גְּרָר", "Gerar"),
    # A name over two joined words, and a word TAHOT reconstructs from the
    # Septuagint (X), which is not Hebrew text.
    tahot_row("Gen.20.3#01=L", "בֵּֽית\\־", "{H1053G}+\\H9014", "HNpl", "בֵּית שֶׁ֫מֶשׁ", "Beth-shemesh"),
    tahot_row("Gen.20.3#02=L", "שֶׁ֖מֶשׁ", "{H1053G}", "HNpl", "בֵּית שֶׁ֫מֶשׁ", "Beth-shemesh"),
    tahot_row("Gen.20.3#03=X", "אַבְרָהָ֛ם", "{H0085}", "HNpm", "אַבְרָהָם", "Abraham"),
]

TBESH_ROWS = [
    "H0040\tH0040G =\tH0040G\tאֲבִימֶ֫לֶךְ\ta.vi.me.lekh\tN:N-M-P\tAbimelech\tdef",
    "H0085\tH0085 =\tH0085\tאַבְרָהָם\tav.ra.ham\tN:N-M-P\tAbraham\tdef",
    "H0567\tH0567 =\tH0567\tאֱמֹרִי\te.mo.ri\tN:N--LG\tAmorite\tdef",
    "H1053\tH1053G =\tH1053G\tבֵּית שֶׁ֫מֶשׁ\tbeit she.mesh\tN:N--L\tBeth-shemesh\tdef",
    "H1642\tH1642 =\tH1642\tגְּרָר\tge.rar\tN:N--L\tGerar\tdef",
    "H1732\tH1732 =\tH1732\tדָּוִד\tda.vid\tN:N-M-P\tDavid\tdef",
    "H2526\tH2526 =\tH2526\tחָם\tcham\tN:N-M-P\tHam\tdef",
    "H2983\tH2983G =\tH2983G\tיְבוּסִי\tye.vu.si\tN:N--LG\tJebusite\tdef",
    "H3315\tH3315 =\tH3315\tיֶ֫פֶת\tye.fet\tN:N-M-P\tJapheth\tdef",
    "H5146\tH5146 =\tH5146\tנֹחַ\tno.ach\tN:N-M-P\tNoah\tdef",
    "H8035\tH8035 =\tH8035\tשֵׁם\tshem\tN:N-M-P\tShem\tdef",
    "H8283\tH8283 =\tH8283\tשָׂרָה\tsa.rah\tN:N-F-P\tSarah\tdef",
]


def write_hebrew_sources(directory: Path) -> Path:
    """The fixture TAHOT (in the four files the real data has) and TBESH."""
    directory.mkdir(parents=True, exist_ok=True)
    names = ("TAHOT_Gen-Deu.txt", "TAHOT_Jos-Est.txt", "TAHOT_Job-Sng.txt", "TAHOT_Isa-Mal.txt")
    header = "TAHOT fixture\nEng (Heb) Ref & Type\tHebrew\n"
    for index, name in enumerate(names):
        rows = TAHOT_ROWS if index == 0 else []
        (directory / name).write_text(header + "\n".join(rows) + "\n", encoding="utf-8")
    (directory / "TBESH.txt").write_text("﻿TBESH fixture\n" + "\n".join(TBESH_ROWS) + "\n", encoding="utf-8")
    return directory
