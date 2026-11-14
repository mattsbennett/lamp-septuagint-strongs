#!/usr/bin/env python3
"""
Builds the Hebrew OT and Greek NT translation modules from STEPBible's
Translators Amalgamated datasets (TAHOT and TAGNT).

These are the first Lamp modules to carry per-word morphology. The schema has
always had the fields — `data.morphology` and `data.lemma` beside
`data.strongs` — and `LampStrongsAnalysis` has always computed parsings and
grammar from them; nothing installed populated them, so those answers came back
empty. The module type is `translation`, not something new: an
original-language text is a translation whose language is Hebrew or Greek, and
it inherits import, sync, search and passage reading for nothing.

Two editorial decisions, because the amalgamated files hold more than one text:

  * Greek: a word is included when its type contains N or n, which is the
    Nestlé-Aland reading. The file's own note says the text follows NA27, "which
    is followed by almost all translations". Words only in the Textus Receptus
    (K) or in other editions (O) are left out of the running text rather than
    interleaved with it, which would give a verse the same word twice.
  * Hebrew: the text translators read, which is the dataset's own reading
    text. A Leningrad word (L) is included as it stands. Where the scribes'
    margin corrects the text, the dataset's row is the Qere (Q), with the
    Ketiv it replaces recorded as a variant: the row is included and the Ketiv
    becomes a footnote, so a word written but not read is a footnote alone.
    The verses Leningrad lacks and its parallels supply (R; Joshua 21:36-37
    and the end of Nehemiah 7:68) are included with a footnote saying so.
    Words reconstructed from the Septuagint (X) are not Hebrew manuscript
    text and are left out.

The variants are in the source for anyone who wants them. This module is a
reading text with its morphology, and says so in its description.

Usage:
    python stepbible_to_translation_schema.py --source-dir ../sources/stepbible \
        --output-dir ../outputs

    # One book, to check a change quickly:
    python stepbible_to_translation_schema.py --books Jhn --output-dir /tmp

Attribution, which the licence requires and which the module record carries:
Tyndale House, Cambridge and STEPBible.org, CC BY 4.0,
https://STEPBible.github.io/STEPBible-Data.
"""

import argparse
import io
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# One book table and one Strong's normaliser for every converter, so a key this
# module writes is a key the lexicons and the other translations answer to.
from sword_to_translation_schema import BOOK_INFO, normalize_strongs

# Every book code the datasets use, so the check below runs over the real list
# rather than over whatever the mapping happens to mention.
STEPBIBLE_CODES = (
    "Gen Exo Lev Num Deu Jos Jdg Rut 1Sa 2Sa 1Ki 2Ki 1Ch 2Ch Ezr Neh Est Job Psa "
    "Pro Ecc Sng Isa Jer Lam Ezk Dan Hos Jol Amo Oba Jon Mic Nam Hab Zep Hag Zec "
    "Mal Mat Mrk Luk Jhn Act Rom 1Co 2Co Gal Eph Php Col 1Th 2Th 1Ti 2Ti Tit Phm "
    "Heb Jas 1Pe 2Pe 1Jn 2Jn 3Jn Jud Rev"
).split()

# STEPBible's three-letter codes against the OSIS abbreviations the schema uses.
# Only the ones that differ need naming; the rest match.
STEPBIBLE_TO_OSIS = {
    "Exo": "Exod", "Deu": "Deut", "Jos": "Josh", "Jdg": "Judg",
    "1Sa": "1Sam", "2Sa": "2Sam", "1Ki": "1Kgs", "2Ki": "2Kgs",
    "1Ch": "1Chr", "2Ch": "2Chr", "Ezr": "Ezra", "Est": "Esth",
    "Psa": "Ps", "Pro": "Prov", "Ecc": "Eccl", "Sng": "Song",
    "Ezk": "Ezek", "Jol": "Joel", "Oba": "Obad", "Nam": "Nah",
    "Zec": "Zech", "Mrk": "Mark", "Luk": "Luke", "Jhn": "John",
    "1Co": "1Cor", "2Co": "2Cor", "Php": "Phil", "1Th": "1Thess",
    "2Th": "2Thess", "1Ti": "1Tim", "2Ti": "2Tim", "Tit": "Titus",
    "Phm": "Phlm", "Jas": "Jas", "1Pe": "1Pet", "2Pe": "2Pet",
    "1Jn": "1John", "2Jn": "2John", "3Jn": "3John", "Jud": "Jude",
    "Rut": "Ruth", "Amo": "Amos", "Jon": "Jonah", "Zep": "Zeph",
    "Mat": "Matt", "Act": "Acts",
}

HEADER = re.compile(r"^([1-3]?[A-Za-z]{2,3})\.(\d+)[^.]*\.(\d+)[^#]*#(\d+)(?:=(.*))?$")


def osis(code: str) -> str:
    return STEPBIBLE_TO_OSIS.get(code, code)


def parse_header(field: str):
    """`Gen.31.55(32.1)#03=L` -> ("Gen", 31, 55, 3, "L").

    The reference is English versification with the Hebrew in brackets, so the
    bracketed part is dropped: Lamp's references are the English ones.
    """
    match = HEADER.match(field.strip())
    if not match:
        return None
    code, chapter, verse, word, word_type = match.groups()
    return code, int(chapter), int(verse), int(word), (word_type or "").strip()


def greek_is_text(word_type: str) -> bool:
    """A Nestlé-Aland word. See the editorial note at the top of this file."""
    return "N" in word_type.upper()


def hebrew_is_text(word_type: str) -> bool:
    """A Leningrad word, a Qere, or restored text, as against a reconstruction
    from the Septuagint. See the editorial note at the top of this file."""
    return word_type[:1].upper() in ("L", "Q", "R")


# Where the restored verses come from, in English numbering: the dataset's
# header names the parallels as 1Ch.6.63-64 and Ezr.2.66 in Hebrew numbering.
RESTORED_FROM = {
    "Jos": "1 Chronicles 6:78–79",
    "Neh": "Ezra 2:66",
}


def clean_gloss(gloss: str) -> str:
    """`<in>/ before` -> `before`. Angle brackets hold what is in the Hebrew
    but not translated, square brackets what is implied and translated, and
    the slashes mirror the Hebrew's morphemes."""
    gloss = re.sub(r"<[^>]*>", "", gloss).replace("[", "").replace("]", "")
    parts = [part.strip() for part in gloss.split("/") if part.strip()]
    # A noun and its pronoun suffix, `foot/ his`, read in English order.
    if len(parts) == 2 and parts[1] in POSSESSIVES:
        parts.reverse()
    return re.sub(r"\s+", " ", " ".join(parts))


POSSESSIVES = {"my", "your", "his", "her", "its", "our", "their"}


def ketiv_note(columns: list[str], read: bool) -> str | None:
    """The footnote giving the Ketiv a Qere replaces, or None if the row has
    none. A Ketiv that changes the meaning is in the meaning-variant column
    with its gloss, `K= tom.Ri (תֹּאמְרִ֥י) "you are saying" (H0559=HVqi2fs)`;
    one that changes only the spelling is in the spelling column,
    `L= ... ¦ ;K= בֵּינ/וֹ`."""
    meaning = columns[6] if len(columns) > 6 else ""
    spelling = columns[7] if len(columns) > 7 else ""
    match = re.search(r'K=[^(]*\(([^)]*)\)\s*"([^"]*)"', meaning)
    if match:
        written, gloss = clean_hebrew(match.group(1))[0], clean_gloss(match.group(2))
    else:
        match = re.search(r"K=\s*([^;¦]+)", spelling)
        if not match:
            return None
        written, gloss = clean_hebrew(match.group(1).strip())[0], ""
    if not written:
        return None
    label = "Written (Ketiv)" if read else "Written but not read (Ketiv)"
    return f"{label}: {written}" + (f", “{gloss}”" if gloss else "")


def strongs_keys(field: str) -> list[str]:
    """The Strong's keys a word's dStrongs field names, stems first.

    Hebrew words are segmented, so the field can read `H9003/{H7225G}`: a
    prefix code and the stem in braces. The braces mark the word proper, and
    STEPBible's H9xxx codes for prefixes and suffixes are its own extension
    rather than Strong's, so they are not offered as lexicon keys. What is
    dropped here survives in the morphology, which keeps the whole segmented
    tag.
    """
    stems = re.findall(r"\{([HG][0-9]+[A-Za-z]?)\}", field)
    if stems:
        return [normalize_strongs(key) for key in stems]
    plain = [
        key for key in re.findall(r"[HG][0-9]+[A-Za-z]?", field)
        if not re.match(r"H9\d{3}", key)
    ]
    return [normalize_strongs(key) for key in plain]


def lemma_from(expanded: str, dictionary_form: str) -> str | None:
    """The dictionary form, from whichever column the dataset puts it in.

    Greek has a `Dictionary form = Gloss` column (`βίβλος=book`). Hebrew has
    `Expanded Strong tags`, where the stem appears as `{H7225G=רֵאשִׁית=...}`.
    """
    if expanded:
        stem = re.search(r"\{[HG][0-9]+[A-Za-z]?=([^=}]+)=", expanded)
        if stem:
            return stem.group(1).strip() or None
    if dictionary_form:
        form = dictionary_form.split("=", 1)[0].strip()
        return form or None
    return None


def clean_greek(word: str) -> str:
    """`Βίβλος (Biblos)` -> `Βίβλος`. The transliteration is a column of its own
    in the Hebrew files and parenthesised in the Greek ones; the verse text is
    the Greek."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", word).strip()


# Hebrew words carry two markers that are not spelling. `/` divides the
# morphemes the analysis found, which is information about the word rather than
# part of it, and the segmented form survives in the morphology tag. `\` says
# that what follows is a mark rather than a letter: maqaf, sof pasuq, paseq, the
# setuma and petucha letters, nun hafucha.
MAQAF = "\u05be"


def clean_hebrew(word: str) -> tuple[str, str]:
    """`הָ/אָֽרֶץ\׃` -> (`הָאָֽרֶץ`, `׃`): the word, and the marks after it.

    Returned apart because the marks belong to the verse — a text with no sof
    pasuq is not the text — while the annotation marks the word. Trimming them
    off the span by Unicode category would not do: setuma and petucha are the
    letters ס and פ, so a category test keeps them inside the word.
    """
    head, _, tail = word.partition("\\")
    return head.replace("/", "").strip(), tail.replace("\\", "").replace("/", "").strip()


def read_words(path: Path, greek: bool):
    """Yields (osis, chapter, verse, text, marks, strongs, morphology, lemma,
    note) per word. A word whose text is empty is a footnote alone: a Ketiv
    the Qere does not read."""
    restored = None
    with io.open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 6:
                continue
            header = parse_header(columns[0])
            if not header:
                continue
            code, chapter, verse, _, word_type = header
            note = None
            if greek:
                if not greek_is_text(word_type):
                    continue
                text, marks = clean_greek(columns[1]), ""
                strongs_field, grammar = split_strongs_grammar(columns[3])
                lemma = lemma_from("", columns[4] if len(columns) > 4 else "")
            else:
                if not hebrew_is_text(word_type):
                    continue
                text, marks = clean_hebrew(columns[1])
                strongs_field, grammar = columns[4], columns[5]
                lemma = lemma_from(columns[11] if len(columns) > 11 else "", "")
                kind = word_type[:1].upper()
                if kind == "Q":
                    note = ketiv_note(columns, read=bool(text))
                elif kind == "R" and restored != (code, chapter, verse):
                    # One note per restored verse, on its first word.
                    note = ("Not in the Leningrad Codex; restored from the parallel in "
                            f"{RESTORED_FROM[code]}.")
                restored = (code, chapter, verse) if kind == "R" else None
            if not text and not note:
                continue
            yield (
                osis(code), chapter, verse, text, marks,
                strongs_keys(strongs_field), grammar.strip() or None, lemma, note,
            )


def word_span(text: str, start: int) -> tuple[int, int, str]:
    """The offsets of the word itself inside `text`, punctuation trimmed.

    The verse keeps its punctuation — `λόγος,` is how the clause reads — but the
    annotation marks the word, so a reader tapping it highlights `λόγος` and not
    the comma after it. Trimming is by Unicode category rather than by a list of
    marks, so Hebrew vowel points and cantillation (which are combining marks,
    not letters) are kept while a maqaf or a full stop is not.
    """
    first, last = 0, len(text)
    while first < last and unicodedata.category(text[first])[0] in "PZ":
        first += 1
    while last > first and unicodedata.category(text[last - 1])[0] in "PZ":
        last -= 1
    if first == last:
        return start, start + len(text), text
    return start + first, start + last, text[first:last]


def split_strongs_grammar(field: str) -> tuple[str, str]:
    """Greek packs both into one column: `G0976=N-NSF`."""
    if "=" in field:
        keys, grammar = field.split("=", 1)
        return keys, grammar
    return field, ""


def build(paths: list[Path], greek: bool, wanted: set[str] | None):
    """Collects words into the schema's book/chapter/verse shape."""
    books: dict[str, dict] = {}
    words = footnotes = 0
    for path in paths:
        for code, chapter, verse, text, marks, keys, grammar, lemma, note in read_words(path, greek):
            if wanted and code not in wanted:
                continue
            book = books.setdefault(code, {})
            verses = book.setdefault(chapter, {})
            entry = verses.setdefault(verse, {"text": "", "annotations": [], "footnotes": []})
            if not text:
                # A Ketiv not read: the note marks where it is written.
                footnotes += 1
                entry["footnotes"].append((str(footnotes), len(entry["text"]), note))
                continue
            # A maqaf joins what follows it with no space, the way a Hebrew
            # Bible prints it; everything else takes one.
            if entry["text"] and not entry["text"].endswith(MAQAF):
                entry["text"] += " "
            start = len(entry["text"])
            entry["text"] += text + marks
            words += 1
            if note:
                footnotes += 1
                entry["footnotes"].append((str(footnotes), len(entry["text"]) - len(marks), note))
            data = {}
            if keys:
                data["strongs"] = ",".join(keys)
            if grammar:
                data["morphology"] = grammar
            if lemma:
                data["lemma"] = lemma
            if not data:
                continue
            first, last, word = word_span(text, start)
            annotation = {
                "type": "strongs",
                "start": first,
                "end": last,
                "text": word,
                "data": data,
            }
            entry["annotations"].append(annotation)
    return books, words


def to_schema(books: dict[str, dict]) -> list[dict]:
    output = []
    for code in sorted(books, key=lambda c: BOOK_INFO[c][1]):
        name, number, testament = BOOK_INFO[code]
        chapters = []
        for chapter in sorted(books[code]):
            verses = []
            for verse in sorted(books[code][chapter]):
                entry = books[code][chapter][verse]
                content = {"text": entry["text"]}
                if entry["annotations"]:
                    content["annotations"] = entry["annotations"]
                record = {
                    "v": verse,
                    "ref": number * 1_000_000 + chapter * 1_000 + verse,
                    "content": content,
                }
                if entry["footnotes"]:
                    content["footnote_refs"] = [
                        {"id": key, "offset": offset} for key, offset, _ in entry["footnotes"]
                    ]
                    record["footnotes"] = [
                        {"id": key, "type": "manuscript", "content": note}
                        for key, _, note in entry["footnotes"]
                    ]
                verses.append(record)
            chapters.append({"chapter": chapter, "verses": verses})
        output.append({
            "id": code, "name": name, "number": number,
            "testament": testament, "chapters": chapters,
        })
    return output


EDITIONS = {
    "TAHOT": {
        "id": "TAHOT",
        "name": "Translators Amalgamated Hebrew Old Testament",
        "abbreviation": "TAHOT",
        "language": "hbo",
        "languageName": "Ancient Hebrew",
        "textDirection": "rtl",
        "description": (
            "The Hebrew text of the Leningrad Codex with per-word Strong's numbers, "
            "dictionary forms and morphology, from STEPBible's Translators Amalgamated "
            "Hebrew Old Testament. It reads the scribes' corrections (Qere), with the "
            "written text they replace (Ketiv) in footnotes, and supplies Joshua 21:36–37 "
            "and Nehemiah 7:68's last words from their parallels. Other variants are in "
            "the source dataset rather than in this reading text."
        ),
        "greek": False,
        "prefix": "TAHOT_",
    },
    "TAGNT": {
        "id": "TAGNT",
        "name": "Translators Amalgamated Greek New Testament",
        "abbreviation": "TAGNT",
        "language": "grc",
        "languageName": "Koine Greek",
        "textDirection": "ltr",
        "description": (
            "The Greek text of the Nestlé-Aland editions with per-word Strong's numbers, "
            "dictionary forms and morphology, from STEPBible's Translators Amalgamated "
            "Greek New Testament. Words read only by the Textus Receptus or by other "
            "editions are in the source dataset rather than in this reading text."
        ),
        "greek": True,
        "prefix": "TAGNT_",
    },
}


def meta(edition: dict) -> dict:
    return {
        "schemaVersion": "1.0",
        "id": edition["id"],
        "type": "translation",
        "name": edition["name"],
        "abbreviation": edition["abbreviation"],
        "description": edition["description"],
        "language": edition["language"],
        "languageName": edition["languageName"],
        "textDirection": edition["textDirection"],
        "translationPhilosophy": "literal",
        "copyright": "Data created by STEPBible.org based on work at Tyndale House, Cambridge.",
        "license": "CC BY 4.0",
        "attribution": (
            "Tyndale House, Cambridge and STEPBible.org. "
            "Source: https://STEPBible.github.io/STEPBible-Data"
        ),
        "features": {
            "strongs": True,
            "morphology": True,
            "redLetter": False,
            "footnotes": not edition["greek"],
            "sectionHeadings": False,
            "poetry": False,
            "paragraphs": False,
        },
        "versification": "standard",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="sources/stepbible")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--editions", default="TAHOT,TAGNT")
    parser.add_argument("--books", help="OSIS abbreviations to keep, comma-separated")
    options = parser.parse_args()

    # Over every code the datasets use, not only the ones the mapping names, and
    # before 100MB is read: six codes were missing when this was the other way
    # round, two of them Matthew and Acts.
    unresolved = {code: osis(code) for code in STEPBIBLE_CODES if osis(code) not in BOOK_INFO}
    if unresolved:
        sys.exit(f"These book codes do not resolve to the book table: {unresolved}")
    if len(STEPBIBLE_CODES) != 66:
        sys.exit(f"Expected 66 book codes, have {len(STEPBIBLE_CODES)}.")

    source = Path(options.source_dir)
    output = Path(options.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    wanted = set(options.books.split(",")) if options.books else None

    for name in options.editions.split(","):
        edition = EDITIONS[name.strip()]
        paths = sorted(source.glob(f"{edition['prefix']}*.txt"))
        if not paths:
            sys.exit(f"No {edition['prefix']}*.txt in {source}.")
        print(f"{edition['id']}: reading {len(paths)} file(s)")
        books, words = build(paths, edition["greek"], wanted)
        if not books:
            sys.exit(f"{edition['id']}: no words matched.")
        document = {"meta": meta(edition), "books": to_schema(books)}
        destination = output / f"{edition['id']}.json"
        with io.open(destination, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False)
        verses = sum(
            len(chapter["verses"]) for book in document["books"] for chapter in book["chapters"]
        )
        print(
            f"{edition['id']}: {len(document['books'])} books, {verses} verses, "
            f"{words} words -> {destination} "
            f"({destination.stat().st_size / 1_000_000:.1f} MB)"
        )


if __name__ == "__main__":
    main()
