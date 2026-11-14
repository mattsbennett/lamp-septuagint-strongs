#!/usr/bin/env python3
"""
Convert SWORD Bible modules to the translation_schema.json format.
Extracts all available annotations: Strong's, morphology, paragraphs, poetry,
red letter, divine names, added words, footnotes, section headings, etc.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Optional


# H853 is the untranslated Hebrew direct-object marker. Some English SWORD
# modules attach it to the adjacent translated object, which makes that visible
# word look as though it represents H853. Keep the source alignment useful by
# omitting only this non-lexical click target.
UNTRANSLATED_ENGLISH_STRONGS = {"H853"}

# Book data: abbreviation, full name, number, testament
BOOKS = [
    ("Gen", "Genesis", 1, "OT"), ("Exod", "Exodus", 2, "OT"), ("Lev", "Leviticus", 3, "OT"),
    ("Num", "Numbers", 4, "OT"), ("Deut", "Deuteronomy", 5, "OT"), ("Josh", "Joshua", 6, "OT"),
    ("Judg", "Judges", 7, "OT"), ("Ruth", "Ruth", 8, "OT"), ("1Sam", "1 Samuel", 9, "OT"),
    ("2Sam", "2 Samuel", 10, "OT"), ("1Kgs", "1 Kings", 11, "OT"), ("2Kgs", "2 Kings", 12, "OT"),
    ("1Chr", "1 Chronicles", 13, "OT"), ("2Chr", "2 Chronicles", 14, "OT"), ("Ezra", "Ezra", 15, "OT"),
    ("Neh", "Nehemiah", 16, "OT"), ("Esth", "Esther", 17, "OT"), ("Job", "Job", 18, "OT"),
    ("Ps", "Psalms", 19, "OT"), ("Prov", "Proverbs", 20, "OT"), ("Eccl", "Ecclesiastes", 21, "OT"),
    ("Song", "Song of Solomon", 22, "OT"), ("Isa", "Isaiah", 23, "OT"), ("Jer", "Jeremiah", 24, "OT"),
    ("Lam", "Lamentations", 25, "OT"), ("Ezek", "Ezekiel", 26, "OT"), ("Dan", "Daniel", 27, "OT"),
    ("Hos", "Hosea", 28, "OT"), ("Joel", "Joel", 29, "OT"), ("Amos", "Amos", 30, "OT"),
    ("Obad", "Obadiah", 31, "OT"), ("Jonah", "Jonah", 32, "OT"), ("Mic", "Micah", 33, "OT"),
    ("Nah", "Nahum", 34, "OT"), ("Hab", "Habakkuk", 35, "OT"), ("Zeph", "Zephaniah", 36, "OT"),
    ("Hag", "Haggai", 37, "OT"), ("Zech", "Zechariah", 38, "OT"), ("Mal", "Malachi", 39, "OT"),
    ("Matt", "Matthew", 40, "NT"), ("Mark", "Mark", 41, "NT"), ("Luke", "Luke", 42, "NT"),
    ("John", "John", 43, "NT"), ("Acts", "Acts", 44, "NT"), ("Rom", "Romans", 45, "NT"),
    ("1Cor", "1 Corinthians", 46, "NT"), ("2Cor", "2 Corinthians", 47, "NT"), ("Gal", "Galatians", 48, "NT"),
    ("Eph", "Ephesians", 49, "NT"), ("Phil", "Philippians", 50, "NT"), ("Col", "Colossians", 51, "NT"),
    ("1Thess", "1 Thessalonians", 52, "NT"), ("2Thess", "2 Thessalonians", 53, "NT"),
    ("1Tim", "1 Timothy", 54, "NT"), ("2Tim", "2 Timothy", 55, "NT"), ("Titus", "Titus", 56, "NT"),
    ("Phlm", "Philemon", 57, "NT"), ("Heb", "Hebrews", 58, "NT"), ("Jas", "James", 59, "NT"),
    ("1Pet", "1 Peter", 60, "NT"), ("2Pet", "2 Peter", 61, "NT"), ("1John", "1 John", 62, "NT"),
    ("2John", "2 John", 63, "NT"), ("3John", "3 John", 64, "NT"), ("Jude", "Jude", 65, "NT"),
    ("Rev", "Revelation", 66, "NT")
]

# Map pysword book names to our standard names
PYSWORD_BOOK_MAP = {
    "Genesis": "Gen", "Exodus": "Exod", "Leviticus": "Lev", "Numbers": "Num",
    "Deuteronomy": "Deut", "Joshua": "Josh", "Judges": "Judg", "Ruth": "Ruth",
    "I Samuel": "1Sam", "II Samuel": "2Sam", "I Kings": "1Kgs", "II Kings": "2Kgs",
    "I Chronicles": "1Chr", "II Chronicles": "2Chr", "Ezra": "Ezra", "Nehemiah": "Neh",
    "Esther": "Esth", "Job": "Job", "Psalms": "Ps", "Proverbs": "Prov",
    "Ecclesiastes": "Eccl", "Song of Solomon": "Song", "Isaiah": "Isa", "Jeremiah": "Jer",
    "Lamentations": "Lam", "Ezekiel": "Ezek", "Daniel": "Dan", "Hosea": "Hos",
    "Joel": "Joel", "Amos": "Amos", "Obadiah": "Obad", "Jonah": "Jonah", "Micah": "Mic",
    "Nahum": "Nah", "Habakkuk": "Hab", "Zephaniah": "Zeph", "Haggai": "Hag",
    "Zechariah": "Zech", "Malachi": "Mal", "Matthew": "Matt", "Mark": "Mark",
    "Luke": "Luke", "John": "John", "Acts": "Acts", "Romans": "Rom",
    "I Corinthians": "1Cor", "II Corinthians": "2Cor", "Galatians": "Gal",
    "Ephesians": "Eph", "Philippians": "Phil", "Colossians": "Col",
    "I Thessalonians": "1Thess", "II Thessalonians": "2Thess", "I Timothy": "1Tim",
    "II Timothy": "2Tim", "Titus": "Titus", "Philemon": "Phlm", "Hebrews": "Heb",
    "James": "Jas", "I Peter": "1Pet", "II Peter": "2Pet", "I John": "1John",
    "II John": "2John", "III John": "3John", "Jude": "Jude", "Revelation of John": "Rev"
}

BOOK_INFO = {abbr: (name, num, test) for abbr, name, num, test in BOOKS}


@dataclass
class Annotation:
    type: str
    start: int
    end: int
    text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class ParsedVerse:
    clean_text: str
    annotations: list
    footnotes: list
    paragraph: bool = False
    poetry_indent: int = 0
    stanza_break: bool = False


def normalize_strongs(s: str) -> str:
    """Normalize Strong's number: H07225 -> H7225"""
    if not s:
        return s
    match = re.match(r'([HG])0*(\d+)', s)
    if match:
        return match.group(1) + match.group(2)
    return s


def reader_represented_strongs(keys: list[str]) -> list[str]:
    """Return normalized, unique keys represented by visible English text."""
    seen = set()
    represented = []
    for raw_key in keys:
        key = normalize_strongs(raw_key)
        if key in UNTRANSLATED_ENGLISH_STRONGS or key in seen:
            continue
        seen.add(key)
        represented.append(key)
    return represented


def parse_osis_verse(raw_text: str, book_num: int) -> ParsedVerse:
    """
    Parse OSIS XML markup and extract clean text with annotations.
    Uses a two-pass approach: first tokenize, then build clean text with annotations.
    """
    if not raw_text:
        return ParsedVerse("", [], [])

    annotations = []
    footnotes = []
    footnote_counter = [0]

    # Track state
    paragraph_start = False
    poetry_indent = 0
    stanza_break = False

    # Check for paragraph markers at start
    if '<milestone marker="¶"' in raw_text or re.search(r'<div[^>]*sID=', raw_text):
        paragraph_start = True

    # Check for poetry markers
    line_match = re.search(r'<l\s+level="(\d+)"', raw_text)
    if line_match:
        poetry_indent = int(line_match.group(1))

    # Check for stanza break
    if '<lg sID=' in raw_text:
        stanza_break = True

    # Remove all title elements (section headings are extracted separately)
    # This includes parallel passage references and regular section headings
    raw_text = re.sub(r'<title[^>]*>.*?</title>', '', raw_text, flags=re.DOTALL)

    # Extract footnotes
    def extract_footnote(match):
        footnote_counter[0] += 1
        fid = str(footnote_counter[0])
        content = match.group(1)
        content = re.sub(r'<[^>]+>', '', content)
        if content.strip():
            footnotes.append({
                "id": fid,
                "type": "translation",
                "content": content.strip()
            })
        return ''

    raw_text = re.sub(r'<note[^>]*>(.*?)</note>', extract_footnote, raw_text, flags=re.DOTALL)

    # Tokenize the text into a list of (type, content, attrs) tuples
    tokens = []

    # Patterns
    patterns = [
        # SWORD uses self-closing `<w .../>` elements for source-language words
        # that have no visible English span. Match these first: otherwise the
        # paired-word regex consumes from this tag through the next `</w>` and
        # incorrectly assigns the untranslated key to that following text.
        ('w_empty', re.compile(r'<w\s+([^>]*)/\s*>', re.DOTALL)),
        ('w_tag', re.compile(r'<w\s+([^>]*)>(.*?)</w>', re.DOTALL)),
        ('divine', re.compile(r'<divineName>(.*?)</divineName>')),
        ('added', re.compile(r'<transChange[^>]*type="added"[^>]*>(.*?)</transChange>')),
        ('red_start', re.compile(r'<q[^>]*who="Jesus"[^>]*>')),
        ('red_end', re.compile(r'</q>')),
        ('any_tag', re.compile(r'<[^>]+>')),
    ]

    pos = 0
    while pos < len(raw_text):
        # Try each pattern
        matched = False
        for ptype, pattern in patterns:
            m = pattern.match(raw_text, pos)
            if m:
                if ptype == 'w_empty':
                    # There is deliberately no visible range to annotate.
                    pass
                elif ptype == 'w_tag':
                    attrs = m.group(1)
                    content = m.group(2)
                    # Handle divine name inside word
                    dn_match = re.search(r'<divineName>(.*?)</divineName>', content)
                    if dn_match:
                        # Split into: text before, divine name, text after
                        before = content[:dn_match.start()]
                        divine_text = dn_match.group(1)
                        after = content[dn_match.end():]
                        if before.strip():
                            tokens.append(('word', re.sub(r'<[^>]+>', '', before), attrs))
                        tokens.append(('divine_word', divine_text, attrs))
                        if after.strip():
                            tokens.append(('word', re.sub(r'<[^>]+>', '', after), attrs))
                    else:
                        clean_content = re.sub(r'<[^>]+>', '', content)
                        if clean_content:
                            tokens.append(('word', clean_content, attrs))
                elif ptype == 'divine':
                    tokens.append(('divine', m.group(1), None))
                elif ptype == 'added':
                    tokens.append(('added', m.group(1), None))
                elif ptype == 'red_start':
                    tokens.append(('red_start', '', None))
                elif ptype == 'red_end':
                    tokens.append(('red_end', '', None))
                # Skip any_tag silently
                pos = m.end()
                matched = True
                break

        if not matched:
            # Regular text character
            tokens.append(('text', raw_text[pos], None))
            pos += 1

    # Build clean text and annotations from tokens
    result = []
    current_pos = 0
    in_red_letter = False
    red_letter_start = 0

    for token_type, content, attrs in tokens:
        if token_type == 'text':
            result.append(content)
            current_pos += len(content)

        elif token_type == 'word' or token_type == 'divine_word':
            word_start = current_pos
            result.append(content)
            current_pos += len(content)
            word_end = current_pos

            # Extract Strong's numbers
            strongs_list = []
            lemma_value = None

            if attrs:
                lemma_match = re.search(r'lemma="([^"]*)"', attrs)
                if lemma_match:
                    lemma_str = lemma_match.group(1)
                    for s in re.findall(r'strong:([HG]\d+)', lemma_str):
                        strongs_list.append(normalize_strongs(s))
                    # Extract original lemma
                    lemma_words = re.findall(r'lemma\.(?:BSBlex|TR):([^\s"]+)', lemma_str)
                    if lemma_words:
                        lemma_value = lemma_words[0]

            strongs_list = reader_represented_strongs(strongs_list)

            # Extract morphology
            morph_list = []
            if attrs:
                morph_match = re.search(r'morph="([^"]*)"', attrs)
                if morph_match:
                    morphs = re.findall(r'(?:strongMorph|robinson):([^\s"]+)', morph_match.group(1))
                    morph_list.extend(morphs)

            # Create annotation
            if strongs_list:
                ann_data = {"strongs": strongs_list[0] if len(strongs_list) == 1 else ','.join(strongs_list)}
                if morph_list:
                    ann_data["morphology"] = morph_list[0] if len(morph_list) == 1 else ','.join(morph_list)
                if lemma_value:
                    ann_data["lemma"] = lemma_value

                annotations.append(Annotation(
                    type="strongs",
                    start=word_start,
                    end=word_end,
                    text=content,
                    data=ann_data
                ))

            # Add divine name annotation
            if token_type == 'divine_word':
                annotations.append(Annotation(
                    type="divine-name",
                    start=word_start,
                    end=word_end,
                    text=content
                ))

        elif token_type == 'divine':
            word_start = current_pos
            result.append(content)
            current_pos += len(content)
            annotations.append(Annotation(
                type="divine-name",
                start=word_start,
                end=current_pos,
                text=content
            ))

        elif token_type == 'added':
            word_start = current_pos
            result.append(content)
            current_pos += len(content)
            annotations.append(Annotation(
                type="added",
                start=word_start,
                end=current_pos,
                text=content
            ))

        elif token_type == 'red_start':
            in_red_letter = True
            red_letter_start = current_pos

        elif token_type == 'red_end':
            if in_red_letter and current_pos > red_letter_start:
                annotations.append(Annotation(
                    type="red-letter",
                    start=red_letter_start,
                    end=current_pos,
                    text=''.join(result[red_letter_start:])
                ))
            in_red_letter = False

    raw_text_built = ''.join(result)

    # Build position mapping from raw to normalized text
    # This handles leading/trailing whitespace removal and whitespace collapse
    normalized_parts = []
    raw_to_norm = {}  # Maps raw position to normalized position
    norm_pos = 0
    raw_pos = 0
    in_whitespace = True  # Start true to skip leading whitespace

    for char in raw_text_built:
        if char.isspace():
            if in_whitespace:
                # Collapse consecutive whitespace / skip leading
                raw_to_norm[raw_pos] = norm_pos
            else:
                # First whitespace after content - include one space
                raw_to_norm[raw_pos] = norm_pos
                normalized_parts.append(' ')
                norm_pos += 1
                in_whitespace = True
        else:
            raw_to_norm[raw_pos] = norm_pos
            normalized_parts.append(char)
            norm_pos += 1
            in_whitespace = False
        raw_pos += 1

    # Handle end position
    raw_to_norm[raw_pos] = norm_pos

    clean_text = ''.join(normalized_parts).rstrip()  # Remove trailing space if any

    # Final cleanup: remove any remaining XML/HTML tags that slipped through
    # This handles edge cases where tags weren't matched by the tokenizer
    if '<' in clean_text:
        clean_text = re.sub(r'<[^>]+>', '', clean_text)
        # Re-normalize whitespace after tag removal
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    # Adjust annotation positions
    annotations_out = []
    for ann in annotations:
        new_start = raw_to_norm.get(ann.start, ann.start)
        new_end = raw_to_norm.get(ann.end, ann.end)

        # Clamp to valid range
        new_start = max(0, min(new_start, len(clean_text)))
        new_end = max(new_start, min(new_end, len(clean_text)))

        # Extract actual text at new position
        actual_text = clean_text[new_start:new_end]

        ann_dict = {
            "type": ann.type,
            "start": new_start,
            "end": new_end,
            "text": actual_text
        }
        if ann.data:
            ann_dict["data"] = ann.data
        annotations_out.append(ann_dict)

    return ParsedVerse(
        clean_text=clean_text,
        annotations=annotations_out,
        footnotes=footnotes,
        paragraph=paragraph_start,
        poetry_indent=poetry_indent,
        stanza_break=stanza_break
    )


def parse_config(config_text: str) -> dict:
    """Parse SWORD .conf file to extract metadata."""
    meta = {}
    for line in config_text.split('\n'):
        line = line.strip()
        if '=' in line and not line.startswith('#'):
            key, value = line.split('=', 1)
            meta[key.strip()] = value.strip()
    return meta


# Modules whose Strong's numbers were aligned by machine, about half on the
# wrong word: they are read without them.
DROP_STRONGS = {"ASV"}


def get_translation_meta(config: dict, module_id: str) -> dict:
    """Build translation metadata from SWORD config."""
    # Map known translations to their philosophy
    philosophy_map = {
        "KJV": "formal",
        "BSB": "optimal",
        "ASV": "formal",
        "YLT": "literal",
        "WEB": "formal",
        "BBE": "dynamic"
    }

    # Clean translation names
    name_map = {
        "KJV": "King James Version",
        "BSB": "Berean Standard Bible",
        "ASV": "American Standard Version",
        "YLT": "Young's Literal Translation",
        "WEB": "World English Bible",
        "BBE": "Bible in Basic English"
    }

    # Determine features from config
    has_strongs = "StrongsNumbers" in config.get("Feature", "") or "OSISStrongs" in config.get("GlobalOptionFilter", "")
    features = {
        "strongs": has_strongs,
        "morphology": "OSISMorph" in config.get("GlobalOptionFilter", ""),
        "redLetter": "OSISRedLetterWords" in config.get("GlobalOptionFilter", ""),
        "footnotes": "OSISFootnotes" in config.get("GlobalOptionFilter", ""),
        "sectionHeadings": "OSISHeadings" in config.get("GlobalOptionFilter", ""),
        "poetry": True,  # Most have this
        "paragraphs": "NoParagraphs" not in config.get("Feature", "")
    }

    # A module whose source has Strong's keeps the "s" on its id, since the
    # apps key what people save by it, even where the tags are dropped. The
    # name and abbreviation people see have no suffix: features.strongs says
    # whether the module has the tags.
    display_name = name_map.get(module_id, config.get("Description", module_id))
    trans_id = f"{module_id}s" if has_strongs else module_id
    if module_id in DROP_STRONGS:
        features["strongs"] = False

    meta = {
        "schemaVersion": "1.0",
        "id": trans_id,
        "type": "translation",
        "name": display_name,
        "abbreviation": module_id,
        "description": config.get("About", "").replace("\\par", "\n").strip() if config.get("About") else "",
        "language": config.get("Lang", "en"),
        "languageName": "English" if config.get("Lang", "en") == "en" else config.get("Lang", "en"),
        "textDirection": "ltr",
        "translationPhilosophy": philosophy_map.get(module_id, "formal"),
        "features": features,
        "versification": config.get("Versification", "standard").lower()
    }

    # Add copyright info if available
    if config.get("DistributionLicense"):
        meta["license"] = config.get("DistributionLicense")
    if config.get("Copyright"):
        meta["copyright"] = config.get("Copyright")

    return meta


def extract_headings_from_verse(raw_text: str) -> list:
    """Extract section headings from verse preverse content."""
    headings = []

    # Look for title elements (not canonical psalm titles)
    title_pattern = re.compile(r'<title(?![^>]*canonical="true")[^>]*>(.*?)</title>', re.DOTALL)
    for match in title_pattern.finditer(raw_text):
        title_text = match.group(1)
        # Skip parallel passage references
        if 'type="parallel"' in match.group(0):
            continue
        # Clean up the title text
        title_text = re.sub(r'<[^>]+>', '', title_text).strip()
        if title_text and not title_text.startswith('('):
            headings.append({"text": title_text, "level": 1})

    return headings


def convert_module(source_file: str, module_id: str, output_file: str):
    """Convert a SWORD module to translation schema JSON."""
    from pysword.modules import SwordModules

    print(f"Loading SWORD module: {module_id} from {source_file}")

    modules = SwordModules(source_file)
    modules.parse_modules()
    bible = modules.get_bible_from_module(module_id)

    # Get config for metadata
    import zipfile
    with zipfile.ZipFile(source_file, 'r') as zf:
        conf_files = [f for f in zf.namelist() if f.endswith('.conf')]
        if conf_files:
            config_text = zf.read(conf_files[0]).decode('utf-8')
            config = parse_config(config_text)
        else:
            config = {}

    meta = get_translation_meta(config, module_id)

    # Get book structure
    structure = bible.get_structure()
    all_books = structure._books['ot'] + structure._books['nt']

    books_output = []

    for pysword_book in all_books:
        book_name = pysword_book.name
        std_abbr = PYSWORD_BOOK_MAP.get(book_name)

        if not std_abbr:
            print(f"  Warning: Unknown book name '{book_name}', skipping")
            continue

        book_info = BOOK_INFO.get(std_abbr)
        if not book_info:
            print(f"  Warning: No info for book '{std_abbr}', skipping")
            continue

        full_name, book_num, testament = book_info
        print(f"  Processing {full_name}...")

        chapters_output = []

        for chapter_num in range(1, pysword_book.num_chapters + 1):
            verses_in_chapter = pysword_book.get_indicies(chapter_num)
            verses_output = []
            headings_output = []

            for verse_num in range(1, len(verses_in_chapter) + 1):
                try:
                    raw_text = bible.get(
                        books=[book_name],
                        chapters=[chapter_num],
                        verses=[verse_num],
                        clean=False
                    )
                except Exception as e:
                    print(f"    Error getting {book_name} {chapter_num}:{verse_num}: {e}")
                    continue

                if not raw_text:
                    continue

                # Extract headings before parsing verse
                verse_headings = extract_headings_from_verse(raw_text)
                for h in verse_headings:
                    headings_output.append({
                        "text": h["text"],
                        "beforeVerse": verse_num,
                        "level": h.get("level", 1)
                    })

                # Parse verse content
                parsed = parse_osis_verse(raw_text, book_num)

                if not parsed.clean_text:
                    continue

                # Build verse reference (BBCCCVVV)
                verse_ref = book_num * 1000000 + chapter_num * 1000 + verse_num

                verse_obj = {
                    "v": verse_num,
                    "ref": verse_ref,
                    "content": {
                        "text": parsed.clean_text
                    }
                }

                if parsed.annotations:
                    verse_obj["content"]["annotations"] = parsed.annotations

                if parsed.footnotes:
                    verse_obj["footnotes"] = parsed.footnotes

                if parsed.paragraph:
                    verse_obj["paragraph"] = True

                if parsed.poetry_indent > 0 or parsed.stanza_break:
                    verse_obj["poetry"] = {}
                    if parsed.poetry_indent > 0:
                        verse_obj["poetry"]["indent"] = parsed.poetry_indent
                    if parsed.stanza_break:
                        verse_obj["poetry"]["stanzaBreak"] = True

                verses_output.append(verse_obj)

            chapter_obj = {
                "chapter": chapter_num,
                "verses": verses_output
            }

            if headings_output:
                chapter_obj["headings"] = headings_output

            chapters_output.append(chapter_obj)

        book_obj = {
            "id": std_abbr,
            "name": full_name,
            "number": book_num,
            "testament": testament,
            "chapters": chapters_output
        }

        books_output.append(book_obj)

    if module_id in DROP_STRONGS:
        for book in books_output:
            for chapter in book["chapters"]:
                for verse in chapter["verses"]:
                    content = verse.get("content")
                    if isinstance(content, dict) and "annotations" in content:
                        content["annotations"] = [a for a in content["annotations"] if a["type"] != "strongs"]
                        if not content["annotations"]:
                            del content["annotations"]

    output = {
        "meta": meta,
        "books": books_output
    }

    print(f"Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Done! Converted {len(books_output)} books.")


def main():
    parser = argparse.ArgumentParser(description="Convert SWORD Bible module to translation schema JSON")
    parser.add_argument('--source', '-s', required=True, help='Path to SWORD module zip file')
    parser.add_argument('--module', '-m', required=True, help='Module ID (e.g., KJV, BSB)')
    parser.add_argument('--output', '-o', required=True, help='Output JSON file path')

    args = parser.parse_args()

    convert_module(args.source, args.module, args.output)


if __name__ == "__main__":
    main()
