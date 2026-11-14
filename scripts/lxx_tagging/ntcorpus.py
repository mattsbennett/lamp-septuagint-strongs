"""STEPBible's tagged New Testament as training and test data for the chooser,
with candidate sets made the way a Septuagint token's are.

WHY THIS IS NEEDED

The chooser learns from TAGNT (CC BY 4.0) only. To learn and to be measured
honestly, each New Testament token must reach it as a Septuagint token would:
as Brenton-style text tokenised by `text.tokenize`, looked up by the same
providers, through the same case filter and paradigm expansion, against a form
table that does not contain the token's own row. So:

  * The running text is the Nestle-Aland text (`greek_is_text`, the words the
    shipped NT module has), one verse at a time, punctuation kept, the
    paragraph marks ¶ ¬ and brackets removed.
  * Capitals are set the way Brenton sets them, not as NA prints them: the
    first word of a verse and of a sentence (after . ; ! ?), every word TAGNT
    tags as a name (-P -L -PG -LG) or title (-T: θεός, κύριος), and NA's own
    capital where it says nothing (after a comma or ano teleia: the start of
    quoted speech). Any other capital is lowered. This is the evidence the
    case filter reads in Brenton, where names and the titles of God are
    capitalised, so the filter behaves on the NT as on the LXX.
  * Folds: the books are split into `k` folds (`assign_folds`). A fold's
    tokens get their candidates from a form table built from the rows of the
    *other* folds only, every edition's rows included, as the LXX's table has
    every edition's. So a held-out form either is unseen (no candidate, or a
    TBESG candidate, or a tolerant hint, or the analyser's), or carries only
    the readings the rest of the NT gives it, with the rest of the NT's
    counts. That is the LXX's situation: most of its tokens are forms the NT
    has, in other contexts, and many are forms the NT lacks.
  * The case profile (which keys Brenton capitalises mid-sentence) is read
    from the whole simulated text, as the LXX's is from the whole LXX.

Joined words split by the tokenizer (ἱνατί) take the parts' readings from
TAGNT's compound row; a crasis TAGNT keeps whole (κἀγώ) keeps its compound
code. A row whose surface tokenises to several words (μαράνα θά) gives its
reading to the first and none to the rest.
"""

from __future__ import annotations

import io
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from stepbible_to_translation_schema import (  # noqa: E402
    clean_greek, greek_is_text, lemma_from, parse_header, split_strongs_grammar, strongs_keys,
)
from lxx_tagging import ntforms  # noqa: E402
from lxx_tagging.candidates import CaseProfile, Tagger, TokenCandidates  # noqa: E402
from lxx_tagging.lexicon import Lexicon  # noqa: E402
from lxx_tagging.ntforms import NTForms, Reading, Row, reading_kind  # noqa: E402
from lxx_tagging.text import SENTENCE_BREAKS, Token, bare, tokenize  # noqa: E402

# The order of the New Testament books, and which are narrative.
GOSPELS = ("Mat", "Mrk", "Luk", "Jhn")
NARRATIVE = GOSPELS + ("Act",)
_REMOVE = str.maketrans("", "", "¶¬[]")


@dataclass
class NTVerse:
    """One verse of the simulated NT: its Brenton-style text, tokens, and each
    token's TAGNT reading (None where the row gives none)."""

    book: str
    chapter: int
    verse: int
    text: str
    tokens: list[Token]
    gold: list[Reading | None]
    fold: int = -1

    @property
    def osis(self) -> str:
        return f"{self.book}.{self.chapter}.{self.verse}"


@dataclass
class _Word:
    book: str
    chapter: int
    verse: int
    printed: str          # with punctuation
    row: Row
    in_text: bool


def _read_words(path: Path) -> Iterable[_Word]:
    """Every word row of a TAGNT file with its printed form (punctuation kept)
    and its parsed `Row`, in file order."""
    with io.open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            columns = line.rstrip("\n").split("\t")
            if len(columns) < 6:
                continue
            header = parse_header(columns[0])
            if not header:
                continue
            book, chapter, verse, _, word_type = header
            printed = unicodedata.normalize("NFC", clean_greek(unicodedata.normalize("NFC", columns[1])))
            printed = printed.translate(_REMOVE).strip()
            surface = ntforms.surface_of(columns[1])
            keys, code = split_strongs_grammar(columns[3])
            normalised = strongs_keys(keys)
            lemma = lemma_from("", columns[4])
            if not surface or not printed or not normalised or not code.strip() or not lemma:
                continue
            reading = Reading(",".join(normalised), lemma, code.strip())
            row = Row(columns[0].strip(), word_type, surface, reading,
                      ntforms._compound_parts(columns[3], columns[4]))
            yield _Word(book, chapter, verse, printed, row, greek_is_text(word_type))


def _set_case(word: str, capital: bool) -> str:
    letters = [index for index, char in enumerate(word) if char.isalpha()]
    if not letters:
        return word
    first = letters[0]
    char = word[first].upper() if capital else word[first].lower()
    return word[:first] + char + word[first + 1:]


def _brenton_style(words: list[_Word]) -> tuple[str, list[tuple[int, int]]]:
    """A verse's text with Brenton's capitals, and each word's span in it."""
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    offset = 0
    previous = ""
    for number, word in enumerate(words):
        printed = word.printed
        kind = reading_kind(word.row.reading.morph)
        letters = [char for char in printed if char.isalpha()]
        na_capital = bool(letters) and letters[0].isupper()
        initial = number == 0 or (previous and previous[-1] in SENTENCE_BREAKS)
        pause = bool(previous) and previous[-1] in ",·:"
        capital = kind in ("name", "title") or initial or (na_capital and pause)
        # A word NA prints in capitals throughout is kept as it is.
        if not (len(letters) > 1 and all(char.isupper() for char in letters)):
            printed = _set_case(printed, capital)
        if pieces:
            pieces.append(" ")
            offset += 1
        pieces.append(printed)
        spans.append((offset, offset + len(printed)))
        offset += len(printed)
        previous = printed
    return "".join(pieces), spans


def read(paths: Sequence[Path] | None = None) -> tuple[list[NTVerse], list[tuple[str, int, Row]]]:
    """The NA text as `NTVerse`s, and every row of every edition as (book,
    chapter, row), for building form tables."""
    paths = list(paths) if paths is not None else ntforms.tagnt_paths()
    by_verse: dict[tuple[str, int, int], list[_Word]] = {}
    order: list[tuple[str, int, int]] = []
    rows: list[tuple[str, int, Row]] = []
    for path in paths:
        for word in _read_words(path):
            rows.append((word.book, word.chapter, word.row))
            if not word.in_text:
                continue
            reference = (word.book, word.chapter, word.verse)
            if reference not in by_verse:
                by_verse[reference] = []
                order.append(reference)
            by_verse[reference].append(word)
    verses = []
    for reference in order:
        words = by_verse[reference]
        text, spans = _brenton_style(words)
        tokens = tokenize(text)
        gold: list[Reading | None] = []
        cursor = 0
        used: set[int] = set()
        for token in tokens:
            while cursor < len(spans) and spans[cursor][1] <= token.word_start:
                cursor += 1
            reading = None
            if cursor < len(spans) and spans[cursor][0] <= token.word_start < spans[cursor][1]:
                row = words[cursor].row
                if token.joined and row.parts and len(row.parts) == token.parts:
                    reading = row.parts[token.part]
                elif token.joined:
                    reading = None
                elif cursor not in used:
                    reading = row.reading
                    used.add(cursor)
            gold.append(reading)
        verses.append(NTVerse(*reference, text, tokens, gold))
    return verses, rows


# ---------------------------------------------------------------------------
# Folds

def assign_folds(verses: Sequence[NTVerse], k: int = 5, holdout: str = "stratified") -> dict[tuple[str, int], int]:
    """(book, chapter) -> fold.

    "stratified" (the default): each Gospel and Acts is cut into `k` runs of
    whole chapters, one per fold (the run a fold gets rotates from book to
    book), and the epistles and Revelation go whole, largest first, to the
    fold with the fewest tokens so far. So every fold has Gospel, Acts and
    epistle material, and most of its tokens are in books or long runs the
    training folds never saw.
    "book": whole books only, largest first to the smallest fold.
    """
    if k < 2:
        raise ValueError("at least two folds are needed")
    sizes: Counter = Counter()
    chapters: dict[str, Counter] = defaultdict(Counter)
    book_order: list[str] = []
    for verse in verses:
        if verse.book not in chapters:
            book_order.append(verse.book)
        sizes[verse.book] += len(verse.tokens)
        chapters[verse.book][verse.chapter] += len(verse.tokens)
    assignment: dict[tuple[str, int], int] = {}
    load = [0] * k
    whole = list(book_order)
    if holdout == "stratified":
        whole = [book for book in book_order if book not in NARRATIVE]
        for rotation, book in enumerate(book for book in book_order if book in NARRATIVE):
            numbers = sorted(chapters[book])
            total = sum(chapters[book].values())
            run, running = 0, 0
            for number in numbers:
                fold = (run + rotation) % k
                assignment[(book, number)] = fold
                load[fold] += chapters[book][number]
                running += chapters[book][number]
                if running >= total * (run + 1) / k and run < k - 1:
                    run += 1
    elif holdout != "book":
        raise ValueError(f"holdout must be 'stratified' or 'book', not {holdout!r}")
    for book in sorted(whole, key=lambda b: (-sizes[b], book_order.index(b))):
        fold = min(range(k), key=lambda f: (load[f], f))
        for number in chapters[book]:
            assignment[(book, number)] = fold
        load[fold] += sizes[book]
    for verse in verses:
        verse.fold = assignment[(verse.book, verse.chapter)]
    return assignment


def table_without(rows: Sequence[tuple[str, int, Row]], assignment: dict[tuple[str, int], int],
                  excluded: Iterable[int]) -> NTForms:
    """The form table of every row outside the excluded folds."""
    excluded = set(excluded)
    return NTForms(row for book, chapter, row in rows
                   if assignment.get((book, chapter), -1) not in excluded)


# ---------------------------------------------------------------------------
# Candidate sets

TaggerFactory = Callable[..., Tagger]


def profile_of(verses: Iterable[NTVerse]) -> CaseProfile:
    return CaseProfile.from_tokens(token for verse in verses for token in verse.tokens)


def vocabulary_of(verses: Iterable[NTVerse]) -> set[str]:
    """The letters of the verses' words: what an analyser must be able to read."""
    return {bare(token.key) for verse in verses for token in verse.tokens}


def candidate_sets(verses: Sequence[NTVerse], forms: NTForms, lexicon: Lexicon, profile: CaseProfile,
                   factory: TaggerFactory) -> list[list[TokenCandidates]]:
    """Each verse's `TokenCandidates`, from a tagger over `forms` (and, if the
    factory builds one, an analyser over `forms` for these verses' words)."""
    tagger = factory(forms, lexicon, profile, vocabulary=vocabulary_of(verses))
    return [[tagger.candidates_for(token) for token in verse.tokens] for verse in verses]


# Shared with forked workers (see `jackknife`).
_SHARED: dict = {}


def _fold_sets(task: tuple[int, tuple[int, ...]]) -> list[list[TokenCandidates]]:
    fold, excluded = task
    verses, rows, assignment, lexicon, profile, factory = (
        _SHARED[name] for name in ("verses", "rows", "assignment", "lexicon", "profile", "factory"))
    forms = table_without(rows, assignment, set(excluded) | {fold})
    return candidate_sets([v for v in verses if v.fold == fold], forms, lexicon, profile, factory)


def jackknife(verses: Sequence[NTVerse], rows, assignment, lexicon: Lexicon, profile: CaseProfile,
              factory: TaggerFactory, folds: Iterable[int], excluded: Iterable[int] = (),
              jobs: int = 1, tasks: Sequence[tuple[int, tuple[int, ...]]] | None = None
              ) -> dict:
    """For each fold f in `folds`, the candidate sets of f's verses from the
    table of every fold but f and `excluded`. Returned by fold, verses in order.

    `tasks`, instead, lists (fold, excluded folds) pairs and the result is
    keyed by pair. `jobs` > 1 builds them in forked worker processes (an
    analyser takes a minute or so to build); the result is the same."""
    if tasks is None:
        excluded_tuple = tuple(sorted(set(excluded)))
        work = [(fold, excluded_tuple) for fold in folds]
    else:
        work = list(tasks)
    _SHARED.update(verses=verses, rows=rows, assignment=assignment, lexicon=lexicon, profile=profile,
                   factory=factory)
    try:
        if jobs > 1 and len(work) > 1:
            import multiprocessing
            with multiprocessing.get_context("fork").Pool(min(jobs, len(work))) as pool:
                results = pool.map(_fold_sets, work, chunksize=1)
        else:
            results = [_fold_sets(task) for task in work]
    finally:
        _SHARED.clear()
    if tasks is None:
        return {fold: sets for (fold, _), sets in zip(work, results)}
    return {task: sets for task, sets in zip(work, results)}


def same_reading(candidate_reading: tuple[str, str, str | None], gold: Reading) -> bool:
    """Whether a candidate's reading is TAGNT's, lemmas compared in NFC. A
    candidate without morphology (a name or loanword whose case its form does
    not show, which the policy tags without it) matches on number and lemma."""
    strongs, lemma, morph = candidate_reading
    if strongs != gold.strongs or unicodedata.normalize("NFC", lemma) != unicodedata.normalize("NFC", gold.lemma):
        return False
    return morph == gold.morph or morph is None
