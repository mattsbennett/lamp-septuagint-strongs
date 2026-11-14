# File formats

All files are UTF-8 with Unix line endings. The tables are tab-separated with a header row and no
quoting; no field contains a tab or a line break.

## `data/tagged/NN-Book.tsv`

One file per book, one row per word, in the order the words stand. `NN` is the book's number in the
66-book order (so 16, Nehemiah, is absent: Brenton prints it as chapters 11–23 of 2 Esdras, which
is `15-Ezra`).

### The tag

| Column | Meaning |
|---|---|
| `ref` | The verse as `BBCCCVVV`: book, chapter, verse. `1001001` is Genesis 1:1. Chapters and verses are numbered as the Septuagint numbers them, the Psalms by the Greek count. |
| `osis` | The same reference as `Gen.1.1`. |
| `token` | The word's position in its verse, from 0. |
| `start`, `end` | Where the word is in the verse's text in `data/source/LXX.json`, as character offsets (Unicode code points; `end` is exclusive). |
| `surface` | The word as printed. |
| `strongs` | The Strong's key: `G` and a number in STEPBible's extended set for a Greek word, or `H` and a number for a proper name identified by the Hebrew name it renders (see [METHOD.md](METHOD.md#4-names)). Empty if the word is untagged. |
| `lemma` | The dictionary form. For a Greek key it is STEPBible's; for an `H` key it is the name's own nominative where the Greek declines it, otherwise the form as printed. |
| `morph` | The parse, in the codes STEPBible's Greek New Testament uses (Robinson's scheme): `N-DSF` is a noun, dative singular feminine; `V-2AAI-3S` a verb, second aorist active indicative, third person singular. Empty for names whose case the form does not show, and for every `H` key. |
| `confidence` | How likely the whole tag (number, lemma and parse) is to be right, from 0 to 1, as measured on held-out data. See [SCORES.md](SCORES.md#confidence) for how far to trust it. |

### How the tag was reached

| Column | Meaning |
|---|---|
| `method` | `unique`: the form has one reading. `model`: the statistical chooser picked among several readings in the verse's context. `tolerant`: the word matched a New Testament name only when accents and breathings were ignored. `tolerant-mismatch`: the same, and the borrowed parse does not fit the printed form; the number is kept at very low confidence. `hebrew-verse`, `hebrew-nearby`, `hebrew-order`, `hebrew-consistency`, `hebrew-association`: the five ways a name is matched to a Hebrew name. Empty if untagged. |
| `margin` | How much worse the best alternative scored. For `model` it is the chooser's score difference; for the name methods the match score's. `inf` when there was no alternative. |
| `tier` | What the candidates looked like before any choice: `a` one reading; `b` one word, several parses; `c` several words; `d` looks like a name and has no acceptable Greek reading; `e` no candidate at all; `u` only an unconfirmed match. |
| `reason` | The tier in words. For a word read through the misprint table it begins "assumed misprint" and names the word assumed. |
| `key` | The word as it was looked up, after the spelling rules in `scripts/lxx_tagging/text.py` (lower case, grave to acute, final sigma folded, and so on). |
| `n_readings` | How many readings were considered. |
| `candidates` | Every reading considered, separated by ` ; `. Each is `strongs\|lemma\|morph\|source\|weight`. The source says where the reading came from: `nt-form` (the form occurs in the New Testament with this reading; the weight is how often), `nt-caps`, `nt-compound`, `tbesg` (the lexicon's entry is the form), `analyser` (generated from the lexicon entry's paradigm; the weight is relative to the best reading), `nt-tolerant`, `tbesg-tolerant`. A suffix after `:` names a spelling rule that was needed (`nt-form:movable-nu`), and `+paradigm` marks a parse added because the paradigm gives the same form that parse too. |
| `runner_up` | The reading that came second, in the same shape. |
| `set_aside` | Readings ruled out before the choice, with the reason as a sixth field: for example a common-word reading of a word Brenton capitalises mid-sentence. |
| `part` | For a word Brenton prints joined that is tagged as its parts (`Ἱνατί` as ἵνα and τί): `1/2`, `2/2`. |

### Names

Filled only for words the name stage looked at.

| Column | Meaning |
|---|---|
| `name_hebrew` | The Hebrew name matched: its verse and position, Strong's number as STEPBible writes it, the Hebrew word and STEPBible's English for it. |
| `name_score` | How well the Greek spelling fits that Hebrew name as a transliteration, as a log-likelihood ratio against chance. Higher is better; below 3 is no evidence. |
| `name_grammar` | The case, number and gender the form and its article or preposition show, in the shape of a morphology code with `?` for what is unknown. It is kept here because an `H` key carries no `morph`. |
| `name_key` | `tahot`: the key is the Hebrew number. `tbesg`: the word is a New Testament name in a form the New Testament lacks, and takes that name's Greek number. |
| `name_note` | For a word left untagged, why; for a loosely matched New Testament name, whether the Hebrew confirms it. |
| `name_replaces` | The loose New Testament match the Hebrew overruled, if any. |
| `name_relation` | `transliterates`: the Greek word is the Hebrew name spelt in Greek. `stands-for`: the Septuagint prints a different name at that place, and the key is the Hebrew name it stands for. |

## `data/untagged.tsv`

Every word without a tag: `ref`, `osis`, `token`, `surface`, `tier`, `reason`, `name_note`.
[GAPS.md](GAPS.md) summarises them.

## `data/report.json`

The tagging run's own report: words and tagged words by book and tier, the count for each method,
source and spelling rule, confidence bands, the commonest untagged forms, and the name stage's
settings, calibration and counts.

## `data/MANIFEST.json`

Size and SHA-256 of the two inputs under `data/source/` and of every output, including
`outputs/LXX_tagged.json`.

## `outputs/LXX_tagged.json` (built, not committed)

The build also writes the tagged text as one JSON document, in the shape the Lamp Bible app reads:
`data/source/LXX.json` unchanged, with each verse given a list of annotations
(`start`, `end`, `text`, and `data` holding `strongs`, `lemma` and, where there is one,
`morphology`). It is about 185 MB, too large to commit, and is attached to each release; its
checksum is in the manifest.

## `data/source/`

`LXX.json` is Brenton's Greek text, the build's input: books, chapters and verses with their text.
`versemap_lxx_to_english.json` maps each Septuagint `BBCCCVVV` reference to the English reference
it corresponds to, or `null` where there is none.
