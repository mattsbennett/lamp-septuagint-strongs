# How this differs from other tagged Septuagints

This is a description, not a ranking. Several of the projects named here are better than this one
at what they set out to do: the CATSS/CCAT analysis was made and corrected by scholars over
decades, and the PTNK treebank is checked by hand. What this tagging offers is a particular
combination that, as far as we could find, did not exist: Strong's numbers, a clean public-domain
text, no reliance on restricted data, and published evidence of how good it is.

Everything said about another project below is what its own repository or documentation stated
when we read it on **13 November 2026**. Projects change; check the source before relying on it.

## 1. It does not rest on licence-restricted data

Almost every tagged Septuagint descends from the CATSS/CCAT *Morphologically Analyzed Septuagint*,
whose user declaration forbids commercial use and redistribution without the owners' consent. That
is true of data offered under open licences too, one step removed:

| Project | Its licence label | What it rests on, in its own words |
|---|---|---|
| [OpenScriptorium/lxx-morph](https://github.com/OpenScriptorium/lxx-morph) | data CC BY 4.0 | "derived from the Eliran Wong… edition of Rahlfs", which is the CCAT text and carries CCAT's user declaration |
| [Open Scriptures LxxLemmas](https://github.com/openscriptures/GreekResources) | CC BY 4.0 | "structured to conform to the text files of the lxxmorph": corrections to CCAT's lemmas |
| [Pedalion trees](https://github.com/perseids-publications/pedalion-trees), Septuagint Genesis | MIT / CC BY-SA | "We were happy to rely on the part-of-speech annotation by Kraft, R., ed. 1988. Morphologically Analyzed Septuagint… (CATSS)" |
| [GLAUx](https://github.com/alekkeersmaekers/glaux) | CC BY-SA | Tagged automatically by a tagger whose training data includes the Pedalion trees, and so the CATSS-based Genesis |
| [UD Ancient Greek PTNK](https://github.com/UniversalDependencies/UD_Ancient_Greek-PTNK) | CC BY-SA 4.0 | Text "extracted from https://greekdoc.github.io", with thanks to John Barach "for granting permission to use his morphological annotations"; how that annotation was first made is not documented |
| STEPBible TAGOT | CC BY 4.0 (repository) | Not yet published; listed as planned, with "Morpology based on CCAT" |

One open tagging we found does avoid CCAT:
[QuantForgeSoftware/lxx](https://github.com/QuantForgeSoftware/lxx) (CC BY-SA), which runs the
Morpheus analyser over the First1KGreek text. Points 3, 4 and 6 below are where this tagging
differs from it.

This tagging reads STEPBible's New Testament, lexicons and Hebrew Old Testament, and nothing else.
[PROVENANCE.md](PROVENANCE.md) lists the rules that kept it so.

## 2. Its accuracy is measured and the method is published

[SCORES.md](SCORES.md) gives agreement with four other taggings, on a half of the books that
was held back from development, with the list of naming differences that were counted as
agreement and how often each applied. It also gives how well those reference taggings agree with
each other on the same words, which is the only fair yardstick for agreement between any two.

## 3. It carries Strong's numbers

Each word has a Strong's number in STEPBible's extended set, the same scheme as STEPBible's Greek
New Testament. A search for G26 finds ἀγάπη in both Testaments, and a Septuagint word opens the
same lexicon entry as its New Testament occurrences. CCAT and the treebanks carry lemmas but no
Strong's numbers. The Apostolic Bible Polyglot has its own numbering, which extends Strong's
differently and is not openly licensed.

## 4. Parses are chosen in context

Where a form can be parsed more than one way (τό is nominative or accusative; λύετε indicative or
imperative), the reading is chosen by a model that looks at the verse. A dictionary-style analyser
gives a form the same parse wherever it stands: QuantForgeSoftware/lxx says of its engine that
"Morpheus is form-deterministic", and tags τὸ in Psalm 2:7 as nominative where it is the object.

## 5. Every word records how it was tagged

The tables give, for each word, the source of its reading, the method that chose it, the other
readings considered, the runner-up and a confidence. A user can keep only the high-confidence
tags, or find every place a particular rule was applied.

## 6. The text is clean, complete for its books, and in the public domain

Brenton's Greek as published by eBible.org has all 39 books of the Hebrew canon and is clean: a
scan of the whole text found no words mixing Greek and Latin letters and no stray digits or
sigla, and the tagging turned up only a handful of suspected misprints ([GAPS.md](GAPS.md)). The other open full Septuagint text, First1KGreek's Swete, is uncorrected OCR in places
(we counted about one word in a hundred wrong in Genesis 1, and a collaborator on the project
reports that "an hour's worth of reading Genesis produced 74 errors/omissions over 56 verses",
[issue 2579](https://github.com/OpenGreekAndLatin/First1KGreek/issues/2579)) and has a catalogue
entry for Ecclesiastes with no text. GLAUx tags the Septuagint
text of el.wikisource, whose printed edition is not named. Rahlfs' and the Göttingen texts are
better critical texts than Brenton's; they are not openly available apart from CCAT.

## 7. Nothing in it is share-alike or NonCommercial

Most Ancient Greek taggers and treebanks are CC BY-SA or NonCommercial. None was used, so the only
condition on reusing this tagging is STEPBible's attribution. This project's own contribution is
CC0.

## 8. It can be rebuilt

The program is plain Python with no dependencies, the inputs are pinned by commit and checksum, and
the same inputs give the same bytes. `./build.sh` rebuilds every file under `data/`.

## 9. Its gaps are listed

Every untagged word is in `data/untagged.tsv` with the reason. [GAPS.md](GAPS.md) describes what
is missing and what is weakest.

## What it is not

- **Not hand-checked.** It is the output of a program, with samples read by way of checking. The
  CATSS/CCAT analysis and the PTNK treebank were corrected by people and will be right in places
  where this is wrong.
- **Not a critical text**, and not the whole Septuagint: the books outside the Hebrew canon are
  not included.
- **Not a syntactic analysis.** It gives each word's number, dictionary form and parse, with no
  trees and no word senses.
