# How this was made

An account of why the tagging exists, the decisions behind it, and what happened at each stage,
including what did not work. It is here so that a reader can judge the result knowing how it was
produced.

## Who made it

The tagging was made for Lamp Bible, a free Bible study app, on
13 November 2026. The program, the measurements and these documents were written by an AI coding
assistant (Anthropic's Claude) working under the direction of the app's developer, who set the
rules: which sources may be used, that every word that can be analysed should be tagged, that
restricted taggings may be used for measurement only, and that the result should be published with
its evidence. Samples of the output were read by the assistant, not by a Greek scholar. Read the
accuracy figures with that in mind.

## Why

Lamp Bible links each word of its original-language texts to a lexicon by Strong's number. For
the Hebrew Old Testament and the Greek New Testament it uses STEPBible's data. For the Septuagint
there was nothing it could use:

- Every tagged Septuagint we found descends from the CATSS/CCAT morphological analysis, whose user
  declaration asks recipients not to use the material commercially or pass it on without the
  owners' written consent. The project does not have that consent.
- STEPBible has announced a tagged Septuagint (TAGOT) but had not published it, and describes its
  morphology as based on CCAT.
- The taggings offered under open licences either rest on CCAT one step removed, or lack Strong's
  numbers, or parse each form without regard to context ([COMPARISON.md](COMPARISON.md)).

So the project decided to tag the text itself, from data it is free to use.

## Decisions taken at the start

1. **Text: Brenton's Greek** from eBible.org. A survey of open Septuagint texts found none better
   that is free to use. First1KGreek's Swete has about one word in a hundred wrong and no
   Ecclesiastes; a clean transcription of Swete exists but only under the GPL, with no record of
   the transcriber's grant; Rahlfs' text is not openly available apart from CCAT.
2. **Tag sources: STEPBible's CC BY data and our own rules, nothing else.** No share-alike or
   NonCommercial data, which ruled out Morpheus, Wiktionary, the Perseus lexica, the treebanks and
   every tagger trained on them.
3. **Tag every word that can be analysed**, each with a confidence, rather than only the certain
   ones.
4. **Use the restricted taggings to measure, never to tag**, with half the books held back as a
   test set that development never looks at word by word.
5. **Names**, where STEPBible's Greek lexicon has no number, are identified from STEPBible's tagged
   Hebrew. The project generally refuses to label one text from another; this exception was made
   deliberately, for proper names only, and each such tag is marked as one.

## Stages

Figures are agreement with CCAT on the held-back half of the books, with naming conventions set
aside, unless stated. [SCORES.md](SCORES.md) explains the measures.

### 1. Lookup against the New Testament

Normalise each word, find it among the forms of the tagged New Testament, tag it if it has exactly
one reading. This tagged **59.6%** of the words; of those, the dictionary form agreed with CCAT
99.7% of the time and the full parse about 99.4%. The scoring programs and the division of the
books into development and test halves were written at this stage.

### 2. The context model

A statistical model, trained on the New Testament, to choose among several readings of a form.
Tagged words rose to **84.9%**. Full-parse agreement over all tagged words was 95.8%; where the
model had chosen between parses of one word it was about 87%. The two commonest errors were
already the ones that remain: neuter nominative against accusative, and masculine against neuter
in the genitive and dative.

### 3. The analyser

A generator of Greek paradigms from the lexicon's entries, for forms the New Testament never uses
(about 12% of the words). Tagged words rose to **96.8%**. Full parse fell slightly, to 95.5%: the
new forms brought new confusions, notably middle against passive voice.

### 4. First accuracy round

The model was retrained on training text that had been passed through the analyser, so that it
met analyser-made ambiguity in training, and given more grammar: subject and verb agreement, the
gender of a pronoun's antecedent, the conjunctions that take a subjunctive. Full parse rose from
95.5% to **95.8%**, and from 88.0% to 89.0% where the model chooses. Confusions of vocative, person
and tense fell by a quarter to a half. The largest class, neuter nominative against accusative,
barely moved.

**What did not help.** Features that read clause boundaries from commas made the Septuagint's
choices worse without helping the New Testament, because Brenton prints nearly twice as many
commas as the New Testament text the model learns from. Clauses are now cut at sentence breaks
only.

### 5. Comparing the references with each other

Until this point every disagreement with CCAT had been counted as our error. Measuring CCAT
against GLAUx on the same words showed that two independent taggings agree with each other on the
full parse about 97% of the time, and no more often than we agree with them on dictionary form and
part of speech. It also showed that GLAUx is not independent of CCAT in Genesis, where part of its
annotation descends from a CCAT-based treebank. See [SCORES.md](SCORES.md).

### 6. Names

About 13,000 words (2.7%) were names the New Testament does not use. They were matched to the
names of the corresponding Hebrew verse by a model of how the Septuagint transliterates Hebrew.
Tagged words rose to **99.1%**. On names held out from the model's fitting, 98% of matches agreed
with the name links STEPBible's lexicon already records, and every disagreement read was a case
where the lexicon's link, made for the New Testament bearer of a name, did not fit that
occurrence. In a sample of 150 read by hand, two were wrong.

Faults found by reading samples and fixed: a gentilic tagged as the tribe's founder (Λευΐ for
Λευΐτης); dictionary forms grouped across gentilic endings; one Greek name consuming two Hebrew
occurrences; names assigned by consistency where the Septuagint had read a common Hebrew word as a
name.

### 7. Second accuracy round

Aimed at the agreement of article, adjective and noun, at the names matched loosely to New
Testament names, and at the words still untagged.

- **Agreement: no gain on the Septuagint.** Four new kinds of evidence were added to the model
  (whether the nearest verb still needs a subject or an object, coordination across καί, a clean
  start after a subordinating conjunction, agreement looked for further ahead). Cross-validation on
  the New Testament improved slightly (ambiguous words 90.7% to 90.9%). Agreement with CCAT did not
  move, and the commonest error classes were unchanged. A fifth idea, looking back from an
  adjective to the noun before it, made the New Testament worse and was dropped. A noun-gender
  lexicon was not built: too few Septuagint-only nouns show their gender unambiguously.
- **Names matched loosely to New Testament names**: reading their case from the word itself and
  its neighbours raised their full-parse agreement from 53% to 70% on the held-back half.
- **Untagged words**: capitalised common words, names at the start of a sentence that had never
  reached the name stage, more irregular forms in the analyser, lexicon entries printed without a
  breathing, and a table of evident misprints together took the tagged share from 99.1% to
  **99.3%**.
- **Substituted names** were marked `stands-for` in the tables.

**Something we undid.** This round also fitted a correction to the confidence figures from how
often each confidence band agreed with CCAT on the development half. It made the middle bands more
honest. It was removed before publication, because it put numbers derived from a restricted
reference into the output, and the rule is that the references measure and nothing more. The
uncorrected calibration is reported in [SCORES.md](SCORES.md#confidence).

### 8. Making it reproducible

Preparing this repository, we ran the tagger twice and compared checksums. They differed: the name
stage gave slightly different results from run to run, a few dozen tags out of 480,000. The claim
that the same inputs give the same bytes had been tested only by running twice inside one process,
which cannot see this kind of fault. Two causes were found, both a matter of the
order in which Python walks a set. One Hebrew name, Jerusalem, has a letter with two vowel points,
and which was read first varied from run to run; because Jerusalem is common, that shifted the
transliteration model. And two lexicon rows for one Greek name tied in a sort. Both now have a
stated rule. Four full runs under different hash seeds gave identical bytes, a test now checks the
name stage under several seeds, and the files in `data/` were built from this repository's own copy
of the code and compared with the development copy's output.

## What we would do next

- Real clause structure, to tell subjects from objects among neuters.
- The books outside the Hebrew canon, which have no Hebrew to identify names from.
- A second, independent reading of a sample by someone who reads Greek.
