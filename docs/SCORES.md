# How good is it?

**This tagging was made by a program and has not been checked by hand.** What follows is the
evidence for how often it is right: agreement with other tagged Septuagints, cross-validation on
the New Testament, and samples read one by one. None of it is a guarantee about any particular
word.

In short, on the half of the books held back from development:

| | Agreement |
|---|---|
| Dictionary form, against CCAT | 99.5% |
| Part of speech, against CCAT | 99.8% |
| Full parse, against CCAT | 95.8% |
| Full parse, against GLAUx | 96.5% |
| Strong's number, against the Apostolic Bible Polyglot | 99.5% |
| CCAT and GLAUx with each other, full parse, same words | 96.8% |

So about one word in 25 has a parse that CCAT disagrees with, and about one in 200 a dictionary
form. The references disagree with each other nearly as often, so the share that is actually
wrong is smaller: our best estimate for the parse is one word in 30 to 50.

## What it was compared with

| Reference | What it is | Words compared |
|---|---|---|
| **CCAT** | The CATSS/CCAT *Morphologically Analyzed Septuagint* (Rahlfs' text): dictionary form and parse for every word, made and corrected by scholars | 96.5% of Brenton's words |
| **ABP** | The Apostolic Bible Polyglot: a Strong's-style number for every word | 91.2% |
| **GLAUx** | A machine tagging of the whole Septuagint by a tagger trained on Greek treebanks: dictionary form and parse | 98.2% |
| **PTNK** | The Universal Dependencies treebank of Genesis and Ruth, checked by hand: dictionary form and parse | Genesis and Ruth, 95.8% |

Each tags a different edition of the Greek, so the texts were aligned verse by verse and word by
word, and **only words spelt the same on both sides are compared**. Words Brenton's text has and
the reference's lacks, or spells differently, are not scored.

None of these was used to make the tags, and nothing from them is in this repository
([PROVENANCE.md](PROVENANCE.md)). GLAUx is not fully independent of CCAT: part of its training
data for Genesis descends from CCAT, and the two agree unusually closely there (98.9% on the full
parse, against 95–97% in other books).

## How agreement is counted

- **Held-back books.** The books were divided once into a development half (Genesis, Leviticus,
  Judges, 1–2 Samuel, 1 Chronicles, Ezra–Nehemiah, Job, Psalms 1–75, Isaiah, Ezekiel, Hosea to
  Micah) and a test half (the rest). While the program was being improved, only error totals from
  the development half were looked at. The figures on this page are from the test half unless they
  say otherwise. The two halves score within a tenth or two of a point of each other.
- **Strict and with conventions.** Two taggings can describe a word identically and still write it
  differently. CCAT files οὐ, μή and καί under one class where STEPBible's codes distinguish
  particle, adverb and conjunction; one calls a form middle where the other calls it passive in
  tenses where the two are spelt alike; one lists εἶπον under λέγω and the other under ἔπω. A
  *strict* figure counts every such difference as a disagreement. A figure *with conventions* sets
  aside differences that fall under a named rule. The rules and how often each applied are listed
  [below](#the-conventions). Both figures are given wherever it matters.
- **Uncertain words** (a reference code that could not be read, a word the two texts divide
  differently) are left out of both figures. They are a quarter of a percent of words or fewer.
- **Untagged words** are not counted as errors here; they are 0.66% of the text and are counted in
  [GAPS.md](GAPS.md).

## Results on the held-back half

| | Against CCAT | Against GLAUx | Against PTNK (Ruth) |
|---|---|---|---|
| Words compared | 235,053 | 238,606 | 2,000 |
| Dictionary form, strict | 96.9% | 96.4% | 95.3% |
| Dictionary form, with conventions | 99.5% | 99.0% | 99.4% |
| Part of speech, with conventions | 99.8% | 99.8% | 99.8% |
| Full parse, strict | 90.6% | 90.6% | 82.0% |
| Full parse, with conventions | 95.8% | 96.5% | 97.3% |

The low strict figures against the treebank are almost all its different way of writing the same
analysis.

### Against the references' agreement with each other

On the words that CCAT, GLAUx and this tagging all cover (about 222,000 in the test half), with
conventions:

| | Ours and CCAT | Ours and GLAUx | CCAT and GLAUx |
|---|---|---|---|
| Dictionary form | 99.5% | 99.0% | 98.9% |
| Part of speech | 99.8% | 99.8% | 99.8% |
| Full parse | 95.8% | 96.5% | 96.8% |

- On dictionary form and part of speech this tagging agrees with each reference as well as they
  agree with each other. What is left there is mostly convention.
- On the full parse there is a real gap: about one point against CCAT and a third of a point
  against GLAUx.
- Where this tagging disagrees with *both* (5,529 words, 2.4%), the two references agree with each
  other nine times in ten. That 2.1% is the best estimate of this tagging's outright parse errors.
- On 93.9% of words every reference that covers the word agrees with the tag, and on 0.7% every
  one disagrees.

### Strong's numbers

Against the Apostolic Bible Polyglot, on 219,991 words of the test half: the same number on
89.0% strictly and **99.5%** with conventions. The conventions are the places where the two
numbering schemes differ by design (ABP numbers every personal pronoun 1473; Strong's gave some
forms their own numbers). Two classes cannot be scored at all and are left out: words outside
Strong's original list, which the two schemes number independently (7,290 words), and proper
names, which ABP does not number (10,233 words).

## Where it is weaker

By how the tag was reached (test half, against CCAT, with conventions):

| Kind of word | Share of text | Dictionary form | Full parse |
|---|---|---|---|
| The form has one reading | 65% | 99.8% | 99.3% |
| One word, several possible parses: the model chose | 26% | 99.6% | 88.2% |
| Several possible words: the model chose | 5% | 97.4% | 92.0% |
| Names matched to the Hebrew | 2.3% | 97.8% | no parse given |
| Names matched loosely to a New Testament name | 0.2% | 82.3% | 69.6% |

The errors are concentrated where a form is genuinely ambiguous and only context decides. On the
development half (226,000 words) the commonest disagreements with CCAT were:

| Disagreement | Words |
|---|---|
| Neuter nominative taken for accusative, or the reverse (τό, τά, αὐτό, πάντα…) | 3,479 |
| Masculine taken for neuter, or the reverse, in the genitive and dative (τοῦ, αὐτῷ, αὐτῶν…) | 2,283 |
| Middle or passive, where CCAT has middle (partly convention) | 2,099 |
| Singular for plural, or the reverse | 796 |
| Imperative where CCAT has indicative (second person plural) | 260 |
| Second for third person | 196 |

Adjectives are the weakest part of speech: their full parse agrees with CCAT 82% of the time,
against 87% between CCAT and GLAUx.

## Confidence

Every tag has a confidence between 0 and 1: the measured chance that the whole tag (number,
dictionary form and parse) is right. It is measured on New Testament text the model had not seen,
not on the Septuagint, so the question is whether it holds there. On the test half, against CCAT,
with conventions:

| Confidence | Share of tagged words | Full parse agrees |
|---|---|---|
| 0.99 – 1.00 | 66% | 99.5% |
| 0.95 – 0.99 | 16% | 96.1% |
| 0.90 – 0.95 | 7% | 89.9% |
| 0.80 – 0.90 | 6% | 82.2% |
| 0.70 – 0.80 | 3% | 72.7% |
| 0.50 – 0.70 | 2% | 64.4% |
| below 0.50 | 0.5% | 59.1% |

It tracks reality in order and roughly in size. Between 0.80 and 0.95 it is a few points
optimistic. Keeping only tags at 0.95 and above keeps 82% of the tagged words at about 98.8% parse
agreement.

We chose not to correct the confidence using these figures. Doing so would put a number derived
from a restricted reference into the published data, and the rule here is that the references
measure and nothing more.

## Names

Names the New Testament does not use are matched to the Hebrew names of the corresponding verse
([METHOD.md](METHOD.md#4-names)). They were checked three ways.

- **Held-out names.** STEPBible's lexicon already records which Hebrew name some Greek names
  render. A quarter of those names were kept out of the transliteration model's fitting, and the
  matcher was run on their occurrences as if they were unknown. On 50 such names
  (3,231 words) it matched 96% of them, and 98.3% of its matches were the name the lexicon records.
  The disagreements were read one by one at an earlier stage (53 then, 52 now): each was a
  different bearer of the name correctly identified, where the lexicon's link was made for the New
  Testament's bearer (Ἀμώς is Amoz, Isaiah's father, in Isaiah 1:1 and Amos in Amos 7).
- **Read by hand.** Two samples of 150 name tags were read one by one against the Hebrew. The first found no wrong name among the same-verse matches but several systematic
  faults, which were fixed ([DEVELOPMENT.md](DEVELOPMENT.md)). In the second, drawn afresh
  afterwards, 2 of 150 were wrong, both from the methods that look beyond the verse. Later, all 38
  names matched by association were read and judged right, one of them arguable.
- **Against ABP.** Of the name-tagged words that line up with a word of the Apostolic Bible
  Polyglot, all but a handful are marked as names there too.

A match says which Hebrew name stands at that place. In 174 cases the Septuagint has a
different name there, and the tables mark those `stands-for` and not `transliterates`
([FORMAT.md](FORMAT.md)).

## Cross-validation on the New Testament

The same program can be tested openly on the New Testament, where STEPBible's own tags are the
answer key. The New Testament was divided into five parts; each part in turn was tagged by a
program that had seen neither its forms nor its tags, and compared with STEPBible.

| | Whole tag right |
|---|---|
| All words | 96.3% |
| Words with more than one possible reading | 90.9% |
| Words whose form the program had never seen | 84.8% |

This is a harder test than the Septuagint in one way (a fifth of the forms are withheld) and an
easier one in another (it is the kind of Greek the model was trained on). `validate_nt.py` reruns
it.

## The conventions

Differences from CCAT that were counted as agreement, with the number of words each applied to on
the development half (226,000 words). `scripts/lxx_tagging/ccat_codes.py` defines each one.

| Convention | Words | What it covers |
|---|---|---|
| Indeclinable class | 8,413 | Conjunction, particle, adverb, preposition or interjection with the same dictionary form: CCAT classes by use in the sentence (adverbial καί as an adverb), STEPBible by the word (δέ a conjunction, οὐ a negative particle) |
| Middle/passive label | 2,239 | Middle against passive in the present, imperfect, perfect and pluperfect, where the forms are identical; STEPBible's "either" and deponent labels against CCAT's middle or passive |
| Attic/Koine spelling | 1,277 | γίγνομαι/γίνομαι, γιγνώσκω/γινώσκω, -ττ-/-σσ-, -ρρ-/-ρσ- |
| Deponent headword | 1,026 | A verb listed under -ομαι by one and -ω by the other |
| Suppletive verb headword | 795 | εἶδον/οἶδα/ὁράω, εἶπον/ἐρῶ/λέγω, φάγω/ἐσθίω and the like listed under different dictionary forms |
| Name spelling | 733 | Proper names differing by ει/ι, a doubled consonant or one letter between Rahlfs, Brenton and the Strong's dictionary form |
| Substantival adjective | 663 | Noun against adjective with the same dictionary form, for an adjective regularly used as a noun |
| Substantive headword | 525 | Masculine or neuter, singular or plural dictionary form of the same noun (ἄκρον/ἄκρος) |
| Spelling variant | 452 | ει/ι and single or double consonants in a dictionary form, and listed pairs (βορρᾶς/βορέας, θέλω/ἐθέλω) |
| Degree marking | 251 | A comparative or superlative marked by one side and not the other (CCAT marks only irregular ones) |
| Pronominal adjective | 247 | Possessives, correlatives and reciprocals: adjectives in CCAT, pronouns in STEPBible |
| Adverbial adjective | 221 | Adverb against adjective of the same stem (πρῶτον/πρῶτος) |
| Crasis headword | 202 | κἀγώ against καί + ἐγώ |
| Frozen case form | 183 | An adverb or preposition against the case form it froze from (κύκλῳ as adverb or as dative of κύκλος) |
| Others (11 rules) | 662 | Compound codes, contracted, negative and comparative headwords, participles used as nouns, numerals, interjections, movable final letters, -μι/-ω verbs, pronoun and impersonal headwords |

## What these measures cannot tell you

- **Agreement is not correctness.** Where a reference is wrong and this tagging is right, it counts
  against it; where both are wrong the same way, it counts for it.
- **Only words spelt alike are compared.** Brenton's text differs from Rahlfs' in places, and those
  words (3.5% against CCAT) are unscored. They are likelier than average to be unusual forms.
- **The conventions are judgements.** A different list would move the with-conventions figures.
  The strict figures are given so you can discount them as you see fit.
- **The reader of the samples** was an AI assistant, not a scholar of Greek or Hebrew
  ([DEVELOPMENT.md](DEVELOPMENT.md)).
