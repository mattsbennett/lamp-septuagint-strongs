# The Septuagint with Strong's numbers

Brenton's Greek Septuagint, every word tagged with a Strong's number, a dictionary form and a
parse, made entirely from openly licensed data and free to use for anything.

```
osis      surface   strongs  lemma    morph     confidence  method
Gen.1.1   ἘΝ        G1722    ἐν       PREP      0.9965      unique
Gen.1.1   ἀρχῇ      G746     ἀρχή     N-DSF     0.9965      unique
Gen.1.1   ἐποίησεν  G4160    ποιέω    V-AAI-3S  0.9965      unique
Gen.1.1   ὁ         G3588    ὁ        T-NSM     0.9940      model
Gen.1.1   Θεὸς      G2316    θεός     N-NSM-T   0.9982      model
Gen.10.2  Ἰάφεθ     H3315    Ἰάφεθ              0.9950      hebrew-verse
```

**This is a machine tagging. It has not been checked by hand.** A program chose every tag; its
accuracy was measured against other tagged Septuagints and by reading samples, and the figures
below are the result. Expect roughly one word in 30 to have a wrong parse and one in 200 a
wrong dictionary form. Each tag carries a confidence so you can set your own threshold.

## What is in it

| | |
|---|---|
| Text | Brenton's Greek (1851), the 39 books of the Hebrew canon, public domain |
| Words | 479,567 |
| Tagged | 476,414 (99.34%) |
| Numbers | STEPBible's extended Strong's numbers, the same scheme as its Greek New Testament |
| Names | Proper names the New Testament lacks carry the Strong's number of the Hebrew name they render |
| Parses | The morphology codes STEPBible's Greek New Testament uses |
| Per word | Source of the reading, method, other readings considered, confidence |

## How good it is

Agreement with other tagged Septuagints on the half of the books held back from development,
after setting aside differences of naming convention. The last column is how often two of those
references agree with *each other* on the same words, which is the realistic ceiling.

| | Against CCAT | Against GLAUx | CCAT against GLAUx |
|---|---|---|---|
| Dictionary form | 99.5% | 99.0% | 98.9% |
| Part of speech | 99.8% | 99.8% | 99.8% |
| Full parse | 95.8% | 96.5% | 96.8% |

Strong's numbers agree with the Apostolic Bible Polyglot's on 99.5% of the words the two schemes
can be compared on. Names matched to the Hebrew agreed with STEPBible's own name links 98% of the
time on names held out for the purpose.

[docs/SCORES.md](docs/SCORES.md) has the method, the conventions, the figures by kind of word and
by confidence, and what the errors are.

## Why it exists

Every other tagged Septuagint we could find either descends from the CATSS/CCAT analysis, whose
terms forbid commercial use and redistribution, or lacks Strong's numbers, or parses each form
without looking at its context. This one uses only STEPBible's CC BY data and Brenton's
public-domain text, and says exactly how each tag was reached and how often it is likely to be
wrong. [docs/COMPARISON.md](docs/COMPARISON.md) sets out the differences.

## Using it

- **`data/tagged/NN-Book.tsv`**: one row per word. Columns are described in
  [docs/FORMAT.md](docs/FORMAT.md). For most purposes you need `ref`, `surface`, `strongs`,
  `lemma`, `morph` and `confidence`.
- **`data/untagged.tsv`**: the words left untagged and why.
- **Releases** carry the whole tagged text as one JSON file in the shape the Lamp Bible app reads.

Things to know before you rely on it:

- Verses are numbered as the Septuagint numbers them. The Psalms follow the Greek count, and
  Nehemiah is chapters 11–23 of `15-Ezra`.
- A key beginning `H` is a Hebrew Strong's number on a proper name, and has no parse.
- The books outside the Hebrew canon (Tobit, Maccabees and the rest) are not included.
- [docs/GAPS.md](docs/GAPS.md) lists what is missing and what is least reliable.

## Licence

Our work here is given away without conditions: the data and documents under CC0 1.0, the code
under 0BSD. The Strong's numbers and dictionary forms come from STEPBible's data, which is CC BY
4.0, so **if you republish the tagged text, credit STEPBible.org and Tyndale House, Cambridge**:

> Strong's numbers, lemmas and morphology codes from data created by
> [STEPBible.org](https://www.stepbible.org) based on work at Tyndale House, Cambridge
> (CC BY 4.0), <https://github.com/STEPBible/STEPBible-Data>.

Crediting this repository is welcome and not required. See [LICENSE.md](LICENSE.md) and
[NOTICE.md](NOTICE.md).

## Rebuilding it

```sh
./build.sh
```

Plain Python 3 with no packages. The script downloads STEPBible's files at a pinned commit, tags
the text, rewrites `data/` and checks the result against what is committed. See
[docs/PROVENANCE.md](docs/PROVENANCE.md).

## Documents

| | |
|---|---|
| [docs/METHOD.md](docs/METHOD.md) | How a word gets its tag |
| [docs/SCORES.md](docs/SCORES.md) | How accuracy was measured, and the results |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | Sources, versions, and what was kept out |
| [docs/COMPARISON.md](docs/COMPARISON.md) | How this differs from other tagged Septuagints |
| [docs/GAPS.md](docs/GAPS.md) | Untagged words and known weaknesses |
| [docs/FORMAT.md](docs/FORMAT.md) | File formats |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | Why and how it was made, stage by stage |

## Corrections

Errors are certain to be there. If you find one, an issue naming the reference and the word is the
most useful report.
