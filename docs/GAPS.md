# Gaps and weak spots

What is missing, what is least reliable, and what the text itself gets wrong. Every untagged word
is in [`data/untagged.tsv`](../data/untagged.tsv) with its reason.

## Not included at all

- **The books outside the Hebrew canon**: Tobit, Judith, Wisdom, Sirach, Baruch, the Letter of
  Jeremiah, Susanna, Bel and the Dragon, 1–4 Maccabees, 1 Esdras, the Prayer of Manasseh. Brenton
  prints them; this tagging does not cover them.
- **Other text forms.** Daniel is Theodotion's version only, Judges is one text, and the tags
  describe Brenton's Greek, not Rahlfs' or the Göttingen editions'.
- **Verse letters.** 3 Kingdoms 12:24a–z and Esther's additions are merged into the numbered
  verses they follow.

## Untagged words: 3,153 of 479,567 (0.66%)

| Why | Words | Examples |
|---|---|---|
| The dictionary form is not in STEPBible's lexicon, or the form could not be analysed | 1,530 | δαβίρ, ἴν, φύλαξ, ἐσωτέρας, σῖτα, πεντηκονταετοῦς |
| A name with no Hebrew name that matches | 1,116 | Ναθινίμ, Αἰθιοπίας, Ἀρταξέρξου, Ἰδουμαίαν, Κύρου, Λίβυες |
| A name in a verse whose Hebrew has no name left to match | 218 | Βααλίμ, Σαριρά, Σουσακίμ, Ἄδερ |
| A name that fits two Hebrew names equally | 151 | Ἰεβοσθέ, Ἰωάς, Σαβά, Καάθ |
| A name whose Hebrew counterpart was claimed by another Greek word | 58 | Χαβραθά, Ἐσεβών, Ἀβεσσαλώμ |
| A lower-case word whose only reading is a name | 38 | χλόης, δορκάς, σατάν |
| A capitalised word that transliterates a common Hebrew word | 22 | Σηλώμ, Θαιμάν |
| Other | 20 | Ὁ printed with a capital mid-sentence; names in verses with no Hebrew counterpart |

Notes on these:

- **No number is invented.** If STEPBible's lexicon has no entry for a word, it stays untagged
  however obvious its meaning. Transliterated Hebrew words (δαβίρ, θεραφίν, ἐφώδ) are the largest
  group.
- **Some of the 1,530 are the analyser's failures**, not the lexicon's: φύλαξ and ἐσωτέρας are
  ordinary words it should read.
- **Greek-origin and translated names** (Αἰθιοπία for Cush, Ἰδουμαία for Edom, Κῦρος) cannot be
  found by transliteration. A few of the commonest are matched another way
  ([METHOD.md](METHOD.md#4-names)); most are not.
- In a sample of 50 untagged names read by hand, about half had a Hebrew counterpart the matcher
  missed, usually through an unusual spelling.

## Tagged, but without a parse

24,091 tagged words (5.0%) carry a number and dictionary form and no parse.

- **10,864 names with a Hebrew number.** By design: the number is a Hebrew one, and a Greek parse
  beside it would be misread. Their case, where it shows, is in the `name_grammar` column.
- **13,227 Greek-numbered words**: names and loanwords whose form does not show a case (Ἀδάμ),
  and words found only as the lexicon's own entry (Εὐφράτης).

## Tagged, and least reliable

In rough order of how much they matter. Figures are from [SCORES.md](SCORES.md).

1. **Ambiguous case and gender.** Where one form serves two cases or genders and only the sentence
   decides, the model is right about 88% of the time. Neuter nominative against accusative and
   masculine against neuter in the genitive and dative account for most parse errors. Two rounds of
   work on this moved it little.
2. **Adjectives**: 82% full-parse agreement with CCAT, against 87% between references.
3. **Middle against passive**, where the forms are identical. Much of the disagreement here is
   convention, but not all.
4. **Names matched loosely to a New Testament name** (about 1,000 words, method `tolerant`): the
   name is usually right (82–98% agreement on the dictionary form in the two halves) and the parse
   often is not (70%). 240 of them keep a borrowed parse at low confidence because nothing in the
   sentence showed their case.
5. **Names placed by list order** (method `hebrew-order`, 203 words) and **names that stand for a
   different Hebrew name** (`name_relation` = `stands-for`, 174 words). In a sample of 48
   `stands-for` rows read by hand, about 40 were right, 2 were wrong (loose transliterations
   mistaken for substitutions) and 6 were arguable.
6. **Confidence between 0.80 and 0.95** is a few points optimistic on the Septuagint.
7. **Dictionary forms of names with a Hebrew number.** Where the nominative is not attested and
   could not safely be built, the dictionary form is an oblique form as printed.
8. **Verses the verse map gets wrong.** Where the Septuagint orders its text differently (Exodus
   35–40, Jeremiah's oracles, parts of 3 Kingdoms) names are found by searching nearby, which is
   less sure. Additions with no Hebrew (Esther, Daniel 3) can only be matched by spelling
   elsewhere.

## The text's own errors

The digitised text has a small number of evident misprints. They are not corrected. Forty-two are
listed in a table (`text.MISPRINT_TABLE`; the list with references is under `misprints` in
[`data/report.json`](../data/report.json)) and the 90 words they cover are read as the word
assumed, with "assumed misprint" in their `reason`. Eighty-nine of them are tagged; εὐώδο in
Genesis 39:23, read as εὐώδου, is a form the analyser cannot read either way. The commonest:

| Printed | Read as | Times |
|---|---|---|
| γἱοὶ | υἱοὶ | 38 |
| μον | μου | 3 |
| τῶς | τῶν | 3 |
| λέλει | λέγει | 2 |
| θυλάτηρ | θυγάτηρ | 2 |

Others certainly remain among the untagged words. If a form in `data/untagged.tsv` looks like a
misprint to you, it probably is.

## What would close the gaps

- A lexicon entry for each missing word. That is STEPBible's to add, not ours to invent.
- A second, independent reading of a sample by someone who reads Greek and Hebrew.
- A model of clause structure, for subjects and objects among neuters.
