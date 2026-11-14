# Notices

## STEPBible data (CC BY 4.0)

The Strong's numbers, dictionary forms and morphology codes in this repository's tagged text are
drawn from data created by [STEPBible.org](https://www.stepbible.org), based on work at Tyndale
House, Cambridge, and distributed under the
[Creative Commons Attribution 4.0 International licence](https://creativecommons.org/licenses/by/4.0/).
The source is <https://github.com/STEPBible/STEPBible-Data>, read at commit
`5de0a47f35f0fc13dfe1430d104e5db8a4285f9c` (2 November 2026). Four datasets are used:

- **TAGNT**, the Translators Amalgamated Greek New Testament: the forms of the New Testament with
  their Strong's numbers, dictionary forms and morphology codes.
- **TBESG**, the Translators Brief lexicon of Extended Strongs for Greek: the extended Strong's
  numbers and dictionary forms of Septuagint words the New Testament lacks, and which Greek entries
  render which Hebrew names.
- **TAHOT**, the Translators Amalgamated Hebrew Old Testament: the proper names of each Hebrew
  verse with their Strong's numbers.
- **TBESH**, the Translators Brief lexicon of Extended Strongs for Hebrew: which Hebrew numbers name
  the same person or place.

**What was changed.** None of STEPBible's files is reproduced here; `scripts/fetch_sources.py`
downloads them from STEPBible's own repository. What this repository carries is new: STEPBible's
numbers, dictionary forms and codes applied by this project's own program to the words of a
different text, Brenton's Greek Septuagint, which STEPBible has not tagged. Strong's keys are
written without leading zeros or disambiguation letters (`G0976` as `G976`, `H0040G` as `H40`).
Which reading each word received was decided by this project's rules and statistical model, not by
STEPBible, and STEPBible has not reviewed or endorsed the result. Errors in the tagging are this
project's.

## SIL versification tables (MIT)

`data/source/versemap_lxx_to_english.json` maps each Septuagint verse reference to the English
verse it corresponds to. It was computed from the standard versification tables in SIL's libpalaso
(`SIL.Scripture/Resources/*.vrs.txt`, commit `b4f7a6da7b868f9ae909e40fb0708f5fd4efd782`), which are
distributed under this licence:

> The MIT License (MIT)
>
> Copyright (c) 2007-2026 SIL Global
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
> associated documentation files (the "Software"), to deal in the Software without restriction,
> including without limitation the rights to use, copy, modify, merge, publish, distribute,
> sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or
> substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
> NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
> DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
> OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Brenton's Septuagint (public domain)

The Greek text is *The Septuagint Version of the Old Testament*, compiled by Sir Lancelot C. L.
Brenton (1851), as digitised and published by [eBible.org](https://ebible.org/details.php?id=grcbrent)
(`grcbrent`), which marks it public domain. eBible.org does not say who digitised it.

## Texts used only to measure accuracy

The scores in `docs/SCORES.md` compare this tagging with four others: the CATSS/CCAT morphologically
analysed Septuagint, the Apostolic Bible Polyglot, the Universal Dependencies PTNK treebank and
GLAUx. Those are other people's work under their own terms, some of which forbid redistribution or
commercial use. Nothing from them is in this repository, no tag here was taken or learnt from them,
and no word-by-word comparison with them is published; only totals are. `docs/PROVENANCE.md` says how
that separation was kept.
