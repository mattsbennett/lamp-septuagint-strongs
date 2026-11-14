# How the tagging is done

The program reads Brenton's Greek verse by verse and gives each word a Strong's number, a
dictionary form and a parse. It never invents a reading. Every tag is one of a set of candidate
readings found in open data or generated from it by stated rules, and where there are several the
program chooses among them by context. The steps run in this order; each is one module under
`scripts/lxx_tagging/`, and each module's opening comment is the full specification.

## 1. Words and lookup keys (`text.py`)

A verse is split into words. Each word is reduced to a lookup key by rules that fold only
differences which never distinguish two Greek words: capitals, grave for acute, a second accent
thrown back by an enclitic, final sigma, breathings on rho, diaeresis. Breathings on vowels, the
place and kind of accent, and iota subscript are kept, because those do distinguish words (ἡ/ἥ/ᾗ,
εἰς/εἷς).

Some differences are tried only when the key itself finds nothing, and only for the kind of word
the rule is about: movable nu on a verb's third person or a dative plural, οὕτω/οὕτως, ἐκ/ἐξ,
οὐ/οὐκ/οὐχ, Brenton's ὁρᾷν for ὁρᾶν, λήψομαι for λήμψομαι. Phrases Brenton prints as one word
(διατοῦτο, ἱνατί) are split by an explicit list and each part is tagged.

Whether a word is capitalised, and where in its sentence, is remembered. Brenton capitalises names
and the titles of God, so a capital in mid-sentence is evidence that a word is a name.

## 2. Candidate readings (`candidates.py`, `ntforms.py`, `lexicon.py`)

Readings are gathered from three places, in order.

1. **The Greek New Testament.** STEPBible's TAGNT gives every form of the New Testament its
   Strong's number, dictionary form and parse. A Septuagint word spelt the same as a New Testament
   form takes that form's readings. About 83% of the Septuagint's words are found this way.
   Because the New Testament uses a form only in some of its possible parses, parses that every
   paradigm gives the same form are added (a neuter's nominative, accusative and vocative).
2. **The lexicon.** STEPBible's TBESG lists Septuagint words the New Testament lacks under extended
   Strong's numbers. A word that is its own only form (an adverb, a particle, an indeclinable name)
   is found there directly.
3. **The analyser** (`analyser.py`, `morph_*.py`). For an inflected form the New Testament never
   uses, the program generates the paradigms of the lexicon's verbs, nouns, adjectives and pronouns
   and looks the form up in them. Stems come, strongest first, from the word's own New Testament
   forms, from other compounds of the same verb, from tables of irregular principal parts, and from
   the regular rules. A word whose dictionary form is not in the lexicon gets no reading: no number
   is invented.

Readings of the wrong kind for the word's capital are set aside: a lower-case word is not a name,
and a word capitalised in mid-sentence is not a common noun. Set-aside readings stay in the output
with the reason.

## 3. Choosing in context (`chooser.py`, `ntcorpus.py`)

About two thirds of words have one reading and are tagged with it. For the rest, a statistical
model picks one reading per word for the whole verse at once.

The model is a linear model over candidate readings, trained with the averaged structured
perceptron and decoded by beam search. Its features are grammatical, built from the parts of the
parse and not from the particular word: agreement of article, adjective and noun; the case a
preposition governs; a nominative's number and person against the verb; the conjunction before a
verb and the mood it takes; a pronoun's antecedent. So what it learns carries over to words the New
Testament never has.

It is trained only on the tagged New Testament. To make that training resemble the task, the New
Testament is set in Brenton's capitalisation and each part of it is looked up in tables built from
the *other* parts, so its words are ambiguous or unseen in the way Septuagint words are.

Each choice gets a margin (how much worse the best alternative scores) and the margin is turned
into a confidence by a table measured on held-out New Testament text.

## 4. Names

Modules: `names.py`, `translit.py`, `hebrew_names.py`.

Names the New Testament uses (Ἀβραάμ, Δαυίδ) are tagged with their Greek numbers by the steps
above. STEPBible's Greek lexicon has no numbers for the thousands of names the New Testament does
not use. Those are identified from the Hebrew:

- STEPBible's tagged Hebrew Old Testament (TAHOT) says which words of each Hebrew verse are proper
  names, with their Strong's numbers.
- A table of how the Septuagint transliterates Hebrew (ב as β, ח as χ or nothing, and so on),
  fitted to the name pairs STEPBible's lexicon already records, scores how well a Greek spelling
  fits a Hebrew name. Greek case endings are scored separately so an ending is never evidence.
- The Greek names of a verse are assigned one to one to the Hebrew names of the corresponding
  verse, so a list of names lines up as a whole. A match is accepted only if it is clearly better
  than the alternatives, and not if a common Hebrew word of the verse fits the Greek better (the
  Septuagint sometimes transliterates a common noun).
- Where the Septuagint orders its text differently from the Hebrew, neighbouring verses and then
  the chapter are searched at stricter thresholds. A name left alone between two matched names in a
  list takes the Hebrew name left between theirs. A spelling that always matches one Hebrew name
  takes it in the few verses where the Hebrew lacks it.

A name tagged this way carries the **Hebrew** Strong's number (`H3315` on Ἰάφεθ) and no parse. The
divine names the Septuagint translates (יהוה, אֲדֹנָי) are never matched. This is the one place the
program labels the Greek text from another text, and it is confined to proper names and gentilics.

A Greek name whose verses nearly always have one particular Hebrew name that nothing else in
them accounts for takes that name even though it does not transliterate it (Αἰθιοπία for Cush,
Ἀρταξέρξης for Ahasuerus). The threshold is strict and this tags few words.

The match says which Hebrew name stands at that place. Usually the Greek is that name
transliterated. Sometimes the Septuagint has a different name there (2 Chronicles 24:20 reads
Ἀζαρίαν where the Hebrew has Zechariah). The `name_relation` column says which: `transliterates`
or `stands-for`.

## 5. After the names

Three small steps run last, each recorded in the word's row.

- **Common words Brenton capitalises** (Βοῤῥᾶν, Νότον, Οὐαί) look like names and have no Hebrew
  name to match. They are tagged from their lower-case readings, but only when Brenton also prints
  the same word in lower case elsewhere and the reading is not a guess (`capitals.py`).
- **Names matched loosely to a New Testament name** (the same letters with other accents) first
  borrowed that New Testament form's case, which is often wrong. Their case is now read from the
  Septuagint word itself: an iota subscript, the article before it, a preposition that takes one
  case, the ending, a preceding word of kinship. Where none applies the borrowed parse is kept at
  low confidence (`name_parse.py`).
- **Misprints.** The digitised text has a few evident misprints (γἱοὶ for υἱοὶ 38 times, Δέγει
  for Λέγει). The text is never changed. A printed form is read as another word only when an
  explicit table lists it (`text.MISPRINT_TABLE`, 42 entries; the list is in `data/report.json`),
  and the row's `reason` then says "assumed misprint". Anything not in the table is left untagged.

## What the program does not do

- It does not change the text. A suspected misprint is left as printed; its tag is for the word
  assumed, and says so.
- It does not use any other tagged Septuagint, nor any model trained on one. See
  [PROVENANCE.md](PROVENANCE.md).
- It was not checked by hand, beyond the samples described in [SCORES.md](SCORES.md).
