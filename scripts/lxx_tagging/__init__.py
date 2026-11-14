"""Tagging Brenton's Septuagint with Strong's numbers, dictionary forms and
morphology from open data.

The package is a pipeline of small parts, each with one job:

  text        tokenises a verse and normalises each word to a lookup key
  ntforms     the forms of STEPBible's tagged Greek New Testament (TAGNT)
  lexicon     STEPBible's Greek lexicon (TBESG), including its Septuagint words
  candidates  readings for a token from an ordered list of providers, and the
              policy (or a plugged-in chooser) that picks one
  hebrew_names  the names of STEPBible's tagged Hebrew Old Testament (TAHOT)
              by verse, their sounds, and the Septuagint's verses mapped onto
              the Hebrew's
  translit    how likely a Greek spelling is as the rendering of a Hebrew name
  names       the name stage: the names the New Testament never uses, tagged
              with the Hebrew Strong's number of the name they render
  capitals    common words Brenton capitalises, tagged as common words
  name_parse  the case of a name tagged from a loose NT match, read from the
              Septuagint word
  tag_septuagint  the command line: LXX.json in, LXX_tagged.json and reports out

Only STEPBible's TAGNT and TBESG and rules written here are used for the
Greek words. Proper names the New Testament does not use (and gentilics) are
identified from STEPBible's TAHOT and TBESH (all four CC BY 4.0), by the
Hebrew name of the same verse they render: an owner-approved exception to
the rule against labelling one text from another, recorded as its own
methods in the decisions file and the report. Nothing derived from the
CCAT/CATSS Septuagint morphology may be read.
"""
