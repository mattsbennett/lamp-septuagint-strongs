"""The tagger as the command line builds it, over any form table.

One function builds the tagger for the Septuagint and for every training fold
of the chooser, so the chooser learns from candidates made exactly as the
LXX's are: the TAGNT form table, TBESG, the morphological analyser (built
over the same form table, so a fold's analyser knows nothing of the held-out
rows), and the tolerant lookups, through the same case filter.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SCRIPTS = Path(__file__).resolve().parents[1]
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lxx_tagging.candidates import CaseProfile, Tagger, default_tagger  # noqa: E402
from lxx_tagging.lexicon import Lexicon  # noqa: E402
from lxx_tagging.ntforms import NTForms  # noqa: E402


@dataclass(frozen=True)
class TaggerSpec:
    """How to build the tagger. Called as `spec(forms, lexicon, profile,
    vocabulary)`: `vocabulary` is the letters (`text.bare`) of the words that
    will be tagged, which the analyser needs to build quickly."""

    tbesg_path: Path | None = None
    paradigm: bool = True
    analyser: bool = True

    def __call__(self, forms: NTForms, lexicon: Lexicon, profile: CaseProfile | None,
                 vocabulary: Iterable[str] | None = None) -> Tagger:
        provider = None
        if self.analyser:
            from lxx_tagging.analyser import Analyser
            provider = Analyser(lexicon, forms, vocabulary=vocabulary, tbesg_path=self.tbesg_path)
        return default_tagger(forms, lexicon, profile, paradigm=self.paradigm, analyser=provider)

    def settings(self) -> dict:
        """What the trained chooser depends on, for its cache key."""
        return {"paradigm": self.paradigm, "analyser": self.analyser}
