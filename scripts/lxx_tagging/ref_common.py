"""Shared pieces for the private reference-set scorers (``validate_refs.py``).

Scoring-only, behind the same firewall as the CCAT harness: reference files
live under ``.cache/lxx_validation/<name>/`` and nothing in the tagger may
import this module or any ``ref_*`` module.

A reference is read into ``RefVerse`` objects (chapter, verse, tokens) in
the reference's own order and numbering; ``ccat_align.align_book`` then
aligns them with Brenton by content, exactly as for CCAT.
"""

from __future__ import annotations

import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

try:
    from .ccat_greek import form_key
except ImportError:  # run as a script, or with this directory on sys.path
    from ccat_greek import form_key


@dataclass
class RefToken:
    """One word of a reference text with its annotation (all optional)."""

    form: str
    lemma: str = ""
    tag: str = ""            # the reference's own POS/morphology code or feature string
    numbers: tuple = ()      # Strong's-style numbers (ABP)
    parts: list = field(default_factory=list)  # sub-tokens of a multiword token
    key: str = ""

    def __post_init__(self) -> None:
        if not self.key:
            self.key = form_key(self.form)


@dataclass
class RefVerse:
    label: str
    chapter: int | None
    verse: int | None
    letter: str = ""
    tokens: list = field(default_factory=list)


def fetch(url: str, dest: Path, *, insecure: bool = False, delay: float = 1.0) -> bool:
    """Download one file if it is not already there (sequential, polite)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "lxx-validation/1.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=120) as r:
        data = r.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    time.sleep(delay)
    return True


def group_verses(rows) -> list[RefVerse]:
    """rows: iterable of (chapter, verse, token) in text order -> RefVerses,
    a new verse whenever (chapter, verse) changes."""
    out: list[RefVerse] = []
    cur = None
    for ch, v, tok in rows:
        if cur is None or (cur.chapter, cur.verse) != (ch, v):
            cur = RefVerse(f"{ch}:{v}", ch, v)
            out.append(cur)
        cur.tokens.append(tok)
    return out
