#!/usr/bin/env python3
"""Fetch the STEPBible data the tagger reads, at the commit it was built from.

STEPBible asks that copies of its data not be posted elsewhere, so that its
own repository stays the one current copy. This repository therefore carries
none of it: this script downloads the eight files from STEPBible-Data at one
pinned commit into `sources/stepbible/` and checks each against the SHA-256
recorded here. A file that is already present and matches is left alone.

    python3 scripts/fetch_sources.py
    python3 scripts/fetch_sources.py --check     # verify only, download nothing

The data is by STEPBible.org, based on work at Tyndale House, Cambridge, under
CC BY 4.0. See NOTICE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = REPOSITORY_ROOT / "sources" / "stepbible"

COMMIT = "5de0a47f35f0fc13dfe1430d104e5db8a4285f9c"  # 2026-11-02
BASE = "https://raw.githubusercontent.com/STEPBible/STEPBible-Data/" + COMMIT + "/"

AMALGAMATED = "Translators Amalgamated OT+NT/"
LEXICONS = "Lexicons/"

# local name: (path in STEPBible-Data, bytes, SHA-256)
FILES = {
    "TAGNT_Mat-Jhn.txt": (
        AMALGAMATED + "TAGNT Mat-Jhn - Translators Amalgamated Greek NT - STEPBible.org CC-BY.txt",
        14188364, "d4c0c820fdf9f2b450c9f96ce6103d6875028affd2e44094d753f1329ca1fd85"),
    "TAGNT_Act-Rev.txt": (
        AMALGAMATED + "TAGNT Act-Rev - Translators Amalgamated Greek NT - STEPBible.org CC-BY.txt",
        15938967, "122040eeeaeba50d6abbf67b39cacaff687211bda69b080e5d53927c830d9968"),
    "TAHOT_Gen-Deu.txt": (
        AMALGAMATED + "TAHOT Gen-Deu - Translators Amalgamated Hebrew OT - STEPBible.org CC BY.txt",
        18190495, "7763ef367bb0e6e1c1a95fb9b652e1d611b2481827611433e82abbf229216d70"),
    "TAHOT_Jos-Est.txt": (
        AMALGAMATED + "TAHOT Jos-Est - Translators Amalgamated Hebrew OT - STEPBible.org CC BY.txt",
        24500357, "b79add01a8c95673e31ec3ab8ad86183d2312cd96421ddf21f3d1f74377526a7"),
    "TAHOT_Job-Sng.txt": (
        AMALGAMATED + "TAHOT Job-Sng - Translators Amalgamated Hebrew OT - STEPBible.org CC BY.txt",
        9540663, "e0416880bb78f46715bb17ce526b623c3051da0ef83f7bf83beaad350e6272e8"),
    "TAHOT_Isa-Mal.txt": (
        AMALGAMATED + "TAHOT Isa-Mal - Translators Amalgamated Hebrew OT - STEPBible.org CC BY.txt",
        17977558, "d5a609f5a4b9f7757a2615e55877b07f9f6931a8afccef012c0b4bf662caf540"),
    "TBESG.txt": (
        LEXICONS + "TBESG - Translators Brief lexicon of Extended Strongs for Greek - STEPBible.org CC BY.txt",
        4736983, "734a5b64a1152f67604bf09d32f15c0ba537151961752fcf168087b41b4d93ea"),
    "TBESH.txt": (
        LEXICONS + "TBESH - Translators Brief lexicon of Extended Strongs for Hebrew - STEPBible.org CC BY.txt",
        3288115, "848eef2ac16753a43694897dbb9e8f266725e7cc9da4809f367ee5b990f05dbb"),
}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def fetch(name: str, check_only: bool) -> bool:
    upstream, size, expected = FILES[name]
    path = DESTINATION / name
    if path.exists() and digest(path) == expected:
        print(f"ok       {name}")
        return True
    if check_only:
        print(f"{'differs ' if path.exists() else 'missing '} {name}")
        return False
    url = BASE + urllib.parse.quote(upstream)
    request = urllib.request.Request(url, headers={"User-Agent": "fetch-sources/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = response.read()
    found = hashlib.sha256(data).hexdigest()
    if len(data) != size or found != expected:
        print(f"REJECTED {name}: {len(data)} bytes, sha256 {found}; expected {size} bytes, {expected}")
        return False
    DESTINATION.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    temporary.write_bytes(data)
    temporary.replace(path)
    print(f"fetched  {name}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="verify what is present; download nothing")
    options = parser.parse_args()
    results = [fetch(name, options.check) for name in FILES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
