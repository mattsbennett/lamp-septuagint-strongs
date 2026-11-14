#!/bin/sh
# Rebuild everything under data/ from the inputs: Brenton's text and the verse
# map in data/source/, and STEPBible's files fetched at a pinned commit.
# Needs Python 3 (built and tested with 3.13) and nothing else. The first run trains the
# chooser (several minutes on four cores); later runs reuse it from .cache/.
set -eu
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

"$PYTHON" scripts/fetch_sources.py
"$PYTHON" scripts/lxx_tagging/tag_septuagint.py \
    --input data/source/LXX.json \
    --verse-map data/source/versemap_lxx_to_english.json \
    --output-dir outputs
"$PYTHON" scripts/export_data.py
"$PYTHON" scripts/export_data.py --check
