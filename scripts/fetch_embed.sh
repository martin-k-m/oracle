#!/usr/bin/env bash
# fetch_embed.sh: download all-MiniLM-L6-v2 and convert it to a Twill weight
# tree under models/embed/, for `oracle ask --semantic`.
#
# The weights are the open sentence-transformers release (Apache-2.0), hosted by
# Hugging Face. They are NOT trained here; only src/embed.tw is. The download is
# about 90 MB and the converted tree is about 180 MB (kept in full precision, as
# the model is small), so neither is committed to git; this runs once.
#
# Requirements: curl, and a Python 3 with numpy for the one-time conversion.
# Set PYTHON to an interpreter that has numpy if the default lacks it.

set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDIR="${EMBED_DIR:-$ROOT/models/embed}"
BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
PYTHON="${PYTHON:-python3}"

mkdir -p "$DDIR"

if ! "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
  echo "fetch-embed needs a Python 3 with numpy for the one-time conversion." >&2
  echo "Install it (pip install numpy), or set PYTHON to an interpreter that has it:" >&2
  echo "  PYTHON=/path/to/python oracle fetch-embed" >&2
  exit 1
fi

fetch() {
  local remote="$1" local_name="$2"
  if [ -s "$DDIR/$local_name" ]; then
    echo "have $local_name"
  else
    echo "downloading $local_name ..."
    curl -L --fail -o "$DDIR/$local_name" "$BASE/$remote"
  fi
}

echo "fetching all-MiniLM-L6-v2 ..."
fetch config.json bert_config.json
fetch vocab.txt vocab.txt
fetch model.safetensors model.safetensors

echo "converting to a Twill weight tree ..."
"$PYTHON" "$ROOT/scripts/convert_embed.py" "$DDIR"
echo "done. oracle ask will now rerank with the encoder; force it with --semantic."
