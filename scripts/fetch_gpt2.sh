#!/usr/bin/env bash
# fetch_gpt2.sh: download the open GPT-2 (124M) release and convert it into
# Twill-loadable checkpoints under models/gpt2/.
#
# The weights are OpenAI's open GPT-2 release, hosted by Hugging Face. They are
# NOT trained in Twill; only Oracle's runtime is. The download is ~500MB and the
# converted fp64 checkpoint is ~1GB, so neither is committed to git; this runs
# once on your machine. After it finishes:
#
#   oracle gpt2 "The quick brown fox"
#
# Requirements: curl, and a Python 3 with numpy for the one-time conversion.
# Set PYTHON to point at an interpreter that has numpy if the default lacks it.

set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDIR="${GPT2_DIR:-$ROOT/models/gpt2}"
BASE="https://huggingface.co/gpt2/resolve/main"
PYTHON="${PYTHON:-python3}"

mkdir -p "$DDIR"

fetch() {
  local name="$1"
  if [ -s "$DDIR/$name" ]; then
    echo "have $name"
  else
    echo "downloading $name ..."
    curl -L --fail -o "$DDIR/$name" "$BASE/$name"
  fi
}

fetch config.json
fetch vocab.json
fetch merges.txt
fetch model.safetensors

echo "converting to Twill checkpoints ..."
"$PYTHON" "$ROOT/scripts/convert_gpt2.py" "$DDIR"

echo "done. Try:  oracle gpt2 \"The quick brown fox\""
