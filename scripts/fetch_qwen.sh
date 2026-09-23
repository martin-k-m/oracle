#!/usr/bin/env bash
# fetch_qwen.sh: download Qwen2.5-Coder-0.5B-Instruct and convert it to an int8
# Twill tree under models/qwen/.
#
# The weights are Qwen's open release (Apache-2.0), hosted by Hugging Face. They
# are NOT trained in Twill; only Oracle's runtime is. The download is about 1 GB
# of bf16 weights and the int8 tree it converts to is about 475 MB, so neither is
# committed to git; this runs once on your machine. After it finishes:
#
#   oracle code "def fibonacci(n):"
#
# Requirements: curl, and a Python 3 with numpy for the one-time conversion.
# Set PYTHON to an interpreter that has numpy if the default lacks it.

set -euo pipefail

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDIR="${QWEN_DIR:-$ROOT/models/qwen}"
BASE="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/main"
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
fetch tokenizer.json
fetch model.safetensors

echo "converting to an int8 Twill tree ..."
"$PYTHON" "$ROOT/scripts/convert_qwen.py" "$DDIR"
"$PYTHON" "$ROOT/scripts/convert_qwen_tok.py" "$DDIR"
