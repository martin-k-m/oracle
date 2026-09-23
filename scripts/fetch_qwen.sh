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

# The model size, one of 0.5B (default), 1.5B, 3B, 7B. Larger is more capable and
# needs proportionally more memory and time; 0.5B and 1.5B are the comfortable
# laptop sizes. The runtime is config-driven, so any of them runs unchanged.
SIZE="${1:-0.5B}"

ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DDIR="${QWEN_DIR:-$ROOT/models/qwen}"
BASE="https://huggingface.co/Qwen/Qwen2.5-Coder-${SIZE}-Instruct/resolve/main"
PYTHON="${PYTHON:-python3}"

mkdir -p "$DDIR"

# Check the conversion prerequisite before a multi-gigabyte download, not after.
if ! "$PYTHON" -c "import numpy" >/dev/null 2>&1; then
  echo "fetch-qwen needs a Python 3 with numpy for the one-time conversion." >&2
  echo "Install it (pip install numpy), or set PYTHON to an interpreter that has it:" >&2
  echo "  PYTHON=/path/to/python oracle fetch-qwen ${SIZE}" >&2
  exit 1
fi

fetch() {
  local name="$1"
  local required="${2:-yes}"
  if [ -s "$DDIR/$name" ]; then
    echo "have $name"
  elif [ "$required" = "no" ]; then
    curl -L --fail -o "$DDIR/$name" "$BASE/$name" 2>/dev/null || return 0
  else
    echo "downloading $name ..."
    curl -L --fail -o "$DDIR/$name" "$BASE/$name"
  fi
}

echo "fetching Qwen2.5-Coder-${SIZE}-Instruct ..."
fetch config.json
fetch tokenizer.json

# Larger checkpoints are sharded: a model.safetensors.index.json lists the
# shards. Try the single file first; if it is absent, download every shard the
# index names.
if fetch model.safetensors no && [ -s "$DDIR/model.safetensors" ]; then
  echo "have single-file weights"
else
  echo "downloading sharded weights ..."
  curl -L --fail -o "$DDIR/model.safetensors.index.json" "$BASE/model.safetensors.index.json"
  "$PYTHON" - "$DDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
for shard in sorted(set(idx["weight_map"].values())):
    print(shard)
PY
  while read -r shard; do fetch "$shard"; done < <("$PYTHON" - "$DDIR" <<'PY'
import json, os, sys
d = sys.argv[1]
idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
for shard in sorted(set(idx["weight_map"].values())):
    print(shard)
PY
)
fi

echo "converting to an int8 Twill tree ..."
"$PYTHON" "$ROOT/scripts/convert_qwen.py" "$DDIR"
"$PYTHON" "$ROOT/scripts/convert_qwen_tok.py" "$DDIR"
