#!/usr/bin/env bash
# fetch_corpus.sh: regenerate data/corpus.txt deterministically.
#
# Oracle trains on a small public-domain text: the opening ~60 KB of the
# "tiny shakespeare" file (the complete works of William Shakespeare are public
# domain). This script fetches the source and trims it to a whole number of
# lines, producing exactly the data/corpus.txt committed to the repo. The corpus
# is committed too, so training works offline; this script is here so the
# derivation is reproducible rather than magic.
set -euo pipefail

URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
BYTES=60000
OUT="$(dirname "$0")/../data/corpus.txt"

tmp="$(mktemp)"
curl -sSL "$URL" | head -c "$BYTES" > "$tmp"

# Trim to the last complete line so the corpus ends cleanly.
python3 - "$tmp" "$OUT" <<'PY'
import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()
i = s.rfind("\n")
open(out, "w", encoding="utf-8").write(s[:i] + "\n")
print("wrote", out, len(s[:i] + "\n"), "bytes")
PY

rm -f "$tmp"
