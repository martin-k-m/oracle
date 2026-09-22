# Changelog

All notable changes to Oracle are recorded here. Versions follow the three
phases the project was built in: a working model, then efficient inference, then
a public release with a unified CLI and automation.

## [0.3.0] - 2026-09-22

Public release.

- Renamed the project to Oracle throughout the code, the checkpoints, the docs,
  and the history.
- Added a unified `oracle` CLI (`bin/oracle`) with `train`, `generate`,
  `quantize`, `bench`, and `check`. It locates the twill binary from `$TWILL`,
  `PATH`, or the Go install path, and finds the repository from its own location,
  so it runs from any directory.
- Added a CI workflow that shape-checks every `.tw` file on push and pull
  request, so the check that gates merges is the same one you run locally.
- Added a tag-triggered release workflow. Pushing a `vX.Y.Z` tag builds a source
  tarball and publishes a GitHub Release whose notes are this file's section for
  that version, with the checkpoints attached as assets.
- Polished the README for a public audience: a 60-second quickstart, the
  weaving metaphor stated plainly, the full benchmark tables, and an honest
  statement of scope.

## [0.2.0] - 2026-09-22

Efficient inference.

- Added int8 weight quantization in pure Twill (`src/quant.tw`, `quantize.tw`):
  every dense weight is packed with a per-row scale. The checkpoint is 6.3x
  smaller on disk and the model is 6.3x smaller in memory.
- Added a KV-cache generation path in `src/model.tw`. It is about 8x faster than
  re-running the whole context each step, and produces token-for-token identical
  greedy output within the context length.
- Added `bench.tw`: it measures parameters, checkpoint size, model footprint,
  cached and uncached tokens per second, the fast-matmul option, peak memory, and
  held-out cross-entropy and perplexity for both the fp64 and int8 models.
- The int8 quality cost is about 0.0008 of perplexity, within the noise of the
  model itself, so int8 is the efficient default with no meaningful regression.

## [0.1.0] - 2026-09-22

The first trained, generating model.

- A from-scratch, character-level, decoder-only transformer written entirely in
  Twill: token and learned positional embeddings, pre-norm blocks with masked
  multi-head self-attention and a gelu feed-forward, a final layernorm, and a
  tied output head (`src/model.tw`).
- A character tokenizer (`src/tokenizer.tw`) and a committed public-domain
  corpus (`data/corpus.txt`), regenerable with `scripts/fetch_corpus.sh`.
- A deterministic training loop with Adam over random windows (`train.tw`), and a
  seeded sampler with temperature and top-k decoding (`generate.tw`).
- About 609,000 parameters. The default 800 steps train in a few minutes on a
  laptop CPU and reach a cross-entropy loss of roughly 1.7.
