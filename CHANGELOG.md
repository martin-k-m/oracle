# Changelog

All notable changes to Oracle are recorded here. Versions follow the phases the
project was built in: a working model, then efficient inference, then a public
release with a unified CLI and automation, a modernized architecture, and now a
leaner int8 load.

## [0.5.0] - 2026-09-22

Efficiency release: a leaner int8 load.

- Fixed the int8 generation memory spike. Loading the int8 checkpoint rebuilt
  each packed weight by appending every value into one Twill list, and `append`
  copies its whole list on each call, so one weight cost O((rows*cols)^2) copies
  and left that much transient garbage. Peak resident set on a 300-token int8
  generation was ~163 MB, far above the ~1.9 MB steady footprint and above the
  ~88 MB fp64 peak. `src/quant.tw` now streams the reconstruction one row at a
  time: each row becomes a `[1, cols]` tensor, the rows are stacked once with
  `concat`, and only one f64 weight is expanded at a time. The int8 peak drops to
  ~64 MB, below the fp64 peak, and the load is much faster too because the
  quadratic copying is gone.
- The quantization math is unchanged. The row-by-row reconstruction produces the
  exact same QTensor as before, so the int8 perplexity (5.09793, delta 0.006836
  over fp64) and the KV-cache correctness (0 mismatched tokens past the training
  context) are identical to 0.4.0. This release changes load mechanics, not the
  numbers.
- Measured int4 rather than assuming it. Quantizing every dense weight to 4-bit
  blocks took perplexity to 5.243 (delta 0.152, about 22x the int8 delta) while
  shrinking the in-memory model only from 1.98 MB to 1.48 MB, because the f64
  token table and norms already dominate the footprint. Int4 stays unshipped: a
  25 percent memory saving for a 22x larger quality hit is not worth a second
  format at this scale.
- A native twill builtin that read packed int8 codes straight into a quantized
  tensor would remove even the one-weight reconstruction; the `quantize` builtin
  ingests only a full 2-D tensor in 1.18.0, so that stays a twill-side follow-up
  and this release ships the best in-language fix.

## [0.4.0] - 2026-09-22

Architecture modernization and a larger model.

- Replaced the learned positional embedding table with rotary position
  embeddings (RoPE) applied to the query and key projections inside attention,
  in both the training forward pass and the KV-cache decode path. Position is now
  a computed rotation rather than a table with a last row, so generation and the
  KV-cache run past the training context instead of stopping at a fixed cap. The
  cached and uncached greedy continuations still agree token for token, now
  verified on a run longer than the training context.
- Added top-p (nucleus) sampling and a repetition penalty on top of the existing
  temperature and top-k, all exposed through `bin/oracle generate` as `--topp`,
  `--rep`, `--temp`, `--topk`, `--steps`, and `--seed`. Sampling stays seeded and
  reproducible.
- Grew the model from about 609,000 parameters to about 1.79 million: d_model
  192, 6 heads, 4 layers, and a 128-character training context, up from 128 / 4 /
  3 / 64. Training now reads a larger slice of the full ~1.1 MB tiny-shakespeare
  corpus (committed in full, regenerable with `scripts/fetch_corpus.sh`).
- Retrained and re-quantized. The new samples are clearly more coherent than the
  0.3.0 baseline. See the README BENCHMARKS section for the measured parameters,
  training time, loss, checkpoint sizes, perplexity delta, and tokens per second.

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
