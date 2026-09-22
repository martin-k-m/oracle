# Oracle

[![CI](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/martin-k-m/oracle?sort=semver)](https://github.com/martin-k-m/oracle/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Written in Twill](https://img.shields.io/badge/written%20in-Twill-6e5494.svg)](https://github.com/twill-lang/twill)

Oracle is a small language model written entirely in [Twill](https://github.com/twill-lang/twill). Twill is the language and the loom; Oracle is the cloth woven on it. The model, the tokenizer, the training loop, and the sampler are all Twill source. There is no Python, no C++, and no external model weights. You train it and run it with the single `twill` binary.

It is a from-scratch, character-level decoder-only transformer of about 1.8 million parameters. It is small on purpose: it trains on a laptop CPU in about half an hour, and everything about it is meant to be read and understood rather than treated as a black box. It is not GPT. It learns the letter-by-letter shape of its training text and continues a prompt in that hand. At this scale the output is text-like and often word-like, with short real words and the cadence of the source, not fluent English. That is the honest ceiling of a model this size, and watching it reach that ceiling is the point.

Position is carried by rotary embeddings (RoPE), not a learned table, so there is no fixed context cap: a long prompt and a long generation run past the 128-character training context, and the KV-cache runs past it too.

## Quickstart

Sixty seconds from a clean machine to generated text. The repository ships the trained checkpoints, so you can generate before you train.

```
go install github.com/twill-lang/twill/cmd/twill@v1.18.0   # get the toolchain
git clone git@github.com:martin-k-m/oracle.git
cd oracle
bin/oracle generate "To be, or not to be"                  # sample from the shipped checkpoint
```

If `twill` is not on your `PATH` after the install, either add your `GOBIN` to `PATH` or point the CLI straight at the binary with `TWILL=/path/to/twill bin/oracle generate "..."`.

## The oracle CLI

`bin/oracle` is one entrypoint over the whole workflow. It finds the twill binary (from `$TWILL`, your `PATH`, or the Go install path) and finds the repository from its own location, so it runs from any directory.

| Command | What it does |
| --- | --- |
| `oracle train` | train the model and write `models/oracle.bin` |
| `oracle generate "<prompt>"` | sample a continuation from the checkpoint |
| `oracle generate "<prompt>" models/oracle-int8.bin` | sample from a specific checkpoint |
| `oracle quantize` | pack `models/oracle.bin` to `models/oracle-int8.bin` |
| `oracle bench` | measure size, speed, memory, and quality |
| `oracle check` | static shape-check every `.tw` file |

The `make` targets do the same things if you prefer them; the CLI and the Makefile are both thin convenience over `twill run`.

## What is inside

The architecture is the standard decoder-only transformer, defined in `src/model.tw`:

- A token embedding table. Position is not a table: it enters through rotary embeddings (RoPE) applied to the query and key projections inside attention, so it has no length limit.
- A stack of pre-norm blocks, each with masked multi-head self-attention and a gelu feed-forward.
- A final layernorm.
- A tied output head: the logits come from reading the token-embedding table a second time, so the map from a token to its vector and back is one relationship learned once.

The default configuration:

| Setting | Value |
| --- | --- |
| Vocabulary | 62 characters (derived from the training slice) |
| Model width (d_model) | 192 |
| Attention heads | 6 |
| Decoder blocks | 4 |
| Training context | 128 characters (RoPE, so not a runtime cap) |
| Parameters | 1,788,672 (about 1.79 million) |

Oracle builds on Twill's standard library for the low-level pieces (embedding, layernorm, causal attention with RoPE, gelu, dense layers, Adam, cross-entropy, and the sampling filters including top-k and top-p), but the block, the stacked model, the loss, the KV-cache, the repetition penalty, and the generation loop are Oracle's own code in `src/model.tw`. The character tokenizer is in `src/tokenizer.tw`.

## The corpus

Training uses a public-domain text: the full "tiny shakespeare" file, about 1.1 MB (the works of William Shakespeare are public domain). It is committed in full at `data/corpus.txt` so training works offline and is reproducible, and `scripts/fetch_corpus.sh` regenerates that exact file from the source, so the derivation is not magic.

Character-level tokenization of the whole 1.1 MB file in the interpreter is slow, so `train.tw` reads a deterministic prefix (`TRAIN_CHARS`, 200,000 characters by default) to keep the wall clock inside the budget. That is still several times the 60 KB the 0.3.0 model trained on, and the prefix is a fixed function of the committed file, so the run stays reproducible. Raise `TRAIN_CHARS` for a longer, slower run. The corpus is deliberately far larger than the model, so what Oracle learns is the structure of the text rather than a memorized copy of it.

## Requirements

- The Twill toolchain, version 1.18.0. Install a released build with `go install github.com/twill-lang/twill/cmd/twill@v1.18.0` and put your `GOBIN` on `PATH`, then confirm `twill --version` prints `1.18.0`.

## Train

From the repository root:

```
make train
```

or directly:

```
twill run train.tw
```

This reads the corpus prefix, builds the vocabulary, trains with Adam over random windows, prints the loss as it goes, and saves the weights, the config, and the vocabulary to `models/oracle.bin`. Training is a pure function of two fixed seeds, so a second run reproduces the first. The default 650 steps took about 26 minutes on this laptop CPU (measured 2026-09-22, 1,579 seconds wall clock) and reached a training-batch cross-entropy loss around 1.6. The held-out cross-entropy is 1.63 (see BENCHMARKS). That is inside the roughly 35-minute budget the hyperparameters are tuned for; raise `STEPS` or `TRAIN_CHARS` for a longer, slower, slightly better run.

## Generate

```
make generate PROMPT="To be, or not to be"
```

or directly:

```
twill run generate.tw "To be, or not to be"
```

This loads `models/oracle.bin`, encodes the prompt with the saved vocabulary, and samples a continuation. Decoding combines temperature, top-k, top-p (nucleus), and a repetition penalty, all with sensible defaults and all overridable as flags:

```
bin/oracle generate "To be, or not to be" --temp 0.7 --topk 40 --topp 0.95 --rep 1.15 --steps 300
```

Sampling is seeded (`--seed`, default 42), so a given prompt and settings give the same continuation every run. Because position is rotary, a long prompt and a large `--steps` run past the 128-character training context rather than being truncated to it. The flags map to `ORACLE_*` environment variables that `generate.tw` reads, so the defaults live at the top of that file.

## Quantize

Oracle can pack its weights to int8, which shrinks the checkpoint on disk and the model in memory at a quality cost that is negligible at this scale. From the repository root:

```
make quantize
```

or directly:

```
twill run quantize.tw
```

This loads `models/oracle.bin`, packs every dense weight into int8 with a per-row scale (in pure Twill, so the file is small too, not just the model in memory), and writes `models/oracle-int8.bin`. The token table and the layernorm parameters stay f64; there is no position table to carry, because RoPE places positions inside attention. `generate.tw` runs either checkpoint; pass the int8 one as a second argument:

```
twill run generate.tw "To be, or not to be" models/oracle-int8.bin
```

It detects the int8 file and rebuilds the packed weights into Twill's int8 matmul kernel before sampling.

## Before and after

The same prompt, "To be, or not to be", greedy-adjacent settings, showing the 0.3.0 model (about 609,000 parameters, 60 KB corpus, learned positions) against this 0.4.0 model (about 1.79 million parameters, 200 KB corpus, RoPE):

Version 0.3.0:

```
To be, or not to be ronte
moner the allinn crustiong-thantesp: and oo the their,
has thale ound soulve of thought weat ilas.
```

Version 0.4.0:

```
To be, or not to be
There am, the dignereful but capsing.

MENENIUS:
Your rendule, I'll what heart hight: fares!
```

Both are still small-model output with invented words, which is the honest ceiling at this scale. The 0.4.0 text holds line and speaker structure, punctuates, and reaches for real words more often. Your exact continuation depends on the decoding flags and seed.

## Benchmarks

Measured on this machine (Apple Silicon, macOS, Twill 1.18.0, CPU) on 2026-09-22, from `bench.tw`. Reproduce with:

```
make bench                     # strict matmul
make bench MATMUL=fast         # fast matmul microkernels
```

Model and checkpoint:

| Quantity | fp64 | int8 |
| --- | --- | --- |
| Parameters | 1,788,672 | 1,788,672 |
| Checkpoint on disk | 14,311,867 B (13.65 MiB) | 1,982,749 B (1.89 MiB) |
| Model footprint (`nbytes`) | 14,309,376 B | 1,978,368 B |

The int8 checkpoint is 7.22x smaller on disk and the model is 7.23x smaller in memory. Only the dense weights are packed; the small f64 token table and norms are carried through, which is why the ratio is a little under the 8x of a pure int8-for-f64 swap. Removing the learned positional table in this release moved the ratio closer to 8x, because the table was one of the f64 tensors that did not shrink.

Generation speed, 56 tokens continued from a short prompt, tokens per second:

| Path | strict matmul | fast matmul |
| --- | --- | --- |
| Uncached (re-run whole context each step) | 96.6 | 87.8 |
| KV-cache | 702.4 | 780.2 |

The KV-cache is 7.3x faster than re-running the whole context (8.9x with fast matmul), and the cached and uncached greedy continuations are identical token for token (0 mismatches) on a run of 149 characters, which is past the 128-character training context: RoPE removes the old ceiling, so the cache is exercised beyond the length it trained on and still agrees exactly. `TWILL_MATMUL=fast` is a small win for the cached path (about 11 percent) and a small loss for the uncached path (about 9 percent): the hand-written microkernels amortize over large matmuls, and Oracle's are 192 wide, still on the small side for the setup to fully pay off. The flag is wired up and reported so the effect is visible rather than assumed.

Quality proxy, mean cross-entropy and perplexity over 30 held-out 128-character windows drawn from a slice of the corpus past the training prefix, so the windows are unseen (lower is better):

| Model | cross-entropy | perplexity |
| --- | --- | --- |
| fp64 | 1.62749 | 5.09109 |
| int8 | 1.62884 | 5.09793 |

Int8 costs about 0.007 of perplexity, which is within the noise of the model itself. At this scale int8 is the efficient default with no meaningful quality regression. Int4 is not shipped: the twill primitive exists, but at these dense-layer widths the per-block 4-bit scheme does not buy enough over int8 to justify a second format for this model, so the honest choice is one good quantization rather than two.

Peak process memory, from `/usr/bin/time -l` on a short generation:

| Run | peak resident set |
| --- | --- |
| fp64 generate | ~52 MB |
| int8 generate | ~139 MB |

The int8 steady footprint is smaller (see `nbytes` above), but loading the int8 checkpoint reconstructs each packed weight through a Twill list before handing it to the int8 kernel, and that reconstruction transiently allocates well above the model it produces. A native builtin that reads packed codes straight into a quantized tensor would remove the spike; it is a phase-3 item, not a property of the format.

## Repository layout

```
oracle/
  bin/
    oracle          one CLI over train, generate, quantize, bench, check
  src/
    model.tw        Oracle's own decoder-only transformer (with the KV-cache path)
    tokenizer.tw    character-level tokenizer
    quant.tw        int8 weight quantization and reconstruction
  data/
    corpus.txt      the committed public-domain training text
  models/
    oracle.bin      the trained f64 checkpoint (written by train.tw)
    oracle-int8.bin the int8 checkpoint (written by quantize.tw)
  scripts/
    fetch_corpus.sh regenerate the corpus deterministically
  train.tw          train and save a checkpoint
  generate.tw       load a checkpoint (f64 or int8) and sample text
  quantize.tw       pack a checkpoint to int8
  bench.tw          measure size, speed, memory, and quality
  Makefile          thin convenience over the twill commands
  CHANGELOG.md      the per-version history
  .github/workflows CI (twill check) and tag-triggered releases
```

## Honest limits

Oracle is a teaching-scale model. At about 1.8 million parameters trained for about half an hour on 200 KB of text, it learns spelling, spacing, common short words, speaker labels, and the rough cadence of the corpus, and it holds a line of pseudo-dialogue together better than the 0.6 million parameter 0.3.0 model did. It does not learn grammar, meaning, or facts, and it will still produce nonsense words and broken sentences. It has no instruction following, no chat behavior, and no knowledge of anything outside its training text. It is a small, honest, from-scratch demonstration of how a transformer language model is built and trained, all the way down, in one language. Bigger and longer-trained would read better, but the point is to stay home-runnable and readable, not to chase fluency.

Phase 1 built a working, trained, generating model. Phase 2 made inference efficient and measured it: int8 quantization, a KV-cache, the fast-matmul option, and the benchmarks above. Phase 3 was the first public release: the rename to Oracle, the unified `oracle` CLI, CI that shape-checks every file, tag-triggered releases, and this documentation. Phase 4 (this release, 0.4.0) modernized the architecture: rotary positions in place of the learned table, which removes the context cap, plus top-p and a repetition penalty in the sampler, a larger model, and a larger corpus. What is left for a future phase is a native packed-int8 load path to remove the reconstruction memory spike (see the peak-memory table), which is what 0.5.0 should pick up.

## License

MIT, Copyright (c) 2026 Martin Muskov. See `LICENSE`.
