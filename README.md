# Oracle

[![CI](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/martin-k-m/oracle?sort=semver)](https://github.com/martin-k-m/oracle/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Written in Twill](https://img.shields.io/badge/written%20in-Twill-6e5494.svg)](https://github.com/twill-lang/twill)

Oracle is a small language model written entirely in [Twill](https://github.com/twill-lang/twill). Twill is the language and the loom; Oracle is the cloth woven on it. The model, the tokenizer, the training loop, and the sampler are all Twill source. There is no Python, no C++, and no external model weights. You train it and run it with the single `twill` binary.

It is a from-scratch, character-level decoder-only transformer of about 0.6 million parameters. It is small on purpose: it trains on a laptop CPU in a few minutes, and everything about it is meant to be read and understood rather than treated as a black box. It is not GPT. It learns the letter-by-letter shape of its training text and continues a prompt in that hand. At this scale the output is text-like and often word-like, not fluent English. That is the honest ceiling of a model this size, and watching it reach that ceiling is the point.

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

- Token and learned positional embeddings.
- A stack of pre-norm blocks, each with masked multi-head self-attention and a gelu feed-forward.
- A final layernorm.
- A tied output head: the logits come from reading the token-embedding table a second time, so the map from a token to its vector and back is one relationship learned once.

The default configuration:

| Setting | Value |
| --- | --- |
| Vocabulary | 59 characters (derived from the corpus) |
| Model width (d_model) | 128 |
| Attention heads | 4 |
| Decoder blocks | 3 |
| Context length | 64 characters |
| Parameters | about 609,000 |

Oracle builds on Twill's standard library for the low-level pieces (embedding, layernorm, causal attention, gelu, dense layers, Adam, cross-entropy, and the sampling filters), but the block, the stacked model, the loss, and the generation loop are Oracle's own code in `src/model.tw`. The character tokenizer is in `src/tokenizer.tw`.

## The corpus

Training uses a small public-domain text: the opening 60 KB of "tiny shakespeare" (the works of William Shakespeare are public domain). It is committed at `data/corpus.txt` so training works offline and is reproducible. `scripts/fetch_corpus.sh` regenerates that exact file from the source, so the derivation is not magic.

The corpus is deliberately larger than the model, so what Oracle learns is the structure of the text rather than a memorized copy of it.

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

This reads the corpus, builds the vocabulary, trains with Adam over random windows, prints the loss as it goes, and saves the weights, the config, and the vocabulary to `models/oracle.bin`. Training is a pure function of two fixed seeds, so a second run reproduces the first. The default 800 steps take about six minutes on a recent laptop CPU and reach a cross-entropy loss of roughly 1.8.

## Generate

```
make generate PROMPT="To be, or not to be"
```

or directly:

```
twill run generate.tw "To be, or not to be"
```

This loads `models/oracle.bin`, encodes the prompt with the saved vocabulary, and samples a continuation with temperature and top-k decoding. Sampling is seeded, so a given prompt gives the same continuation every run. The decoding settings live at the top of `generate.tw`.

## Quantize

Oracle can pack its weights to int8, which shrinks the checkpoint on disk and the model in memory at a quality cost that is negligible at this scale. From the repository root:

```
make quantize
```

or directly:

```
twill run quantize.tw
```

This loads `models/oracle.bin`, packs every dense weight into int8 with a per-row scale (in pure Twill, so the file is small too, not just the model in memory), and writes `models/oracle-int8.bin`. The token and position tables and the layernorm parameters stay f64. `generate.tw` runs either checkpoint; pass the int8 one as a second argument:

```
twill run generate.tw "To be, or not to be" models/oracle-int8.bin
```

It detects the int8 file and rebuilds the packed weights into Twill's int8 matmul kernel before sampling.

## Benchmarks

Measured on this machine (Apple Silicon, macOS, Twill 1.18.0, CPU) on 2026-09-22, from `bench.tw`. Reproduce with:

```
make bench                     # strict matmul
make bench MATMUL=fast         # fast matmul microkernels
```

Model and checkpoint:

| Quantity | fp64 | int8 |
| --- | --- | --- |
| Parameters | 609,280 | 609,280 |
| Checkpoint on disk | 4,876,330 B (4.65 MiB) | 776,632 B (0.74 MiB) |
| Model footprint (`nbytes`) | 4,874,240 B | 773,120 B |

The int8 checkpoint is 6.28x smaller on disk and the model is 6.30x smaller in memory. Only the dense weights are packed; the small f64 tables and norms are carried through, which is why the ratio is a little under the 8x of a pure int8-for-f64 swap.

Generation speed, 56 tokens continued from a short prompt within the 64-token context, tokens per second:

| Path | strict matmul | fast matmul |
| --- | --- | --- |
| Uncached (re-run whole context each step) | ~290 | ~234 |
| KV-cache | ~2,390 | ~2,580 |

The KV-cache is about 8x faster than re-running the whole context, and the cached and uncached greedy continuations are identical token for token within the context length (0 mismatches), so the speed is not bought with a changed output. `TWILL_MATMUL=fast` is not a win at this size: it slows the uncached path by about 20 percent and helps the cached path by under 10 percent. The hand-written microkernels are built to amortize over large matmuls, and Oracle's are 128 wide, too small for the setup to pay off. The flag is wired up and reported so the effect is visible rather than assumed.

Quality proxy, mean cross-entropy and perplexity over 30 held-out 64-character windows from the tail of the corpus (lower is better):

| Model | cross-entropy | perplexity |
| --- | --- | --- |
| fp64 | 1.68857 | 5.41172 |
| int8 | 1.68872 | 5.41255 |

Int8 costs about 0.0008 of perplexity, which is within the noise of the model itself. At this scale int8 is the efficient default with no meaningful quality regression. Int4 is not shipped: the twill primitive exists, but at 128-wide dense layers the per-block 4-bit scheme does not buy enough over int8 to justify a second format for this model, so the honest choice is one good quantization rather than two.

Peak process memory, from `/usr/bin/time -l` on a 300-token generation:

| Run | peak resident set |
| --- | --- |
| fp64 generate | ~31 MB |
| int8 generate | ~105 MB |

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

Oracle is a teaching-scale model. At about 0.6 million parameters trained for a few minutes on 60 KB of text, it learns spelling, spacing, common short words, and the rough cadence of the corpus. It does not learn grammar, meaning, or facts, and it will produce nonsense words and broken sentences. It has no instruction following, no chat behavior, and no knowledge of anything outside its training text. It is a small, honest, from-scratch demonstration of how a transformer language model is built and trained, all the way down, in one language.

Phase 1 built a working, trained, generating model. Phase 2 made inference efficient and measured it: int8 quantization, a KV-cache, the fast-matmul option, and the benchmarks above. Phase 3 is this public release: the rename to Oracle, the unified `oracle` CLI, CI that shape-checks every file, tag-triggered releases, and this documentation. What is left for a future phase is a native packed-int8 load path to remove the reconstruction memory spike, and rotary positions if generation is to run past the 64-token context without re-basing.

## License

MIT, Copyright (c) 2026 Martin Muskov. See `LICENSE`.
