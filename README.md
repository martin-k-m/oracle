# Oracle

Oracle is a small language model written entirely in [Twill](https://github.com/twill-lang/twill). Twill is the language and the loom; Oracle is the cloth woven on it. The model, the tokenizer, the training loop, and the sampler are all Twill source. There is no Python, no C++, and no external model weights. You train it and run it with the single `twill` binary.

It is a from-scratch, character-level decoder-only transformer of about 0.6 million parameters. It is small on purpose: it trains on a laptop CPU in a few minutes, and everything about it is meant to be read and understood rather than treated as a black box. It is not GPT. It learns the letter-by-letter shape of its training text and continues a prompt in that hand. At this scale the output is text-like and often word-like, not fluent English. That is the honest ceiling of a model this size, and watching it reach that ceiling is the point.

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

## Repository layout

```
oracle/
  src/
    model.tw        Oracle's own decoder-only transformer
    tokenizer.tw    character-level tokenizer
  data/
    corpus.txt      the committed public-domain training text
  models/
    oracle.bin      the trained checkpoint (written by train.tw)
  scripts/
    fetch_corpus.sh regenerate the corpus deterministically
  train.tw          train and save a checkpoint
  generate.tw       load a checkpoint and sample text
  Makefile          thin convenience over the twill commands
```

## Honest limits

Oracle is a teaching-scale model. At about 0.6 million parameters trained for a few minutes on 60 KB of text, it learns spelling, spacing, common short words, and the rough cadence of the corpus. It does not learn grammar, meaning, or facts, and it will produce nonsense words and broken sentences. It has no instruction following, no chat behavior, and no knowledge of anything outside its training text. It is a small, honest, from-scratch demonstration of how a transformer language model is built and trained, all the way down, in one language.

This is phase 1: a working, trained, generating model. Efficient quantized inference and benchmarks are planned for a later phase.

## License

MIT, Copyright (c) 2026 Martin Muskov. See `LICENSE`.
