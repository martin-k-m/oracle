# Oracle

[![CI](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml/badge.svg)](https://github.com/martin-k-m/oracle/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/martin-k-m/oracle?sort=semver)](https://github.com/martin-k-m/oracle/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Written in Twill](https://img.shields.io/badge/written%20in-Twill-6e5494.svg)](https://github.com/twill-lang/twill)

Oracle is a small language model written entirely in [Twill](https://github.com/twill-lang/twill). Twill is the language and the loom; Oracle is the cloth woven on it. The model, the tokenizer, the training loop, and the sampler are all Twill source. The from-scratch model uses no Python, no C++, and no external weights: you train it and run it with the single `twill` binary.

Oracle also runs real pretrained open models through runtimes written in the same Twill. The headline one is a **Qwen2.5-Coder-0.5B runtime**: a genuinely code-capable, instruction-following assistant that runs on a laptop CPU. Ask it in plain English and it writes working, commented code. See [The Qwen coder](#the-qwen-coder). There is also a **GPT-2 124M runtime** that writes coherent English, kept as the simpler first example. Those weights are the open Qwen (Apache-2.0) and GPT-2 releases, not trained here; only the runtimes are Twill.

So Oracle is three things: a small model you can train from scratch and read end to end, a GPT-2 runtime, and a real code assistant you self-host at home, all in Twill.

## The Qwen coder

`oracle code "<prompt>"` answers a plain-English request with real code, running Qwen2.5-Coder-0.5B-Instruct entirely through a Twill runtime (`src/qwen.tw`, `src/qwen_tok.tw`). It is a faithful Qwen2 implementation: RMSNorm, rotary embeddings at theta 1,000,000, grouped-query attention (14 query heads over 2 key/value heads), a SwiGLU feed-forward, and a tied head. At 0.5B parameters it only fits in a laptop's memory as int8 (about 475 MB), rebuilt into the int8 kernel by twill 1.18.4's `quantize_packed`. The tokenizer reproduces Qwen's token ids exactly, and the prompt is wrapped in Qwen's ChatML template, which is what turns raw completion into instruction following.

```
oracle fetch-qwen                                   # one time: ~1 GB download, needs python3 with numpy
oracle code "Write a Python function that returns True if a number is prime."
```

For noticeably better answers, install the 1.5B model instead: `oracle fetch-qwen 1.5B` (about 3 GB, ~8 tokens per second). Once it is installed, oracle uses the most capable model you have by default, so `oracle code` and `oracle chat` pick it up with no flag.

Give it a file as context to ask about real code: `oracle code "what bug could this have?" --file mycode.py`. Or start `oracle chat` and use `/file <path>` to load one or more files into a running conversation, then ask about them. Pass --file more than once to give it several files, or pipe one in: `cat mycode.py | oracle code "add tests"`.

For common tasks there are shortcuts: `oracle explain --file x.py`, `oracle review --file x.py`, and `git diff --staged | oracle commit` to draft a commit message. All read a file or piped stdin and take --model.

Real output, unedited:

```python
Here's a simple Python function to check if a number is prime:

def is_prime(n):
    # Check if the number is less than 2
    if n < 2:
        return False
    # Check for factors from 2 up to the square root of n
    for i in range(2, int(n**0.5) + 1):
        ...
```

Honest scope: Qwen-0.5B is a small model. It is a capable coding assistant that writes functions, explains code, and follows instructions, but it is not a frontier model and will make mistakes on hard problems. Measured on an Apple laptop CPU it generates about sixteen tokens per second once the prompt is read (roughly 0.06 seconds per token), so a short answer takes a few seconds. The speed comes from the twill 1.18.4 int8 kernel, which parallelises a single-token step across every core, from projecting only the last position for the first token, and from twill 1.18.5's sampler, which selects the top-k and the nucleus without sorting the whole 150,000-token vocabulary each step (that sort alone had been costing as much as the model itself). Run a bigger, more capable model with `oracle fetch-qwen 1.5B` (or `3B`): the runtime is config-driven, so a larger Qwen2.5-Coder drops in unchanged, for more capability at proportionally more memory and time. 1.5B is the next comfortable laptop size. Once fetched, select it per command with `oracle code --model 1.5B "..."` or `oracle chat --model 1.5B`; sizes live in their own directories and coexist. The weights are Qwen's open Apache-2.0 release; `oracle fetch-qwen` downloads and converts them once and does not commit them to git.

### Working with your code

The coder reads real code, from a file, several files, or piped stdin, and there are shortcuts for the common tasks:

```
oracle code "add type hints and a docstring" --file util.py
oracle explain --file parser.py
oracle review --file server.py
oracle fix --file parser.py "IndexError on empty input"
oracle tests --file util.py
oracle sh "find every TODO under src and show the file and line"
git diff --staged | oracle commit
cat a.py b.py | oracle code "how do these interact?"
```

`oracle fix` diagnoses a bug and returns corrected code; add the error message or
a description as trailing words and it uses that to locate the problem. `oracle
tests` writes unit tests for a file, and `oracle sh` turns a plain request into a
single shell command. Every task also takes trailing words as extra guidance, so
`oracle review --file server.py "focus on error handling"` narrows the review.

For a session that keeps context, `oracle chat` is an interactive workspace: type a question, watch the reply stream, and use `/file <path>` to load code into the conversation, `/reset` to clear it, `/help` for the list.

### A local web console

If you would rather point and click, `oracle serve` opens a small web console at `http://127.0.0.1:8080`. It is a streaming chat that keeps its context, with a row of tools (write code, explain, review, fix, tests, shell, commit) for one-shot tasks on pasted code. The model is loaded once and held live in a Twill host process (`serve.tw`), so replies stream in token by token and every message after the first pays no load cost. The bridge (`scripts/serve.py`) uses only the Python standard library, binds to localhost, and exposes only the fixed set of tasks.

```
oracle serve                    # http://127.0.0.1:8080
oracle serve --port 9000 --model 1.5B
```

The server is also a model host for the command line. Each `oracle code` or
`oracle fix` normally starts twill and loads the weights again, about two seconds
before it even begins (more for a bigger model). Point the CLI at a running
server and it hands the work there instead, so the model is loaded once for the
server's life and repeated calls skip the reload:

```
oracle serve &                                   # once, in the background
export ORACLE_SERVER=http://127.0.0.1:8080
oracle fix --file parser.py "IndexError"         # answered by the live model
git diff --staged | oracle commit
```

It streams the same way, and falls back to running locally if the server is not
up, so it is safe to leave `ORACLE_SERVER` set. Set it per command with
`--server URL` instead of the environment variable if you prefer. The console has
Style and Length controls, and `--temp` and `--steps` carry through to the server
too, so each request can set its own creativity and reply length. When more than
one model size is installed, a selector in the console switches between them live,
loading the chosen weights without restarting the server.

## The from-scratch model

It is a from-scratch, decoder-only transformer of about 3.4 million parameters over a byte-level BPE vocabulary. It is small on purpose: it trains on a laptop CPU in about half an hour, and everything about it is meant to be read and understood rather than treated as a black box. It is not GPT. It learns the token-by-token shape of its training text and continues a prompt in that hand. At this scale the output is text-like and often word-like, with real words, speaker labels, and the cadence of the source, not fluent English. That is the honest ceiling of a model this size, and watching it reach that ceiling is the point.

The tokenizer is byte-level Byte Pair Encoding: it starts from the 256 byte values, so every input encodes with no unknown token and every id decodes back to exact bytes, and it learns merges by pair frequency up to a target vocabulary. Because BPE packs several characters into each token, a token context covers far more text than a character context did, which is the main quality lift in this release.

Position is carried by rotary embeddings (RoPE), not a learned table, so there is no fixed context cap: a long prompt and a long generation run past the 256-token training context, and the KV-cache runs past it too.

## Quickstart

Sixty seconds from a clean machine to generated text. The repository ships the trained checkpoints, so you can generate before you train.

```
go install github.com/twill-lang/twill/cmd/twill@v1.18.5   # get the toolchain
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
| `oracle test` | run the tokenizer round-trip tests |
| `oracle fetch-gpt2` | download and convert the open GPT-2 124M weights |
| `oracle gpt2 "<prompt>"` | continue a prompt with the GPT-2 runtime |
| `oracle fetch-qwen` | download and convert Qwen2.5-Coder-0.5B (the code model) |
| `oracle code "<prompt>" [--file F]` | ask the Qwen coder about code, optionally giving it a file as context |

The `make` targets do the same things if you prefer them; the CLI and the Makefile are both thin convenience over `twill run`.

## The GPT-2 runtime

Oracle's own model is small and honest about it. To show what the same Twill can do with real weights, Oracle also implements a faithful [GPT-2](https://openai.com/research/better-language-models) 124M inference engine, in `src/gpt2.tw` and `src/gpt2_tok.tw`. It is the full modern decoder architecture: learned token and position embeddings, twelve pre-norm blocks of causal multi-head attention and a gelu feed-forward, a bias on every projection, and a tied output head. The byte-level BPE tokenizer reproduces GPT-2's token ids exactly, checked against the reference encoder, so the pipeline is Twill end to end: encode, forward, decode, with no Python at run time.

The weights are OpenAI's open GPT-2 release. They are downloaded and converted to a Twill tensor tree once, on your machine, by a one-line command; they are not committed to this repository and they are not trained here.

```
oracle fetch-gpt2                              # one time: ~500 MB download, needs python3 with numpy
oracle gpt2 "Dear team, I wanted to update you on the timeline."
```

Real output, unedited:

> Dear team, I wanted to update you on the project timeline. Our plan is that we will be working in conjunction with our community partners and their organizations until January of 2019 (when a full transition period begins). We believe this needs your help

That is genuine, fluent English from a prompt, running on a CPU. What it is honest about: GPT-2 small writes coherent short prose and email-shaped text, and it is **weak at code**. Asked to continue `def fibonacci(n):` it produces Python-shaped but wrong output, because GPT-2 predates the code-heavy training that makes today's models good at programming. It has no instruction following and no facts you should trust. It is a 2019 model at the smallest size, run at home for the interest of running it.

Requirements and cost, measured on an Apple laptop CPU: the converted fp64 weights are about 1 GB on disk, generation uses roughly 2.3 GB of RAM, and it produces about 1.8 tokens per second with the key/value cache. `oracle fetch-gpt2` needs `curl` and a Python 3 with `numpy` for the one-time conversion; the conversion is data prep, not the runtime.

There is an int8 path (`oracle quantize-gpt2`, then `oracle gpt2 --int8`), and it is honest about a tradeoff rather than a free win. It cuts the weights from about 1 GB to 126 MB on disk, a 7.5x reduction, and generation stays coherent with a small memory saving. But in twill 1.18.4 it runs about six times slower than fp64, because the int8 kernels are reconstructed from the packed codes at load and that reconstruction is linear in the model's 124 million parameters. So fp64 is the default for the GPT-2 runtime, and int8 is there for when disk or bandwidth matters more than speed. The real fix is a native twill builtin that reads packed codes straight into the int8 kernel, which is a change to twill itself, not to Oracle.

## What is inside

The architecture is the standard decoder-only transformer, defined in `src/model.tw`:

- A token embedding table. Position is not a table: it enters through rotary embeddings (RoPE) applied to the query and key projections inside attention, so it has no length limit.
- A stack of pre-norm blocks, each with masked multi-head self-attention and a gelu feed-forward.
- A final layernorm.
- A tied output head: the logits come from reading the token-embedding table a second time, so the map from a token to its vector and back is one relationship learned once.

The default configuration:

| Setting | Value |
| --- | --- |
| Vocabulary | 1024 BPE tokens (256 byte tokens + 768 learned merges) |
| Model width (d_model) | 256 |
| Attention heads | 8 |
| Decoder blocks | 4 |
| Training context | 256 BPE tokens (RoPE, so not a runtime cap) |
| Parameters | 3,417,600 (about 3.42 million) |

Oracle builds on Twill's standard library for the low-level pieces (embedding, layernorm, causal attention with RoPE, gelu, dense layers, Adam, cross-entropy, and the sampling filters including top-k and top-p), but the block, the stacked model, the loss, the KV-cache, the repetition penalty, and the generation loop are Oracle's own code in `src/model.tw`. The byte-level BPE tokenizer, including its merge trainer, its encoder and decoder, and its save/load, is Oracle's own code in `src/tokenizer.tw`.

### The tokenizer

`src/tokenizer.tw` is byte-level Byte Pair Encoding, and it is built so the encode/decode machinery is separable from where the merges come from. `train_from_corpus` learns a merge table on Oracle's corpus by iterated most-frequent-pair merging; `from_merges` builds the identical tokenizer record from any ordered list of id pairs. The encoder pre-tokenizes text into whitespace-attached chunks so a token never spans two words, then applies the merges greedily; the decoder expands each id to its bytes and joins them, which is exact because every id ultimately expands to byte tokens. That separation is deliberate: the next release ships an open-weights GPT-2 runtime, and GPT-2's tokenizer is byte-level BPE with an externally supplied vocab and merges, so a loader that parses GPT-2's tables into id pairs reuses this file's encoder and decoder unchanged. `tests/tokenizer_test.tw` (`oracle test`) checks the round-trip identity, save/load, and the `from_merges` reuse path.

## The corpus

Training uses a public-domain text: the full "tiny shakespeare" file, about 1.1 MB (the works of William Shakespeare are public domain). It is committed in full at `data/corpus.txt` so training works offline and is reproducible, and `scripts/fetch_corpus.sh` regenerates that exact file from the source, so the derivation is not magic.

Learning BPE merges over the whole 1.1 MB file in the interpreter is slow, so `train.tw` reads a deterministic prefix (`TRAIN_CHARS`, 250,000 characters by default) to learn the merges and to train on, which keeps the wall clock inside the budget. The prefix is a fixed function of the committed file, so the run stays reproducible, and the tokenizer itself is deterministic (ties broken by the lowest pair key), so its 768 merges reproduce too. Raise `TRAIN_CHARS` for a longer, slower run. The corpus is deliberately far larger than the model, so what Oracle learns is the structure of the text rather than a memorized copy of it.

## Requirements

- The Twill toolchain, version 1.18.5. Install a released build with `go install github.com/twill-lang/twill/cmd/twill@v1.18.5` and put your `GOBIN` on `PATH`, then confirm `twill --version` prints `1.18.5`.

## Train

From the repository root:

```
make train
```

or directly:

```
twill run train.tw
```

This reads the corpus prefix, learns the byte-level BPE tokenizer and saves it to `models/oracle-tok.bin`, encodes the corpus to token ids, trains with Adam over random windows, prints the loss as it goes, and saves the weights and the config to `models/oracle.bin`. The tokenizer and the checkpoint are two files that load together. Training is a pure function of two fixed seeds, so a second run reproduces the first. The default run (learning 768 merges, then 360 Adam steps at batch 6) took about 34 minutes on this laptop CPU (measured 2026-09-22, 2,027 seconds wall clock) and reached a training-batch cross-entropy around 4.45 over the 1024-token vocabulary. That is inside the roughly 35-minute budget the hyperparameters are tuned for; raise `STEPS` or `TRAIN_CHARS` for a longer, slower, better run.

## Generate

```
make generate PROMPT="To be, or not to be"
```

or directly:

```
twill run generate.tw "To be, or not to be"
```

This loads `models/oracle.bin` and the tokenizer from `models/oracle-tok.bin`, encodes the prompt, and samples a continuation. Decoding combines temperature, top-k, top-p (nucleus), and a repetition penalty, all with sensible defaults and all overridable as flags:

```
bin/oracle generate "To be, or not to be" --temp 0.7 --topk 40 --topp 0.95 --rep 1.15 --steps 200
```

`--steps` now counts BPE tokens, not characters, so a token is roughly three characters and the default 200 tokens is a longer sample than the old 300 characters. Sampling is seeded (`--seed`, default 42), so a given prompt and settings give the same continuation every run. Because position is rotary, a long prompt and a large `--steps` run past the 256-token training context rather than being truncated to it. The flags map to `ORACLE_*` environment variables that `generate.tw` reads, so the defaults live at the top of that file.

## Quantize

Oracle can pack its weights to int8, which shrinks the checkpoint on disk and the model in memory at a quality cost that is negligible at this scale. From the repository root:

```
make quantize
```

or directly:

```
twill run quantize.tw
```

This loads `models/oracle.bin`, packs every dense weight into int8 with a per-row scale (in pure Twill, so the file is small too, not just the model in memory), and writes `models/oracle-int8.bin`. The token table and the layernorm parameters stay f64; there is no position table to carry, because RoPE places positions inside attention. `generate.tw` runs either checkpoint (both load the same `models/oracle-tok.bin` tokenizer); pass the int8 one as a second argument:

```
twill run generate.tw "To be, or not to be" models/oracle-int8.bin
```

It detects the int8 file and rebuilds the packed weights into Twill's int8 matmul kernel before sampling.

## Before and after

The same prompt, "To be, or not to be", the same seed and default sampling, showing the 0.5.0 model (character-level tokenizer, about 1.79 million parameters) against this 0.6.0 model (byte-level BPE, about 3.42 million parameters):

Version 0.5.0 (character-level):

```
To be, or not to befrour his
There am, the dignereful but capsing.

MENENIUS:
Your rendule, I'll what heart hight: fares!
```

Version 0.6.0 (byte-level BPE):

```
To be, or not to bein.

Second Servingman: I is it:
And so in the chearsed I cannot had: my cow for.

Third Citizen:
Stance no ra.

HASTINGS: sir,
Whwell of me; and it is him to your good mister me?

MENENIUS:
Not,
Well, that wound;
```

Both are still small-model output with some invented words, which is the honest ceiling at this scale. The 0.6.0 text is clearly more word-coherent: whole real words dominate ("I cannot had", "it is him to your good", "that wound"), the speaker labels are real character names, and the dialogue structure holds. BPE is the reason: the model composes from word-shaped tokens rather than spelling every word one character at a time. Your exact continuation depends on the decoding flags and seed.

## Benchmarks

Measured on this machine (Apple Silicon, macOS, Twill 1.18.4, CPU) on 2026-09-22, from `bench.tw`. Reproduce with:

```
make bench                     # strict matmul
make bench MATMUL=fast         # fast matmul microkernels
```

Model and checkpoint:

| Quantity | fp64 | int8 |
| --- | --- | --- |
| Parameters | 3,417,600 | 3,417,600 |
| Checkpoint on disk | 27,342,719 B (26.08 MiB) | 5,398,241 B (5.15 MiB) |
| Model footprint (`nbytes`) | 27,340,800 B | 5,394,432 B |

The int8 checkpoint is 5.07x smaller on disk and the model is 5.07x smaller in memory. Only the dense weights are packed; the f64 token table and norms are carried through. The ratio is further from the 8x of a pure int8-for-f64 swap than the 0.5.0 char model's 7.2x, and the reason is the vocabulary: the f64 token-embedding table is now 1024 rows of width 256 rather than 62 rows of width 192, so the un-quantized table is a much larger fixed share of the int8 model. That is a real and explainable cost of a bigger vocabulary, not a regression in the packing.

Generation speed, 56 tokens continued from a short prompt, tokens per second:

| Path | strict matmul | fast matmul |
| --- | --- | --- |
| Uncached (re-run whole context each step) | 71.6 | 60.8 |
| KV-cache | 462.2 | 524.7 |

The KV-cache is 6.45x faster than re-running the whole context (8.63x with fast matmul), and the cached and uncached greedy continuations are identical token for token (0 mismatches) on a run of 144 tokens. `TWILL_MATMUL=fast` is a win for the cached path (about 14 percent) and a small loss for the uncached path: the hand-written microkernels amortize over large matmuls, and even at d_model 256 Oracle's are on the small side for the setup to fully pay off on the uncached path. The flag is wired up and reported so the effect is visible rather than assumed.

Quality proxy, mean cross-entropy and perplexity over 30 held-out 256-token windows drawn from a slice of the corpus past the training prefix, so the windows are unseen (lower is better):

| Model | cross-entropy | perplexity |
| --- | --- | --- |
| fp64 | 5.13203 | 169.360 |
| int8 | 5.13155 | 169.280 |

**This perplexity is over BPE tokens and is NOT comparable to the 0.5.0 char-level perplexity of 5.09.** A token carries more information than a single character, so a per-token perplexity of 169 is not worse than a per-character perplexity of 5; they are different units over different alphabets (1024 tokens versus 62 characters), and the only honest comparison between the two models is the sample text, not the number. Within this release the comparison that does hold is fp64 versus int8: int8 is 0.08 of perplexity lower here, which is within the noise of the measurement, so int8 remains the efficient default with no meaningful quality regression.

Peak process memory, from `/usr/bin/time -l` on a 200-token generation:

| Run | peak resident set |
| --- | --- |
| fp64 generate | ~103 MB |
| int8 generate | ~72 MB |

The int8 peak stays below the fp64 peak, so the row-by-row int8 reconstruction from 0.5.0 (which removed the old ~163 MB load spike) still holds at the larger vocabulary and model size.

Before 0.5.0 the int8 load reconstructed each packed weight by appending every
value into one Twill list, and `append` copies its whole list on each call, so a
single weight cost O((rows*cols)^2) copies and left that much transient garbage.
The int8 peak was ~163 MB, well above both the ~1.9 MB steady footprint (see
`nbytes` above) and the fp64 peak. 0.5.0 streams the reconstruction one row at a
time: each row becomes a `[1, cols]` tensor, the rows are stacked once with
`concat`, and only that one f64 weight is ever expanded at a time. The peak drops
to ~64 MB, below the fp64 peak, and the load is also much faster because the
quadratic copying is gone. A native twill builtin that read packed codes straight
into a quantized tensor (the `quantize` builtin only ingests a full 2-D tensor in
1.18.4) would remove even the one-weight f64 reconstruction; that stays a
twill-side follow-up, not a property of this format.

## Repository layout

```
oracle/
  bin/
    oracle          one CLI over train, generate, quantize, bench, check, test
  src/
    model.tw        Oracle's own decoder-only transformer (with the KV-cache path)
    tokenizer.tw    byte-level BPE tokenizer (trainer, encoder, decoder, save/load)
    quant.tw        int8 weight quantization and reconstruction
  tests/
    tokenizer_test.tw  round-trip, save/load, and reuse tests for the tokenizer
  data/
    corpus.txt      the committed public-domain training text
  models/
    oracle.bin      the trained f64 checkpoint (written by train.tw)
    oracle-int8.bin the int8 checkpoint (written by quantize.tw)
    oracle-tok.bin  the learned BPE tokenizer (written by train.tw)
  scripts/
    fetch_corpus.sh regenerate the corpus deterministically
  train.tw          learn the tokenizer, train, and save a checkpoint
  generate.tw       load a checkpoint (f64 or int8) plus the tokenizer and sample
  quantize.tw       pack a checkpoint to int8
  bench.tw          measure size, speed, memory, and quality
  Makefile          thin convenience over the twill commands
  CHANGELOG.md      the per-version history
  .github/workflows CI (twill check plus tokenizer tests) and tag-triggered releases
```

## Honest limits

Oracle is a teaching-scale model. At about 3.4 million parameters trained for about half an hour on 250 KB of text, it learns spelling, spacing, real words, speaker labels, punctuation, and the rough cadence of the corpus, and byte-level BPE lets it compose from word-shaped tokens so its output is clearly more word-coherent than the character-level 0.5.0 model. It still does not learn grammar, meaning, or facts, and it will still produce invented words and broken sentences. It has no instruction following, no chat behavior, and no knowledge of anything outside its training text. It is a small, honest, from-scratch demonstration of how a transformer language model and its tokenizer are built and trained, all the way down, in one language. Bigger and longer-trained would read better, but the point is to stay home-runnable and readable, not to chase fluency.

Phase 1 built a working, trained, generating model. Phase 2 made inference efficient and measured it: int8 quantization, a KV-cache, the fast-matmul option, and the benchmarks above. Phase 3 was the first public release: the rename to Oracle, the unified `oracle` CLI, CI that shape-checks every file, tag-triggered releases, and this documentation. Phase 4 (0.4.0) modernized the architecture: rotary positions in place of the learned table, which removes the context cap, plus top-p and a repetition penalty in the sampler, a larger model, and a larger corpus. Phase 5 (0.5.0) was an efficiency release: it removed the int8 load memory spike by streaming the weight reconstruction one row at a time. Phase 6 (0.6.0, this work) is the quality release: a byte-level BPE tokenizer written in Twill and a base-size 3.4M-parameter model, which lift the quality ceiling above character-level and, by design, leave the tokenizer's encode/decode reusable for an external merge table. That reuse is the groundwork for the next step, an open-weights GPT-2 runtime that loads GPT-2's own byte-level BPE tables through the same encoder and decoder.

## License

MIT, Copyright (c) 2026 Martin Muskov. See `LICENSE`.
