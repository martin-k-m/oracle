# Changelog

All notable changes to Oracle are recorded here. Versions follow the phases the
project was built in: a working model, then efficient inference, then a public
release with a unified CLI and automation, a modernized architecture, a leaner
int8 load, and now byte-level BPE with a base-size model.

## [Unreleased]

## [0.40.0] - 2026-09-24

Act on your computer, in text instead of screenshots.

- Added oracle agent: a text-based agent that does things on the machine without
  the token cost of screen images. The model proposes one shell command, the user
  confirms it (default no, obviously destructive commands flagged), it runs in the
  directory oracle was invoked from, and the model sees the command's text output
  and chooses the next step, up to a step limit. The whole loop lives in one Twill
  process (agent.tw) with the model loaded once and the run builtin executing the
  commands, so it is efficient in both tokens and time: a command's output is a
  few tokens where a screenshot is a thousand. src/qwen_tok gained chat_ids_with
  for the agent's own system prompt. agent.tw is covered by oracle check and CI.

## [0.39.0] - 2026-09-24

Reusable retrieval, and the console embeds on its live encoder.

- Refactored retrieve.py's orchestration into retrieve_passages (rank a repo's
  passages for a query) and format_context (turn them into numbered context and a
  sources list), with an injectable embedder (args.embed_fn). The CLI prints the
  result and the console grounds its Ask answer in it through the same code.
- The console's Ask now retrieves in process, embedding on the server's already
  loaded encoder instead of spawning a subprocess that reloaded the 181 MB model
  each question. Measured, retrieval dropped to about 0.02s; the remaining latency
  of a grounded answer is the model prefilling the context, so the console uses a
  leaner context budget than the CLI to stay responsive. 26 tests now.

## [0.38.0] - 2026-09-24

Ask a codebase from the web console.

- The console gained an Ask repo tool: it answers a question about the project
  oracle serve was started in (or --repo D), retrieving the relevant passages
  with the server's live encoder and citing them, with a general question
  answered from general knowledge. The server retrieves through the same
  retrieve.py the CLI uses, embedding with a local encoder in a subprocess so the
  live hosts are never touched re-entrantly.
- Fixed a robustness bug found while building this: if a Server-Sent-Events
  client disconnects mid-reply, the handler now drains the model host to the end
  of its response instead of abandoning it, so the host's wire protocol stays in
  sync for the next request. Also route the console's Ask to the directory serve
  was launched from, matching the CLI's invoke-directory behaviour.

## [0.37.0] - 2026-09-24

Tests that guard the CLI dispatch logic.

- Added tests/cli_test.py: it pins the directory logic and routing that a green
  Python suite had missed, the source of the 0.36.0 bug. It checks that oracle
  ask defaults to the directory you invoked it from, that a relative --repo
  resolves against that directory while an absolute one is kept, and that a bare
  quoted question routes to ask with its flags intact. An ORACLE_DEBUG_REPO hook
  makes ask print the resolved repository and stop before loading any model, so
  the tests need no weights and run in CI. 25 tests now.

## [0.36.0] - 2026-09-23

Search the project you are in, and a bare-question shorthand.

- Fixed a real bug: ask, index and chat --repo defaulted the repository to ".",
  but oracle cd's into its own install directory before running, so from any
  other directory they searched Oracle's own source instead of the user's
  project. They now resolve the repository against the directory you invoked
  oracle from, so `oracle ask` from inside a project searches that project.
- A bare quoted question is now shorthand for ask: `oracle "why does X..."`
  routes to `oracle ask "why does X..."`, keeping any flags. No subcommand
  contains a space, so a spaced first argument is unambiguously a question.

Interactive chat that understands a codebase.

- oracle chat can now answer from a repository. Start it with `oracle chat
  --repo .` (or `/repo <dir>` in the session) and each question retrieves the
  relevant passages and answers from them with a sources line, the same as oracle
  ask, while general questions in the same session are answered from general
  knowledge. chat.tw shells out to the retrieval driver per question through
  Twill's `run` builtin, so the model stays loaded once and follow-ups are
  immediate. The banner no longer hardcodes a model size.
- Raised the relevance gate from 0.2 to 0.3 cosine (ask and chat). Measured on
  the encoder, general questions top out around 0.15 to 0.22 against code while
  real code questions score 0.38 to 0.41, so 0.2 occasionally grounded a physics
  question on incidental shared words (a "why is the sky blue" answer citing a
  CSS color); 0.3 separates them cleanly. Tunable with ORACLE_MIN_SCORE.

## [0.34.0] - 2026-09-23

1.5B is the recommended default model.

- oracle fetch-qwen now downloads the 1.5B Qwen2.5-Coder by default instead of
  0.5B: noticeably better answers across code and general engineering while still
  running on a laptop CPU (about 1.4 GB int8, a few tokens per second). 0.5B
  stays one command away (oracle fetch-qwen 0.5B) as the smallest, fastest
  option, and sizes coexist, with oracle using the most capable one installed
  unless --model says otherwise. Updated the quickstart, the Qwen section, the
  honest-scope note, and the CLI help to lead with 1.5B.

## [0.33.0] - 2026-09-23

A general engineering assistant, not just a coder.

- Broadened the assistant's system prompt from a code-only framing ("use Python
  unless...") to a general engineering one: it now helps with programming,
  physics, mathematics, and mechanical, electrical and CAD questions, and gives
  concrete numbers, formulas or steps as well as code. The same lightweight
  Qwen-Coder-0.5B already carried this breadth; the prompt was holding it back.
- oracle ask is now the one entrypoint for day-to-day questions. A relevance gate
  (cosine --min-score, default 0.2) decides per question whether the repository
  is actually relevant: if it is, the answer is grounded in the code and cited as
  before; if it is not (a physics or CAD question in a code repo), no context is
  used and the model answers from its own knowledge. Tested.

## [0.32.0] - 2026-09-23

A README that reflects what the project has become.

- Rewrote the top of the README: a Highlights map of the real capabilities (three
  Twill model runtimes, the coder CLI, repo-aware semantic answers, the live
  server) with links, and a Quickstart that shows the code assistant and semantic
  ask, not just the from-scratch model. Completed the CLI reference (it was
  missing explain, review, fix, tests, sh, chat, ask, serve, index and
  fetch-embed) and corrected the repository layout, which still described only the
  original from-scratch files.

## [0.31.0] - 2026-09-23

Line-anchored citations in ask answers.

- oracle ask now numbers the retrieved passages [1], [2] in the context, asks the
  model to cite them inline, and prints a Sources footer after the answer listing
  each passage's exact path:line-range. The footer is built from the retrieved
  passages, not written by the model, so the line ranges are always correct even
  when a small model's prose is not, and you can jump straight to the code an
  answer came from. retrieve.py gained a --number flag (tested); the footer is
  assembled in the CLI and printed after both local and server answers.

## [0.30.1] - 2026-09-23

- oracle index no longer refuses to run when the encoder is not installed
  locally but ORACLE_SERVER points at a server that has it; it embeds through the
  server. The index tests are now independent of a local model (they stand in a
  fake encoder), so they pass in CI, which has no model.

## [0.30.0] - 2026-09-23

A test suite for the retrieval pipeline, and it runs in CI.

- Added Python tests for the parts that had none: the WordPiece tokenizer (its
  lowercasing, accent stripping, punctuation splitting, greedy longest-match and
  truncation, against a small fixed vocab), the BM25 keyword retrieval (the right
  file rises, source outranks keyword-stuffed docs, ignored directories and
  binaries are skipped, header line ranges), and the vector index's caching (a
  build embeds every chunk, a rebuild re-embeds only changed files and reuses the
  rest, deletions drop out, search ranks the matching chunk first). The index
  tests use a deterministic fake encoder, so the whole suite needs no model and
  no network. 18 tests, run with `oracle test-py` or in CI.
- CI now runs those tests alongside the shape-check, so a regression in the
  retrieval or tokenizer logic fails the build rather than surfacing as quietly
  worse search. Fixed two unclosed-file handles the tests flushed out.

## [0.29.0] - 2026-09-23

Hold the encoder live too, and build the index in one pass.

- oracle serve gained an /embed endpoint backed by a live encoder host, so with
  ORACLE_SERVER set, ask and oracle index embed against the loaded model instead
  of starting one each time: a query now embeds in a few milliseconds rather than
  paying the model load. The encoder host starts on the first embedding request,
  so a coder-only server never loads it, and retrieve.py falls back to a local
  encoder if the server has none.
- Building the index now embeds every changed chunk in a single encoder pass
  instead of starting the encoder once per file, which on a first build was the
  bulk of the time. Incremental rebuilds are unchanged: only files whose size or
  mtime moved are re-embedded.

## [0.28.0] - 2026-09-23

A persistent embedding index, so semantic ask scales and embeds only the question.

- Added oracle index: it embeds every chunk of a repository once with the Twill
  encoder and caches the vectors, so a semantic ask then embeds only the question
  and scores it against the whole repository, instead of re-embedding keyword
  candidates every time. This also improves ranking, because it ranks the whole
  repo rather than reordering a BM25 shortlist.
- The index is incremental: re-running oracle index re-embeds only the files
  whose size or mtime changed (a no-change rebuild reuses everything and loads no
  model). It lives in a cache under ~/.cache/oracle/index keyed by the repository
  path, so it never writes into the repository being searched. ask uses the index
  automatically when one exists and falls back to on-the-fly reranking otherwise.

## [0.27.0] - 2026-09-23

Semantic retrieval, with a sentence encoder written in Twill.

- oracle ask can now rank passages by meaning, not just keywords. src/embed.tw is
  a full runtime for all-MiniLM-L6-v2, a 6-layer BERT sentence encoder, written
  in Twill the same way the GPT-2 and Qwen runtimes are: learned position and
  token-type embeddings, bidirectional attention, post-LayerNorm, GELU, and a
  mean-pooled, L2-normalised 384-dim output. Its embeddings match a reference
  forward pass to six decimal places, and similar sentences score high while
  unrelated ones score near zero.
- Retrieval is now hybrid: BM25 finds candidates (recall), then the encoder
  reranks them by cosine similarity (precision), so a passage about the same
  thing under different words rises even with no shared keyword. `oracle
  fetch-embed` downloads and converts the open MiniLM weights (Apache-2.0), and
  once present ask uses the encoder automatically; --no-semantic forces
  keyword-only. WordPiece tokenisation lives in the Python retrieval driver
  (scripts/wordpiece.py); the encoder consumes the ids. src/embed.tw is covered
  by oracle check and CI.

## [0.26.0] - 2026-09-23

Ask about a whole repository, not just a file you name.

- Added oracle ask: it answers a question about a codebase by finding the
  relevant passages itself and handing those to the coder as context, so you no
  longer pass --file. scripts/retrieve.py walks the repo, splits files into line
  windows, and ranks them with BM25 keyword relevance, splitting camelCase and
  snake_case so "get user" finds getUser; documentation is down-weighted so
  source ranks first. No model, embedding, or on-disk index is involved, and
  nothing is written out. Flags: --repo (default the current directory), --k
  (passages), --budget (their size). It works locally and through ORACLE_SERVER,
  and --model picks the size that reasons over the context. Honest scope: lexical
  retrieval into a small model is for "where is X" and "how does Y work"
  navigation, not deep cross-file reasoning.

## [0.25.0] - 2026-09-23

Switch models live in the console.

- The web console gained a model selector, shown when more than one size is
  installed, that loads a different model without restarting the server: the host
  stops and reopens on the chosen weights, and the conversation starts fresh. The
  server exposes GET /models (installed sizes and the current one) and POST
  /model to switch. A single installed model keeps the selector hidden.
- Also recorded the conclusion of the decode-speed investigation: the int8
  matmul is memory-latency bound on this class of machine, not floating-point or
  bandwidth bound, so neither a SIMD kernel, int4 weights (measured slower), nor
  W8A8 activation quantization would speed it up. The per-token cost is at its
  floor; the useful levers were the sampler (0.21.0) and skipping the model
  reload (0.23.0), both already shipped.

## [0.24.0] - 2026-09-23

Per-request creativity and length, in the console and the CLI.

- The model host (serve.tw) now takes a parameter line with each message, so a
  request sets its own temperature and reply length instead of the whole server
  sharing one setting fixed at startup. The web console gained Style (Precise,
  Balanced, Creative) and Length (Short, Medium, Long) controls, and the CLI's
  --temp and --steps now carry through to a server in ORACLE_SERVER mode the same
  as they do locally. Values are clamped to sane bounds server-side, and a blank
  field keeps the host default.

## [0.23.0] - 2026-09-23

Reuse a live model from the command line, and a measured look at the decode floor.

- The CLI can now hand its work to a running `oracle serve` instead of loading
  the model again. Set ORACLE_SERVER (or --server URL) and code, explain, review,
  fix, tests, sh and commit stream their reply from the live model, skipping the
  roughly two seconds each call otherwise spends starting twill and loading the
  weights (more for a bigger model). It falls back to running locally if the
  server is unreachable, so it is safe to leave set. A small stdlib client,
  scripts/client.py, does the streaming.
- Investigated the per-token decode cost (the int8 matmuls) as a possible SIMD
  target and, after profiling, did not change it: the kernel is not
  floating-point bound (float32 and int32 inner loops are no faster in pure Go),
  not memory-bandwidth bound (it moves about 10 GB/s of a ~100 GB/s machine), and
  already saturates near six cores, so a wider SIMD kernel would not help. The
  honest lever at this scale is the per-call model load, which the server reuse
  above removes; a materially faster kernel would need activation quantization
  (W8A8) or int4 weights, both of which trade accuracy and belong to their own
  validated change.

## [0.22.0] - 2026-09-23

Every task streams now, not just chat.

- oracle code, explain, review, fix, tests, sh and commit stream their reply
  token by token as it is generated, the same way oracle chat already did, so a
  one-shot task shows its answer as it is written instead of pausing for the
  whole generation and printing it at the end. qwen_gen.tw switched from
  generate_cached to generate_stream with a delta-decode that never splits a
  multi-byte character. Same tokens, same output, just visible sooner.

## [0.21.0] - 2026-09-23

Faster generation: the sampler no longer sorts the whole vocabulary.

- Profiling showed the token sampler was costing as much per step as the entire
  24-layer model: top-k and nucleus (top-p) each ran a full sort over the
  150,000-token vocabulary every step. Fixed in twill 1.18.5, which selects the
  top-k and the nucleus without that sort, and Oracle now pins it. Measured on a
  0.5B int8 model, steady-state generation went from about 6.5 to about 16
  tokens per second, with bit-identical token choices. The stale "about twenty
  tokens per second" figure in the README is corrected to the measured rate.

## [0.20.0] - 2026-09-23

The model stays loaded: a live server, streaming, and multi-turn chat.

- Added serve.tw, a persistent Twill model host driven over a stdin/stdout line
  protocol: it loads the weights once and answers many requests, streaming each
  reply token by token with a length-framed wire format that keeps code and
  newlines intact. It keeps the conversation in memory, with a RESET command to
  clear it.
- Rebuilt oracle serve on top of it. scripts/serve.py now starts serve.tw once
  and holds it live, so the browser no longer reloads the model on every click;
  replies stream over Server-Sent Events, and a chat keeps its history between
  messages. The console is now a streaming chat with a row of one-shot tools,
  not a form. Still stdlib only, still bound to localhost, still a fixed set of
  tasks. This is the big cost win: after the first load, every turn is just
  generation.
- serve.tw is now covered by oracle check and CI.

## [0.19.0] - 2026-09-23

A lightweight local web console.

- Added oracle serve: a small web UI at http://127.0.0.1:8080 over the same
  tasks (ask, explain, review, fix, tests, shell, commit). Pick a task, type a
  request, paste code, read the answer in the page. It is a thin bridge in
  scripts/serve.py that shells out to bin/oracle, uses only the Python standard
  library, binds to localhost, and only exposes the fixed set of tasks. Flags:
  --port and --model. Each answer reloads the model, so it is meant for local
  use, not for serving traffic.

## [0.18.0] - 2026-09-23

More of an engineer's daily toolkit: fix, tests, sh, and per-task guidance.

- Added oracle fix (diagnose a bug and return corrected code, taking an error
  message or description as a hint), oracle tests (write unit tests for a file),
  and oracle sh (turn a plain request into one shell command). All go through the
  same shared runner, so they read a --file or piped stdin like the rest.
- Every fixed-instruction task now accepts trailing words as extra guidance, so
  `oracle review --file server.py "focus on error handling"` or
  `oracle fix --file p.py "IndexError on empty input"` steer the task without a
  new command.

## [0.17.0] - 2026-09-23

Fix file context: the file's real contents now reach the model.

- read_file returns a Res[Str, Str] (Ok(contents) or Err(message)), and the
  file-context path was stringifying the whole result, so every --file and every
  /file in chat fed the model `Ok(<contents>)` with a literal Ok( prefix and a
  trailing ) wrapped around the code. Both qwen_gen.tw and chat.tw now match on
  the result and use the unwrapped contents, and report a read error instead of
  silently embedding one. This makes explain, review, code --file, and chat's
  /file actually reason about the file you gave them.

## [0.16.0] - 2026-09-23

Task commands for daily engineering work.

- Added oracle explain, oracle review, and oracle commit, thin wrappers over the
  coder with the right instruction. explain and review take a file (--file) or
  piped stdin; commit reads a diff from stdin, so `git diff --staged |
  oracle commit` drafts a message. All accept --model to use a bigger model.
  Refactored the code path into a shared runner so every task reads files and
  stdin the same way.

## [0.15.0] - 2026-09-23

Ask about several files at once.

- oracle code takes --file more than once, so a question can span multiple
  files (each is given to the model as its own fenced context block). Piped
  stdin and a single --file still work.

## [0.14.0] - 2026-09-23

Pipe code straight in.

- oracle code reads piped stdin as context when no --file is given, so
  `cat file.py | oracle code "add tests"` and `oracle code "explain" < file.py`
  work the way a shell tool should. An explicit --file still takes precedence.

## [0.13.0] - 2026-09-22

Select the model per command, and a system-prompt fix.

- oracle code and oracle chat take --model (a size like 1.5B, or a directory).
  Sizes fetched with oracle fetch-qwen live in their own directories
  (models/qwen for 0.5B, models/qwen-<size> for the rest) and coexist, so a
  bigger model is one flag away without re-fetching. The runtime reads the model
  directory from ORACLE_QWEN_DIR.
- Fixed the system prompt. It said "You are Oracle", which the small model
  confused with the Oracle database and answered some questions in SQL. It now
  describes a software engineering assistant and defaults to Python, so answers
  are on-topic.

## [0.12.0] - 2026-09-22

Make the chat a workspace: load code and manage the session.

- oracle chat now takes slash commands. /file <path> loads a file into the
  conversation as context, so you can load one or more files and then ask about
  them; /reset clears the conversation and loaded files; /help lists the
  commands; /exit quits. A plain line is still a question, and the reply still
  streams. This turns the chat from a single question box into a place to work
  through real code with the model.

## [0.11.0] - 2026-09-22

Ask about real code, not only a prompt.

- oracle code takes an optional --file, whose contents are given to the model as
  context wrapped in a code fence, so a question can be about actual code. For
  example: oracle code "what bug could this have?" --file mycode.py. Small model,
  so the reasoning is limited; the 1.5B model answers such questions better.

## [0.10.0] - 2026-09-22

About 3 to 4x faster generation, and the larger model validated.

- Generation went from about 5 to about 20 tokens per second on a CPU. Two
  changes: twill 1.18.4 int8 kernel parallelises a one-token step over every
  core (it used to run single-threaded during decoding), and the runtime now
  projects only the last hidden state for the first token instead of the whole
  prompt. Applied to the Qwen and GPT-2 runtimes.
- Verified Qwen2.5-Coder-1.5B end to end (oracle fetch-qwen 1.5B): a different
  shape (hidden 1536, 28 layers) that the config-driven runtime runs unchanged,
  about 1.4 GB int8 at roughly 8 tokens per second, with noticeably better
  answers than 0.5B.
- fetch-qwen and fetch-gpt2 check for numpy before the multi-gigabyte download
  rather than failing after it.
- to_qtensor now loads packed int8 straight through quantize_packed, so the
  from-scratch and GPT-2 int8 paths load in one pass too.
- Pinned twill 1.18.4.

## [0.9.0] - 2026-09-22

Run larger, more capable Qwen coders.

- `oracle fetch-qwen [SIZE]` now takes a model size (0.5B default, or 1.5B, 3B,
  7B). The runtime is config-driven, so a larger Qwen2.5-Coder runs unchanged;
  the fetcher and converter handle both single-file and sharded safetensors, so
  the bigger checkpoints download and convert too. 1.5B is the next comfortable
  laptop size for noticeably better code.

## [0.8.0] - 2026-09-22

An interactive, streaming chat CLI for the Qwen coder.

- Added `oracle chat` (`chat.tw`): a small terminal UI over the Qwen runtime with
  a coloured banner, a prompt, the reply streamed token by token as it is
  generated, and a running multi-turn conversation. The whole loop, model and
  all, runs in one Twill process, using the `read_line` builtin from twill
  1.18.3, so the weights load once. `generate_stream` in `src/qwen.tw` emits each
  token for live output, and `src/qwen_tok.tw` gained multi-turn ChatML.
- Bumped the twill pin to 1.18.3.

## [0.7.0] - 2026-09-22

The Qwen coder: a real, instruction-following code assistant you self-host.

- Added a Qwen2.5-Coder-0.5B-Instruct runtime written in Twill (`src/qwen.tw`,
  `src/qwen_tok.tw`): RMSNorm, rotary embeddings at theta 1e6, grouped-query
  attention, SwiGLU, and a tied head, running int8 (~475 MB) through
  quantize_packed. The tokenizer reproduces Qwen ids exactly and wraps prompts
  in the ChatML template, so plain-English requests get worked answers with code.
  `oracle fetch-qwen` downloads and converts the open weights; `oracle code`
  runs it. About five tokens per second on a CPU.
- Depends on twill 1.18.2 for quantize_packed, which loads the int8 weights in
  one pass. Without it a 0.5B model could not load fast or fit in RAM.

## [0.6.1] - 2026-09-22

An int8 path for the GPT-2 runtime, measured honestly rather than assumed.

- Added int8 quantization for the GPT-2 weights (`src/gpt2_quant.tw`,
  `oracle quantize-gpt2`): every block weight and the token table pack to int8
  with a per-row scale, reusing the row-streaming quantizer. The weights drop
  from about 1 GB to 126 MB on disk, a 7.5x reduction, and generation stays
  coherent.
- Kept fp64 as the default for the GPT-2 runtime and made int8 opt-in
  (`oracle gpt2 --int8`). In twill 1.18.0 the int8 model runs about six times
  slower than fp64, because the int8 kernels are reconstructed from the packed
  codes at load and that reconstruction is linear in the 124 million parameters.
  This is a measurement, not a guess. A native twill builtin that reads packed
  codes straight into the int8 kernel would remove the cost; that is a change to
  twill, not to Oracle.
- Routed the embedding through a helper so the fp and int8 paths share one
  forward pass; the int8 model keeps the token table packed and reads only the
  rows a sequence needs, so it never holds the full table in f64.

## [0.6.0] - 2026-09-22

The big release. Two things: a byte-level BPE tokenizer with a base-size
from-scratch model, and a faithful GPT-2 124M inference runtime written in Twill
that loads the open GPT-2 weights and generates genuinely coherent English.

GPT-2 runtime:

- Added a GPT-2 124M inference engine in Twill (`src/gpt2.tw`): learned token and
  position embeddings, twelve pre-norm causal multi-head attention and gelu
  feed-forward blocks, a bias on every projection, a tied output head, a
  key/value cache, and top-k/top-p/repetition-penalty sampling. The forward pass
  is faithful to GPT-2 (the tanh gelu, the layernorm epsilon, the Conv1D weight
  transpose handled in conversion), so it produces coherent English. Example: the
  prompt "Dear team," continues into a fluent short email.
- Added GPT-2's byte-level BPE tokenizer in Twill (`src/gpt2_tok.tw`): it loads
  the converted vocab and merge tables and reproduces GPT-2's token ids exactly,
  checked against the reference encoder, so the runtime is Twill end to end with
  no Python at run time.
- Added `scripts/fetch_gpt2.sh` and `scripts/convert_gpt2.py` to download OpenAI's
  open GPT-2 release and convert it (parsing safetensors directly) into a Twill
  tensor tree. The weights are not trained here and are not committed to git; the
  fetch runs once on the user's machine. Wired `oracle fetch-gpt2` and
  `oracle gpt2 "<prompt>"` into the CLI.
- Honest scope: GPT-2 small writes coherent short prose and email-shaped text and
  is weak at code, with no instruction following and no reliable facts. At fp64 it
  needs about 2.3 GB of RAM and runs about 1.8 tokens per second on a CPU; an int8
  path is the next release.

Byte-level BPE and base model:

- Replaced the character-level tokenizer with a byte-level BPE tokenizer written
  in Twill (`src/tokenizer.tw`). It starts from the 256 byte tokens, so there is
  no unknown token and `decode(encode(s)) == s` for any input, and learns merges
  by pair frequency up to a target vocabulary. Encoding pre-tokenizes into
  whitespace-attached chunks (a leading space binds to the word after it), then
  applies the learned merges greedily; decoding expands each id back to its
  bytes. The default vocabulary is 1024 (256 byte tokens plus 768 learned
  merges), and on tiny-shakespeare BPE packs about 2.82 characters into each
  token, so a 256-token context now covers roughly 700 characters.
- Made the tokenizer reusable and source-agnostic. The encode/decode machinery is
  a pure function of a merges table and knows nothing about where the merges came
  from: `train_from_corpus` learns them on Oracle's corpus, and `from_merges`
  builds the identical tokenizer record from any ordered id-pair list. That is the
  seam the next release needs, where a loader parses GPT-2's `vocab.json` and
  `merges.txt` into id pairs and reuses this file's encode and decode unchanged.
- Added tokenizer round-trip tests (`tests/tokenizer_test.tw`, `oracle test`,
  `make test`): decode(encode) identity across text, empty, whitespace, unseen
  bytes, tabs and newlines; save/load identity; and that a tokenizer rebuilt from
  a bare merges list behaves the same. CI shape-checks and runs them.
- Saved the learned tokenizer (vocab plus merges) to its own file,
  `models/oracle-tok.bin`, alongside the checkpoint. `train.tw` writes it,
  `generate.tw` and `bench.tw` load it, and the checkpoint no longer carries a
  vocabulary of its own. Only the merges and vocab size are stored; the rest of
  the tokenizer is rebuilt on load.
- Retrained a base-size model on BPE tokens: vocabulary 1024, d_model 256, 8
  attention heads, 4 decoder blocks, context 256 tokens, 3,417,600 parameters
  (up from 1,788,672). Trained 360 Adam steps at batch 6 (1,536 tokens per step,
  the same throughput as the 0.5.0 char model) over the first 250,000 characters
  of the corpus. Wall clock 2,027 seconds (about 34 minutes) on this laptop CPU,
  inside the roughly 35-minute budget; final training-batch cross-entropy about
  4.45.
- Re-quantized to int8 and confirmed it generates. The int8 checkpoint is
  5,398,241 B versus 27,342,719 B for fp64, a 5.07x shrink, and the in-memory
  footprint drops from 27,340,800 B to 5,394,432 B. The ratio is lower than
  0.5.0's 7.2x because the f64 token-embedding table grew with the vocabulary
  (1024 rows now, not 62) and it is not quantized, so it is a larger fixed share
  of the int8 model.
- BPE perplexity is NOT comparable to the old char-level perplexity, and the
  benchmark and README say so. Perplexity is now measured over BPE tokens: on 30
  held-out 256-token windows past the training prefix, fp64 is 169.36 and int8
  169.28 (delta -0.08, within noise, so int8 is again effectively free). A token
  carries more information than a character, so a per-token perplexity of 169 is
  not worse than the old per-character 5.09; the two are different units.
- The KV-cache still matches the uncached path exactly (0 mismatched tokens), and
  generation runs 6.45x faster cached than uncached (8.63x with fast matmul).

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
