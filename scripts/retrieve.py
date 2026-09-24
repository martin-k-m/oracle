#!/usr/bin/env python3
# retrieve.py: pull the passages of a repository most relevant to a question.
#
# A 0.5B model cannot read a whole codebase, so `oracle ask` does not hand it
# one. This walks a repository, splits every text file into line windows, ranks
# them against the question with BM25 (a standard keyword-relevance score), and
# prints the top few within a character budget, each under a `path:start-end`
# header. It is lexical, not semantic: it matches the words in the question, with
# camelCase and snake_case split so "get user" finds getUser and get_user. That
# is enough to answer "where is X" and "how does Y work" without any model,
# embedding, or index on disk.
#
# Usage: retrieve.py --repo DIR --query "..." [--k N] [--budget CHARS]

import argparse
import array
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "out", "target", "vendor",
    ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache", ".next",
    ".cache", "coverage", ".idea", ".gradle", "bin", "obj", "models",
}
# Extensions that are not source text worth searching.
SKIP_EXT = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".bin", ".so", ".dylib", ".dll", ".a", ".o", ".class", ".jar",
    ".mp4", ".mov", ".mp3", ".wav", ".woff", ".woff2", ".ttf", ".eot",
    ".lock", ".min.js", ".map", ".svg",
}
MAX_FILE_BYTES = 200 * 1024
CHUNK_LINES = 40
MAX_CHUNKS = 20000

# Prose files repeat the words of a question far more than source does, so a
# "how does X work" query tends to surface the changelog over the code that
# answers it. Down-weight documentation so real source ranks first, without
# dropping docs that are strongly on point.
DOC_EXT = {".md", ".markdown", ".rst", ".txt", ".adoc"}

_word = re.compile(r"[A-Za-z0-9_]+")
_camel = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


def tokenize(text):
    toks = []
    for w in _word.findall(text):
        toks.append(w.lower())
        # Split identifiers so a query word matches inside a compound name.
        for part in _camel.findall(w):
            p = part.lower()
            if p != w.lower() and len(p) > 1:
                toks.append(p)
    return [t for t in toks if len(t) > 1]


def is_binary(path):
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(2048)
    except OSError:
        return True


def walk(repo):
    for root, dirs, files in os.walk(repo):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in SKIP_EXT or name.endswith(".min.js"):
                continue
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            if is_binary(path):
                continue
            yield path


def chunk_file(path, repo):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    rel = os.path.relpath(path, repo)
    for start in range(0, max(len(lines), 1), CHUNK_LINES):
        block = lines[start:start + CHUNK_LINES]
        text = "".join(block)
        if not text.strip():
            continue
        yield rel, start + 1, start + len(block), text


DIMS = 384  # all-MiniLM-L6-v2


def _here():
    return os.path.dirname(os.path.abspath(__file__))


def embed_ready(args):
    root = os.path.dirname(_here())
    embed_dir = args.embed_dir or os.path.join(root, "models", "embed")
    return (
        os.path.exists(os.path.join(embed_dir, "embed.bin"))
        and os.path.exists(os.path.join(embed_dir, "vocab.txt"))
        and os.path.exists(os.path.join(root, "src", "embed.tw"))
    )


def _embed_via_server(server, texts):
    # Ask a running `oracle serve` to embed the texts with its live encoder, so no
    # model is loaded here. Returns None on any failure so the caller falls back.
    import json as _json
    import urllib.error
    import urllib.request
    body = _json.dumps({"texts": texts}).encode("utf-8")
    req = urllib.request.Request(
        server.rstrip("/") + "/embed", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req, timeout=600)
        data = _json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    vecs = data.get("vectors")
    if not isinstance(vecs, list) or len(vecs) != len(texts):
        return None
    return vecs


def embed_texts(texts, args):
    # Embed each text and return a list of 384-float vectors, or None if no
    # encoder is available. A running server (its live encoder) is used first when
    # given; otherwise the Twill encoder runs in one process here, loading the
    # weights once. Empty texts embed to an empty vector, kept for alignment.
    if not texts:
        return None
    # An in-process embedder (the server passes its live encoder here) is used
    # first, then a running server over HTTP, then a local encoder subprocess.
    fn = getattr(args, "embed_fn", None)
    if fn is not None:
        try:
            return fn(texts)
        except Exception:
            return None
    if getattr(args, "server", ""):
        vecs = _embed_via_server(args.server, texts)
        if vecs is not None:
            return vecs
        # fall through to a local encoder if the server has no encoder or is down
    if not embed_ready(args):
        return None
    root = os.path.dirname(_here())
    embed_dir = args.embed_dir or os.path.join(root, "models", "embed")
    vocab = os.path.join(embed_dir, "vocab.txt")
    script = os.path.join(root, "src", "embed.tw")
    sys.path.insert(0, _here())
    try:
        import wordpiece
    except ImportError:
        return None
    lines = "\n".join(" ".join(str(i) for i in wordpiece.encode(t, vocab)) for t in texts) + "\n"
    env = dict(os.environ)
    env["ORACLE_EMBED_DIR"] = embed_dir
    try:
        out = subprocess.run(
            [args.twill or "twill", "run", script],
            input=lines, capture_output=True, text=True, cwd=root, env=env,
        )
    except (OSError, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    # One output line per input text; a blank line is an empty vector.
    vecs = [[float(x) for x in line.split()] for line in out.stdout.strip("\n").split("\n")]
    if len(vecs) != len(texts):
        return None
    return vecs


def semantic_rerank(query, cands, args):
    # Reorder BM25 candidates by meaning, for when there is no index: embed the
    # query and each candidate and score by cosine (vectors are L2-normalised, so
    # cosine is a dot product). None means "fall back to keyword ranking".
    vecs = embed_texts([query] + [c[4] for c in cands], args)
    if vecs is None or not vecs[0]:
        return None
    qv = vecs[0]
    rescored = []
    for i, c in enumerate(cands):
        cv = vecs[i + 1]
        sim = sum(a * b for a, b in zip(qv, cv)) if cv else -1.0
        rescored.append((sim, c[1], c[2], c[3], c[4]))
    rescored.sort(key=lambda x: x[0], reverse=True)
    min_score = getattr(args, "min_score", 0.0)
    if min_score > 0:
        rescored = [r for r in rescored if r[0] >= min_score]
    return rescored


# -- Persistent vector index -------------------------------------------------
#
# Embedding every chunk each query is wasteful and limits ranking to the BM25
# candidates. The index embeds every chunk once, caches the vectors keyed by the
# repository path, and re-embeds only files whose size or mtime changed. A query
# then embeds itself and scores against the whole repository. The index is a
# cache under the user's home, so it never touches the repository being searched.

def index_dir(repo):
    key = hashlib.sha1(os.path.abspath(repo).encode("utf-8")).hexdigest()[:16]
    base = os.environ.get("ORACLE_INDEX_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "oracle", "index"
    )
    return os.path.join(base, key)


def load_index(repo):
    idir = index_dir(repo)
    meta_path = os.path.join(idir, "meta.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("dims") != DIMS:
            return None
        with open(os.path.join(idir, "vectors.f32"), "rb") as f:
            vectors = f.read()
        chunks = []
        with open(os.path.join(idir, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                chunks.append((r["p"], r["s"], r["e"]))
    except (OSError, ValueError, KeyError):
        return None
    return {"files": meta["files"], "vectors": vectors, "chunks": chunks, "dims": DIMS}


def build_index(repo, args):
    if not embed_ready(args) and not getattr(args, "server", ""):
        print("index: the encoder is not installed. Run: oracle fetch-embed", file=sys.stderr)
        return False
    old = load_index(repo)
    old_files = old["files"] if old else {}
    stride = DIMS * 4

    # Pass 1: decide per file whether its cached vectors can be reused, and gather
    # every chunk that must be (re-)embedded so they all go through the encoder in
    # one call, loading the model once instead of once per file.
    plan = []  # (rel, sig, reuse, chunks-or-None)
    pending_texts = []
    for path in sorted(walk(repo)):
        rel = os.path.relpath(path, repo)
        try:
            st = os.stat(path)
        except OSError:
            continue
        sig = [int(st.st_mtime_ns), st.st_size]
        prev = old_files.get(rel)
        if old and prev and prev["sig"] == sig:
            plan.append((rel, sig, True, None))
        else:
            chunks = [(c[0], c[1], c[2]) for c in chunk_file(path, repo)]
            texts = [c[3] for c in chunk_file(path, repo)]
            plan.append((rel, sig, False, chunks))
            pending_texts.extend(texts)

    pending_vecs = []
    if pending_texts:
        pending_vecs = embed_texts(pending_texts, args)
        if pending_vecs is None or len(pending_vecs) != len(pending_texts):
            print("index: embedding failed; aborting.", file=sys.stderr)
            return False

    # Pass 2: assemble the index in file order, reusing old vector blocks and
    # consuming the freshly embedded vectors in the order they were gathered.
    new_files = {}
    new_vectors = bytearray()
    new_chunks = []
    reused = 0
    embedded = 0
    pi = 0
    for rel, sig, reuse, chunks in plan:
        off = len(new_chunks)
        if reuse:
            p = old_files[rel]
            o, n = p["off"], p["n"]
            new_vectors += old["vectors"][o * stride:(o + n) * stride]
            new_chunks.extend(old["chunks"][o:o + n])
            new_files[rel] = {"sig": sig, "off": off, "n": n}
            reused += n
        else:
            for c in chunks:
                v = pending_vecs[pi]
                pi += 1
                if len(v) != DIMS:
                    print("index: unexpected vector width; aborting.", file=sys.stderr)
                    return False
                new_vectors += struct.pack("<%df" % DIMS, *v)
                new_chunks.append(c)
            new_files[rel] = {"sig": sig, "off": off, "n": len(chunks)}
            embedded += len(chunks)

    idir = index_dir(repo)
    os.makedirs(idir, exist_ok=True)
    with open(os.path.join(idir, "vectors.f32"), "wb") as f:
        f.write(new_vectors)
    with open(os.path.join(idir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for rel, s, e in new_chunks:
            f.write(json.dumps({"p": rel, "s": s, "e": e}) + "\n")
    with open(os.path.join(idir, "meta.json"), "w") as f:
        json.dump({"dims": DIMS, "files": new_files}, f)
    print("index: %d chunks (%d embedded, %d reused) at %s" % (len(new_chunks), embedded, reused, idir), file=sys.stderr)
    return True


def index_search(repo, query, args, k, budget):
    idx = load_index(repo)
    if not idx or not idx["chunks"]:
        return None
    qv = embed_texts([query], args)
    if qv is None or not qv[0]:
        return None
    q = qv[0]
    allv = array.array("f")
    allv.frombytes(idx["vectors"])
    n = len(idx["chunks"])
    scored = []
    for i in range(n):
        base = i * DIMS
        s = 0.0
        for d in range(DIMS):
            s += q[d] * allv[base + d]
        rel, st, en = idx["chunks"][i]
        scored.append((s, rel, st, en))
    scored.sort(key=lambda x: x[0], reverse=True)
    # A relevance gate: when the best passages are only weakly similar to the
    # question, the question is probably not about this repo, so return nothing
    # and let the caller answer from general knowledge instead.
    min_score = getattr(args, "min_score", 0.0)
    if min_score > 0:
        scored = [s for s in scored if s[0] >= min_score]
    return scored[:max(k * 4, 40)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--query", default="")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--semantic", action="store_true", help="rank with the Twill encoder")
    ap.add_argument("--prefilter", type=int, default=48, help="BM25 candidates to rerank without an index")
    ap.add_argument("--twill", default="", help="twill binary for --semantic")
    ap.add_argument("--embed-dir", default="", help="directory holding embed.bin and vocab.txt")
    ap.add_argument("--server", default="", help="URL of a running oracle serve, to embed on its live encoder")
    ap.add_argument("--build-index", action="store_true", help="build/update the persistent vector index and exit")
    ap.add_argument("--number", action="store_true", help="prefix each passage header with [N] for citations")
    ap.add_argument("--min-score", type=float, default=0.0, help="drop semantic hits below this cosine (0 keeps all)")
    args = ap.parse_args()

    if args.build_index:
        sys.exit(0 if build_index(args.repo, args) else 1)

    if not args.query.strip():
        return
    emit(retrieve_passages(args.query, args), args)


def retrieve_passages(query, args):
    # Rank a repository's passages for a question and return them best-first as
    # (rel, start, end, text). With an index, the whole repository is ranked by
    # meaning from one query embedding; otherwise BM25 keyword relevance ranks the
    # chunks and, when semantic is on, the encoder reranks the top candidates.
    # This is the shared entry point: the CLI prints the result, the server grounds
    # its Ask answer in it, and both can inject an embedder via args.embed_fn.
    if getattr(args, "semantic", False) and load_index(args.repo) is not None:
        hits = index_search(args.repo, query, args, args.k, args.budget)
        if hits is not None:
            out = []
            for _score, rel, st, en in hits:
                text = _read_span(os.path.join(args.repo, rel), st, en)
                if text is not None:
                    out.append((rel, st, en, text))
            return out

    q_terms = list(dict.fromkeys(tokenize(query)))  # unique, keep order
    if not q_terms:
        return []

    chunks = []  # (rel, start, end, text, tf-dict, length)
    df = {}
    for path in walk(args.repo):
        for rel, start, end, text in chunk_file(path, args.repo):
            toks = tokenize(text)
            if not toks:
                continue
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for t in tf:
                df[t] = df.get(t, 0) + 1
            chunks.append((rel, start, end, text, tf, len(toks)))
            if len(chunks) >= MAX_CHUNKS:
                break
        if len(chunks) >= MAX_CHUNKS:
            break

    n = len(chunks)
    if n == 0:
        return []
    avg_len = sum(c[5] for c in chunks) / n
    k1, b = 1.5, 0.75

    scored = []
    for rel, start, end, text, tf, length in chunks:
        s = 0.0
        for t in q_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * length / avg_len))
        # A small bonus when the file's own path matches a query term (a file
        # literally named after what is asked is usually the right one).
        path_toks = set(tokenize(rel))
        s += 0.5 * sum(1 for t in q_terms if t in path_toks)
        if os.path.splitext(rel)[1].lower() in DOC_EXT:
            s *= 0.55
        if s > 0:
            scored.append((s, rel, start, end, text))

    scored.sort(key=lambda x: x[0], reverse=True)

    if getattr(args, "semantic", False) and scored:
        reranked = semantic_rerank(query, scored[:args.prefilter], args)
        if reranked is not None:
            scored = reranked

    return [(rel, st, en, text) for _score, rel, st, en, text in scored]


def format_context(passages, args, number=True):
    # Turn ranked passages into the numbered context blocks the model reads and a
    # matching list of "[N] path:start-end" source references, within the budget.
    parts = []
    sources = []
    used = 0
    shown = 0
    for rel, start, end, text in passages:
        label = ("[%d] " % (shown + 1)) if number else ""
        header = "===== " + label + rel + ":" + str(start) + "-" + str(end) + " =====\n"
        block = header + text.rstrip("\n") + "\n\n"
        if used + len(block) > args.budget and shown > 0:
            break
        parts.append(block)
        sources.append((label + rel + ":" + str(start) + "-" + str(end)).strip())
        used += len(block)
        shown += 1
        if shown >= args.k:
            break
    return "".join(parts), sources


def _read_span(path, start, end):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return None
    return "".join(lines[start - 1:end])


def emit(passages, args):
    # Print the ranked passages as context blocks, numbered when --number is set.
    context, _sources = format_context(passages, args, number=getattr(args, "number", False))
    if context:
        print(context, end="")


if __name__ == "__main__":
    main()
