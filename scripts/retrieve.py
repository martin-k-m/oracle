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
import math
import os
import re
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


def semantic_rerank(query, cands, args):
    # Reorder the BM25 candidates by meaning: embed the query and each candidate
    # with the Twill encoder (one process, weights loaded once) and score by
    # cosine similarity. BM25 gives recall (it will not miss a strong keyword
    # match); this gives precision (a passage about the same thing under
    # different words rises). Returns None to signal "fall back to BM25" if the
    # model or the toolchain is not available.
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    embed_dir = args.embed_dir or os.path.join(root, "models", "embed")
    vocab = os.path.join(embed_dir, "vocab.txt")
    script = os.path.join(root, "src", "embed.tw")
    if not (os.path.exists(os.path.join(embed_dir, "embed.bin")) and os.path.exists(vocab) and os.path.exists(script)):
        return None
    twill = args.twill or "twill"
    sys.path.insert(0, here)
    try:
        import wordpiece
    except ImportError:
        return None

    texts = [query] + [c[4] for c in cands]
    lines = "\n".join(" ".join(str(i) for i in wordpiece.encode(t, vocab)) for t in texts) + "\n"
    env = dict(os.environ)
    env["ORACLE_EMBED_DIR"] = embed_dir
    try:
        out = subprocess.run(
            [twill, "run", script], input=lines, capture_output=True, text=True, cwd=root, env=env
        )
    except (OSError, FileNotFoundError):
        return None
    if out.returncode != 0:
        return None
    vecs = []
    for line in out.stdout.strip().split("\n"):
        parts = line.split()
        vecs.append([float(x) for x in parts] if parts else [])
    if len(vecs) != len(texts) or not vecs[0]:
        return None
    qv = vecs[0]
    rescored = []
    for i, c in enumerate(cands):
        cv = vecs[i + 1]
        sim = sum(a * b for a, b in zip(qv, cv)) if cv else -1.0  # both L2-normalised
        rescored.append((sim, c[1], c[2], c[3], c[4]))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--semantic", action="store_true", help="rerank with the Twill encoder")
    ap.add_argument("--prefilter", type=int, default=48, help="BM25 candidates to rerank")
    ap.add_argument("--twill", default="", help="twill binary for --semantic")
    ap.add_argument("--embed-dir", default="", help="directory holding embed.bin and vocab.txt")
    args = ap.parse_args()

    q_terms = list(dict.fromkeys(tokenize(args.query)))  # unique, keep order
    if not q_terms:
        return

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
        return
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

    if args.semantic and scored:
        reranked = semantic_rerank(args.query, scored[:args.prefilter], args)
        if reranked is not None:
            scored = reranked
        else:
            print("retrieve: semantic model unavailable, using keyword ranking.", file=sys.stderr)

    used = 0
    shown = 0
    for s, rel, start, end, text in scored:
        header = "===== " + rel + ":" + str(start) + "-" + str(end) + " =====\n"
        block = header + text.rstrip("\n") + "\n\n"
        if used + len(block) > args.budget and shown > 0:
            break
        print(block, end="")
        used += len(block)
        shown += 1
        if shown >= args.k:
            break


if __name__ == "__main__":
    main()
