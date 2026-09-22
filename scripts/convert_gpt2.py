# convert_gpt2.py: turn the open GPT-2 release into Twill-loadable checkpoints.
#
# This is DATA PREP, not the runtime. It parses OpenAI's GPT-2 (124M) weights and
# tokenizer files and rewrites them in Twill's native save format ("RSTR"), which
# the Twill runtime reads with its built-in `load`. No torch, no numpy-free-lunch:
# just the safetensors byte layout and struct/numpy.
#
# It reads, from the directory given as argv[1] (default models/gpt2):
#   model.safetensors   the fp32 weights
#   vocab.json          token string -> id
#   merges.txt          ordered BPE merges
# and writes, into the same directory:
#   gpt2.bin            the weight tree (float64 tensors), for src/gpt2.tw
#   gpt2-tok.bin        the tokenizer tables, for src/gpt2_tok.tw
#
# GPT-2 stores its linear weights as Conv1D, shape [in, out], and applies x @ W.
# The rest of the runtime multiplies with `linear(X, W)` = X @ W^T, so every
# such weight is transposed here to [out, out_in] = [out, in]. The combined QKV
# projection c_attn [768, 2304] is transposed to [2304, 768] and split by row
# into Wq, Wk, Wv.

import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rstr


def load_safetensors(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        data = f.read()
    dtypes = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "BF16": None}
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        dt = meta["dtype"]
        s, e = meta["data_offsets"]
        shape = meta["shape"]
        if dt == "BF16":
            raw = np.frombuffer(data[s:e], dtype="<u2").astype(np.uint32) << 16
            arr = raw.view(np.float32).reshape(shape)
        else:
            arr = np.frombuffer(data[s:e], dtype=dtypes[dt]).reshape(shape)
        out[name] = arr.astype(np.float64)
    return out


def build_weights(st, ddir):
    # infer depth
    depth = 0
    while f"h.{depth}.ln_1.weight" in st:
        depth += 1
    dim = st["wte.weight"].shape[1]
    blocks = []
    for i in range(depth):
        p = f"h.{i}."
        c_attn_w = st[p + "attn.c_attn.weight"].T  # [2304, 768]
        c_attn_b = st[p + "attn.c_attn.bias"]      # [2304]
        Wq, Wk, Wv = c_attn_w[0:dim], c_attn_w[dim:2 * dim], c_attn_w[2 * dim:3 * dim]
        bq, bk, bv = c_attn_b[0:dim], c_attn_b[dim:2 * dim], c_attn_b[2 * dim:3 * dim]
        blocks.append({
            "ln1": {"gamma": st[p + "ln_1.weight"], "beta": st[p + "ln_1.bias"]},
            "attn": {
                "Wq": np.ascontiguousarray(Wq), "bq": bq,
                "Wk": np.ascontiguousarray(Wk), "bk": bk,
                "Wv": np.ascontiguousarray(Wv), "bv": bv,
                "Wo": np.ascontiguousarray(st[p + "attn.c_proj.weight"].T), "bo": st[p + "attn.c_proj.bias"],
            },
            "ln2": {"gamma": st[p + "ln_2.weight"], "beta": st[p + "ln_2.bias"]},
            "ff": {
                "fc1": {"W": np.ascontiguousarray(st[p + "mlp.c_fc.weight"].T), "b": st[p + "mlp.c_fc.bias"]},
                "fc2": {"W": np.ascontiguousarray(st[p + "mlp.c_proj.weight"].T), "b": st[p + "mlp.c_proj.bias"]},
            },
        })
    tree = {
        "cfg": {
            "vocab": int(st["wte.weight"].shape[0]),
            "dim": int(dim),
            "heads": 12,
            "depth": int(depth),
            "max_T": int(st["wpe.weight"].shape[0]),
        },
        "wte": st["wte.weight"],
        "wpe": st["wpe.weight"],
        "blocks": blocks,
        "lnf": {"gamma": st["ln_f.weight"], "beta": st["ln_f.bias"]},
    }
    rstr.save(tree, os.path.join(ddir, "gpt2.bin"))
    print("wrote gpt2.bin (depth=%d dim=%d vocab=%d)" % (depth, dim, tree["cfg"]["vocab"]))


def bytes_to_unicode():
    # The exact GPT-2 mapping: 256 printable code points, one per byte value.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def build_tok(ddir):
    with open(os.path.join(ddir, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)  # token-string -> id
    decoder = {v: k for k, v in vocab.items()}
    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}

    byte_to_id = np.zeros(256, dtype=np.float64)
    for b in range(256):
        byte_to_id[b] = vocab[b2u[b]]

    with open(os.path.join(ddir, "merges.txt"), "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    a_ids, b_ids, new_ids = [], [], []
    for line in lines:
        if not line:
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        a, b = parts
        merged = a + b
        if a in vocab and b in vocab and merged in vocab:
            a_ids.append(vocab[a])
            b_ids.append(vocab[b])
            new_ids.append(vocab[merged])

    n_vocab = len(vocab)
    id_to_bytes = [None] * n_vocab
    for tid in range(n_vocab):
        s = decoder[tid]
        raw = bytes(u2b[c] for c in s)
        id_to_bytes[tid] = raw

    rec = {
        "merges_a": np.array(a_ids, dtype=np.float64),
        "merges_b": np.array(b_ids, dtype=np.float64),
        "merges_new": np.array(new_ids, dtype=np.float64),
        "byte_to_id": byte_to_id,
        "id_to_bytes": id_to_bytes,
    }
    rstr.save(rec, os.path.join(ddir, "gpt2-tok.bin"))
    print("wrote gpt2-tok.bin (merges=%d vocab=%d)" % (len(a_ids), n_vocab))


def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else "models/gpt2"
    st = load_safetensors(os.path.join(ddir, "model.safetensors"))
    build_weights(st, ddir)
    build_tok(ddir)


if __name__ == "__main__":
    main()
