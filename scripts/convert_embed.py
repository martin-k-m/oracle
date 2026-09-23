# convert_embed.py: convert all-MiniLM-L6-v2 into a Twill weight tree.
#
# Data prep, not the runtime. It reads the open MiniLM weights (BERT encoder,
# 6 layers, hidden 384) and writes them as an RSTR tree for src/embed.tw. The
# model is tiny (about 22M parameters), so the weights are kept in full precision
# rather than quantized: correctness of the embedding matters more than the few
# tens of milliseconds a chunk takes, and f64 keeps the runtime a faithful match
# to the reference. The weights are the open sentence-transformers release
# (Apache-2.0); only the runtime is Twill.
#
# Reads models/embed/model.safetensors and bert_config.json; writes embed.bin.

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
    dtypes = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8"}
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


def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else "models/embed"
    cfg = json.load(open(os.path.join(ddir, "bert_config.json")))
    w = load_safetensors(os.path.join(ddir, "model.safetensors"))

    def ln(prefix):
        return {"gamma": w[prefix + ".weight"], "beta": w[prefix + ".bias"]}

    layers = []
    for i in range(cfg["num_hidden_layers"]):
        p = "encoder.layer.%d." % i
        layers.append({
            "qw": w[p + "attention.self.query.weight"], "qb": w[p + "attention.self.query.bias"],
            "kw": w[p + "attention.self.key.weight"], "kb": w[p + "attention.self.key.bias"],
            "vw": w[p + "attention.self.value.weight"], "vb": w[p + "attention.self.value.bias"],
            "ow": w[p + "attention.output.dense.weight"], "ob": w[p + "attention.output.dense.bias"],
            "aln": ln(p + "attention.output.LayerNorm"),
            "iw": w[p + "intermediate.dense.weight"], "ib": w[p + "intermediate.dense.bias"],
            "dw": w[p + "output.dense.weight"], "db": w[p + "output.dense.bias"],
            "oln": ln(p + "output.LayerNorm"),
        })

    tree = {
        "cfg": {
            "layers": cfg["num_hidden_layers"],
            "hidden": cfg["hidden_size"],
            "heads": cfg["num_attention_heads"],
            "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
            "inter": cfg["intermediate_size"],
        },
        "word_emb": w["embeddings.word_embeddings.weight"],
        "pos_emb": w["embeddings.position_embeddings.weight"],
        "type_emb": w["embeddings.token_type_embeddings.weight"],
        "emb_ln": ln("embeddings.LayerNorm"),
        "layers": layers,
    }
    out = os.path.join(ddir, "embed.bin")
    rstr.save(tree, out)
    print("wrote %s (%.0f MB)" % (out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
