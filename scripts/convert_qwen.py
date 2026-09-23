# convert_qwen.py: convert Qwen2.5-Coder-0.5B-Instruct into an int8 Twill tree.
#
# This is data prep, not the runtime. It reads the open Qwen weights (bf16
# safetensors) and quantizes every 2-D weight to int8 with a per-row scale, in
# the packed form the Twill runtime reads with quantize_packed. At 0.5B
# parameters the model is about 4 GB in f64 and does not fit in a laptop's RAM,
# so int8 is the only way it runs at home; the packed tree is about 500 MB.
#
# Reads from the directory given as argv[1] (default models/qwen):
#   model.safetensors   the bf16 weights
#   config.json         the architecture
# Writes:
#   qwen-int8.bin       the int8 weight tree, for src/qwen.tw
#
# Qwen2 stores linear weights as [out, in], which is what linear(X, W) = X @ W^T
# wants, so no transpose is needed (unlike GPT-2's Conv1D). Attention has a bias
# on q, k, v and none on the output; the MLP is SwiGLU (gate, up, down) with no
# bias; the norms are RMSNorm (a weight, no bias); the output head is tied to the
# embedding table.

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
    dtypes = {"F32": "<f4", "F16": "<f2", "F64": "<f8"}
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


def pack_i8(w):
    # Per-row (per output channel) symmetric int8, matching twill's QuantizeI8.
    rows, cols = w.shape
    maxabs = np.abs(w).max(axis=1)
    scale = np.where(maxabs == 0.0, 0.0, maxabs / 127.0)
    inv = np.where(scale == 0.0, 0.0, 1.0 / scale)
    codes = np.clip(np.round(w * inv[:, None]), -128, 127).astype(np.int16)
    q = (codes + 128).astype(np.uint8).tobytes()
    return {"q": q, "scale": scale.astype(np.float64), "rows": int(rows), "cols": int(cols)}


def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen"
    cfg = json.load(open(os.path.join(ddir, "config.json")))
    st = load_safetensors(os.path.join(ddir, "model.safetensors"))

    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = hidden // heads
    depth = cfg["num_hidden_layers"]

    layers = []
    for i in range(depth):
        p = f"model.layers.{i}."
        layers.append({
            "ln1": st[p + "input_layernorm.weight"],
            "attn": {
                "q": pack_i8(st[p + "self_attn.q_proj.weight"]),
                "qb": st[p + "self_attn.q_proj.bias"],
                "k": pack_i8(st[p + "self_attn.k_proj.weight"]),
                "kb": st[p + "self_attn.k_proj.bias"],
                "v": pack_i8(st[p + "self_attn.v_proj.weight"]),
                "vb": st[p + "self_attn.v_proj.bias"],
                "o": pack_i8(st[p + "self_attn.o_proj.weight"]),
            },
            "ln2": st[p + "post_attention_layernorm.weight"],
            "mlp": {
                "gate": pack_i8(st[p + "mlp.gate_proj.weight"]),
                "up": pack_i8(st[p + "mlp.up_proj.weight"]),
                "down": pack_i8(st[p + "mlp.down_proj.weight"]),
            },
        })

    tree = {
        "cfg": {
            "vocab": int(cfg["vocab_size"]),
            "hidden": int(hidden),
            "heads": int(heads),
            "kv_heads": int(kv_heads),
            "head_dim": int(head_dim),
            "depth": int(depth),
            "inter": int(cfg["intermediate_size"]),
            "rope_theta": float(cfg["rope_theta"]),
            "eps": float(cfg["rms_norm_eps"]),
            "eos": int(cfg["eos_token_id"]),
        },
        "embed": pack_i8(st["model.embed_tokens.weight"]),
        "layers": layers,
        "norm": st["model.norm.weight"],
        "quant": 8,
    }
    rstr.save(tree, os.path.join(ddir, "qwen-int8.bin"))
    print("wrote qwen-int8.bin (depth=%d hidden=%d heads=%d kv=%d vocab=%d)"
          % (depth, hidden, heads, kv_heads, tree["cfg"]["vocab"]))


if __name__ == "__main__":
    main()
