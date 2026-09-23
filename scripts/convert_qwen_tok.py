# convert_qwen_tok.py: convert Qwen's tokenizer.json into a Twill tokenizer tree.
#
# Qwen uses the same byte-level BPE as GPT-2 (the "Ġ is a space" unicode mapping),
# so this reuses that scheme: it writes byte_to_id (256 base ids), the merge
# table as id triples, id_to_bytes for decoding, and the ChatML special-token ids
# the runtime needs to build a chat prompt. Written to models/qwen/qwen-tok.bin
# for src/qwen_tok.tw.

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import rstr


def bytes_to_unicode():
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


def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else "models/qwen"
    tj = json.load(open(os.path.join(ddir, "tokenizer.json")))
    vocab = tj["model"]["vocab"]           # token-string -> id
    merges = tj["model"]["merges"]         # "a b" strings
    added = {a["content"]: a["id"] for a in tj["added_tokens"]}

    b2u = bytes_to_unicode()
    u2b = {u: b for b, u in b2u.items()}

    byte_to_id = np.array([vocab[b2u[b]] for b in range(256)], dtype=np.float64)

    a_ids, b_ids, new_ids = [], [], []
    for m in merges:
        a, b = m.split(" ")
        ab = a + b
        if a in vocab and b in vocab and ab in vocab:
            a_ids.append(vocab[a])
            b_ids.append(vocab[b])
            new_ids.append(vocab[ab])

    # id_to_bytes for every id, base and special. The full range is the config
    # vocab size; specials decode to their literal content bytes.
    max_id = max(max(vocab.values()), max(added.values()))
    decoder = {v: k for k, v in vocab.items()}
    special_ids = set(added.values())
    id_to_bytes = [b""] * (max_id + 1)
    for tid in range(max_id + 1):
        if tid in decoder and tid not in special_ids:
            id_to_bytes[tid] = bytes(u2b[c] for c in decoder[tid])
        else:
            content = next((c for c, i in added.items() if i == tid), "")
            id_to_bytes[tid] = content.encode("utf-8")

    rec = {
        "merges_a": np.array(a_ids, dtype=np.float64),
        "merges_b": np.array(b_ids, dtype=np.float64),
        "merges_new": np.array(new_ids, dtype=np.float64),
        "byte_to_id": byte_to_id,
        "id_to_bytes": id_to_bytes,
        "im_start": int(added["<|im_start|>"]),
        "im_end": int(added["<|im_end|>"]),
        "endoftext": int(added["<|endoftext|>"]),
    }
    rstr.save(rec, os.path.join(ddir, "qwen-tok.bin"))
    print("wrote qwen-tok.bin (merges=%d vocab=%d im_start=%d im_end=%d)"
          % (len(a_ids), max_id + 1, rec["im_start"], rec["im_end"]))


if __name__ == "__main__":
    main()
