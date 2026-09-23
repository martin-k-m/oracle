# wordpiece.py: the BERT (uncased) tokenizer, in the Python retrieval driver.
#
# Tokenisation is a deterministic string step, so it lives here rather than in
# the Twill encoder: basic tokenisation (lowercase, strip accents, split on
# whitespace and punctuation) then greedy WordPiece against the model's vocab.
# It reproduces the ids the reference model would produce, which is what makes
# the Twill encoder's embeddings match the reference.

import os
import unicodedata

_vocab = None
_ids = None


def _is_punct(ch):
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def load_vocab(path):
    global _vocab, _ids
    if _vocab is not None:
        return _vocab
    _vocab = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            _vocab[line.rstrip("\n")] = i
    _ids = {
        "unk": _vocab.get("[UNK]", 100),
        "cls": _vocab.get("[CLS]", 101),
        "sep": _vocab.get("[SEP]", 102),
    }
    return _vocab


def _basic_tokenize(text):
    # NFD, drop combining marks (accent stripping), lowercase, split on space and
    # punctuation.
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn").lower()
    tokens = []
    for word in text.split():
        buf = ""
        for ch in word:
            if _is_punct(ch):
                if buf:
                    tokens.append(buf)
                    buf = ""
                tokens.append(ch)
            else:
                buf += ch
        if buf:
            tokens.append(buf)
    return tokens


def _wordpiece(token, vocab, unk):
    if len(token) > 100:
        return [unk]
    out = []
    start = 0
    n = len(token)
    while start < n:
        end = n
        cur = None
        while start < end:
            sub = token[start:end]
            if start > 0:
                sub = "##" + sub
            if sub in vocab:
                cur = sub
                break
            end -= 1
        if cur is None:
            return [unk]
        out.append(vocab[cur])
        start = end
    return out


def encode(text, vocab_path, max_len=256):
    vocab = load_vocab(vocab_path)
    ids = [_ids["cls"]]
    for tok in _basic_tokenize(text):
        for piece in _wordpiece(tok, vocab, _ids["unk"]):
            ids.append(piece)
            if len(ids) >= max_len - 1:
                break
        if len(ids) >= max_len - 1:
            break
    ids.append(_ids["sep"])
    return ids
