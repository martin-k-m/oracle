# wordpiece_test.py: the BERT tokenizer's algorithm, on a small fixed vocab.
#
# The real model vocab is 30k tokens and is not committed, so these test the
# tokenisation logic (lowercasing, accent stripping, punctuation splitting,
# greedy longest-match WordPiece, [UNK], truncation) against a tiny vocab where
# every expected id is obvious. A regression in any of those silently degrades
# retrieval, so it is worth pinning.

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import wordpiece

VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]",   # 0..3
    "hello", "world", ",", "!",           # 4..7
    "un", "##aff", "##able",              # 8..10
    "cafe", "the", "play", "##ing",       # 11..14
]
CLS, SEP, UNK = 2, 3, 1


class WordPieceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fd, cls.vocab_path = tempfile.mkstemp(suffix="_vocab.txt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(VOCAB) + "\n")
        wordpiece._vocab = None  # reset the module cache for the test vocab

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.vocab_path)

    def enc(self, text, max_len=256):
        wordpiece._vocab = None
        return wordpiece.encode(text, self.vocab_path, max_len=max_len)

    def test_wraps_with_cls_and_sep(self):
        self.assertEqual(self.enc("hello"), [CLS, 4, SEP])

    def test_lowercases_and_splits_punctuation(self):
        # "Hello, World!" -> hello , world !
        self.assertEqual(self.enc("Hello, World!"), [CLS, 4, 6, 5, 7, SEP])

    def test_greedy_wordpiece_continuation(self):
        # "unaffable" -> un ##aff ##able
        self.assertEqual(self.enc("unaffable"), [CLS, 8, 9, 10, SEP])

    def test_unknown_word_is_unk(self):
        self.assertEqual(self.enc("xyzzy"), [CLS, UNK, SEP])

    def test_strips_accents(self):
        # "Café" -> cafe
        self.assertEqual(self.enc("Café"), [CLS, 11, SEP])

    def test_playing_splits(self):
        self.assertEqual(self.enc("playing"), [CLS, 13, 14, SEP])

    def test_truncates_to_max_len(self):
        ids = self.enc("hello hello hello hello", max_len=4)
        self.assertLessEqual(len(ids), 4)
        self.assertEqual(ids[0], CLS)
        self.assertEqual(ids[-1], SEP)

    def test_empty_text(self):
        self.assertEqual(self.enc("   "), [CLS, SEP])


if __name__ == "__main__":
    unittest.main()
