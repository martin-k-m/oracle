# index_test.py: the persistent vector index's caching and search logic.
#
# The encoder is replaced by a deterministic fake (same text always maps to the
# same vector), so these test the parts that hold the bugs without the model:
# that a build embeds every chunk, that a rebuild re-embeds only the files that
# changed and reuses the rest, that the on-disk layout round-trips, and that a
# search ranks the chunk whose text matches the query first.

import hashlib
import math
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import retrieve

EMBED_LOG = []  # texts embedded by the fake, per call, for asserting reuse


def fake_embed(texts, args):
    EMBED_LOG.append(list(texts))
    out = []
    for t in texts:
        h = hashlib.sha256(t.encode("utf-8")).digest()
        raw = (h * (retrieve.DIMS // len(h) + 1))[:retrieve.DIMS]
        v = [b - 127.5 for b in raw]
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


class IndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.home = tempfile.mkdtemp()
        os.environ["ORACLE_INDEX_HOME"] = self.home
        self.args = types.SimpleNamespace(twill="", embed_dir="", server="", repo=self.tmp)
        self._real = retrieve.embed_texts
        self._real_ready = retrieve.embed_ready
        retrieve.embed_texts = fake_embed
        retrieve.embed_ready = lambda args: True  # the fake encoder stands in for the model
        EMBED_LOG.clear()

    def tearDown(self):
        retrieve.embed_texts = self._real
        retrieve.embed_ready = self._real_ready
        os.environ.pop("ORACLE_INDEX_HOME", None)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.home, ignore_errors=True)

    def test_build_embeds_all_and_roundtrips(self):
        write(self.tmp, "a.py", "def alpha():\n    return 1\n")
        write(self.tmp, "b.py", "def beta():\n    return 2\n")
        self.assertTrue(retrieve.build_index(self.tmp, self.args))
        idx = retrieve.load_index(self.tmp)
        self.assertIsNotNone(idx)
        rels = {c[0] for c in idx["chunks"]}
        self.assertEqual(rels, {"a.py", "b.py"})
        # vectors.f32 holds exactly one DIMS-vector per chunk
        self.assertEqual(len(idx["vectors"]), len(idx["chunks"]) * retrieve.DIMS * 4)

    def test_incremental_reuses_unchanged(self):
        write(self.tmp, "a.py", "def alpha():\n    return 1\n")
        write(self.tmp, "b.py", "def beta():\n    return 2\n")
        retrieve.build_index(self.tmp, self.args)
        EMBED_LOG.clear()
        # Change only b.py (new content -> new size and mtime).
        write(self.tmp, "b.py", "def beta():\n    return 2 + 2 + 2\n")
        retrieve.build_index(self.tmp, self.args)
        embedded = [t for call in EMBED_LOG for t in call]
        self.assertTrue(embedded, "a changed file should be re-embedded")
        self.assertTrue(all("beta" in t for t in embedded), "only b.py should be re-embedded")

    def test_no_change_embeds_nothing(self):
        write(self.tmp, "a.py", "def alpha():\n    return 1\n")
        retrieve.build_index(self.tmp, self.args)
        EMBED_LOG.clear()
        retrieve.build_index(self.tmp, self.args)
        self.assertEqual(EMBED_LOG, [], "an unchanged repo should embed nothing")

    def test_search_ranks_matching_chunk_first(self):
        text_a = "def alpha():\n    return 1\n"
        write(self.tmp, "a.py", text_a)
        write(self.tmp, "b.py", "def beta():\n    return 2\n")
        retrieve.build_index(self.tmp, self.args)
        # The query equals a.py's text, so its fake vector matches a.py's chunk.
        hits = retrieve.index_search(self.tmp, text_a, self.args, k=2, budget=8000)
        self.assertTrue(hits)
        self.assertEqual(hits[0][1], "a.py")

    def test_deleted_file_drops_out(self):
        write(self.tmp, "a.py", "def alpha():\n    return 1\n")
        write(self.tmp, "b.py", "def beta():\n    return 2\n")
        retrieve.build_index(self.tmp, self.args)
        os.remove(os.path.join(self.tmp, "b.py"))
        retrieve.build_index(self.tmp, self.args)
        idx = retrieve.load_index(self.tmp)
        rels = {c[0] for c in idx["chunks"]}
        self.assertEqual(rels, {"a.py"})


if __name__ == "__main__":
    unittest.main()
