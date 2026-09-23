# retrieve_test.py: the keyword (BM25) retrieval, on a synthetic repository.
#
# These run retrieve.py end to end over a temporary repo (no model, no
# --semantic) and check the ranking behaviour that oracle ask depends on: the
# right file rises, source outranks documentation that merely repeats the words,
# skipped directories and binaries are ignored, and the chunk headers are the
# path:line form the model is given.

import os
import re
import subprocess
import sys
import tempfile
import unittest

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "retrieve.py")


def write(root, rel, text):
    path = os.path.join(root, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb" if isinstance(text, bytes) else "w") as f:
        f.write(text)


def run(root, query, extra=None):
    cmd = [sys.executable, SCRIPT, "--repo", root, "--query", query, "--k", "5", "--budget", "8000"]
    if extra:
        cmd += extra
    out = subprocess.run(cmd, capture_output=True, text=True)
    return out.stdout, out.stderr


def headers(stdout):
    return re.findall(r"^===== (.+?):(\d+)-(\d+) =====$", stdout, re.M)


class RetrieveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_relevant_file_ranks_first(self):
        write(self.tmp, "auth.py", "def login(user, password):\n    return check_password(user, password)\n")
        write(self.tmp, "math.py", "def add(a, b):\n    return a + b\n")
        stdout, _ = run(self.tmp, "how does login check the password")
        h = headers(stdout)
        self.assertTrue(h, "expected at least one result")
        self.assertEqual(h[0][0], "auth.py")

    def test_source_outranks_documentation(self):
        # A doc that repeats the query words should not beat the source that
        # defines the thing, because docs are down-weighted.
        write(self.tmp, "cache.py", "class Cache:\n    def evict(self):\n        drop_oldest_entry()\n")
        write(self.tmp, "NOTES.md", "cache cache cache evict evict evict eviction eviction the cache evicts\n" * 3)
        stdout, _ = run(self.tmp, "how does the cache evict entries")
        h = headers(stdout)
        self.assertEqual(h[0][0], "cache.py", "source should outrank the keyword-stuffed doc")

    def test_skips_ignored_dirs_and_binaries(self):
        write(self.tmp, "real.py", "def widget_handler():\n    return build_widget()\n")
        write(self.tmp, "node_modules/dep.py", "def widget_handler():\n    pass\n")
        write(self.tmp, "blob.bin", b"\x00\x01widget_handler\x00")
        stdout, _ = run(self.tmp, "widget_handler")
        files = {h[0] for h in headers(stdout)}
        self.assertIn("real.py", files)
        self.assertNotIn("node_modules/dep.py", files)
        self.assertNotIn("blob.bin", files)

    def test_header_line_range(self):
        write(self.tmp, "f.py", "\n".join("line %d token%d" % (i, i) for i in range(10)) + "\n")
        stdout, _ = run(self.tmp, "token3")
        h = headers(stdout)
        self.assertTrue(h)
        rel, s, e = h[0]
        self.assertEqual(rel, "f.py")
        self.assertEqual(int(s), 1)
        self.assertGreaterEqual(int(e), int(s))

    def test_no_match_no_output(self):
        write(self.tmp, "f.py", "def add(a, b):\n    return a + b\n")
        stdout, _ = run(self.tmp, "zzzznothingmatchesthisquery")
        self.assertEqual(headers(stdout), [])

    def test_number_flag_labels_passages(self):
        write(self.tmp, "a.py", "def login():\n    pass\n")
        write(self.tmp, "b.py", "def login_helper():\n    login()\n")
        stdout, _ = run(self.tmp, "login", extra=["--number"])
        labels = re.findall(r"^===== (\[\d+\]) .+?:\d+-\d+ =====$", stdout, re.M)
        self.assertTrue(labels, "expected numbered headers")
        self.assertEqual(labels[0], "[1]")
        # Numbering is 1-based and sequential across the emitted passages.
        self.assertEqual(labels, ["[%d]" % (i + 1) for i in range(len(labels))])


if __name__ == "__main__":
    unittest.main()
