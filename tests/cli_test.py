# cli_test.py: the bin/oracle dispatch logic that has no model in it.
#
# These pin the behaviour a real user depends on and that green Python tests once
# missed: `oracle ask` searches the directory you invoked it from (not Oracle's
# own install directory, which the script cd's into to run twill), a --repo is
# resolved relative to that directory, and a bare quoted question routes to ask.
# ORACLE_DEBUG_REPO makes ask print the resolved repo and question, then stop
# before loading any model, so the whole thing runs with no weights and no
# network.

import os
import subprocess
import tempfile
import unittest

ORACLE = os.path.join(os.path.dirname(__file__), "..", "bin", "oracle")


def run_from(cwd, args):
    env = dict(os.environ)
    env["ORACLE_DEBUG_REPO"] = "1"
    out = subprocess.run(
        [ORACLE, *args], cwd=cwd, env=env, capture_output=True, text=True
    )
    fields = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k] = v
    return fields, out


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.realpath(tempfile.mkdtemp())
        if not (os.environ.get("TWILL") or _twill_findable()):
            self.skipTest("twill not available to run bin/oracle")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_ask_defaults_to_invoking_directory(self):
        fields, out = run_from(self.tmp, ["ask", "how is retry handled"])
        self.assertEqual(fields.get("REPO"), self.tmp, out.stderr)
        self.assertEqual(fields.get("QUESTION"), "how is retry handled")

    def test_bare_question_routes_to_ask(self):
        fields, out = run_from(self.tmp, ["why does a flywheel store energy"])
        self.assertEqual(fields.get("REPO"), self.tmp, out.stderr)
        self.assertEqual(fields.get("QUESTION"), "why does a flywheel store energy")

    def test_bare_question_keeps_flags(self):
        # A flag after a bare question must survive the routing to ask.
        fields, out = run_from(self.tmp, ["explain the design here", "--k", "3"])
        self.assertEqual(fields.get("REPO"), self.tmp, out.stderr)
        self.assertEqual(fields.get("QUESTION"), "explain the design here")

    def test_relative_repo_resolves_against_invoking_dir(self):
        sub = os.path.join(self.tmp, "sub")
        os.makedirs(sub)
        fields, out = run_from(self.tmp, ["ask", "q", "--repo", "sub"])
        self.assertEqual(fields.get("REPO"), os.path.realpath(sub), out.stderr)

    def test_absolute_repo_is_kept(self):
        other = os.path.realpath(tempfile.mkdtemp())
        try:
            fields, out = run_from(self.tmp, ["ask", "q", "--repo", other])
            self.assertEqual(fields.get("REPO"), other, out.stderr)
        finally:
            import shutil
            shutil.rmtree(other, ignore_errors=True)


def _twill_findable():
    import shutil
    if shutil.which("twill"):
        return True
    gobin = os.environ.get("GOBIN") or os.path.join(
        os.environ.get("GOPATH", os.path.expanduser("~/go")), "bin"
    )
    return os.path.exists(os.path.join(gobin, "twill"))


if __name__ == "__main__":
    unittest.main()
