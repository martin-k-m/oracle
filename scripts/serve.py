#!/usr/bin/env python3
# serve.py: a small local web console for Oracle.
#
# It is a thin GUI over the same `bin/oracle` command line, nothing more: the
# browser posts a task, a prompt, and optional pasted code, and this server runs
# the matching CLI task and returns its output. The model still runs on the CPU
# through the Twill runtime; this only puts a text box in front of it. It binds
# to localhost so it is never exposed off the machine, and it runs one task per
# request (each reloads the model), so it is meant for local use, not a service.
#
# Started by `oracle serve`. Stdlib only, so it needs no install beyond the
# Python that `oracle fetch-qwen` already required.

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORACLE = os.path.join(ROOT, "bin", "oracle")

# Only these tasks are reachable from the browser, and the task name is chosen
# from this set, never taken as free text, so the request cannot name an
# arbitrary command.
TASKS = {"code", "explain", "review", "fix", "tests", "sh", "commit"}

PORT = int(os.environ.get("ORACLE_PORT", "8080"))
MODEL = os.environ.get("ORACLE_MODEL", "")


def run_task(task, prompt, code):
    argv = [ORACLE, task]
    if prompt:
        argv.append(prompt)
    if MODEL:
        argv += ["--model", MODEL]
    env = dict(os.environ)
    # Keep GUI answers a touch shorter than the CLI default so a click returns
    # promptly; a caller can still raise it with ORACLE_STEPS in the environment.
    env.setdefault("ORACLE_STEPS", "160")
    try:
        proc = subprocess.run(
            argv,
            input=(code or ""),
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
    except FileNotFoundError:
        return "", "could not find bin/oracle at " + ORACLE
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if proc.returncode != 0 and not out:
        return "", err or ("oracle exited with status " + str(proc.returncode))
    return out, err


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            with open(os.path.join(ROOT, "gui", "index.html"), encoding="utf-8") as f:
                self._send(200, f.read(), "text/html")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/run":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "bad JSON"}))
            return
        task = req.get("task", "")
        if task not in TASKS:
            self._send(400, json.dumps({"error": "unknown task"}))
            return
        prompt = (req.get("prompt") or "").strip()
        code = req.get("code") or ""
        if not prompt and not code:
            self._send(400, json.dumps({"error": "give a prompt or some code"}))
            return
        out, err = run_task(task, prompt, code)
        self._send(200, json.dumps({"reply": out, "error": err}))

    def log_message(self, *args):
        pass  # keep the terminal quiet; the CLI already prints what matters


def main():
    if not os.path.exists(ORACLE):
        print("serve: cannot find bin/oracle next to this script", file=sys.stderr)
        sys.exit(1)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:" + str(PORT)
    print("Oracle console on " + url + (" (model " + MODEL + ")" if MODEL else ""))
    print("Open it in a browser. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
