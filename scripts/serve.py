#!/usr/bin/env python3
# serve.py: a small local web console for Oracle, backed by one live model.
#
# It starts serve.tw once (the Twill model host), keeps that process running, and
# forwards every browser request to it, so the model is loaded a single time
# instead of once per click. Replies stream back token by token over
# Server-Sent Events, and a chat keeps its history in the host between messages.
# It binds to localhost, exposes only a fixed set of tasks, and runs one request
# at a time (a single model, guarded by a lock). Stdlib only.
#
# Started by `oracle serve`. Wire protocol to serve.tw is documented there.

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWILL = os.environ.get("TWILL") or "twill"

# The task name is chosen from this set, never taken as free text. Each maps to
# the instruction the CLI uses; "code" and "chat" carry none (the prompt is the
# whole request), the rest wrap the pasted code.
INSTRUCTIONS = {
    "chat": "",
    "code": "",
    "explain": "Explain what the following code does, clearly and concisely, step by step where it helps.",
    "review": "Review the following code. Point out bugs, edge cases, and concrete improvements, most important first. Be specific and brief.",
    "commit": "Write a git commit message for the following diff. Use a short imperative subject line under about sixty characters, then a blank line, then a brief body of what changed and why. Output only the commit message.",
    "fix": "Find and fix the bug in the following code. First name the bug in one or two sentences, then give the corrected code. If an error message or description is provided, use it to locate the problem.",
    "tests": "Write focused unit tests for the following code. Cover the main behavior and the important edge cases, using the language's standard test style. Output only the test code.",
    "sh": "Give a single shell command for macOS or Linux that does what is asked. Output only the command on one line, with no explanation and no code fence.",
    "ask": "Answer the question using the repository code provided as context. Point to the file paths the answer comes from. If the context does not contain the answer, say so rather than guessing.",
}

PORT = int(os.environ.get("ORACLE_PORT", "8080"))
MODEL = os.environ.get("ORACLE_MODEL", "")

SOH, STX, EOT, EOM = 1, 2, 4, b"__ORACLE_EOM__"


def resolve_twill():
    # Prefer $TWILL, then PATH, then a go-installed binary, matching bin/oracle.
    import shutil
    if os.path.sep in TWILL and os.path.exists(TWILL):
        return TWILL
    found = shutil.which(TWILL)
    if found:
        return found
    gobin = os.environ.get("GOBIN") or os.path.join(
        os.environ.get("GOPATH", os.path.expanduser("~/go")), "bin"
    )
    cand = os.path.join(gobin, "twill")
    return cand if os.path.exists(cand) else TWILL


# The sizes in the order the CLI prefers them, each with the directory it lives
# in (0.5B is the plain models/qwen; the rest are models/qwen-<size>).
MODEL_ORDER = ["0.5B", "1.5B", "3B", "7B", "14B", "32B"]


def dir_for(name):
    if not name or name == "0.5B":
        return os.path.join(ROOT, "models", "qwen")
    return os.path.join(ROOT, "models", "qwen-" + name)


def installed_models():
    # The sizes that are actually downloaded, so the console only offers models
    # the machine can load.
    return [n for n in MODEL_ORDER if os.path.exists(os.path.join(dir_for(n), "qwen-int8.bin"))]


def default_model():
    if MODEL and os.path.exists(os.path.join(dir_for(MODEL), "qwen-int8.bin")):
        return MODEL
    have = installed_models()
    return have[-1] if have else "0.5B"  # the largest installed, like the CLI


class Host:
    # One long-lived serve.tw process, guarded so only one request drives the
    # model at a time. Respawns if the child has exited, and can switch to a
    # different installed model on request.
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.twill = resolve_twill()
        self.name = default_model()

    def start(self):
        env = dict(os.environ)
        env["ORACLE_QWEN_DIR"] = dir_for(self.name)
        self.proc = subprocess.Popen(
            [self.twill, "run", "serve.tw"],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )

    def switch(self, name):
        # Load a different model: stop the current host and start one on the new
        # weights. The conversation does not carry across a model change.
        if name not in installed_models():
            return False
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            self.name = name
            self.start()
        return True

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _read_exact(self, n):
        out = self.proc.stdout
        buf = b""
        while len(buf) < n:
            chunk = out.read(n - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    def _drain_to_eot(self):
        # Read and discard one framed response (used for the RESET ack).
        for _ in self._frames():
            pass

    def _frames(self):
        # Yield decoded text deltas until the EOT byte ends the response.
        out = self.proc.stdout
        while True:
            b = out.read(1)
            if not b:
                return
            code = b[0]
            if code == EOT:
                return
            if code == SOH:
                num = b""
                while True:
                    d = out.read(1)
                    if not d or d[0] == STX:
                        break
                    num += d
                try:
                    n = int(num.decode("ascii"))
                except ValueError:
                    return
                yield self._read_exact(n).decode("utf-8", "replace")
            # any other byte is protocol noise; ignore it

    def stream(self, task, prompt, code, temp=None, steps=None):
        # Serialize the whole exchange: send the request, then relay deltas. The
        # parameter line lets each request pick its own creativity and length; a
        # blank field means the host keeps its default.
        body = compose(task, prompt, code)
        params = ("" if temp is None else str(temp)) + " " + ("" if steps is None else str(steps))
        with self.lock:
            if not self._alive():
                self.start()
            stdin = self.proc.stdin
            if task != "chat":
                # Stateless tasks start from a clean conversation each time.
                stdin.write(b"RESET\n")
                stdin.flush()
                self._drain_to_eot()
            stdin.write(b"MSG\n")
            # Positional "<temp> <steps>": keep the space so an empty temp still
            # leaves steps in the second field.
            stdin.write(params.encode("utf-8"))
            stdin.write(b"\n")
            stdin.write(body.encode("utf-8"))
            stdin.write(b"\n")
            stdin.write(EOM + b"\n")
            stdin.flush()
            for delta in self._frames():
                yield delta

    def reset(self):
        with self.lock:
            if not self._alive():
                return
            self.proc.stdin.write(b"RESET\n")
            self.proc.stdin.flush()
            self._drain_to_eot()


def clamp_params(temp, steps):
    # Keep request-supplied generation controls inside sane bounds; None means
    # "use the host default". temp is creativity, steps is the reply length cap.
    t = None
    if temp is not None:
        try:
            t = max(0.0, min(2.0, float(temp)))
        except (TypeError, ValueError):
            t = None
    s = None
    if steps is not None:
        try:
            s = max(1, min(1024, int(steps)))
        except (TypeError, ValueError):
            s = None
    return t, s


def compose(task, prompt, code):
    parts = []
    instr = INSTRUCTIONS.get(task, "")
    if instr:
        parts.append(instr)
    if prompt:
        parts.append(prompt)
    if code and code.strip():
        parts.append("```\n" + code + "\n```")
    return "\n\n".join(parts) if parts else "Hello."


class EmbedHost:
    # A live all-MiniLM encoder (src/embed.tw), held so semantic ask and index
    # embed against a loaded model instead of starting one each time. It is
    # started lazily on the first /embed request, so a server used only for the
    # coder never loads it. The encoder is line-oriented: one id-line in, one
    # vector line out, so a batch of N texts sends N lines and reads N back.
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.twill = resolve_twill()
        self.embed_dir = os.path.join(ROOT, "models", "embed")
        self.vocab = os.path.join(self.embed_dir, "vocab.txt")

    def available(self):
        return (
            os.path.exists(os.path.join(self.embed_dir, "embed.bin"))
            and os.path.exists(self.vocab)
            and os.path.exists(os.path.join(ROOT, "src", "embed.tw"))
        )

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        env = dict(os.environ)
        env["ORACLE_EMBED_DIR"] = self.embed_dir
        self.proc = subprocess.Popen(
            [self.twill, "run", "src/embed.tw"],
            cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )

    def _readline(self):
        buf = b""
        while True:
            c = self.proc.stdout.read(1)
            if not c or c == b"\n":
                return buf
            buf += c

    def embed(self, texts):
        import wordpiece
        with self.lock:
            if not self._alive():
                self.start()
            payload = "".join(
                " ".join(str(i) for i in wordpiece.encode(t, self.vocab)) + "\n" for t in texts
            )
            self.proc.stdin.write(payload.encode("utf-8"))
            self.proc.stdin.flush()
            out = []
            for _ in range(len(texts)):
                parts = self._readline().decode("utf-8", "replace").split()
                out.append([float(x) for x in parts])
            return out


HOST = Host()
EMBED = EmbedHost()


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
        elif self.path == "/models":
            self._send(200, json.dumps({"models": installed_models(), "current": HOST.name}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return None

    def do_POST(self):
        if self.path == "/reset":
            HOST.reset()
            self._send(200, json.dumps({"ok": True}))
            return
        if self.path == "/model":
            req = self._read_json() or {}
            name = req.get("name", "")
            if HOST.switch(name):
                self._send(200, json.dumps({"ok": True, "current": HOST.name}))
            else:
                self._send(400, json.dumps({"error": "model not installed: " + str(name)}))
            return
        if self.path == "/embed":
            req = self._read_json() or {}
            texts = req.get("texts")
            if not isinstance(texts, list):
                self._send(400, json.dumps({"error": "texts must be a list"}))
                return
            if not EMBED.available():
                self._send(400, json.dumps({"error": "encoder not installed on server; run oracle fetch-embed"}))
                return
            try:
                vectors = EMBED.embed(texts) if texts else []
                self._send(200, json.dumps({"vectors": vectors}))
            except Exception as e:
                self._send(500, json.dumps({"error": str(e)}))
            return
        if self.path not in ("/run", "/stream"):
            self._send(404, json.dumps({"error": "not found"}))
            return
        req = self._read_json()
        if req is None:
            self._send(400, json.dumps({"error": "bad JSON"}))
            return
        task = req.get("task", "")
        if task not in INSTRUCTIONS:
            self._send(400, json.dumps({"error": "unknown task"}))
            return
        prompt = (req.get("prompt") or "").strip()
        code = req.get("code") or ""
        if not prompt and not code.strip():
            self._send(400, json.dumps({"error": "give a prompt or some code"}))
            return
        temp, steps = clamp_params(req.get("temp"), req.get("steps"))

        if self.path == "/run":
            try:
                reply = "".join(HOST.stream(task, prompt, code, temp, steps))
                self._send(200, json.dumps({"reply": reply.strip()}))
            except Exception as e:  # keep the server up on a single bad request
                self._send(500, json.dumps({"error": str(e)}))
            return

        # /stream: Server-Sent Events, one JSON delta per event, then done.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for delta in HOST.stream(task, prompt, code, temp, steps):
                payload = json.dumps({"t": delta})
                self.wfile.write(("data: " + payload + "\n\n").encode("utf-8"))
                self.wfile.flush()
            self.wfile.write(b"event: done\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-stream

    def log_message(self, *args):
        pass


def main():
    if not os.path.exists(os.path.join(ROOT, "serve.tw")):
        print("serve: cannot find serve.tw at the repo root", file=sys.stderr)
        sys.exit(1)
    mdir = dir_for(HOST.name)
    if not os.path.exists(os.path.join(mdir, "qwen-int8.bin")):
        rel = os.path.relpath(mdir, ROOT)
        print("serve: the Qwen weights are not present at " + rel + ".", file=sys.stderr)
        print("Run once:  oracle fetch-qwen " + HOST.name, file=sys.stderr)
        sys.exit(3)
    HOST.start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = "http://127.0.0.1:" + str(PORT)
    others = [m for m in installed_models() if m != HOST.name]
    extra = (" +" + ",".join(others)) if others else ""
    print("Oracle console on " + url + " (model " + HOST.name + extra + ")")
    print("The model is loaded and held live. Open it in a browser. Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
        if HOST.proc:
            HOST.proc.terminate()


if __name__ == "__main__":
    main()
