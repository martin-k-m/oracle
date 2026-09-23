#!/usr/bin/env python3
# client.py: send one task to a running `oracle serve` and stream the reply.
#
# This is what lets the CLI reuse a live model instead of loading it again.
# `oracle code`, `explain`, `fix` and the rest each pay about two seconds to
# start twill and load the weights; with a server already running they hand the
# work to it over HTTP and the model is loaded only once, for the server's life.
#
# Usage:  client.py URL TASK PROMPT      (the code/context is read from stdin)
# It prints the reply to stdout as it arrives and exits non-zero if the server
# cannot be reached, so the caller can fall back to running the model locally.

import json
import sys
import urllib.error
import urllib.request


def main():
    if len(sys.argv) < 4:
        print("client.py: usage: client.py URL TASK PROMPT", file=sys.stderr)
        return 2
    base, task, prompt = sys.argv[1].rstrip("/"), sys.argv[2], sys.argv[3]
    code = sys.stdin.read() if not sys.stdin.isatty() else ""
    body = json.dumps({"task": task, "prompt": prompt, "code": code}).encode("utf-8")
    req = urllib.request.Request(
        base + "/stream", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.URLError as e:
        # Connection refused, wrong port, server down: let the caller fall back.
        print("client.py: could not reach " + base + " (" + str(e.reason) + ")", file=sys.stderr)
        return 7
    except Exception as e:
        print("client.py: " + str(e), file=sys.stderr)
        return 7
    wrote = False
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\n")
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        t = obj.get("t")
        if t:
            sys.stdout.write(t)
            sys.stdout.flush()
            wrote = True
        if obj.get("error"):
            print("\nserver error: " + obj["error"], file=sys.stderr)
            return 1
    if wrote:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
