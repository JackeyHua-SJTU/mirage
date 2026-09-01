#!/usr/bin/env python3
"""Stand-in for the MPK serving endpoints, used to test the harness itself.

Speaks just enough of ``launch_server``'s surface (``/openapi.json``,
``/v1/chat/completions``, ``/v1/completions``) to be driven by
``online_harness.py`` without a GPU, and can be told to misbehave in one
specific way so the harness's fail-closed logic is provable::

    python mock_server.py --port 8123 --mode dup_done

Modes
    good              behaves correctly
    dup_done          emits two ``data: [DONE]`` lines per stream
    no_done           ends a stream without ``[DONE]``
    chatter_after_done  emits a content chunk after ``[DONE]``
    wrong_marker      echoes another request's marker as well as its own
    nondeterministic  the same prompt yields different text each time
    empty_content     replies with empty content
    hang              accepts completion requests and never answers
    never_ready       ``/openapi.json`` keeps returning 500
    no_shutdown       ignores SIGTERM
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MARKER_RE = re.compile(r"MK7Q2-\d{2}")
MODES = ("good", "dup_done", "no_done", "chatter_after_done", "wrong_marker",
         "nondeterministic", "empty_content", "hang", "never_ready",
         "no_shutdown")

_counter = itertools.count()
_counter_lock = threading.Lock()


def _next_serial() -> int:
    with _counter_lock:
        return next(_counter)


def reply_text(prompt: str, mode: str) -> str:
    """Deterministic canned reply (except in the misbehaving modes)."""
    if mode == "empty_content":
        return ""
    match = MARKER_RE.search(prompt)
    if match:
        own = match.group(0)
        if mode == "wrong_marker":
            foreign = "MK7Q2-%02d" % ((int(own[-2:]) + 1) % 10)
            return f"{own} {foreign}"
        return f"OK {own}"
    text = "mock reply for: " + " ".join(prompt.split())[:60]
    if mode == "nondeterministic":
        text += f" #{_next_serial()}"
    return text


def extract_prompt(body: dict) -> str:
    for message in reversed(body.get("messages") or []):
        if message.get("role") == "user":
            return message.get("content", "")
    return body.get("prompt", "")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "good"
    ready_delay = 0.0
    started_at = 0.0

    # ── plumbing ──────────────────────────────────────────────────────────

    def log_message(self, fmt, *fmt_args):  # noqa: D102 - quieter default log
        sys.stderr.write("[mock] %s\n" % (fmt % fmt_args))

    def _send_json(self, status: int, payload: dict) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _chunk(self, text: str) -> None:
        raw = text.encode("utf-8")
        self.wfile.write(b"%x\r\n" % len(raw) + raw + b"\r\n")
        self.wfile.flush()

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError:
            return {}

    # ── endpoints ─────────────────────────────────────────────────────────

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/openapi.json":
            self._send_json(404, {"detail": "not found"})
            return
        starting = (self.mode == "never_ready"
                    or time.monotonic() - self.started_at < self.ready_delay)
        if starting:
            self._send_json(500, {"detail": "still starting"})
            return
        self._send_json(200, {
            "openapi": "3.1.0",
            "info": {"title": "mock MPK LLM Engine", "version": "0.1.0"},
            "paths": {"/v1/chat/completions": {}, "/v1/completions": {}},
        })

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self._send_json(404, {"detail": "not found"})
            return
        body = self._read_body()
        if self.mode == "hang":
            time.sleep(600)
            return
        text = reply_text(extract_prompt(body), self.mode)
        if body.get("stream"):
            self._stream(text)
        else:
            self._send_json(200, {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
            })

    def _stream(self, text: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        pieces = text.split(" ") if text else [""]
        for piece in pieces:
            payload = json.dumps(
                {"choices": [{"delta": {"content": piece + " "}, "index": 0}]})
            self._chunk(f"data: {payload}\n\n")
        if self.mode != "no_done":
            self._chunk("data: [DONE]\n\n")
        if self.mode == "dup_done":
            self._chunk("data: [DONE]\n\n")
        if self.mode == "chatter_after_done":
            payload = json.dumps(
                {"choices": [{"delta": {"content": "leftover"}, "index": 0}]})
            self._chunk(f"data: {payload}\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--mode", default="good", choices=MODES)
    parser.add_argument("--ready-delay", type=float, default=0.0,
                        help="seconds before /openapi.json answers 200")
    args = parser.parse_args(argv)

    if args.mode == "no_shutdown":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    Handler.mode = args.mode
    Handler.ready_delay = args.ready_delay
    Handler.started_at = time.monotonic()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"[mock] mode={args.mode} listening on "
          f"{args.host}:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
