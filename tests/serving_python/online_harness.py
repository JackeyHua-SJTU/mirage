#!/usr/bin/env python3
"""End-to-end harness for the MPK online-serving HTTP server.

Launches ``python -m mirage.engine.launch_server`` (or attaches to a running
instance), drives it through a fixed battery of serving scenarios, and reports
each case independently.  Uses only the standard library, so it can be copied
next to a server on any box without installing anything.

The harness is *fail-closed*: any phase that does not clearly succeed -- server
launch, readiness, a single case, the final shutdown -- makes the process exit
non-zero.  The spawned server is always killed (TERM, then KILL) on the way out.

Cases
    smoke        one non-stream chat completion returns 200 + non-empty content
    stream       an SSE stream yields >=1 content chunk and exactly one [DONE]
    coldwarm     the same prompt twice in a row returns byte-identical text
    isolation10  10 concurrent requests keep their own marker and only their own
    seq12        12 sequential requests all succeed (request-ring wraparound)
    shutdown     the server exits within --shutdown-timeout of SIGTERM

Usage::

    # launch a server on a free port and run everything
    python tests/serving_python/online_harness.py --model Qwen/Qwen3-0.6B

    # drive a server that is already listening (shutdown case is skipped)
    python tests/serving_python/online_harness.py --attach-port 8000

    # upstream-HEAD baseline, where cross-request leakage is a known bug
    python tests/serving_python/online_harness.py --known-broken-concurrency

Output is a human table followed by a machine-parseable JSON summary fenced
between ``BEGIN_HARNESS_JSON`` / ``END_HARNESS_JSON``; :func:`extract_summary`
parses it back.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
CHAT_PATH = "/v1/chat/completions"
READY_PATH = "/openapi.json"

JSON_BEGIN = "BEGIN_HARNESS_JSON"
JSON_END = "END_HARNESS_JSON"

# Qwen3 answers straight away instead of spending the whole --max-seq-length
# budget on a <think> block; harmless text for models that ignore it.
DEFAULT_PROMPT_SUFFIX = "/no_think"

MARKER_PREFIX = "MK7Q2-"
MARKER_RE = re.compile(re.escape(MARKER_PREFIX) + r"\d{2}")
ISOLATION_N = 10
SEQ_N = 12

EXIT_OK = 0
EXIT_CASE_FAILED = 1
EXIT_SETUP_FAILED = 2


class CaseFailure(Exception):
    """A case observed the server behaving wrongly."""


class SetupFailure(Exception):
    """The server could not be launched or never became ready."""


class SkipCase(Exception):
    """The case does not apply to this run."""


# ── HTTP (stdlib only) ────────────────────────────────────────────────────────


def _post(host: str, port: int, path: str, payload: dict, timeout: float):
    """POST JSON, return ``(status, HTTPResponse, connection)`` unread."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    try:
        conn.request("POST", path, body=body, headers={
            "Content-Type": "application/json",
            "Accept": "*/*",
        })
        resp = conn.getresponse()
    except BaseException:
        conn.close()
        raise
    return resp.status, resp, conn


def post_json(host: str, port: int, path: str, payload: dict,
              timeout: float) -> tuple[int, object]:
    """POST JSON and read the whole (non-streamed) reply."""
    status, resp, conn = _post(host, port, path, payload, timeout)
    try:
        raw = resp.read()
    finally:
        conn.close()
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, raw.decode("utf-8", "replace")


@dataclass
class SseCapture:
    status: int = 0
    content_chunks: list[str] = field(default_factory=list)
    done_count: int = 0
    events_after_done: int = 0
    errors: list[str] = field(default_factory=list)
    eof: bool = False


def post_sse(host: str, port: int, path: str, payload: dict,
             timeout: float) -> SseCapture:
    """POST a streaming request and drain the SSE body to EOF."""
    deadline = time.monotonic() + timeout
    cap = SseCapture()
    status, resp, conn = _post(host, port, path, payload, timeout)
    cap.status = status
    try:
        if status != 200:
            resp.read()
            return cap
        while True:
            if time.monotonic() > deadline:
                raise CaseFailure(
                    f"SSE body still open after {timeout:.1f}s "
                    f"({cap.done_count} [DONE], "
                    f"{len(cap.content_chunks)} content chunks)")
            raw = resp.readline()
            if not raw:
                cap.eof = True
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                cap.done_count += 1
                continue
            if cap.done_count:
                cap.events_after_done += 1
            try:
                event = json.loads(data)
            except ValueError:
                cap.errors.append(f"non-JSON SSE payload: {data[:200]!r}")
                continue
            if isinstance(event, dict) and event.get("error"):
                cap.errors.append(str(event["error"])[:200])
                continue
            try:
                delta = event["choices"][0].get("delta", {})
                text = delta.get("content")
                if text is None:
                    text = event["choices"][0].get("text")
            except (KeyError, IndexError, TypeError):
                cap.errors.append(f"malformed SSE chunk: {data[:200]!r}")
                continue
            if text:
                cap.content_chunks.append(text)
    finally:
        conn.close()
    return cap


def chat_payload(prompt: str, stream: bool = False) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }


def content_of(body: object) -> str:
    """Pull the assistant text out of a chat/text completion reply."""
    if not isinstance(body, dict):
        raise CaseFailure(f"reply is not a JSON object: {str(body)[:200]!r}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise CaseFailure(f"reply has no choices: {str(body)[:200]!r}")
    choice = choices[0]
    text = (choice.get("message") or {}).get("content")
    if text is None:
        text = choice.get("text")
    if not isinstance(text, str):
        raise CaseFailure(f"reply has no text content: {str(body)[:200]!r}")
    return text


def one_completion(ctx: "Context", prompt: str, what: str) -> str:
    """Non-stream completion; raises :class:`CaseFailure` unless it succeeds."""
    try:
        status, body = post_json(ctx.host, ctx.port, CHAT_PATH,
                                 chat_payload(ctx.prompt(prompt)),
                                 ctx.request_timeout)
    except (socket.timeout, TimeoutError):
        raise CaseFailure(
            f"{what}: no reply within {ctx.request_timeout:.1f}s")
    except OSError as exc:
        raise CaseFailure(f"{what}: transport error: {exc!r}")
    if status != 200:
        raise CaseFailure(f"{what}: HTTP {status}: {str(body)[:200]!r}")
    text = content_of(body)
    if not text.strip():
        raise CaseFailure(f"{what}: empty content")
    return text


# ── server process ────────────────────────────────────────────────────────────


def find_free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def port_is_listening(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
        except OSError:
            return False
        return True


class ServerHandle:
    """Owns the server subprocess: launch, readiness, teardown."""

    def __init__(self, cmd: list[str], log_path: str, host: str, port: int):
        self.cmd = cmd
        self.log_path = log_path
        self.host = host
        self.port = port
        self._log = open(log_path, "wb")
        # Own process group so teardown reaps compiler/worker children too.
        self.proc = subprocess.Popen(
            cmd, stdout=self._log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True)

    def _signal_group(self, sig: int) -> None:
        if self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass

    def wait_ready(self, timeout: float, poll_interval: float = 0.5) -> float:
        """Poll ``/openapi.json`` until it answers 200. Returns seconds waited."""
        started = time.monotonic()
        deadline = started + timeout
        last = "no attempt made"
        while time.monotonic() < deadline:
            rc = self.proc.poll()
            if rc is not None:
                raise SetupFailure(
                    f"server exited with code {rc} before becoming ready; "
                    f"log: {self.log_path}")
            try:
                conn = http.client.HTTPConnection(self.host, self.port,
                                                  timeout=5.0)
                try:
                    conn.request("GET", READY_PATH)
                    resp = conn.getresponse()
                    resp.read()
                    if resp.status == 200:
                        return time.monotonic() - started
                    last = f"HTTP {resp.status} from {READY_PATH}"
                finally:
                    conn.close()
            except OSError as exc:
                last = repr(exc)
            time.sleep(poll_interval)
        raise SetupFailure(
            f"server not ready after {timeout:.1f}s (last: {last}); "
            f"log: {self.log_path}")

    def terminate_and_wait(self, timeout: float) -> float:
        """SIGTERM the group and wait. Returns seconds, or -1 on timeout."""
        started = time.monotonic()
        self._signal_group(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return -1.0
        return time.monotonic() - started

    def kill(self) -> None:
        """Unconditional teardown; safe to call more than once."""
        if self.proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        if not self._log.closed:
            self._log.close()

    def log_tail(self, lines: int = 40) -> str:
        try:
            with open(self.log_path, "r", errors="replace") as handle:
                return "".join(handle.readlines()[-lines:])
        except OSError:
            return ""


def server_command(args, port: int) -> list[str]:
    if args.launch_cmd:
        return [part.replace("{port}", str(port))
                for part in shlex.split(args.launch_cmd)]
    cmd = [sys.executable, "-m", "mirage.engine.launch_server",
           "--model", args.model,
           "--max-num-batched-requests", str(args.max_num_batched_requests),
           "--max-num-batched-tokens", str(args.max_num_batched_tokens),
           "--max-seq-length", str(args.max_seq_length),
           "--host", args.host,
           "--port", str(port)]
    if args.model_path:
        cmd += ["--model-path", args.model_path]
    return cmd


# ── cases ─────────────────────────────────────────────────────────────────────


@dataclass
class Context:
    host: str
    port: int
    request_timeout: float
    shutdown_timeout: float
    server: ServerHandle | None
    prompt_suffix: str = ""

    def prompt(self, text: str) -> str:
        """Every prompt the harness sends goes through here."""
        return f"{text} {self.prompt_suffix}".strip() if self.prompt_suffix \
            else text


def case_smoke(ctx: Context) -> str:
    text = one_completion(ctx, "Say hello in three words.", "smoke")
    return f"{len(text)} chars"


def case_stream(ctx: Context) -> str:
    payload = chat_payload(ctx.prompt("Count to three."), stream=True)
    try:
        cap = post_sse(ctx.host, ctx.port, CHAT_PATH, payload,
                       ctx.request_timeout)
    except (socket.timeout, TimeoutError):
        raise CaseFailure(f"stream stalled for {ctx.request_timeout:.1f}s")
    except OSError as exc:
        raise CaseFailure(f"transport error: {exc!r}")
    if cap.status != 200:
        raise CaseFailure(f"HTTP {cap.status}")
    if cap.errors:
        raise CaseFailure(f"stream reported errors: {cap.errors[:3]}")
    if not cap.content_chunks:
        raise CaseFailure("no non-empty content chunk in the stream")
    if cap.done_count != 1:
        raise CaseFailure(f"expected exactly one [DONE], got {cap.done_count}")
    if cap.events_after_done:
        raise CaseFailure(f"{cap.events_after_done} data event(s) after [DONE]")
    if not cap.eof:
        raise CaseFailure("stream body did not close cleanly")
    return f"{len(cap.content_chunks)} content chunks, 1 [DONE]"


def case_coldwarm(ctx: Context) -> str:
    prompt = "Name one primary color. Answer with one word."
    cold = one_completion(ctx, prompt, "coldwarm/cold")
    warm = one_completion(ctx, prompt, "coldwarm/warm")
    if cold != warm:
        raise CaseFailure(
            "the same prompt produced different text: "
            f"cold={cold[:120]!r} warm={warm[:120]!r}")
    return f"identical, {len(cold)} chars"


def _marker(index: int) -> str:
    return f"{MARKER_PREFIX}{index:02d}"


def audit_markers(results: dict[int, str],
                  markers: list[str]) -> tuple[list, list, list]:
    """Classify the markers in each reply.

    Splitting leakage from marker-shaped noise matters: a reply carrying
    *another live request's* marker is the cross-request bug this case exists
    to catch, while an unsent marker or a missing own marker is the model
    failing to echo.  Both are failures; only the first is a serving bug.
    """
    in_flight = set(markers)
    leaked: list[str] = []
    bogus: list[str] = []
    missing: list[str] = []
    for index, text in sorted(results.items()):
        found = set(MARKER_RE.findall(text))
        own = markers[index]
        if own not in found:
            missing.append(f"#{index} lost {own}: {text[:80]!r}")
        others = sorted((found & in_flight) - {own})
        if others:
            leaked.append(f"#{index} saw {others}")
        unknown = sorted(found - in_flight)
        if unknown:
            bogus.append(f"#{index} saw {unknown}")
    return leaked, bogus, missing


def case_isolation10(ctx: Context) -> str:
    markers = [_marker(i) for i in range(ISOLATION_N)]
    results: dict[int, str] = {}
    failures: dict[int, str] = {}
    lock = threading.Lock()

    def worker(index: int) -> None:
        prompt = f"Repeat this code exactly and stop: {markers[index]}"
        try:
            text = one_completion(ctx, prompt, f"isolation10/{index}")
        except CaseFailure as exc:
            with lock:
                failures[index] = str(exc)
            return
        except Exception as exc:
            with lock:
                failures[index] = f"unexpected {type(exc).__name__}: {exc}"
            return
        with lock:
            results[index] = text

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(ISOLATION_N)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=ctx.request_timeout + 30)
    with lock:
        for index in range(ISOLATION_N):
            if index not in results and index not in failures:
                failures[index] = ("request thread never finished"
                                   if threads[index].is_alive()
                                   else "no result recorded")

    leaked, bogus, missing = audit_markers(results, markers)
    problems = []
    if failures:
        problems.append("request failures: " + "; ".join(
            f"#{i}: {msg}" for i, msg in sorted(failures.items())))
    if leaked:
        problems.append("cross-request marker leakage: " + "; ".join(leaked))
    if bogus:
        problems.append("marker-shaped text nobody sent: " + "; ".join(bogus))
    if missing:
        problems.append("own marker not echoed: " + "; ".join(missing))
    if problems:
        raise CaseFailure(" | ".join(problems))
    return f"{ISOLATION_N}/{ISOLATION_N} kept their own marker"


def case_seq12(ctx: Context) -> str:
    for index in range(SEQ_N):
        one_completion(ctx, f"Say the word ok. Attempt {index}.",
                       f"seq12/{index}")
    return f"{SEQ_N}/{SEQ_N} sequential requests succeeded"


def case_shutdown(ctx: Context) -> str:
    if ctx.server is None:
        raise SkipCase("attached to an external server")
    elapsed = ctx.server.terminate_and_wait(ctx.shutdown_timeout)
    if elapsed < 0:
        raise CaseFailure(
            f"server still alive {ctx.shutdown_timeout:.1f}s after SIGTERM")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not port_is_listening(ctx.host, ctx.port):
            return f"exited in {elapsed:.2f}s, port {ctx.port} freed"
        time.sleep(0.2)
    raise CaseFailure(
        f"server exited in {elapsed:.2f}s but port {ctx.port} still accepts "
        "connections")


CASES: list[tuple[str, str, object]] = [
    ("a", "smoke", case_smoke),
    ("b", "stream", case_stream),
    ("c", "coldwarm", case_coldwarm),
    ("d", "isolation10", case_isolation10),
    ("e", "seq12", case_seq12),
    ("f", "shutdown", case_shutdown),
]


def select_cases(spec: str | None):
    """Resolve a ``--cases`` spec (letters or names) to cases, in fixed order."""
    if not spec:
        return list(CASES)
    by_key = {}
    for order, (letter, name, fn) in enumerate(CASES):
        by_key[letter] = by_key[name] = (order, letter, name, fn)
    chosen = set()
    for key in (part.strip() for part in spec.split(",") if part.strip()):
        if key not in by_key:
            raise SystemExit(f"unknown case {key!r}; known: "
                             + ", ".join(n for _, n, _ in CASES))
        chosen.add(by_key[key])
    return [entry[1:] for entry in sorted(chosen, key=lambda e: e[0])]


# ── reporting ─────────────────────────────────────────────────────────────────


def render_table(cases: list[dict]) -> str:
    head = ("case", "status", "seconds", "detail")
    rows = [(c["name"], c["status"].upper(), f"{c['duration_s']:.2f}",
             c["detail"]) for c in cases]
    widths = [max(len(head[i]), *(len(r[i]) for r in rows)) if rows
              else len(head[i]) for i in range(3)]
    out = ["  ".join(head[i].ljust(widths[i]) for i in range(3)) + "  " + head[3]]
    out.append("  ".join("-" * widths[i] for i in range(3)) + "  " + "-" * 8)
    for row in rows:
        out.append("  ".join(row[i].ljust(widths[i]) for i in range(3))
                   + "  " + row[3])
    return "\n".join(out)


def extract_summary(stdout: str) -> dict:
    """Parse the JSON summary out of a harness run's stdout."""
    try:
        body = stdout.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0]
    except IndexError:
        raise ValueError("no JSON summary found in harness output")
    return json.loads(body)


# ── driver ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online-serving harness for the MPK HTTP server.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0,
                        help="port for the launched server (0 = pick a free one)")
    parser.add_argument("--attach-port", type=int, default=None,
                        help="drive a server already listening on this port; "
                             "the shutdown case is skipped")
    parser.add_argument("--launch-cmd", default=None,
                        help="launch this command instead of the mirage server; "
                             "'{port}' is substituted (used by the self-test)")
    parser.add_argument("--max-num-batched-requests", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--ready-timeout", type=float, default=600.0,
                        help="seconds to wait for /openapi.json (the first "
                             "megakernel compile is slow)")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--shutdown-timeout", type=float, default=30.0)
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX,
                        help="appended to every prompt; the default is Qwen3's "
                             "soft switch that suppresses chain-of-thought, so "
                             "greedy replies stay inside --max-seq-length. "
                             "Pass '' for models without it.")
    parser.add_argument("--cases", default=None,
                        help="comma-separated subset, e.g. 'smoke,stream' or 'a,b'")
    parser.add_argument("--known-broken-concurrency", action="store_true",
                        help="treat isolation10 as expected-fail (upstream-HEAD "
                             "baseline, where cross-request leakage is known)")
    parser.add_argument("--server-log", default=None,
                        help="where to write the server's stdout/stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = select_cases(args.cases)

    attach = args.attach_port is not None
    port = args.attach_port if attach else (args.port or find_free_port(args.host))
    log_path = args.server_log or os.path.join(
        tempfile.gettempdir(), f"mpk_online_harness_{port}.log")

    summary = {
        "harness": "online_harness",
        "version": 1,
        "model": args.model,
        "host": args.host,
        "port": port,
        "mode": "attach" if attach else "launch",
        "known_broken_concurrency": bool(args.known_broken_concurrency),
        "prompt_suffix": args.prompt_suffix,
        "server_log": None if attach else log_path,
        "ready_seconds": None,
        "setup_error": None,
        "cases": [],
    }
    results: list[dict] = []
    server: ServerHandle | None = None
    started = time.monotonic()

    def record(name, status, duration, detail, expected_fail=False):
        results.append({
            "name": name, "status": status,
            "duration_s": round(duration, 3), "detail": detail,
            "expected_fail": expected_fail,
        })

    try:
        try:
            if attach:
                if not port_is_listening(args.host, port):
                    raise SetupFailure(
                        f"nothing is listening on {args.host}:{port}")
            else:
                cmd = server_command(args, port)
                print(f"[harness] launching: {' '.join(shlex.quote(c) for c in cmd)}",
                      file=sys.stderr, flush=True)
                server = ServerHandle(cmd, log_path, args.host, port)
                summary["ready_seconds"] = round(
                    server.wait_ready(args.ready_timeout), 2)
                print(f"[harness] server ready in "
                      f"{summary['ready_seconds']}s on port {port}",
                      file=sys.stderr, flush=True)
        except SetupFailure as exc:
            summary["setup_error"] = str(exc)
            for _, name, _fn in selected:
                record(name, "error", 0.0, "not run: setup failed")
            raise

        ctx = Context(host=args.host, port=port,
                      request_timeout=args.request_timeout,
                      shutdown_timeout=args.shutdown_timeout,
                      server=server, prompt_suffix=args.prompt_suffix)

        for _, name, fn in selected:
            expected_fail = (name == "isolation10"
                             and args.known_broken_concurrency)
            case_started = time.monotonic()
            try:
                detail = fn(ctx)
                status = "xpass" if expected_fail else "pass"
            except SkipCase as exc:
                status, detail = "skip", str(exc)
            except CaseFailure as exc:
                status = "xfail" if expected_fail else "fail"
                detail = str(exc)
            except Exception as exc:  # harness-visible server misbehaviour
                status = "xfail" if expected_fail else "fail"
                detail = f"unexpected {type(exc).__name__}: {exc}"
            record(name, status, time.monotonic() - case_started, detail,
                   expected_fail)
            print(f"[harness] {name}: {status.upper()} ({detail})",
                  file=sys.stderr, flush=True)
    except SetupFailure:
        pass
    finally:
        if server is not None:
            server.kill()

    counts = {key: 0 for key in
              ("pass", "fail", "xfail", "xpass", "skip", "error")}
    for result in results:
        counts[result["status"]] += 1
    summary["cases"] = results
    summary["counts"] = counts
    summary["duration_s"] = round(time.monotonic() - started, 2)

    if summary["setup_error"]:
        exit_code = EXIT_SETUP_FAILED
    elif counts["fail"] or counts["error"]:
        exit_code = EXIT_CASE_FAILED
    else:
        exit_code = EXIT_OK
    summary["exit_code"] = exit_code

    if summary["setup_error"]:
        print(f"[harness] SETUP FAILED: {summary['setup_error']}",
              file=sys.stderr, flush=True)
    if exit_code != EXIT_OK and server is not None:
        tail = server.log_tail()
        if tail:
            print("[harness] --- server log tail ---\n" + tail,
                  file=sys.stderr, flush=True)

    print(render_table(results) if results else "(no cases ran)")
    print(JSON_BEGIN)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(JSON_END)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
