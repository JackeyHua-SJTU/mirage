#!/usr/bin/env python3
"""Online-serving benchmark for the MPK HTTP server.

Methodology is borrowed from sglang's ``python/sglang/bench_serving.py`` (and
its ``benchmark/datasets/generated_shared_prefix.py``): Poisson arrivals at a
target request rate, the same metric definitions (TTFT, ITL, TPOT, E2E latency,
request/output throughput with p50/p95/p99), and the same
generated-shared-prefix dataset shape -- ``--groups`` system prompts, each
reused by ``--prompts-per-group`` distinct short questions.  No sglang code is
vendored; this is a standard-library re-implementation shaped to our engine.

Where it deliberately differs, and why
    *No sampling parameters.*  Our engine has no ``max_tokens``: generation
    runs to EOS or ``--max-seq-length``.  Output length is bounded by the
    prompt instead -- short-answer questions plus Qwen3's ``/no_think`` soft
    switch -- so ``--prefix-tokens``/``--question-tokens`` shape the input and
    nothing pins the output.
    *Tiny request rates.*  Prefill advances 8 tokens per iteration
    (``--max-num-batched-tokens 8``), so a rate sglang would call idle already
    saturates us.  0.5-4 rps is the useful band.
    *Approximate token counts.*  There is no tokenizer here.  Prompts are built
    from common English words, which Qwen3 tokenizes at roughly 1.0-1.3 tokens
    per word, so the ``--*-tokens`` targets are word counts (reported as
    ``*_words`` in the JSON).  Output tokens need no tokenizer: the engine puts
    exactly one SSE content chunk on the wire per decoded token, so counting
    chunks counts tokens.  Chunks that decode to empty text (the end-of-stream
    sentinel, a partial-byte token) are skipped, which makes the count a lower
    bound rather than an over-count.
    *ITL has a 2 ms floor.*  The CPU monitor polls the megakernel every 2 ms, so
    several tokens can surface in one poll and appear as one 0 ms gap followed
    by one long one.  ITL percentiles are reported with that quantization noted,
    never as a sub-2 ms measurement.
    *Timing only.*  Greedy decode is not bit-reproducible over long generations
    (split-K reduction order), so this benchmark never compares text.

Workloads
    shared_prefix  G groups x P prompts; the group's system prompt is the
                   shared prefix, so the prefix cache (``--enable-prefix-cache``
                   on the server) can hit on every prompt after the first in a
                   group.
    random         the control: same input shape, but every request gets its
                   own distinct system prompt, so there is nothing to share.

Usage::

    # drive a server the runner already launched (the only supported mode)
    python tests/serving_python/bench_serving.py --attach-port 8000 \\
        --workload both --groups 8 --prompts-per-group 8 \\
        --prefix-tokens 512 --question-tokens 24 --request-rate 1.0

Output is a human table followed by a machine-parseable JSON summary fenced
between ``BEGIN_BENCH_JSON`` / ``END_BENCH_JSON``; :func:`extract_summary`
parses it back.  The process is fail-closed: nothing listening, or any request
that does not complete cleanly, exits non-zero (after printing the numbers).
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import random
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

from online_harness import DEFAULT_PROMPT_SUFFIX, port_is_listening

CHAT_PATH = "/v1/chat/completions"

JSON_BEGIN = "BEGIN_BENCH_JSON"
JSON_END = "END_BENCH_JSON"

# The CPU monitor polls for new tokens every 2 ms, so no inter-token gap below
# this is a real measurement -- see the module docstring.
ITL_POLL_FLOOR_MS = 2.0

WORKLOADS = ("shared_prefix", "random")

EXIT_OK = 0
EXIT_RUN_FAILED = 1
EXIT_SETUP_FAILED = 2


class SetupFailure(Exception):
    """The benchmark could not be started at all."""


# ── prompt text ───────────────────────────────────────────────────────────────

# Common words only: they tokenize stably (~1 token each with a leading space),
# so a word count is a usable stand-in for a token count and the same seed
# reproduces byte-identical prompts on any box.
WORDS = (
    "the field team keeps a short record of every shipment that leaves the "
    "north depot before noon and marks which crates were checked by hand "
    "while the rest wait for the next driver to arrive with a clean manifest "
    "and a signed copy of the weekly summary that the office files under the "
    "current quarter so that later questions about weight route or delay can "
    "be answered from one place without calling the warehouse again during a "
    "busy week when the phone line is already full of other requests"
).split()

TOPICS = (
    "the depot", "the manifest", "the driver", "the crate", "the route",
    "the weight", "the delay", "the office", "the quarter", "the summary",
)


def _words(rng: random.Random, count: int) -> str:
    """``count`` words of plain English, punctuated into short sentences."""
    if count <= 0:
        return ""
    out: list[str] = []
    remaining = count
    while remaining > 0:
        length = min(remaining, rng.randint(8, 14))
        chunk = [rng.choice(WORDS) for _ in range(length)]
        chunk[0] = chunk[0].capitalize()
        out.append(" ".join(chunk) + ".")
        remaining -= length
    return " ".join(out)


def system_prompt(rng: random.Random, tag: str, word_count: int) -> str:
    """A prefix that is stable for its tag and distinct from every other tag."""
    return (f"You are assistant {tag}. Answer every question in at most five "
            f"words. Reference notes follow. {_words(rng, word_count)}")


def question_text(rng: random.Random, index: int, word_count: int) -> str:
    return (f"Note {index}: {_words(rng, word_count)} "
            f"In at most five words, name {rng.choice(TOPICS)}.")


@dataclass
class Prompt:
    index: int
    group: int
    system: str
    question: str

    def messages(self, suffix: str) -> list[dict]:
        user = f"{self.question} {suffix}".strip() if suffix else self.question
        return [{"role": "system", "content": self.system},
                {"role": "user", "content": user}]


def build_prompts(workload: str, groups: int, per_group: int,
                  prefix_words: int, question_words: int, seed: int,
                  ordered: bool) -> list[Prompt]:
    """G x P prompts; ``shared_prefix`` reuses one system prompt per group.

    ``random`` is the control: it keeps the input shape and the total token
    count identical but gives every request its own system prompt, so no two
    requests share a prefix beyond the chat template's own header.
    """
    prompts: list[Prompt] = []
    index = 0
    for group in range(groups):
        # String seeds: ``random.Random`` hashes them itself, so unlike a tuple
        # of ``hash()`` values these reproduce across processes.
        group_rng = random.Random(f"{seed}:prefix:{group}")
        shared = system_prompt(group_rng, f"G{group}", prefix_words)
        for slot in range(per_group):
            item_rng = random.Random(f"{seed}:item:{group}:{slot}")
            if workload == "shared_prefix":
                system = shared
            else:
                system = system_prompt(item_rng, f"G{group}P{slot}",
                                       prefix_words)
            prompts.append(Prompt(
                index=index, group=group, system=system,
                question=question_text(item_rng, index, question_words)))
            index += 1
    if not ordered:
        random.Random(f"{seed}:order").shuffle(prompts)
    return prompts


# ── one streamed request ──────────────────────────────────────────────────────


@dataclass
class Result:
    index: int
    group: int
    status: int = 0
    done: bool = False
    error: str | None = None
    send_ts: float = 0.0
    end_ts: float = 0.0
    event_ts: list[float] = field(default_factory=list)
    chars: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.done and bool(self.event_ts)

    @property
    def output_tokens(self) -> int:
        """One SSE content chunk per decoded token, so chunks are tokens."""
        return len(self.event_ts)

    @property
    def ttft(self) -> float:
        return self.event_ts[0] - self.send_ts

    @property
    def e2e(self) -> float:
        return self.end_ts - self.send_ts

    @property
    def itls(self) -> list[float]:
        """Gaps between consecutive tokens, i.e. everything after the first."""
        return [b - a for a, b in zip(self.event_ts, self.event_ts[1:])]


def _delta_text(event: object) -> str | None:
    if not isinstance(event, dict):
        return None
    try:
        choice = event["choices"][0]
    except (KeyError, IndexError, TypeError):
        return None
    text = (choice.get("delta") or {}).get("content")
    if text is None:
        text = choice.get("text")
    return text if isinstance(text, str) else None


def stream_one(host: str, port: int, payload: dict, timeout: float,
               result: Result) -> Result:
    """POST a streaming chat completion, timestamping every SSE data event."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    try:
        # Connect first so the TCP handshake is not billed to TTFT.
        conn.connect()
    except OSError as exc:
        result.send_ts = result.end_ts = time.monotonic()
        result.error = f"connect failed: {exc!r}"
        conn.close()
        return result
    result.send_ts = time.monotonic()
    try:
        conn.request("POST", CHAT_PATH, body=body, headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        })
        resp = conn.getresponse()
        result.status = resp.status
        if resp.status != 200:
            raw = resp.read()
            result.error = f"HTTP {resp.status}: {raw[:200]!r}"
            return result
        while True:
            raw = resp.readline()
            now = time.monotonic()
            if not raw:
                result.error = "stream ended without [DONE]"
                break
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                result.done = True
                result.end_ts = now
                break
            try:
                event = json.loads(data)
            except ValueError:
                result.error = f"non-JSON SSE payload: {data[:200]!r}"
                break
            if isinstance(event, dict) and event.get("error"):
                result.error = f"server error event: {str(event['error'])[:200]}"
                break
            text = _delta_text(event)
            if text:
                result.event_ts.append(now)
                result.chars += len(text)
    except (socket.timeout, TimeoutError):
        result.error = f"no progress within {timeout:.1f}s"
    except OSError as exc:
        result.error = f"transport error: {exc!r}"
    finally:
        conn.close()
    if not result.end_ts:
        result.end_ts = result.event_ts[-1] if result.event_ts else time.monotonic()
    return result


# ── statistics ────────────────────────────────────────────────────────────────


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile, matching ``numpy.percentile``."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(results: list[Result], duration: float) -> dict:
    """sglang's metric definitions, in milliseconds, over the completed runs."""
    good = [r for r in results if r.ok]
    ttfts = [r.ttft for r in good]
    e2es = [r.e2e for r in good]
    itls = [itl for r in good for itl in r.itls]
    tpots = [(r.e2e - r.ttft) / (r.output_tokens - 1)
             for r in good if r.output_tokens > 1]
    total_output = sum(r.output_tokens for r in good)
    return {
        "completed": len(good),
        "failed": len(results) - len(good),
        "duration_s": round(duration, 3),
        "request_throughput_rps": round(len(good) / duration, 4)
        if duration > 0 else 0.0,
        "output_token_throughput_tps": round(total_output / duration, 3)
        if duration > 0 else 0.0,
        "total_output_tokens": total_output,
        "mean_output_tokens": round(mean([float(r.output_tokens)
                                          for r in good]), 2),
        "ttft_ms": {
            "mean": round(mean(ttfts) * 1e3, 2),
            "p50": round(percentile(ttfts, 50) * 1e3, 2),
            "p95": round(percentile(ttfts, 95) * 1e3, 2),
            "p99": round(percentile(ttfts, 99) * 1e3, 2),
        },
        "itl_ms": {
            "median": round(percentile(itls, 50) * 1e3, 2),
            "p95": round(percentile(itls, 95) * 1e3, 2),
            "samples": len(itls),
            "poll_floor_ms": ITL_POLL_FLOOR_MS,
        },
        "tpot_ms": {"mean": round(mean(tpots) * 1e3, 2), "samples": len(tpots)},
        "e2e_ms": {
            "p50": round(percentile(e2es, 50) * 1e3, 2),
            "p95": round(percentile(e2es, 95) * 1e3, 2),
        },
    }


# ── one workload run ──────────────────────────────────────────────────────────


def arrival_offsets(count: int, rate: float, rng: random.Random) -> list[float]:
    """Poisson arrivals: exponential gaps, first request at t=0 (sglang)."""
    offsets = []
    clock = 0.0
    for _ in range(count):
        offsets.append(clock)
        if rate > 0:
            clock += rng.expovariate(rate)
    return offsets


def run_workload(args, workload: str) -> dict:
    prompts = build_prompts(workload, args.groups, args.prompts_per_group,
                            args.prefix_tokens, args.question_tokens,
                            args.seed, args.ordered)
    offsets = arrival_offsets(len(prompts), args.request_rate,
                              random.Random(args.seed))
    results: list[Result | None] = [None] * len(prompts)
    threads: list[threading.Thread] = []

    def worker(prompt: Prompt) -> None:
        result = Result(index=prompt.index, group=prompt.group)
        payload = {"messages": prompt.messages(args.prompt_suffix),
                   "stream": True}
        try:
            stream_one(args.host, args.port, payload, args.request_timeout,
                       result)
        except Exception as exc:  # a client bug must not vanish into a thread
            result.error = f"unexpected {type(exc).__name__}: {exc}"
        results[prompt.index] = result

    print(f"[bench] {workload}: {len(prompts)} requests at "
          f"{args.request_rate} rps", file=sys.stderr, flush=True)
    started = time.monotonic()
    for offset, prompt in zip(offsets, prompts):
        delay = (started + offset) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        thread = threading.Thread(target=worker, args=(prompt,), daemon=True)
        thread.start()
        threads.append(thread)
    join_deadline = time.monotonic() + args.request_timeout + 60.0
    for thread in threads:
        thread.join(timeout=max(1.0, join_deadline - time.monotonic()))
    for prompt in prompts:
        if results[prompt.index] is None:
            results[prompt.index] = Result(
                index=prompt.index, group=prompt.group,
                send_ts=started, end_ts=time.monotonic(),
                error="request thread never finished")
    finished = [r for r in results if r is not None]
    last_end = max((r.end_ts for r in finished if r.ok), default=time.monotonic())
    duration = last_end - started

    summary = summarize(finished, duration)
    summary.update({
        "workload": workload,
        "requests": len(prompts),
        "groups": args.groups,
        "prompts_per_group": args.prompts_per_group,
        "prefix_words": args.prefix_tokens,
        "question_words": args.question_tokens,
        "request_rate_rps": args.request_rate,
        "seed": args.seed,
        "errors": sorted({r.error for r in finished if r.error})[:10],
        "per_request": [
            {"index": r.index, "group": r.group, "status": r.status,
             "ok": r.ok, "output_tokens": r.output_tokens,
             "chars": r.chars,
             "ttft_ms": round(r.ttft * 1e3, 2) if r.event_ts else None,
             "e2e_ms": round(r.e2e * 1e3, 2) if r.ok else None,
             "error": r.error}
            for r in sorted(finished, key=lambda r: r.index)
        ],
    })
    return summary


# ── reporting ─────────────────────────────────────────────────────────────────

ROWS = (
    ("completed / failed", lambda s: f"{s['completed']} / {s['failed']}"),
    ("duration (s)", lambda s: f"{s['duration_s']:.2f}"),
    ("request throughput (req/s)", lambda s: f"{s['request_throughput_rps']:.3f}"),
    ("output throughput (tok/s)",
     lambda s: f"{s['output_token_throughput_tps']:.2f}"),
    ("mean output tokens", lambda s: f"{s['mean_output_tokens']:.1f}"),
    ("TTFT mean (ms)", lambda s: f"{s['ttft_ms']['mean']:.1f}"),
    ("TTFT p50 (ms)", lambda s: f"{s['ttft_ms']['p50']:.1f}"),
    ("TTFT p95 (ms)", lambda s: f"{s['ttft_ms']['p95']:.1f}"),
    ("TTFT p99 (ms)", lambda s: f"{s['ttft_ms']['p99']:.1f}"),
    ("ITL median (ms)", lambda s: f"{s['itl_ms']['median']:.1f}"),
    ("ITL p95 (ms)", lambda s: f"{s['itl_ms']['p95']:.1f}"),
    ("TPOT mean (ms)", lambda s: f"{s['tpot_ms']['mean']:.1f}"),
    ("E2E p50 (ms)", lambda s: f"{s['e2e_ms']['p50']:.1f}"),
    ("E2E p95 (ms)", lambda s: f"{s['e2e_ms']['p95']:.1f}"),
)


def render_table(runs: list[dict]) -> str:
    label = "metric"
    width = max([len(label)] + [len(name) for name, _ in ROWS])
    columns = [run["workload"] for run in runs]
    colw = [max(12, len(name)) for name in columns]
    out = [label.ljust(width) + "  "
           + "  ".join(c.rjust(w) for c, w in zip(columns, colw))]
    out.append("-" * width + "  " + "  ".join("-" * w for w in colw))
    for name, fmt in ROWS:
        cells = [fmt(run).rjust(w) for run, w in zip(runs, colw)]
        out.append(name.ljust(width) + "  " + "  ".join(cells))
    out.append("")
    out.append(f"output tokens = SSE content chunks (our stream is one chunk "
               f"per decoded token); ITL is quantized by the "
               f"{ITL_POLL_FLOOR_MS:g} ms CPU-monitor poll, so sub-"
               f"{ITL_POLL_FLOOR_MS:g} ms gaps mean a batched poll, not a "
               f"faster token.")
    return "\n".join(out)


def extract_summary(stdout: str) -> dict:
    """Parse the JSON summary out of a benchmark run's stdout."""
    try:
        body = stdout.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0]
    except IndexError:
        raise ValueError("no JSON summary found in benchmark output")
    return json.loads(body)


# ── driver ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="sglang-methodology serving benchmark for the MPK server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--attach-port", type=int, required=True,
                        help="port of a server the runner already launched; "
                             "this benchmark never launches one itself")
    parser.add_argument("--workload", default="both",
                        choices=(*WORKLOADS, "both"),
                        help="'both' runs shared_prefix then random (the "
                             "no-sharing control) against the same server")
    parser.add_argument("--groups", type=int, default=8,
                        help="number of system-prompt groups")
    parser.add_argument("--prompts-per-group", type=int, default=8,
                        help="distinct questions sharing each group's prefix")
    parser.add_argument("--prefix-tokens", type=int, default=512,
                        help="target system-prompt length, in words (~tokens)")
    parser.add_argument("--question-tokens", type=int, default=24,
                        help="target question length, in words (~tokens)")
    parser.add_argument("--request-rate", type=float, default=1.0,
                        help="Poisson arrival rate in req/s; prefill runs at 8 "
                             "tokens/iteration, so 0.5-4 is the useful band")
    parser.add_argument("--seed", type=int, default=0,
                        help="seeds both the prompts and the arrival process")
    parser.add_argument("--ordered", action="store_true",
                        help="send group-major instead of shuffled")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="seconds between the two runs of --workload both")
    parser.add_argument("--prompt-suffix", default=DEFAULT_PROMPT_SUFFIX,
                        help="appended to every question; the default is "
                             "Qwen3's soft switch that suppresses "
                             "chain-of-thought so replies stay short. "
                             "Pass '' for models without it.")
    parser.add_argument("--json-out", default=None,
                        help="also write the JSON summary to this path")
    parser.add_argument("--label", default=None,
                        help="free-form tag recorded in the JSON summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.port = args.attach_port
    workloads = list(WORKLOADS) if args.workload == "both" else [args.workload]

    summary = {
        "bench": "bench_serving",
        "version": 1,
        "methodology": "sglang bench_serving (Poisson arrivals, "
                       "generated-shared-prefix dataset, TTFT/ITL/TPOT/E2E)",
        "host": args.host,
        "port": args.port,
        "label": args.label,
        "prompt_suffix": args.prompt_suffix,
        "output_tokens_are_sse_chunks": True,
        "itl_poll_floor_ms": ITL_POLL_FLOOR_MS,
        "setup_error": None,
        "runs": [],
    }
    started = time.monotonic()
    try:
        if not port_is_listening(args.host, args.port):
            raise SetupFailure(
                f"nothing is listening on {args.host}:{args.port}")
        for position, workload in enumerate(workloads):
            if position:
                time.sleep(args.cooldown)
            summary["runs"].append(run_workload(args, workload))
    except SetupFailure as exc:
        summary["setup_error"] = str(exc)
        print(f"[bench] SETUP FAILED: {exc}", file=sys.stderr, flush=True)

    summary["duration_s"] = round(time.monotonic() - started, 2)
    if summary["setup_error"]:
        exit_code = EXIT_SETUP_FAILED
    elif any(run["failed"] or not run["completed"] for run in summary["runs"]):
        exit_code = EXIT_RUN_FAILED
    else:
        exit_code = EXIT_OK
    summary["exit_code"] = exit_code

    print(render_table(summary["runs"]) if summary["runs"] else "(no runs)")
    print(JSON_BEGIN)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(JSON_END)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
