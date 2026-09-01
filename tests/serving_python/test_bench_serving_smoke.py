"""Smoke-test ``bench_serving.py`` against ``mock_server.py``.

Proves the driver end to end without a GPU: prompts build reproducibly, a tiny
Poisson-arrival run over real HTTP/SSE completes, every metric is populated,
and the fail-closed exits fire.  ``mock_server.py`` needs no changes -- it
already streams one SSE data event per word and terminates with ``[DONE]``, so
the benchmark's per-event timestamping works against it as-is::

    pytest tests/serving_python/test_bench_serving_smoke.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from bench_serving import (
    EXIT_OK,
    EXIT_RUN_FAILED,
    EXIT_SETUP_FAILED,
    build_prompts,
    extract_summary,
    percentile,
)
from online_harness import find_free_port, port_is_listening

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench_serving.py")
MOCK = os.path.join(HERE, "mock_server.py")
HOST = "127.0.0.1"

SMOKE = ["--groups", "2", "--prompts-per-group", "3",
         "--prefix-tokens", "24", "--question-tokens", "6",
         "--request-rate", "8", "--request-timeout", "10", "--cooldown", "0"]


@pytest.fixture
def mock(request):
    """A mock server in ``request.param`` mode (default 'good'); yields a port."""
    mode = getattr(request, "param", "good")
    port = find_free_port(HOST)
    proc = subprocess.Popen(
        [sys.executable, MOCK, "--mode", mode, "--host", HOST,
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if port_is_listening(HOST, port):
                break
            assert proc.poll() is None, "mock server exited during startup"
            time.sleep(0.1)
        else:
            pytest.fail(f"mock server never listened on {port}")
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def run_bench(port: int, *extra: str):
    cmd = [sys.executable, BENCH, "--host", HOST, "--attach-port", str(port),
           *SMOKE, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    return proc, extract_summary(proc.stdout)


def test_both_workloads_run_and_every_metric_is_populated(mock):
    proc, summary = run_bench(mock, "--workload", "both")
    assert proc.returncode == EXIT_OK, proc.stderr[-3000:]
    assert [run["workload"] for run in summary["runs"]] == ["shared_prefix",
                                                            "random"]
    assert summary["output_tokens_are_sse_chunks"] is True
    for run in summary["runs"]:
        assert run["completed"] == 6 and run["failed"] == 0, run["errors"]
        assert run["total_output_tokens"] > 0
        assert run["request_throughput_rps"] > 0
        assert run["output_token_throughput_tps"] > 0
        assert run["ttft_ms"]["p99"] >= run["ttft_ms"]["p50"] > 0
        assert run["e2e_ms"]["p95"] >= run["e2e_ms"]["p50"] > 0
        # The mock streams several tokens per reply, so ITL/TPOT have samples.
        assert run["itl_ms"]["samples"] > 0
        assert run["tpot_ms"]["samples"] == 6
        assert len(run["per_request"]) == 6
        assert all(row["ok"] and row["status"] == 200
                   for row in run["per_request"])


def test_shared_prefix_shares_and_random_does_not():
    """The A/B contrast is in the prompts, so assert it without a server."""
    shared = build_prompts("shared_prefix", 2, 3, 24, 6, seed=0, ordered=True)
    control = build_prompts("random", 2, 3, 24, 6, seed=0, ordered=True)
    assert len({p.system for p in shared}) == 2, "one prefix per group"
    assert len({p.system for p in control}) == 6, "no prefix is reused"
    assert len({p.question for p in shared}) == 6, "questions stay distinct"
    # Same seed, same bytes: reruns and A/B halves are comparable.
    assert [p.system for p in shared] == [
        p.system for p in build_prompts("shared_prefix", 2, 3, 24, 6,
                                        seed=0, ordered=True)]
    assert [p.system for p in shared] != [
        p.system for p in build_prompts("shared_prefix", 2, 3, 24, 6,
                                        seed=1, ordered=True)]


def test_messages_carry_the_prefix_as_a_system_turn():
    prompt = build_prompts("shared_prefix", 1, 1, 12, 4, seed=0,
                           ordered=True)[0]
    messages = prompt.messages("/no_think")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == prompt.system
    assert messages[1]["content"].endswith("/no_think")


@pytest.mark.parametrize("mock", ["no_done"], indirect=True)
def test_a_stream_without_done_fails_the_run(mock):
    proc, summary = run_bench(mock, "--workload", "shared_prefix")
    assert proc.returncode == EXIT_RUN_FAILED
    run = summary["runs"][0]
    assert run["completed"] == 0 and run["failed"] == 6
    assert any("[DONE]" in err for err in run["errors"]), run["errors"]


def test_no_server_on_the_port_is_a_setup_failure():
    port = find_free_port(HOST)
    proc, summary = run_bench(port, "--workload", "shared_prefix")
    assert proc.returncode == EXIT_SETUP_FAILED
    assert "nothing is listening" in summary["setup_error"]
    assert summary["runs"] == []


def test_percentile_matches_linear_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0) == 1.0
    assert percentile(values, 50) == 2.5
    assert percentile(values, 100) == 4.0
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 95) == 7.0
