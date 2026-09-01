"""Self-test: prove ``online_harness.py`` catches every planted failure.

Runs the harness against ``mock_server.py`` -- once in its correct mode, then
once per misbehaviour -- and asserts that the corresponding case turns red and
the process exits non-zero.  No GPU and no model are involved, so this runs
anywhere::

    pytest tests/serving_python/test_online_harness_selftest.py
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time

import pytest

from online_harness import (
    Context,
    EXIT_CASE_FAILED,
    audit_markers,
    EXIT_OK,
    EXIT_SETUP_FAILED,
    extract_summary,
    find_free_port,
    port_is_listening,
)

HERE = os.path.dirname(os.path.abspath(__file__))
HARNESS = os.path.join(HERE, "online_harness.py")
MOCK = os.path.join(HERE, "mock_server.py")
HOST = "127.0.0.1"


def run_harness(mode: str, cases: str, *extra: str, ready_timeout: str = "20",
                run_timeout: float = 120.0):
    """Run the harness against a mock in ``mode``; return (proc, summary)."""
    launch = (f"{shlex.quote(sys.executable)} {shlex.quote(MOCK)} "
              f"--mode {mode} --host {HOST} --port {{port}}")
    cmd = [sys.executable, HARNESS,
           "--launch-cmd", launch,
           "--host", HOST,
           "--cases", cases,
           "--ready-timeout", ready_timeout,
           "--request-timeout", "5",
           "--shutdown-timeout", "3",
           *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=run_timeout)
    return proc, extract_summary(proc.stdout)


def status_of(summary: dict, name: str) -> str:
    for case in summary["cases"]:
        if case["name"] == name:
            return case["status"]
    raise AssertionError(f"case {name!r} not in summary")


def assert_no_orphan(summary: dict) -> None:
    assert not port_is_listening(HOST, summary["port"]), (
        f"a server is still listening on port {summary['port']}")


def test_good_mode_all_cases_pass():
    proc, summary = run_harness("good", "a,b,c,d,e,f")
    assert proc.returncode == EXIT_OK, proc.stderr[-3000:]
    assert summary["counts"]["pass"] == 6, summary["cases"]
    assert summary["counts"]["fail"] == 0
    assert summary["exit_code"] == EXIT_OK
    assert_no_orphan(summary)


def test_ready_delay_is_polled_not_raced():
    """A server that is slow to answer /openapi.json is still picked up."""
    launch = (f"{shlex.quote(sys.executable)} {shlex.quote(MOCK)} "
              f"--mode good --host {HOST} --port {{port}} --ready-delay 2")
    proc = subprocess.run(
        [sys.executable, HARNESS, "--launch-cmd", launch, "--host", HOST,
         "--cases", "smoke", "--ready-timeout", "20", "--request-timeout", "5"],
        capture_output=True, text=True, timeout=120)
    summary = extract_summary(proc.stdout)
    assert proc.returncode == EXIT_OK, proc.stderr[-3000:]
    assert summary["ready_seconds"] >= 2.0
    assert status_of(summary, "smoke") == "pass"


@pytest.mark.parametrize("mode,case", [
    ("empty_content", "smoke"),
    ("hang", "smoke"),
    ("dup_done", "stream"),
    ("no_done", "stream"),
    ("chatter_after_done", "stream"),
    ("nondeterministic", "coldwarm"),
    ("wrong_marker", "isolation10"),
    ("no_shutdown", "shutdown"),
])
def test_planted_failure_is_caught(mode: str, case: str):
    proc, summary = run_harness(mode, case)
    assert status_of(summary, case) == "fail", (
        f"{mode} was not caught by {case}: {summary['cases']}")
    assert proc.returncode == EXIT_CASE_FAILED
    assert summary["cases"][0]["detail"], "a failure must explain itself"
    assert_no_orphan(summary)


def test_failure_detail_names_the_leaked_marker():
    _, summary = run_harness("wrong_marker", "isolation10")
    detail = summary["cases"][0]["detail"]
    assert "cross-request marker leakage" in detail
    assert "MK7Q2-" in detail


def test_known_broken_concurrency_downgrades_only_isolation():
    proc, summary = run_harness("wrong_marker", "a,d,e",
                                "--known-broken-concurrency")
    assert status_of(summary, "isolation10") == "xfail"
    assert status_of(summary, "smoke") == "pass"
    assert status_of(summary, "seq12") == "pass"
    assert proc.returncode == EXIT_OK, proc.stderr[-3000:]


def test_known_broken_concurrency_reports_xpass_when_healthy():
    proc, summary = run_harness("good", "d", "--known-broken-concurrency")
    assert status_of(summary, "isolation10") == "xpass"
    assert proc.returncode == EXIT_OK


def test_strict_mode_is_the_default():
    """Without the flag, the same leakage is a hard failure."""
    proc, summary = run_harness("wrong_marker", "d")
    assert status_of(summary, "isolation10") == "fail"
    assert proc.returncode == EXIT_CASE_FAILED


def test_never_ready_fails_closed_without_hanging():
    started = time.monotonic()
    proc, summary = run_harness("never_ready", "a,b,c,d,e,f",
                                ready_timeout="5", run_timeout=90)
    assert proc.returncode == EXIT_SETUP_FAILED
    assert summary["setup_error"]
    assert summary["counts"]["error"] == 6
    assert summary["counts"]["pass"] == 0
    assert time.monotonic() - started < 45
    assert_no_orphan(summary)


def test_attach_mode_skips_shutdown_and_leaves_server_running():
    port = find_free_port(HOST)
    mock = subprocess.Popen(
        [sys.executable, MOCK, "--mode", "good", "--host", HOST,
         "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while not port_is_listening(HOST, port):
            assert time.monotonic() < deadline, "mock never came up"
            time.sleep(0.1)
        proc = subprocess.run(
            [sys.executable, HARNESS, "--attach-port", str(port),
             "--host", HOST, "--cases", "a,f", "--request-timeout", "5"],
            capture_output=True, text=True, timeout=120)
        summary = extract_summary(proc.stdout)
        assert proc.returncode == EXIT_OK, proc.stderr[-3000:]
        assert status_of(summary, "smoke") == "pass"
        assert status_of(summary, "shutdown") == "skip"
        assert port_is_listening(HOST, port), (
            "attach mode must not kill a server it did not start")
    finally:
        mock.terminate()
        mock.wait(timeout=10)


def test_attach_to_dead_port_is_a_setup_failure():
    port = find_free_port(HOST)
    proc = subprocess.run(
        [sys.executable, HARNESS, "--attach-port", str(port), "--host", HOST,
         "--cases", "a"], capture_output=True, text=True, timeout=60)
    summary = extract_summary(proc.stdout)
    assert proc.returncode == EXIT_SETUP_FAILED
    assert status_of(summary, "smoke") == "error"


def test_prompt_suffix_is_applied_and_optional():
    def context(suffix):
        return Context(host=HOST, port=1, request_timeout=1.0,
                       shutdown_timeout=1.0, server=None, prompt_suffix=suffix)

    assert context("/no_think").prompt("hi") == "hi /no_think"
    assert context("").prompt("hi") == "hi"


def test_audit_markers_separates_leakage_from_noise():
    markers = ["MK7Q2-00", "MK7Q2-01", "MK7Q2-02"]
    leaked, bogus, missing = audit_markers({
        0: "OK MK7Q2-00",                # clean
        1: "MK7Q2-01 and MK7Q2-00",      # another live request's marker
        2: "MK7Q2-02 MK7Q2-77",          # marker-shaped, nobody sent it
    }, markers)
    assert leaked and "#1" in leaked[0] and "MK7Q2-00" in leaked[0]
    assert bogus and "#2" in bogus[0] and "MK7Q2-77" in bogus[0]
    assert not missing

    leaked, bogus, missing = audit_markers({0: "sorry, I forgot"}, markers)
    assert missing and "MK7Q2-00" in missing[0]
    assert not leaked and not bogus
