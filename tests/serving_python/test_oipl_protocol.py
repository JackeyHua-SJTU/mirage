"""Random-interleaving checker for the OIPL page-lifecycle protocol.

Runs the executable model in ``oipl_protocol_sim`` (see
docs/mpk/online_serving_and_prefix_cache.md §6/§7) under a randomized
scheduler and asserts the protocol invariants after every atomic step.  The
whole module is pure Python (no GPU, no torch) and stays under a minute so
it can gate every change to the protocol.

For a longer soak, run the module directly::

    python3 tests/serving_python/oipl_protocol_sim.py --sweep --events 300000 --seeds 4
"""

import pytest

import oipl_protocol_sim as sim_mod
from oipl_protocol_sim import (
    Config,
    InvariantViolation,
    LivenessViolation,
    Sim,
    run_config,
    sweep_configs,
)

EVENTS = 150_000
SEEDS = (0, 1)


@pytest.mark.parametrize("cfg", sweep_configs(), ids=lambda c: c.name)
@pytest.mark.parametrize("seed", SEEDS)
def test_random_interleavings(cfg, seed):
    """Every invariant holds after every atomic step, for every interleaving."""
    sim = run_config(cfg, seed=seed, events=EVENTS, verbose=False)
    assert sim.stats["completions"] == sim.stats["admissions"] > 0
    assert sim.stats["publishes"] == sim.stats["admissions"]


def test_prefix_hits_actually_happen():
    """A pool with cache headroom must produce hits, dedupes and evictions;
    otherwise the sweep above would be checking a cache that never fires."""
    sim = run_config(
        Config(name="roomy", total_pages=40, ring_cap=4, max_prompt=40),
        seed=0,
        events=EVENTS,
        verbose=False,
    )
    assert sim.stats["hits"] > 0
    assert sim.stats["hit_tokens"] > 0
    assert sim.stats["dedupes"] > 0
    assert sim.stats["evictions"] > 0
    # The same cached page in two concurrent rows' page tables (§7 S2).
    assert sim.stats["shared_imports"] > 0


def test_degenerate_prompt_shorter_than_a_page():
    """prompt_len < PAGE_SIZE: nothing is ever cacheable, npp stays 0."""
    sim = run_config(
        Config(name="short", total_pages=12, max_prompt=7),
        seed=0,
        events=EVENTS // 2,
        verbose=False,
    )
    assert sim.stats["completions"] > 0
    assert sim.stats["inserts"] == 0
    assert sim.stats["hits"] == 0


# ── handwritten adversarial schedules ─────────────────────────────────────


def test_toctou_evict_return_publish_between_step0_and_step4():
    """The review's TOCTOU (§10-1): the CPU evicts a page, pushes it to the
    return ring and publishes a request financed by it, all after the GPU has
    already run Step 0 for this iteration.  With the GPU ledger this must be
    safe: admission simply defers until Step 0 drains the page."""
    sim = sim_mod.scenario_toctou()
    assert sim.stats["admission_deferred"] > 0  # the deferral really happened
    assert sim.stats["completions"] == 16  # and no request starved


def test_duplicate_identical_prompts_dedupe_and_then_hit():
    sim = sim_mod.scenario_duplicate_prompts()
    assert sim.stats["dedupes"] >= 1
    assert sim.stats["hits"] >= 1


def test_block_matched_at_publish_survives_eviction():
    sim = sim_mod.scenario_evict_after_match()
    assert sim.stats["hits"] == 1


def test_full_pool_churn():
    """Pool == one worst-case request: every admission is financed by a page
    that had to travel completion ring -> CPU -> return ring -> free queue."""
    sim = sim_mod.scenario_full_pool_churn()
    assert sim.stats["completions"] == 24
    assert sim.stats["returns"] > 0


def test_wrong_prefix_block_is_not_imported():
    """A cached block with matching ids but a different preceding context must
    not be matched; only the parent link in the chained hash rejects it."""
    sim = sim_mod.scenario_wrong_prefix_hit()
    assert sim.stats["hits"] == 1


def test_id_compare_without_the_parent_link_produces_a_wrong_hit():
    """Documents why the lookup verifies the parent link: comparing only the
    matched block's own ids (§6.4 as written) admits a block from a colliding
    chain, and the imported KV then belongs to a different context."""
    with pytest.raises(InvariantViolation) as excinfo:
        sim_mod.scenario_wrong_prefix_hit(bug="ids_only_match")
    assert str(excinfo.value).startswith("I4")


def test_publish_pin_budget_keeps_admission_live():
    sim = sim_mod.scenario_publish_pin_deadlock()
    assert sim.stats["completions"] == 5


def test_unbounded_publish_pins_deadlock_admission():
    """Documents why the publish-time pin budget exists: without it, two
    published-but-unadmitted requests pin enough cached pages that the ring
    head (npp=0) can never be admitted.  Safe, but not live."""
    with pytest.raises(LivenessViolation):
        sim_mod.scenario_publish_pin_deadlock(bug="no_publish_clamp")


# ── the checker must actually bite ────────────────────────────────────────


@pytest.mark.parametrize(
    "bug,invariant",
    [
        ("no_ledger", "I3"),  # admission without the reservation gate
        ("self_free", "I1"),  # Step 1 exports AND self-frees
        ("no_publish_refcount", "I4"),  # match-at-publish takes no refcount
        ("no_refund", "I2"),  # completion drops the reservation refund
        ("raw_last_page_len", "I8"),  # the A10.1 page-boundary bug (§8)
    ],
)
def test_checker_catches_injected_bug(bug, invariant):
    cfg = Config(name="bug", total_pages=12, max_prompt=32, bug=bug)
    with pytest.raises(InvariantViolation) as excinfo:
        run_config(cfg, seed=3, events=EVENTS // 3, verbose=False)
    assert str(excinfo.value).startswith(invariant)


def test_violation_report_carries_the_event_trace():
    cfg = Config(name="bug", total_pages=12, max_prompt=32, bug="self_free")
    with pytest.raises(InvariantViolation) as excinfo:
        run_config(cfg, seed=3, events=EVENTS // 3, verbose=False)
    report = str(excinfo.value)
    assert "--- last events ---" in report
    assert "seed" in report and "gpu.step4 admit" in report
