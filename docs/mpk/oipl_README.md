# OIPL: Prefix Caching for the MPK Online Server

Prefix caching for Mirage's persistent-megakernel (MPK) online serving path, via an **ownership-inverted
page lifecycle (OIPL)**: the GPU stays the sole KV-page *allocator*, the CPU becomes the sole page *freer*,
and the cache holds only the frozen pages of completed requests — so the CPU's cache view is never stale by
construction. This dissolves the CPU/GPU consistency problem that made prefix caching hard for a
never-exiting megakernel (upstream issue #666), rather than synchronising around it.

**Status: implemented and hardware-validated on B200** (branch `oipl-base`). On a shared-prefix online
workload the cache cuts TTFT p50 by ~66 % at low load and lifts served throughput by ~78 % at saturation,
with clean parity for non-sharing traffic.

## The three documents

| Document | Audience | Contents |
|---|---|---|
| [`online_serving_and_prefix_cache.md`](online_serving_and_prefix_cache.md) | **Human** | The design. Mechanism map of the existing serving protocol (CPU/GPU state, timing, memory ordering), the OIPL design and why, correctness argument, compatibility, staged rollout, and the full review record — including every draft claim that hardware or the protocol simulator overturned. |
| [`oipl_repro.md`](oipl_repro.md) | **Agent / engineer** | The runbook. Environment contract, build recipe, commit map, the L0→L2 test pyramid with exact commands and expected results, config constraints, the determinism switch, and known boundaries. Written to be reproduced. |
| [`oipl_benchmarks.md`](oipl_benchmarks.md) | Human | The numbers. sglang-methodology serving benchmark (Poisson arrivals, shared-prefix vs random A/B), the two win regimes, and the honest caveats (the fairness claim that did *not* hold, the 2 ms ITL floor). |

## What landed (branch `oipl-base` = upstream `mpk` + 18 commits)

1. **PR #754 cherry-picked** — race-free request lifecycle (row lease). Prerequisite; without it the server
   wedges permanently under concurrency.
2. **Paging geometry fix** — the `last_page_len` boundary bug that made `page_size=64` (a prefix-cache
   prerequisite) emit out-of-vocab tokens; plus SM100 kernel tests.
3. **Test infrastructure** — an online serving harness (fail-closed, mock-backed self-test) and a
   random-interleaving protocol simulator that is the executable spec (it caught 3 design defects pre-silicon).
4. **OIPL P1** — ownership inversion in two gated commits (shadow-mode channels, then the flip with a
   GPU single-lane reservation ledger + admission gate). Conservation exact, tight-pool non-deadlocking,
   fault-injection fail-closed, throughput at parity.
5. **OIPL P2** — the cache: `prefix_cache.py` (chained hash + parent-link verification + refcount +
   leaf-first LRU + pin budget), a single-owner admission thread, and match-at-publish wiring behind
   `--enable-prefix-cache`.
6. **OIPL P3** — page-attributed backpressure (fires only on genuine page starvation, not row starvation).
7. **A3** — multi-turn chat passthrough, so multi-turn HTTP conversations actually share a growing prefix.

## Provenance

This work was produced with heavy use of Claude Code (design, protocol simulation, staged implementation,
and hardware validation were multi-agent-orchestrated; every stage was gated on B200 validation before the
next began). The design doc's review record and the repro doc's invariant list are the audit trail.
