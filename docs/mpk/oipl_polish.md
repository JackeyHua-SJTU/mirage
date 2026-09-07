<!--
  AUDIENCE: the OIPL author and anyone reviewing the branch that sits on top of `oipl-base`.
  This file records three non-functional changes, what each one measured, and how each was proved
  not to alter behaviour. It also records two findings that were NOT fixed and belong to the
  reviewer's judgement. Design and rationale for OIPL itself live in
  online_serving_and_prefix_cache.md; the build and test recipes live in oipl_repro.md.
-->

# OIPL Polish — three non-functional changes on top of `oipl-base`

## 0. Scope

Three changes on branch `oipl-pinned-write`, based on `oipl-base` @ `0a6f4279`. **No feature is added
and no behaviour is intended to change.** None of them touch the OIPL protocol, the CPU/GPU memory
ordering, the page-ownership model, or any compute kernel.

```
 include/mirage/persistent_kernel/persistent_kernel.cuh | 18 +++++-
 python/mirage/mpk/online_pinned_runtime.py             | 54 +++++++++++------
 python/mirage/mpk/prefix_cache.py                      | 70 ++++++++++++++++++----
 3 files changed, 113 insertions(+), 29 deletions(-)
```

| # | Change | Kind | Pays off |
|---|---|---|---|
| 1 | Batched pinned-buffer writes | CPU owner-thread cost | With a large page pool, or many completions per tick |
| 2 | Lazy-heap eviction | CPU owner-thread cost | With a large resident cache |
| 3 | Batch-sized shared-memory snapshot | **Unblocks a configuration** | `--max-num-pages` above ~4 K |

Change 3 is the only one with a standalone effect at any tested configuration. Changes 1 and 2 are
the prerequisites that keep the CPU owner thread off the critical path once change 3 makes a large
pool reachable; measured end-to-end, they are neutral (§5).

Environment for every number below: one B200 (`sm_100a`), CUDA 12.8, torch 2.11.0+cu128,
Qwen3-0.6B, server geometry `--max-num-batched-requests 4 --max-num-batched-tokens 8 --page-size 64`.

---

## 1. Batched pinned-buffer writes

**File:** `python/mirage/mpk/online_pinned_runtime.py`

### Problem

The completion path already reads its page list as one slice —
`self._comp_pages[base:base + num_pages].tolist()` — but the two write paths were per-element:

```python
for offset, page in enumerate(prefix_pages):        # _publish_request_locked
    self._req_prefix_pages[base + offset] = page
for page in pages:                                  # _return_pages_locked
    self._page_return_ring[tail & self._return_mask] = page
```

Every `tensor[i] = v` on a pinned tensor is a full ATen dispatch. Measured on the reference machine:
**4.14 µs per element.** Both loops run on the owner thread, which also drives admission, and the
owner tick is 0.2 ms. A 32-page return therefore spent **132 µs — two thirds of a tick** — moving
32 integers.

### Change

Each buffer the owner tick *writes* is now held as a numpy view over the same page-locked memory
(`online_pinned_runtime.py:112`), and the writes are slice assignments
(`:368`, `:575`). The page-return ring wraps at most once — the occupancy argument in
`_return_pages_locked`'s docstring bounds a single write by the ring capacity — so two slices always
suffice; a batch that broke that bound now raises a broadcast error instead of silently overwriting
entries the GPU has not drained.

Only the view is kept. The tensor is not stored alongside it, so there is no second, slower way to
write the same buffer, and `reset()` zeroes through the view. The view owns the storage, so no extra
reference is needed.

### Measurement

Real buffer shapes (`RING_CAP=64`, `SLOTS=8`, `MAX_PAGES_PER_REQ=8`), best of 6 after warm-up:

| Path | Before | After | Speedup |
|---|---|---|---|
| `_return_pages_locked`, 8 pages (one request's page table) | 14.53 µs | 0.48 µs | 30× |
| `_return_pages_locked`, 32 pages (`MAX_EVICTIONS_PER_TICK`) | 56.97 µs | 0.91 µs | **62.6×** |
| `_publish_request_locked` (4 scalars + 5-page prefix array) | 15.87 µs | 0.44 µs | 36× |

Wrap-around correctness was checked directly: the new two-slice write produces a byte-identical ring
to the old loop across the wrap point.

---

## 2. Lazy-heap eviction

**File:** `python/mirage/mpk/prefix_cache.py`

### Problem

`PrefixCache.evict()` rescanned the whole resident set for each victim — O(count × N) with
`count` up to `MAX_EVICTIONS_PER_TICK = 32`.

### Change

`self._evictable` (`prefix_cache.py:129`) is a min-heap of `(lru_tick, seq, block)`. A block is pushed
when it *becomes* an unpinned leaf — on insert, on the unpin that drops its last pin, and on the
removal that takes its last child (`_mark_evictable`, `:469`) — so the heap is always a superset of
the evictable set. Entries are never removed when a block stops being evictable; `evict` validates on
pop (`:273`) against membership, refcount, children, and a stale `lru_tick`. `_compact_evictable`
(`:481`) rebuilds the heap once stale entries outnumber the live blocks, bounding its growth.

### Semantics are unchanged, not approximately

The eviction policy is still *leaf-first, minimum `lru_tick` among unpinned leaves*, and
`_remove_block` handing its parent to `_mark_evictable` is what walks a chain leaf-upwards inside a
single call — exactly what the repeated scan did. Two existing tests pin the exact order
(`test_evict_is_lru_leaf_first`, `test_match_refreshes_recency`) and both pass unchanged.

### Measurement

`evict(32)` against N resident blocks, best of 6 after warm-up. The right-hand column is a direct
comparison of the returned page lists:

| Resident blocks | Before | After | Speedup | Order identical |
|---|---|---|---|---|
| 56 (`--max-num-pages 64`) | 59.6 µs | 35.8 µs | 1.7× | yes |
| 128 (benchmark geometry) | 143.2 µs | 39.4 µs | 3.6× | yes |
| 1024 | 1010.4 µs | 66.1 µs | 15.3× | yes |
| 4096 | 4396.6 µs | 89.2 µs | **49.3×** | yes |
| 16384 | 28214.3 µs | 107.1 µs | **263×** | yes |

The old cost is linear in the pool; the new cost is essentially flat.

---

## 3. Batch-sized shared-memory snapshot

**File:** `include/mirage/persistent_kernel/persistent_kernel.cuh`

### Problem — a silent launch failure, not a slow path

Step 2's snapshot buffer was sized by the page pool:

```cpp
__shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
```

Only the one scheduler thread that runs `prepare_next_batch` ever touches it, but `__shared__` is
block-scoped, so **every block in the grid allocates it — workers included**. The page pool therefore
became a static-shared-memory cost paid across the whole megakernel.

At `--page-size 64`, a pool large enough to be worth caching into (8192 pages = 32 KB) pushes the
worker launch to 32 KB static + 201 KB dynamic = **233 KB, past the 228 KB SM limit**. In the split
worker/scheduler path the launches are:

```cpp
worker_kernel   <<<..., MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/, ...>>>   // fails
scheduler_kernel<<<..., 0 /*smem*/, ...>>>                                // succeeds
```

and that path has no launch check (the `cudaDeviceSynchronize()` error check exists only in the
non-split branch). The observable result was **not an error**: the scheduler spun against workers
that never started. GPU at 100 % utilisation, a clean log, and every request hanging forever.

### Change

Step 2 snapshots the page table of the *current batch*, never the pool. The bound is the admission
gate's own: Step 4 reserves `MPK_MAX_PAGES_PER_REQ` per request and there are at most
`MPK_MAX_NUM_BATCHED_REQUESTS` of them.

```cpp
__shared__ int
    smem_kv_indices[MPK_MAX_NUM_BATCHED_REQUESTS * MPK_MAX_PAGES_PER_REQ];   // :506
```

At `mbr=4, max_seq_length=512`: **32 ints = 128 bytes**, and independent of the pool from here on.
An `assert` on `num_pages_total` guards the bound; it compiles out under `NDEBUG`.

This differs from the remedy sketched in `online_serving_and_prefix_cache.md` §11-2 ("move the
snapshot to global, reusing the dead `paged_kv_indices_snapshot`"). Sizing it correctly keeps the
snapshot in shared memory — the pool dependency, not the residency, was the defect.

### Before / after

`--max-num-pages 8192`, everything else as above:

| | Before | After |
|---|---|---|
| Worker kernel | silent launch failure | starts |
| `POST /v1/completions` | hangs (180 s timeout, `code=000`) | **HTTP 200 in 0.72 s** |
| `online_harness.py` | cannot run | **6 pass / 1 skip / 0 fail** |
| Static shared memory | 32 KB | **128 B** |

---

## 4. Validation

| Layer | Command | Result |
|---|---|---|
| L0 | `pytest tests/serving_python/` | **106 passed** |
| L0 | `oipl_protocol_sim.py --sweep --events 300000 --seeds 2` | all green; the 12 config rows are **bit-identical** to the pre-change run |
| L1 | `test_page_geometry` / `test_page_boundary_testmode` / `test_window_skip` | **3/3**, same numerics (`max diff 0.0020`, `7.812e-03`) |
| L2 | `online_harness.py --attach-port P` | **6 pass / 1 skip**, per-case detail strings byte-identical, at both `--max-num-pages 64` and `8192` |

The protocol simulator is the strongest of these: 300 000 randomly interleaved events across 12
configurations produce the same `hit` / `evict` / `dedup` / `backpressure` counts before and after,
which is a behavioural identity check rather than a pass/fail assertion.

### Load benchmark (`bench_serving.py`, same GPU, same seed)

Server geometry per `oipl_repro.md` §8: `--page-size 64 --max-seq-length 1024 --max-num-pages 128`.
Bench: `--workload both --groups 4 --prompts-per-group 8 --prefix-tokens 320 --question-tokens 24
--seed 0`.

**At `--request-rate 1.0`:**

| Workload | Metric | Cache OFF | Cache ON | Δ |
|---|---|---|---|---|
| shared_prefix | TTFT p50 | 103.19 ms | 33.00 ms | **−68.0 %** |
| shared_prefix | throughput | 0.8143 rps | 0.8157 rps | +0.2 % |
| random (control) | TTFT p50 | 102.97 ms | 103.42 ms | +0.4 % |
| random (control) | throughput | 0.8143 rps | 0.8144 rps | +0.0 % |

**At `--request-rate 24` (offered ≫ served):**

| Workload | Metric | Cache OFF | Cache ON | Δ |
|---|---|---|---|---|
| shared_prefix | served throughput | 10.053 rps | 18.195 rps | **+81.0 %** |
| shared_prefix | TTFT p50 | 1006.7 ms | 392.6 ms | −61.0 % |
| random (control) | served throughput | 10.092 rps | 9.243 rps | **−8.4 %** |

Both headline claims in `oipl_benchmarks.md` reproduce with these changes in place: −66 % TTFT
(measured −68 %) and +78 % throughput (measured +81 %). The random-control result at saturation does
not match the documented parity — see §6.1; it is not caused by these changes.

---

## 5. Regression isolation

Because changes 1 and 2 touch the owner thread, the saturation benchmark was repeated against
**unpatched `oipl-base`** on the same GPU with the same seed, so the only variable is this branch.

| Configuration | shared_prefix tput | random tput | shared_prefix TTFT p50 | random TTFT p50 |
|---|---|---|---|---|
| Cache OFF | 10.053 | 10.092 | 1006.7 ms | 986.5 ms |
| Cache ON, `oipl-base` | 18.264 | 9.249 | 386.4 ms | 1116.4 ms |
| Cache ON, this branch | 18.195 | 9.243 | 392.6 ms | 1128.9 ms |

Patched versus unpatched: **−0.4 % / −0.1 % on throughput, +1.6 % / +1.1 % on TTFT p50** — run-to-run
noise. These changes are end-to-end neutral at this configuration, which is expected: with
`--max-num-pages 128` and `mbt 8` the server is GPU-bound on prefill, so tens of microseconds saved
on the owner thread are not observable. The microbenchmarks in §1 and §2 are where their effect is
visible, and change 3 is what makes a configuration where it would matter reachable at all.

---

## 6. Findings not fixed

Both belong to code outside this branch's scope and are recorded rather than changed.

### 6.1 The random control is not at parity under saturation

`oipl_README.md` states "clean parity for non-sharing traffic". That holds at `--request-rate 1.0`
(+0.4 %). At 24 rps offered it does not: served throughput on the random workload is **9.249 rps with
the cache on versus 10.092 with it off, −8.4 %**, with TTFT p50 +13 % and p95 +17 %.

This is present in unpatched `oipl-base` (§5), so it is a property of the feature, not of this branch.
A plausible mechanism, untested: `--workload both` runs `shared_prefix` and then `random` **against the
same server**, so the cache is full when the random phase begins; every random completion then pays
insert → over-capacity → evict → page-return on the owner thread, and at saturation that cost is no
longer hidden. If the published numbers were produced with a server restart between workloads, the two
measurements are not comparable and the parity claim should be scoped to the unsaturated case.

### 6.2 The split worker/scheduler launch is unchecked

`launch_persistent_kernel` checks `cudaDeviceSynchronize()` only in the non-split branch. In the split
branch a worker launch that exceeds the shared-memory limit is silently dropped and presents as a
permanent hang with a clean log (this is how §3's defect manifested). A `cudaGetLastError()` after each
launch would turn an unrecoverable hang into one line of diagnostics. This predates OIPL.

---

## 7. Reproducing the measurements

Environment per `oipl_repro.md` §2. Then:

```bash
# L0 / L1 / L2 — see oipl_repro.md §4
pytest tests/serving_python/
python tests/serving_python/oipl_protocol_sim.py --sweep --events 300000 --seeds 2

# the configuration change 3 unblocks
python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B \
  --max-num-batched-requests 4 --max-num-batched-tokens 8 \
  --page-size 64 --max-seq-length 512 --max-num-pages 8192 \
  --enable-prefix-cache --port 18503 --output-dir ~/oipl/mpk_big
python tests/serving_python/online_harness.py --attach-port 18503

# load benchmark: flip --enable-prefix-cache on the SERVER, keep --seed and the GPU identical
python tests/serving_python/bench_serving.py --attach-port P \
  --workload both --groups 4 --prompts-per-group 8 \
  --prefix-tokens 320 --question-tokens 24 \
  --request-rate 24 --seed 0 --request-timeout 600 --json-out sat.json
```

Use one GPU for the whole ON/OFF pair. On a shared machine a job landing mid-run perturbs the second
workload of `--workload both` in particular: an early attempt at the rate-1.0 pair straddled two GPUs
and reported the random control at +120 %, which the same-GPU repeat resolved to +0.4 %. The random
workload is the built-in validity check — if ON and OFF disagree there at rate 1.0, the pair is not
comparable.
