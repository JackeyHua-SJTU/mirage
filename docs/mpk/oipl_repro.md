<!--
  AUDIENCE: an autonomous coding agent (or a careful engineer) tasked with reproducing, validating, or
  extending the OIPL prefix cache. This file is deliberately dense, imperative, and exhaustive. The
  human-facing rationale lives in online_serving_and_prefix_cache.md; this file is the runbook.
  Every command below was executed on the reference machine. Paths use ~/oipl as the workspace root.
-->

# OIPL Prefix Cache — Reproduction & Validation Runbook (agent-readable)

## 0. What this branch is

`oipl-base` = upstream `mpk` + PR #754 (row-lease lifecycle fix, cherry-picked) + the OIPL prefix cache
(ownership-inverted page lifecycle). Design & rationale: [`online_serving_and_prefix_cache.md`](online_serving_and_prefix_cache.md).
Benchmarks: [`oipl_benchmarks.md`](oipl_benchmarks.md). This file tells you how to build it, prove it, and
change it without breaking the invariants.

The one-sentence model: **the GPU is the sole page allocator, the CPU is the sole page freer, and the cache
domain is only the frozen pages of completed requests** — so the CPU's view of cache contents is never stale
by construction, and #666's consistency problem is dissolved rather than synchronised around.

## 1. Commit map (what each commit does; validate in this order)

```
8437b930 fix(serving): make online request lifecycle race-free        # PR #754 core (row lease, #754)
3b91ce01 fix(serving): bind launcher helpers consistently             # PR #754 followup
7da483d3 test(serving): online-serving harness + mock selftest        # L0/L2 test infra
622c8afe test(serving): random-interleaving protocol checker          # oipl_protocol_sim.py = executable spec
f8d0fb39 docs: mechanism map + OIPL design
7a1816af fix(paging): keep the last KV page on a page boundary        # PR-A: paging geometry (PS=64 prereq)
3b8a5117 feat(serving): shadow-export KV page lifecycle (P1a)         # channels wired, behaviour unchanged
5d3717f9 feat(serving): invert KV page ownership (P1b)                # the flip: GPU allocates, CPU frees
af7312da docs: fold P1 hardware findings
8dad87f0 feat(serving): prefix cache index (P2 core)                  # prefix_cache.py + unit tests
daa50ec6 refactor(serving): single-owner admission thread (P2)        # submit=enqueue; owner thread does all
ea76a34a feat(serving): match-at-publish prefix caching (P2)          # the wiring; --enable-prefix-cache
6863e566 docs: v2.1-c' backpressure + measured TTFT
e95c51d9 fix(serving): page-attributed backpressure (P3)              # backpressure fires only on page starvation
53ea03fc feat(engine): multi-turn chat passthrough (A3)               # messages list; /v1/completions raw
fa7f300b docs: A3 landing + multi-turn TTFT
507af799 test(serving): sglang-methodology serving benchmark
5ef1e0e6 test(serving): report ITL p99 in bench_serving
```

## 2. Environment contract (reference machine)

- **GPU**: NVIDIA B200, `sm_100a`. The qwen3 online path dispatches `tasks/blackwell/attention_sm100.cuh`.
  (Hopper `sm_90` also supported via a different attention header; **Ampere paged-attention has a known
  descoped vectorized-load bug — do not run the online path on Ampere without fixing it**, see §7.)
- **CUDA 12.8** (NOT the box's default 13.2 symlink), paired with **torch 2.11.0+cu128**. The megakernel
  compiler uses `shutil.which("nvcc")`, so `$CUDA_HOME/bin` must be first on PATH or you get nvcc 13.2
  against a cu128 torch.
- **No sudo required.** Build via `pip install -e . -v` (the INSTALL.md / ci-tests-qwen3.yml path), NOT
  `build_mirage_from_source.sh` (that helper needs a *system* libz3 under /usr/lib + apt). `pyproject.toml`
  pins `z3-solver==4.16.0.0` as a build requirement; append the venv's z3-solver `lib/` to
  `LD_LIBRARY_PATH` at runtime instead of symlinking into /usr/lib.
- **Submodules**: init `deps/cutlass` + `deps/json` only. `deps/z3` is intentionally NOT initialized (the
  build uses the pip z3-solver wheel).
- **Footprint**: venv ~7 GB + Qwen3-0.6B ~1.5 GB. Keep compiled megakernel artefacts OUTSIDE the repo tree
  (`--output-dir ~/oipl/mpk_out`) so `git status` stays clean.

### env.sh (source before every command)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
source ~/.cargo/env                      # rust toolchain (transpiler crates)
source ~/oipl/venv/bin/activate
export LD_LIBRARY_PATH="$(python -c 'import z3,os;print(os.path.join(os.path.dirname(z3.__file__),"lib"))'):$LD_LIBRARY_PATH"
export HF_HOME=~/oipl/hf
export CUDA_VISIBLE_DEVICES=0            # OVERRIDE per §3 device-selection discipline
```

### Build

```bash
git clone --branch mpk https://github.com/mirage-project/mirage ~/oipl/mirage
cd ~/oipl/mirage && git fetch origin pull/754/head:pr-754   # or fetch this fork's oipl-base directly
git submodule update --init deps/cutlass deps/json
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e . -v                       # ~a few min; drives cmake + make -j + cargo
python -c "import mirage; print('OK', mirage.__file__)"
```

`.cuh` (header-only) edits need **no rebuild** — the megakernel is nvcc-compiled from the working tree at
every server start (~38 s to ready). Only `src/*.cc` and `python/mirage/_cython/*.pyx` changes need
`pip install -e .` again.

## 3. Device-selection discipline (multi-tenant box)

The reference box is shared and other users grab GPUs without warning. **Before every server launch**:

```bash
for D in 0 4 6 3; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $D)" ]; then
    export CUDA_VISIBLE_DEVICES=$D; break; fi
done
```

`ModelRunner` does `torch.cuda.set_device(rank=0)`, so the chosen physical device maps to logical 0 under
the pinned `CUDA_VISIBLE_DEVICES`. **Never** `kill` a process you did not start. Run **one server at a time**.

## 4. The test pyramid — run bottom-up; each layer gates the next

### L0 — pure Python, no GPU, no network (seconds; run on every change)

```bash
pytest tests/serving_python/                    # prefix_cache, protocol sim, harness selftest,
                                                # template stability, tokenizer, bench smoke
python tests/serving_python/oipl_protocol_sim.py --sweep --events 300000 --seeds 2   # ~32 s
```

`oipl_protocol_sim.py` is **the executable spec**: it models the GPU sub-steps and the CPU owner thread as
atomic steps, randomly interleaves them, and re-checks invariants I1–I9 after *every* step. If you change
any protocol-adjacent logic, this must stay green — and if the code and the sim disagree, **the sim is
right until proven otherwise** (it caught 3 design defects before hardware ever ran). Invariants:
- I1 single page ownership across {free queue, row page table, comp-in-transit, cache, return-in-transit}.
- I2 `avail_uncommitted == (page_queue_tail − page_queue_head) − Σ reserved_remaining`.
- I3 no pop from an empty queue; ledger never negative.
- I4 `npp>0 ⇒ initial_step == npp·PS ≤ prompt_len−1` and imported KV == this request's own prefix.
- I5 comp-ring occupancy ≤ total_inflight; no ring overwrites; return ring can't overflow.
- I6 conservation at quiesce. I7 row lease. I8 attention-derived seq_len == true KV length.
- **I9** `free_estimate` is an UPPER bound on `avail_uncommitted` (the P3 backpressure attribution rests on
  this; if it were ever a lower bound, suppression could silence a real page-starved stall → wedge).

### L1 — SM100 paging kernel tests (one GPU; minutes)

```bash
cd tests/runtime_python/blackwell/sm100_paged_attention && python setup.py build_ext --inplace
python test_page_geometry.py            # PS=64 boundary + first_page_pos>0 vs pytorch reference
python test_page_boundary_testmode.py   # full MPK pipeline; asserts last_page_len=64 on a page-aligned prefill
python test_window_skip.py
```

These prove PR-A. **Pre-fix**, `test_page_boundary_testmode` publishes `last_page_len=0` and the kernel
produces nothing; that is the exact bug that makes PS=64 emit out-of-vocab tokens (HTTP 500 OverflowError)
the moment a sequence crosses a 64-token boundary.

### L2 — end-to-end serving harness (one GPU; ~1 min/run + 38 s server start)

```bash
# launch (background), then run the harness in --attach-port mode
python -m mirage.engine.launch_server --model Qwen/Qwen3-0.6B \
  --max-num-batched-requests 4 --max-num-batched-tokens 8 \
  --page-size 64 --max-seq-length 512 --max-num-pages 64 \
  --enable-prefix-cache --port 18500 --output-dir ~/oipl/mpk_out &
python tests/serving_python/online_harness.py --attach-port 18500        # 7 cases, fail-closed, JSON summary
```

Harness cases (select by NAME, letters shifted when `multiturn` was added): `smoke, stream, coldwarm,
isolation10, seq12, multiturn, shutdown`. `isolation10` sends 10 concurrent requests against 4 rows — it is
the regression test for the #754 row-lease (on upstream HEAD without #754 the server WEDGES here). Expected:
all green, exit 0, at both `--page-size 4096` and `--page-size 64`.

## 5. Config knobs and their hard constraints

| flag (launch_server) | default | meaning | constraint |
|---|---|---|---|
| `--max-num-batched-requests` (mbr) | 4 | buffer rows = max concurrent in-flight requests | **compile-time**; must be ≤ mbt or slots starve; grid_dim assertion depends on it |
| `--max-num-batched-tokens` (mbt) | 8 | per-iteration prefill token budget | **compile-time**; kernels tuned for small M — raising it can produce silently-wrong output on some paths (see upstream #740) |
| `--max-seq-length` | 512 | per-request token cap | sets token buffer stride; worst_need = `ceil(max_seq/page_size)` |
| `--page-size` | 4096 | KV page granularity | **for prefix caching set 64**; requires the PR-A paging fix |
| `--max-num-pages` | 16 | KV page pool | must be ≥ `mbr·ceil(max_seq/page_size)` or the server **refuses to launch** (ValueError) |
| `--enable-prefix-cache` | off | turn the cache on | inert flag when off; regression-tested that OFF == pre-cache behaviour |
| `--prefix-cache-pages` | 0 (=auto) | cache capacity | auto = `max_num_pages − max_pages_per_req` (leaves one worst-case admission slot) |

Changing any compile-time knob triggers a full megakernel recompile (~38 s); `load_mpk` kernel reuse only
matches an identical config (the `pinned_ring_capacity` + geometry macros are now validated — a stale
artefact dir must be recompiled, not reused).

Everything the cache adds is **Python-only + pinned-tensor plumbing**; no compute kernel changed. The GPU
diff is ~single-lane integer code inside `prepare_next_batch` (Step 0 return-ring drain, the reservation
ledger + admission gate, two-phase page-table fill). Compute tasks remain pure readers of device tensors.

## 6. Determinism {#determinism}

Greedy decoding is **nondeterministic run-to-run on long generations** — root-caused (with a discriminating
experiment) to the **SM100 split-K linear's bf16 TMA reduce-add**: `use_splitk = (target_cc == 100)`
([`models/qwen3/builder.py`](../../python/mirage/mpk/models/qwen3/builder.py), for `o_proj`/`down_proj`)
sums 16 K-partials in bf16 in scheduler-arrival order; bf16 is non-associative, so ~896 order-dependent
accumulations/token perturb the residual stream, amplified by a bf16 argmax over 152k logits (ULP 0.125 at
|logit|≈20 — a wide tie band). Short outputs stay stable; long ones fork.

**Determinism switch for byte-exact test gates** (measured: `splits=1` → 5/5 byte-identical vs 0/5 baseline,
same kernel/same TMA path, single variable): set the split factor to 1 in
[`models/utils.py`](../../python/mirage/mpk/models/utils.py) (`return (output_size // 128, 1, 1)`), OR
`use_splitk = False`. Cost: ~15 % decode throughput on long generations (but ~8 % *faster* on short
sequential requests — split-K's 16× task inflation is pure overhead there). **No test in this branch
compares generated text**; use timing/chunk-count assertions, or flip the determinism switch for a byte-exact
run. This is an upstream defect (draft issue exists), independent of the cache.

## 7. Known issues / boundaries (do not "fix" silently)

- **Ampere paged-attention vectorized load** ignores `first_page_pos` (scalar tail is correct). Descoped —
  the B200 online path does not dispatch it, but any model enabling `paged_attention_split_kv_layer` on B200
  runs the Ampere impl. Fix requires a 16-byte-alignment-safe scalar peel.
- **v1 caches prompt-token pages only.** Generated-token pages need cross-turn token-id tracking (detokenize
  → re-tokenize is not identity); out of scope. Multi-turn still benefits: turn k's answer is turn k+1's
  prompt, cached on completion, hit from turn k+2 (one-turn lag, standard).
- **Match granularity is a whole page.** No partial-page / copy-on-write sharing (that would mean writing a
  page another request references). `initial_step = npp·page_size` is always page-aligned; worst re-prefill
  is the ~63-token tail.
- **`on_export()` is dead code** — `insert()` subsumes reconciliation (calling both double-unpins; there is a
  `unpin_underflow` counter test guarding this). Safe to delete in a later cleanup.
- **Fault degradation is fail-closed**: a dead owner/drain thread makes every subsequent request 500 in ~0 ms
  (pre-existing `_raise_drain_error` latch) and freezes the ledger without corruption — availability failure,
  not correctness. A watchdog/self-heal is future hardening, not a safety prerequisite.

## 8. Benchmark reproduction

See [`oipl_benchmarks.md`](oipl_benchmarks.md) §Reproduce. Server geometry `--page-size 64 --max-seq-length
1024 --max-num-batched-requests 4 --max-num-batched-tokens 8 --max-num-pages 128`; drive
`bench_serving.py --attach-port P --workload both` with **identical `--seed`** across the ON/OFF pair; flip
`--enable-prefix-cache` on the *server*, not the bench. Expect TTFT p50 −66 % at low rate and served
throughput +78 % at 24 rps offered on the shared-prefix workload; parity on the random control.

## 9. Extending the design (research directions, from the design doc §4c)

- **C2 in-flight publish**: share a prefix that is still being written, using the per-row progress counter
  (`pinned_step`, already release-ordered with the KV writes) as the publish predicate `step ≥ (j+1)·PS`.
- **C3 preempt == evict**: `config.tokens[row]` already holds the full sequence, so recompute-preemption is a
  page relabel (running → cached), and `prepare_next_batch`'s end-of-graph barrier bounds preemption latency
  to one iteration.
- **C4 elastic task graph**: make attention fan-out a runtime-read count instead of a compile-time constant,
  so dispatch cost scales with *load* not *capacity* — the prerequisite for cheaply raising mbr.
