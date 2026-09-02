# OIPL Prefix Cache — Benchmark Data

All numbers measured on **NVIDIA B200** (`target_cc=100`), Qwen3-0.6B, branch `oipl-base`.
Driver: [`tests/serving_python/bench_serving.py`](../../tests/serving_python/bench_serving.py), whose
methodology mirrors sglang's `bench_serving` (Poisson arrivals via `expovariate`, seeded; TTFT = first
non-empty content SSE event; ITL = inter-event gaps after the first token; E2E = send→last event;
generated-shared-prefix dataset shape). Cache ON and OFF use the **same seed**, so arrival times and prompts
are identical pairs — every delta below is a true A/B, not two independent samples.

Server geometry for every cell: `--page-size 64 --max-seq-length 1024 --max-num-batched-requests 4
--max-num-batched-tokens 8 --max-num-pages 128`, fresh server per (workload, cache) cell.

## Headline

| | cache OFF | cache ON | change |
|---|---|---|---|
| **TTFT p50, shared prefix, 1 rps** | 106.8 ms | **36.6 ms** | **−65.7 %** |
| **E2E p50, shared prefix, 1 rps** | 121.2 ms | **51.2 ms** | **−57.8 %** |
| **Served throughput, 24 rps offered** | 9.6 req/s (saturated) | **17.0 req/s** | **+78 %** |
| **TTFT p50, 24 rps offered** | 1083 ms | **458 ms** | **−58 %** |
| Random (non-shared) control, all rates | — | — | parity ±2 % |

**Two regimes.** Below saturation the cache is a pure **latency** win (TTFT/E2E down ~60 %, throughput
unchanged — the server is idle enough that faster requests just finish sooner). At saturation it becomes a
**capacity** win: a hit turns ~6.4 pages of prefill into ~1.4, so each request holds one of the 4 buffer
rows for less than half as long, and the 4-row pool serves ~78 % more requests.

## W1 — shared prefix (4 groups × 8 prompts, ~320-word system prefix, ~24-word questions)

```
metric                      r1_off       r1_on      r3_off       r3_on
completed / failed          32 / 0       32 / 0      32 / 0       32 / 0
req throughput (req/s)        0.814        0.816       2.428        2.441
out throughput (tok/s)         7.96         7.62       23.75        23.41
mean output tokens              9.8          9.3         9.8          9.6
TTFT mean (ms)                107.6         45.2       109.2         47.4
TTFT p50  (ms)                106.8         36.6       106.2         36.2
TTFT p95  (ms)                108.3        105.8       120.9        107.4
TTFT p99  (ms)                126.5        108.1       159.8        110.5
ITL median (ms)                 2.2          2.2         2.2          2.2
ITL p95   (ms)                  2.5          2.5         2.5          2.5
ITL p99   (ms)                  2.7          2.7         2.6          2.7
TPOT mean (ms)                  1.7          1.7         1.8          1.8
E2E p50   (ms)                121.2         51.2       121.0         51.5
E2E p95   (ms)                128.1        120.8       142.3        122.6
```

Note p95 TTFT is at parity: each group's **first** request must miss (4 of 32 = 12.5 %), and p95 lands
exactly on those cold requests. Positions 1–7 within a group all drop to ~36 ms.

### Same dataset under saturation (added rates — 1/3 rps never built a queue)

```
metric                     r12_off      r12_on     r24_off      r24_on
req throughput (req/s)        9.320        9.655       9.550       17.021   <- OFF ceiling ~9.6
out throughput (tok/s)        90.29        98.66       89.53       164.89
TTFT mean (ms)               274.2         79.8       984.2        463.3
TTFT p50  (ms)               244.3         37.3      1083.0        457.6
TTFT p95  (ms)               481.2        224.0      1663.0        727.4
TTFT p99  (ms)               538.1        261.1      1691.5        761.0
ITL p99   (ms)                 3.0          3.0         4.8          3.8   <- only place a fairness effect shows
E2E p50   (ms)               258.8         58.7      1098.7        476.5
E2E p95   (ms)               502.4        245.7      1678.5        751.5
```

## W2 — random control (32 distinct ~360-word prompts, no shared prefix)

```
metric                      r1_off       r1_on      r3_off       r3_on
req throughput (req/s)        0.814        0.814       2.427        2.426
TTFT p50  (ms)               107.9        107.2       106.8        106.5
TTFT p95  (ms)               109.8        111.8       123.5        111.2
E2E p50   (ms)               121.1        120.9       122.0        120.7
```

Clean parity at every rate → **non-sharing traffic does not regress** with the cache enabled. (The control
is length-matched: each request gets its own distinct system prompt of equal word count, so the A/B isolates
*sharing* rather than confounding it with input length. Both halves still share the chat template's own
header, so a tiny hit is expected even here — reported, not hidden.)

## Interpretation and honest caveats

1. **TTFT win is arithmetic, not assumed.** A ~320-word prefix = 5 pages of 64 tokens;
   `blocks_matched/hits = 140/28 = exactly 5`. Prefill runs 8 tok/iter at ~2 ms/iter, so ~410 tokens of full
   prefill ≈ 51 iters ≈ 105 ms, vs ~90 unmatched tokens ≈ 36 ms. Observed: 106 ms → 36 ms.

2. **The fairness claim is mostly FALSE — this is the headline correction.** The hypothesis was that a
   newcomer's shorter prefill would smooth *neighbours'* decode ITL. It does not, at any rate up to 12 rps
   (ITL p95 deltas +0.4 % / +1.2 % / +1.9 % — zero within the 2 ms quantization). The reason is structural:
   MPK admits only 8 prefill tokens/iteration interleaved with decode, so a long prefill never monopolises
   the megakernel — it costs the *prefilling* request its own TTFT and does not starve concurrent decode.
   Only at 24 rps (row pool genuinely oversubscribed) does an ITL p99 tail effect appear (4.8 → 3.8 ms,
   −22 %). **Do not cite this work as an ITL fairness win at moderate load.**

3. **Saturation is where the capacity story lives.** Cache OFF saturates at ~9.3–9.6 req/s served
   regardless of 12 or 24 rps offered — a hard 4-row ceiling. Cache ON lifts that to 17.0 req/s because
   hit-shortened prefill halves row-occupancy time. This is also the first hardware exercise of P3's
   page-attributed backpressure under real load: **4881 suppressed / 0 fired** (all stalls were row
   starvation, pages stayed plentiful — exactly what the P3 fix is for). Conservation held exactly
   (128/128 pages) at all 12 quiesce points, zero anomalies.

### Measurement notes

- Output tokens are counted as SSE chunks; verified against the engine that one chunk = one decoded token
  (empty-text chunks skipped, so the count is a lower bound). `output_tokens_are_sse_chunks: true` in the JSON.
- `--prefix-tokens` / `--question-tokens` are **word** counts (no tokenizer in a stdlib driver); Qwen3
  tokenizes common English at ~1.0–1.3 tokens/word. Reported as `*_words` in the JSON to avoid false precision.
  For exact npp/hit evidence, pair with an in-process `page_stats` side-run at the same shapes.
- ITL has a **2 ms floor** = the CPU streaming monitor's poll interval; sub-2 ms gaps mean a batched poll,
  not a faster token. Surfaced as `itl_ms.poll_floor_ms` in the JSON.
- Greedy decoding is nondeterministic on long generations (SM100 split-K bf16 reduce-add — see
  [`oipl_repro.md`](oipl_repro.md#determinism) §Determinism); **no benchmark compares text**, only timings
  and chunk counts.

## Reproduce

```bash
# against an already-running server on port P (see oipl_repro.md for launch)
python tests/serving_python/bench_serving.py --attach-port P \
  --workload both --groups 4 --prompts-per-group 8 \
  --prefix-tokens 320 --question-tokens 24 \
  --request-rate 1.0 --seed 0 --json-out w1_r1_on.json --label w1_r1_on
# flip --enable-prefix-cache on the SERVER (not the bench) for the OFF/ON pair; keep --seed identical.
```

Raw JSON (`per_request` rows included for post-hoc analysis) and the paired-delta tables are the artefacts
this file summarises.
