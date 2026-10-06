# off-mgo Implementation Plan

## 0. Goal

Build a clean experimental runtime for **offline, module-batched, multi-GPU MoE CPU offloading**.

The first research question is deliberately narrow:

> When many decode attention micro-batches are accumulated into one large routed MoE batch, does per-batch multi-rank expert materialization reduce layer completion time by balancing expert compute without paying too much activation communication?

This repository starts from measurement, not from a fixed novelty claim.

### Non-goals for the first implementation

- no persistent expert cache;
- no eviction policy;
- no expert substitution or rerouting;
- no future-route prediction;
- no speculative prefetch;
- no expert replication in the first physical baseline;
- no accuracy-changing approximation.

Every routed expert is executed exactly. GPU expert memory is treated as a **layer-scoped staging/materialization buffer**, not a cache.

---

## 1. Execution model

For decode layer `l`:

1. Run attention/router for several local attention micro-batches on every rank.
2. Buffer the post-attention hidden states, top-k expert IDs, weights, and source-rank metadata.
3. After `M` attention micro-batches have accumulated, form one global routed MoE batch.
4. Build the global expert-demand matrix `D[e, r]`, where `D[e, r]` is the number of routes from source rank `r` to expert `e`.
5. Choose a single execution rank for every active expert.
6. Materialize each active expert once from CPU to its assigned GPU rank.
7. Dispatch token activations to the assigned rank(s).
8. Execute rank-local experts as grouped GEMMs.
9. Return exact weighted expert partials to token-origin ranks and combine in native expert order.
10. Release/reuse the layer staging slots and continue to layer `l+1`.

The first characterization mode is intentionally serialized:

```
routing aggregation
  -> placement
  -> required expert H2D
  -> global H2D barrier
  -> forward dispatch
  -> expert compute
  -> global compute barrier
  -> return/combine
```

This makes H2D, communication, and expert execution independently measurable. Overlap is added only after the basic headroom is established.

---

## 2. Batching terminology

Use separate names for the three batching dimensions.

- `R`: number of GPU ranks.
- `b_a`: **local attention micro-batch size per rank** (decode tokens / active sequences processed by one attention call).
- `M`: number of attention micro-batches accumulated before one MoE execution.
- `B = R * b_a * M`: **global aggregated MoE token batch**.
- `k`: router top-k.
- `E`: number of routed experts in a layer.
- average routed rows per expert:
  `n_bar = B * k / E`.

For Qwen3-30B-A3B-Instruct-2507 used by MGO:

- `E = 128`;
- `k = 8`;
- therefore `n_bar = B / 16`.

### Primary R4 sweep

Use physical GPUs already reserved for this project: **0,1,4,5 only**.

Primary attention setting:

- `R = 4`;
- `b_a = 128 tokens/rank`.

Then one attention wave contributes `4 * 128 = 512` global decode tokens.

| M (attention waves) | local accumulated tokens/rank | global B | avg rows/expert |
|---:|---:|---:|---:|
| 1 | 128 | 512 | 32 |
| 2 | 256 | 1,024 | 64 |
| 4 | 512 | 2,048 | 128 |
| 8 | 1,024 | 4,096 | 256 |
| 16 | 2,048 | 8,192 | 512 |
| 32 | 4,096 | 16,384 | 1,024 |
| 64 | 8,192 | 32,768 | 2,048 |
| 128 (stress) | 16,384 | 65,536 | 4,096 |

Do **not** assume that one of these values is optimal. The purpose of the sweep is to find the H2D/compute/communication crossover on the current H100 system.

### Secondary attention-microbatch sweep

After selecting 2-3 useful global `B` points, hold `B` constant and compare:

- `b_a in {64, 128, 256}`

by changing `M` correspondingly. This separates:

1. benefit from a large global expert batch, from
2. efficiency/memory effects of the attention micro-batch itself.

### Expert compute chunk `b_e`

Do not conflate `M` with the expert-kernel row chunk.

Initially:
- execute all rows of an assigned expert in one logical expert group when workspace allows;
- make the grouped executor workspace configurable;
- if an expert exceeds workspace, chunk rows with `b_e in {256, 512, 1024, 2048, 4096}`;
- choose `b_e` using a standalone grouped-GEMM sweep rather than hard-coding a paper value.

---

## 3. Why this sweep is needed

Existing MGO measurements show a Qwen3 expert is approximately **9 MiB**, and the R4 fabric benchmark reports approximately:

- 9 MiB direct H2D: rank medians roughly 23-36 GB/s;
- large R4 NCCL all-to-all payload: roughly 267 GB/s.

If almost all 128 experts become active and are evenly assigned, each rank materializes roughly 32 experts:

`32 * 9 MiB = 288 MiB/rank/layer`.

A first-order, non-overlapped H2D floor is therefore on the order of **8-13 ms/rank/layer** using the existing R4 9 MiB copy measurements. This is only a planning estimate: multi-copy host bandwidth contention and scheduling must be re-measured in the new runtime.

Unlike H2D, expert compute and activation communication grow with `B`. Therefore the experiment should reveal a regime transition:

```
small B:     H2D dominated
medium B:    H2D amortized, compute skew becomes visible
large B:     compute + dispatch/return communication become material
```

The new work should report measured critical-path fractions rather than infer them from bytes alone.

---

## 4. Code reuse strategy

### 4.1 MGO: reuse the validated Qwen3 data plane

Reference source:
- `heonniy/mgo`
- initial reference branch: `codex/policy-regime-20261005`

Reuse/adapt the smallest possible validated pieces:

- checkpoint + host expert store loading;
- pinned/direct expert H2D path;
- slot binding and native Qwen3 BF16 numerics;
- NCCL token dispatch and exact return/combine;
- Triton grouped expert execution;
- NVTX instrumentation and Nsight summarization patterns;
- NUMA/GPU binding and collective warmup.

Useful current files include:

- `mgo_v2/mgo_v2/communicator.py`
- `mgo_v2/mgo_v2/grouped_expert.py`
- `mgo_v2/mgo_v2/executor.py`
- `mgo_v2/mgo_v2/decode_runtime.py`
- `mgo_v2/mgo_v2/model_loader.py`

Do **not** carry over the current persistent-cache control plane:
- cache;
- eviction;
- substitution;
- admission history;
- future-path affinity;
- prefetch predictor.

The new placement object is ephemeral and valid for one `(decode step group, layer)` only.

### 4.2 MoE-Gen / BatchGen: use as batching reference, not as the initial runtime

Reference:
- `batchgen-project/batchgen`
- the original MoE-Gen repository now points to/evolved into this BatchGen codebase.

Use it to understand/reference:
- module-based batching;
- separate attention and MoE decode micro-batch sizes;
- host-side accumulation / coroutine or queue structure;
- module buffer sizing.

Do not initially merge BatchGen's modern model/runtime stack with MGO. BatchGen now contains model-specific EP, quantized kernels, CUDA graphs, and distributed paths that are orthogonal to the Qwen3 characterization. Mixing both runtimes would make attribution and correctness harder.

If source code is copied rather than independently reimplemented, retain the Apache-2.0 attribution/NOTICE requirements.

---

## 5. Repository layout

```
off-mgo/
  README.md
  IMPLEMENTATION_PLAN.md

  off_mgo/
    __init__.py
    config.py                 # R, b_a, M, B, policy, profiling mode
    batching.py               # LayerAccumulator / route buffer
    routes.py                 # D[e,r], expert row counts, source metadata
    placement.py              # static / compute / comm / joint policies
    materialize.py            # ephemeral expert slots + H2D
    transport.py              # narrow adapter around MGO NCCL transport
    expert_executor.py        # grouped Qwen3 expert compute
    layer_runtime.py          # strict and overlapped layer executors
    metrics.py
    profiler.py

  scripts/
    profile_hardware.py
    collect_decode_routes.py
    run_layer_sweep.py
    summarize_layer_sweep.py
    run_e2e_decode.py

  tests/
    test_batch_math.py
    test_route_aggregation.py
    test_placement.py
    test_exact_return.py
    test_layer_parity.py

  experiments/
    phase0_hardware/
    phase1_layer_sweep/
    phase2_e2e/
```

During Phase 0/1, prefer importing the pinned MGO data-plane modules through `PYTHONPATH` to avoid copying code prematurely. Once the interface is stable, either vendor the minimal pieces with provenance or keep a pinned dependency.

---

## 6. Phase 0 — hardware and kernel calibration

Before implementing a placement method, measure the component curves on the exact R4 hardware.

### 6.1 H2D curve

Measure a 9 MiB expert transfer for:
- 1, 2, 4 simultaneous ranks;
- 1, 4, 8, 16, 32 sequential expert copies/rank;
- direct pinned H2D and the currently validated staging variant.

Record:
- per-rank GB/s;
- global host read bandwidth;
- p50/p95 copy time;
- critical-rank total time.

### 6.2 Expert compute curve

For Qwen3 expert shape (hidden 2048, intermediate 768), measure grouped expert execution at rows/expert:

`{16, 32, 64, 128, 256, 512, 1024, 2048, 4096}`.

Also sweep number of concurrently grouped experts/rank:
`{1, 4, 8, 16, 32}`.

Record:
- kernel time;
- achieved TFLOP/s;
- padding/workspace overhead;
- rows/expert where compute utilization saturates.

### 6.3 Communication curve

Use the same dispatch/return implementation as the final runtime.

Sweep:
- hidden-state dispatch payloads corresponding to `B = 512 ... 65536`;
- source/destination fanout 1-4 ranks;
- exact return traffic for top-8 routes.

Record:
- dispatch time;
- return time;
- bytes;
- NCCL bandwidth;
- collective/barrier overhead.

Output:
- `experiments/phase0_hardware/hardware_profile.json`
- CSV curves used by the placement cost model.

---

## 7. Phase 1 — real route aggregation

Implement `LayerAccumulator`.

For every local attention micro-batch:
1. run the normal Qwen3 attention + router;
2. save post-attention hidden rows;
3. save top-k IDs and routing weights;
4. tag rows with source rank and sequence identity;
5. append to the current layer accumulator.

When `M` micro-batches are present:
- concatenate the buffered rows;
- all-gather compact routing metadata;
- compute:
  - active expert set;
  - `D[e,r]`;
  - rows/expert;
  - p50/p95/max rows/expert;
  - source-rank entropy/skew per expert.

For early experiments, support **frozen route replay** so placement policies are compared on exactly the same workload.

---

## 8. Phase 1 placement policies

First physical sweep: one copy of every active expert.

### P0 — Static EP
`owner(e) = e mod R` (or exact existing static mapping).

Purpose: conventional fixed ownership baseline.

### P1 — Compute-only
Sort experts by predicted compute cost / rows and assign with LPT to the currently least-loaded rank.

Purpose: isolate rank compute imbalance.

### P2 — Communication-only
Assign each expert to the rank with the largest local source demand `D[e,r]`.

Purpose: minimize activation movement without considering stragglers.

### P3 — Joint greedy
For each candidate `(e,r)`, estimate the resulting critical rank completion using the Phase-0 curves:

`cost(e,r) = predicted_rank_finish + exposed_remote_dispatch_return_cost`.

Choose the rank that minimizes predicted global makespan. Use communication as a real measured cost, not an arbitrary byte coefficient.

### Optional oracle
After the physical baselines work, add a trace-level CP-SAT/MILP/brute-force oracle for small active-expert subsets or a strong local-search oracle. This is for headroom characterization, not the initial runtime.

### Replication
Explicitly defer replication. Add it only if the single-copy joint policy leaves a clear hot-expert straggler. Then compare `1-copy oracle` vs `replication oracle` before building a practical replication policy.

---

## 9. Phase 1 strict LayerLab executor

Implement a standalone replayable layer executor before full E2E decode.

For each frozen large routed batch:

```
placement
-> H2D active experts into ephemeral slots
-> H2D barrier
-> NCCL forward dispatch
-> grouped expert execution
-> compute barrier
-> NCCL exact return
-> native-order combine
```

Reuse the strict phase separation already validated in MGO so Nsight ranges are easy to audit.

### Required metrics

Per layer/event and per rank:
- `placement_ms`
- `h2d_ms`
- `dispatch_ms`
- `expert_compute_ms`
- `return_ms`
- `combine_ms`
- `layer_makespan_ms`
- expert H2D count/bytes
- dispatch/return bytes
- active expert count
- rows/expert p50/p95/max
- assigned expert count/rank
- routed rows/rank
- predicted compute/rank
- rank completion skew
- remote token->rank packets
- remote expert routes.

Report phase fractions using the **critical-path layer makespan**, not the sum of rank times.

---

## 10. Primary experiment matrix

Hardware:
- R4 only first;
- physical GPUs 0,1,4,5;
- Qwen3-30B-A3B-Instruct-2507 BF16;
- exact top-8 routing.

Batch sweep:
- `b_a = 128/rank`;
- `M = {1,2,4,8,16,32,64}`;
- optional `M=128` stress point.

Policies:
- Static EP;
- Compute-only;
- Communication-only;
- Joint.

For each point:
- fixed frozen workload/routes for policy comparability;
- >= 3 warmup replays;
- >= 10 measured replays for LayerLab;
- record median, p10/p90 or p25/p75, and run-to-run CV.

The first plots should be:

1. **phase fraction vs global B**
   - H2D / dispatch / execute / return;
2. **layer makespan vs global B** by policy;
3. **normalized time/token vs global B**;
4. **rank completion skew vs global B**;
5. **communication bytes vs compute balance** scatter;
6. rows/expert distribution vs `B`.

---

## 11. Go / no-go gates

Do not build the full system until LayerLab shows real headroom.

Suggested interpretation for Joint vs Static EP on the large-batch regime:

- < 5% layer-makespan improvement: stop or substantially reformulate;
- 5-10%: investigate only if a clear operating regime explains it;
- 10-20%: continue to E2E implementation;
- >20%: strong signal; characterize the cause before adding complexity.

Also require:
- policy gain stable across several decode layers/steps;
- no gain caused only by an artificial barrier;
- exact output parity.

---

## 12. Phase 2 — full module-batched decode

Only after Phase 1 passes the go gate.

Replace normal per-microbatch MoE execution with:

```
for layer l:
    for wave in 1..M:
        attention/router(local b_a)
        accumulator.append(...)
    plan global MoE batch
    materialize experts
    dispatch
    grouped expert compute
    return/combine
    restore outputs to each sequence
    advance all accumulated sequences to layer l+1
```

Important correctness issue:
- preserve per-sequence KV state and position IDs;
- preserve the exact native Qwen3 expert accumulation order;
- verify generated tokens against the unbatched exact reference at `M=1` and then larger `M`.

End-to-end metrics:
- batch completion time;
- decode throughput (tokens/s);
- time/token;
- GPU utilization;
- H2D bytes/token;
- communication bytes/token;
- memory footprint;
- queue/accumulator occupancy.

---

## 13. Phase 3 — overlap only after the serialized model is understood

Add double buffering / overlap in controlled steps:

1. overlap H2D of ready experts with remaining attention micro-batches;
2. overlap next-expert H2D with current expert compute;
3. optionally overlap independent rank communication where safe.

For every overlap optimization, retain strict mode as the attribution baseline.

The key question is not merely whether overlap exists, but how much **exposed** H2D remains on the critical path.

---

## 14. Relationship to MoE-Gen / BatchGen

MoE-Gen's central idea is retained:
- attention can use a smaller batch;
- several attention batches are accumulated;
- experts execute on a much larger module batch;
- batch sizes should be selected from hardware behavior rather than fixed universally.

This project adds a different experimental dimension:
- the large routed expert batch is executed on **multiple GPU ranks**;
- expert ownership is **batch-scoped and ephemeral**;
- placement can trade compute-rank balance against dispatch/return communication and H2D contention.

The first implementation should not claim this as novelty until the measured headroom and dominant cause are known.

---

## 15. Immediate implementation order

1. Create the package skeleton and provenance file.
2. Port/import only the MGO Qwen3 store, H2D, transport, and grouped-expert interfaces.
3. Implement Phase-0 microbenchmarks.
4. Implement frozen route format + `LayerAccumulator`.
5. Implement Static/Compute/Comm/Joint single-copy placement.
6. Implement strict LayerLab and exact parity test.
7. Run the `b_a=128, M=1..64` R4 sweep.
8. Decide whether to proceed to full module-batched decode.
9. Only then add overlap and, if justified by oracle headroom, replication.
