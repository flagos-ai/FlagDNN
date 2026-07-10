# Standard SDPA Long-Causal Performance Design

## Background

`benchmark_logs/speedup_lt_0_9_summary.md` was generated on 2026-07-01
and is stale relative to the current `master` branch. Subsequent work brought
the FP8 operators above the benchmark threshold, but standard low-precision
SDPA still has long-causal rows below the required cuDNN-relative speedup of
`0.9` on the repository's H100 80GB test system.

A focused baseline on 2026-07-10 used 20 warmups and 100 measured iterations:

| operator | dtype | shape | cuDNN ms | FlagDNN ms | speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `sdpa` | fp16 | MHA, B2 H16 S2048 D128, causal, stats | 0.087104 | 0.097120 | 0.897 |
| `sdpa` | bf16 | MHA, B2 H16 S2048 D128, causal, stats | 0.085920 | 0.095776 | 0.897 |
| `sdpa` | fp16 | GQA, B1 H32/8 S4096 D128, causal, stats | 0.260768 | 0.300832 | 0.867 |
| `sdpa` | bf16 | GQA, B1 H32/8 S4096 D128, causal, stats | 0.254112 | 0.293920 | 0.865 |
| `sdpa_backward` | fp16 | MHA, B2 H16 S2048 D128, causal | 0.286368 | 0.330528 | 0.866 |
| `sdpa_backward` | bf16 | MHA, B2 H16 S2048 D128, causal | 0.283744 | 0.328032 | 0.865 |
| `sdpa_backward` | fp16 | GQA, B1 H32/8 S4096 D128, causal | 0.862080 | 1.016368 | 0.848 |
| `sdpa_backward` | bf16 | GQA, B1 H32/8 S4096 D128, causal | 0.848160 | 0.973552 | 0.871 |

The benchmark compares a built-once cuDNN graph with FlagDNN compiled-graph
execution. The optimization may reuse compiled launchers and tensor-address
metadata, as a compiled plan normally does, but every invocation must
recompute all mathematical outputs from the current input values.

## Goals

- Raise each of the eight dtype-specific rows above to speedup `>= 0.9`.
- Preserve SDPA and SDPA backward numerical behavior and public APIs.
- Make the improvement apply when a compiled graph is called with new tensor
  storage of the same supported shape and layout.
- Gate new fast paths to the exact contiguous, top-left causal, D128
  low-precision workloads listed in this design.
- Retain only changes that are supported by correctness and repeatable H100
  benchmark evidence.

## Non-Goals

- Do not modify `sdpa_fp8` or `sdpa_fp8_backward` in this work.
- Do not cache or reuse outputs, softmax probabilities, prefix accumulators,
  gradient tensors, or other computed intermediates across calls.
- Do not call cuDNN, PyTorch SDPA, cuBLAS, or another fallback from the
  FlagDNN implementation.
- Do not add a CUDA/CUTLASS/CuTe build dependency in this work.
- Do not broaden mask, padding, bias, dropout, fp32, or non-contiguous-layout
  support as part of the performance fast paths.

## Existing Hot Paths

The two forward shapes use specialized K-descriptor/V-pointer kernels:

- MHA S2048 uses `_sdpa_fwd_mha_causal_desc_kernel`.
- GQA S4096 uses `_sdpa_fwd_gqa_causal_desc_kernel` with four query heads
  grouped per KV head.

Both kernels create the K tensor descriptor inside every CTA and use
`tle.program_id()`, which promotes program indices to int64. Previous work
showed that adding a V descriptor, split-KV execution, wider tiles, and broad
autotune sweeps regress these shapes.

The backward shapes use:

- `_sdpa_bwd_mloop_causal_d128_kernel` for MHA S2048. It owns and directly
  stores dQ, but atomically accumulates dK and dV from every M tile.
- `_sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel` plus
  `_sdpa_bwd_gqa_dkdv_atomic_causal_d128_kernel` for GQA S4096. The first
  kernel owns dQ and stores row delta; the second runs per query head and
  atomically combines dK/dV into each KV head.

PTX/SASS inspection in the existing optimization logs confirms tensor-core
lowering. The remaining backward cost is dominated by global reductions,
atomics, barriers, and high-register causal loops rather than graph wrapper
overhead.

## Forward Design

### Exact host-descriptor variants

Add prepared-graph-only host K-descriptor variants for the two exact forward
shapes. The descriptor presents contiguous K as a flattened `[B * H * S, D]`
matrix and is indexed using a compile-time head-row offset plus the current KV
tile. V remains a pointer-loaded operand because prior measurements showed
that a V descriptor adds synchronization and regresses both shapes.

The prepared runner may retain a descriptor while the K tensor data pointer,
shape, strides, dtype, and device are unchanged. If any of those properties
change, it must rebuild the descriptor before launch. An in-place data update
does not require rebuilding because the descriptor contains address/layout
metadata only; the kernel still reloads and recomputes from K on every call.

The eager/direct path keeps its current implementation. The performance
contract in this work is the repository's compiled-graph benchmark path.

### Bounded index/codegen specialization

The exact variants use native `tl.program_id()` values and keep program, head,
row, and loop indices in int32 until pointer formation. Grid extents are far
below int32 limits. Strides, sequence sizes, group size, and scale remain
compile-time constants.

Retune only the new host-descriptor variants because moving descriptor setup
off device can change the best pipeline depth. The initial narrow candidates
are centered on the retained configurations:

- MHA: `BLOCK_M=64`, `BLOCK_N=64`, 4 warps, stages 2 or 3.
- GQA: `BLOCK_M=16`, `BLOCK_H=4`, `BLOCK_N=64`, 4 warps, stages 2 or 3.

Additional candidates require a measured reason. A candidate is retained only
if it improves both fp16 and bf16 without changing correctness.

## Backward Design

### Stage 1: existing-kernel codegen cleanup

First specialize the three existing long-causal kernels without changing the
algorithm:

- Replace `tle.program_id()` with bounded native `tl.program_id()` indexing.
- Keep loop induction and tile indices in int32.
- Apply alignment/contiguity or `tl.range` lowering controls one at a time,
  retaining only settings that improve both dtypes in repeated measurements.
- Do not revisit tile configurations, eviction hints, or atomic remaps already
  rejected in `.kernel-generate/optimization-log-sdpa_backward.md` during this
  stage. Stage 2 tunes only its newly introduced owner-compute kernel.

If every backward target reaches the acceptance threshold after this stage,
the algorithm rewrite is unnecessary.

### Stage 2: owner-compute causal D128 kernel

If any backward target remains below `0.9`, add a simplified dense-causal
owner-compute path based on the installed PyTorch/Triton flex-attention
backward mapping.

The path has two launches:

1. A row kernel computes `delta = sum(O * dO, axis=-1)` into a temporary fp32
   tensor of shape `[B, HQ, S]`.
2. A combined owner kernel uses one grid containing N-owned and M-owned
   programs.

The owner kernel grid is:

```text
(ceil_div(S, BLOCK_N_DKDV) + Q_PER * ceil_div(S, BLOCK_M_DQ), B, HKV)
```

N-owned programs load one K/V tile, loop over every query head sharing that KV
head and all causally visible M tiles, accumulate dK/dV in fp32, then directly
store each gradient exactly once. M-owned programs load one Q/dO row tile,
loop over causally visible N tiles, accumulate dQ in fp32, then directly store
dQ exactly once.

This mapping has no dK/dV output pre-clear and no gradient atomics. It
recomputes QK, P, dP, and dS independently in the two ownership branches on
every invocation. The trade-off is duplicated score work in exchange for
removing global atomic reduction and synchronization.

Use separate compile-time tiles for the two branches:

- `BLOCK_M_DKDV` and `BLOCK_N_DKDV` for the N-owned branch.
- `BLOCK_M_DQ` and `BLOCK_N_DQ` for the M-owned branch.

Split the causal loop into fully visible tiles and one diagonal masked tile so
the main loop has no element mask. Keep current score scaling and input dtype
for tensor-core dot operands; do not enable reduced-accuracy Q/K prescaling.

### Dispatch

The owner-compute path is eligible only when all of these are true:

- input dtype is fp16 or bf16;
- Q, K, V, O, dO, and stats have the expected contiguous BHSD layouts;
- head dimension and value dimension are 128;
- sequence lengths are equal and tile aligned;
- masking is pure top-left causal with no bias, padding, sink tokens, dropout,
  or variable sequence lengths;
- shape is MHA B2/H16/S2048 or GQA B1/H32/8/S4096.

All other cases keep the current dispatch. The fast path must work for new
same-shape tensor storage, not only the tensors used during compilation.

## Correctness

Add exact-shape correctness cases for both target shapes and both dtypes.
Forward compares output and stats against cuDNN. Backward compares dQ, dK, and
dV against cuDNN using the repository's established low-precision tolerances.

For prepared host descriptors, each exact forward test must run the compiled
graph with two separately allocated input sets. This detects stale descriptor
addresses. Tests must also cover in-place input updates to confirm that
descriptor metadata reuse still reloads current data.

Run the complete standard SDPA forward and backward correctness suites after
targeted tests. FP8 source files are not changed; FP8 tests are optional
regression evidence rather than an acceptance gate for this scoped work.

## Performance Validation

Use H100, a fresh Triton cache for final evidence, 30 warmups, and 300 measured
iterations. Run fp16 and bf16 target rows together to expose shared-resource
and autotune effects, then repeat the command with a separate fresh process.

Acceptance requires:

- every target row has speedup `>= 0.9` in both final runs;
- the implementation target is `>= 0.91` to leave measurement margin;
- no canonical standard SDPA or SDPA backward row regresses by more than 3%;
- no previously passing canonical row falls below `0.9` because of this work;
- correctness and static checks pass.

Performance changes that help only one dtype, depend on output/intermediate
reuse, or produce abnormal replay-only speedups are rejected and reverted.

## Files Expected To Change

- `src/flag_dnn/ops/sdpa.py`
- `src/flag_dnn/graph/prepared/sdpa_forward.py`
- `src/flag_dnn/ops/sdpa_backward.py`
- `src/flag_dnn/graph/prepared/sdpa_backward.py`
- `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml`
- `tests/test_sdpa.py`
- `tests/test_sdpa_backward.py`
- `.kernel-generate/optimization-log-sdpa.md`
- `.kernel-generate/optimization-log-sdpa_backward.md`

Only files needed by retained changes should be modified. FP8 operator and
prepared files are explicitly excluded. The existing untracked optimization
logs may receive append-only experiment notes, but are not included in the
implementation commit unless the user requests it.

## Rollback Rules

- If a forward host descriptor fails to improve both dtypes, keep the current
  K-descriptor/V-pointer kernel for that shape.
- If int32/codegen cleanup is neutral or unstable, revert that individual
  change rather than combining it with unrelated experiments.
- If the owner-compute backward path does not pass correctness or does not
  clear `0.9`, remove its kernels, dispatch, configs, and tests that exist only
  for the rejected path. Keep independently verified improvements.
- Do not weaken tolerances or alter benchmark inputs to make a candidate pass.
