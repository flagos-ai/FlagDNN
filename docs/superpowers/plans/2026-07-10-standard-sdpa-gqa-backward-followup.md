# Standard SDPA GQA Backward Two-Head Fusion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to execute this plan.

**Goal:** Test one previously untried two-query-head dQ+delta fusion for
standard SDPA backward GQA shape ID 5, without changing the retained atomic
dK/dV path.

**Evidence:** The retained ID5 pipeline costs about 0.05 ms for dK/dV clear,
0.40 ms for dQ+delta, and 0.54 ms for atomic dK/dV. Owner compute failed
because it reduced dK/dV parallelism and serialized four query heads. Prior
experiments exhausted dQ tile sizes, dK/dV grouping, scratch reductions,
partial owner variants, atomics, stages, warps, cache hints, and zero tuning.
Sharing K/V loads between two query heads inside only the dQ+delta pass has not
been tested and can save enough time to cross the 0.9 graph gate if it avoids
spill/occupancy loss.

## Global Constraints

- Modify standard `sdpa_backward` only; do not modify forward or FP8 files,
  tests, or configuration.
- Every invocation must recompute delta and dQ/dK/dV from current inputs and
  allocate fresh gradients/workspace. No result or intermediate replay/cache.
- Change only the exact ID5 dQ+delta head grouping. Keep the zero and atomic
  dK/dV kernels byte-for-byte unchanged.
- Limit the candidate to contiguous fp16/bf16
  `B=1,HQ=32,HKV=8,SQ=SKV=4096,D=128`, top-left causal.
- Preserve `(BLOCK_M,BLOCK_N,BLOCK_D)=(64,64,128)`, 4 warps, 2 stages,
  online-softmax/delta math, APIs, masks, tolerances, eager paths, MHA owner,
  and all fallback dispatch.
- Use fixed `BLOCK_H=2`; do not sweep another head group in this task.
- Retain only if both fp16 and bf16 FlagDNN latency improve at least 2% in two
  independent 30/300 runs. The final speedup target is `>=0.9` for both.
- Generated logs and benchmark data remain untracked.

### Task 0: Fuse Two Query Heads In GQA dQ+Delta

**Files:**
- Modify: `tests/test_sdpa_backward.py`
- Modify: `src/flag_dnn/ops/sdpa_backward.py`
- Modify: `src/flag_dnn/graph/prepared/sdpa_backward.py`
- Modify: `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml`

**Interfaces:**
- Consumes `_sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel` and the retained
  three-step GQA prepared pipeline.
- Produces either a retained `BLOCK_H=2` dQ+delta launch or a clean rollback
  with only the expanded replay regression test retained.

- [ ] **Step 1: Expand compile-once GQA replay coverage**

Change `test_sdpa_backward_long_causal_rebinds_storage` to parameterize both
entries of `SDPA_BACKWARD_LONG_CAUSAL_D128_CASES`, with MHA and GQA IDs.

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa_backward.py::test_sdpa_backward_long_causal_rebinds_storage
```

Expected before production edits: four cases pass. This characterizes current
storage rebinding, same-storage mutation, current-input recomputation, and
fresh gradient allocation for both retained paths.

- [ ] **Step 2: Capture two RED performance baselines**

Run twice with distinct fresh Triton caches:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-bwd-gqa-twohead-before.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=30 \
  FLAGDNN_CUDNN_PERF_REPEAT=300 \
  FLAGDNN_CUDNN_SDPA_BACKWARD_PERF_SHAPE_IDS=5 \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype0]' \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype1]'
```

Expected: both speedups remain below 0.9.

- [ ] **Step 3: Add one compile-time head grouping dimension**

Add `BLOCK_H: tl.constexpr` next to the existing block meta arguments of
`_sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel`, and assert:

```python
tl.static_assert(Q_PER % BLOCK_H == 0)
```

Decode the second program axis as grouped query heads:

```python
pid_bhg = tle.program_id(1)
head_groups = HQ // BLOCK_H
off_b = pid_bhg // head_groups
off_h_base = (pid_bhg % head_groups) * BLOCK_H
off_kh = off_h_base // Q_PER
offs_mh = tl.arange(0, BLOCK_H * BLOCK_M)
offs_h = off_h_base + offs_mh // BLOCK_M
rows = start_m + offs_mh % BLOCK_M
```

Because `BLOCK_H=2` divides `Q_PER=4` and head groups start at multiples of 2,
each program remains within one KV head. Replace scalar-head q/o/dO/stats/delta
loads and dQ/delta stores with the flattened `(offs_h, rows)` mapping. K/V
remain loaded once per N tile and reused across the two query heads. Keep the
complete score, mask, exp2, delta, dS, scale, and store math unchanged.

- [ ] **Step 4: Wire the exact prepared grid and YAML meta**

Add `BLOCK_H: block_h` to `sdpa_backward_gqa_dq_delta_d128`, with only:

```yaml
block_h: [2]
```

Read `gqa_dq_block_h` from the singleton config. Require
`q_per_k % gqa_dq_block_h == 0`, change only the dQ grid to:

```python
(cdiv(sq, BLOCK_M), batch * heads // BLOCK_H, 1)
```

Append `BLOCK_H` to the cached dQ tail in exact kernel-signature order and
exclude the final four meta values from first-launch static args. Do not alter
the zero or dK/dV step, context allocation, runtime inputs, or pipeline order.

- [ ] **Step 5: Run correctness before timing**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa_backward.py \
  -k 'long_causal_d128 or long_causal_rebinds_storage'
```

Both shapes and dtypes must pass. A compile error, spill-driven resource
failure, or numerical failure rejects the candidate immediately.

- [ ] **Step 6: Run the two-measurement retention gate**

Repeat Step 2 twice with new cache directories. Compare raw FlagDNN latency
against both baselines. Retain only when every candidate-vs-baseline comparison
improves by at least 2% and both candidate speedups are `>=0.9` in both runs.
Do not try another `BLOCK_H`, tile, warp, stage, or cache hint.

- [ ] **Step 7: Verify and commit**

If retained, run the full standard backward suite and commit all four files:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q tests/test_sdpa_backward.py
git commit -m "perf: fuse gqa backward query heads"
```

If rejected, reverse all three production/config diffs, retain only the GQA
rebind regression expansion, rerun that test, and commit it as:

```bash
git commit -m "test: cover gqa backward input rebinding"
```

In either case run `git diff --check`, all commit hooks, and leave generated
logs untracked.
