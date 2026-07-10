# Standard SDPA GQA Scheduling Follow-up Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to execute this plan.

**Goal:** Falsify or retain one previously untried wave-sized launch swizzle
for standard SDPA forward shape ID 5 before investing in an explicit Hopper
producer/consumer pipeline.

**Evidence:** Two independent 30/300 runs proved that moving the K descriptor
from device construction to host metadata does not improve the ID 5 GQA
kernel. The retained device-descriptor kernel already emits TMA and WGMMA,
does not spill, and is dominated by a serial K-load/QK/softmax/V-load/PV chain.
Prior work exhausted ordinary tile, warp, stage, descriptor, pointer, loop,
mask, grid-axis, and split-KV variants. Only the two extreme grid-axis orders
were tested; a wave-sized grouped mapping remains untried.

## Global Constraints

- Modify standard `sdpa` only; do not modify `sdpa_fp8`, backward operators,
  or their tests/configuration.
- Every invocation must recompute output and stats from current q/k/v values.
- Outputs, stats, probabilities, prefixes, accumulators, and results may not be
  cached or replayed across calls.
- Do not call cuDNN, PyTorch SDPA, cuBLAS, or another fallback from
  implementation code.
- Do not add a build dependency.
- The swizzle is prepared-only and limited to contiguous fp16/bf16
  `B=1,HQ=32,HKV=8,SQ=SKV=4096,D=DV=128`, group 4, top-left causal, with
  generated stats (benchmark shape ID 5).
- Preserve the existing device K-descriptor/V-pointer math, reverse-M order,
  `(BLOCK_M,BLOCK_H,BLOCK_N)=(16,4,64)`, 4 warps, 3 stages, public APIs,
  tolerances, masks, eager dispatch, and fallback dispatch.
- Change only program-to-tile scheduling in this experiment.
- Retain only if both fp16 and bf16 FlagDNN latency improve by at least 2% in
  two independent 30/300 measurements. The ultimate row target remains
  speedup `>=0.9`, with `>=0.91` preferred.
- Generated benchmark data and `.kernel-generate` logs stay untracked.

### Task 0: Test Wave-Sized GQA Launch Swizzle

**Files:**
- Modify: `src/flag_dnn/ops/sdpa.py`
- Modify: `src/flag_dnn/graph/prepared/sdpa_forward.py`
- Verify: `tests/test_sdpa.py`
- Verify: `benchmark/test_sdpa.py`

**Interfaces:**
- Consumes the retained `_sdpa_fwd_gqa_causal_desc_kernel` online-softmax
  body and prepared exact-shape runtime.
- Produces either a retained exact-ID5 grouped launch mapping or a clean
  rejection with no production diff.

- [ ] **Step 1: Record RED performance evidence**

Run twice with independent fresh caches and retain the raw tables:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-gqa-swizzle-before.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=30 \
  FLAGDNN_CUDNN_PERF_REPEAT=300 \
  FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS=5 \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  'benchmark/test_sdpa.py::test_sdpa[dtype0]' \
  'benchmark/test_sdpa.py::test_sdpa[dtype1]'
```

Expected: both rows remain below `0.9`. This is the performance RED for the
schedule-only change.

- [ ] **Step 2: Add an exact compile-time schedule selector**

Extend `_sdpa_fwd_gqa_causal_desc_kernel` with two constexpr arguments placed
after `GROUP`:

```python
BHKV: tl.constexpr,
LAUNCH_GROUP_M: tl.constexpr,
```

Keep the existing mapping when `LAUNCH_GROUP_M == 0`. When it is positive,
flatten the M and batch/KV-head axes and decode one wave-sized group:

```python
flat_pid = tle.program_id(0)
group_span = LAUNCH_GROUP_M * BHKV
group_id = flat_pid // group_span
pid_in_group = flat_pid % group_span
raw_pid_m = group_id * LAUNCH_GROUP_M + pid_in_group % LAUNCH_GROUP_M
pid_bkv = pid_in_group // LAUNCH_GROUP_M
pid_hg = tle.program_id(1)
```

Then keep the existing `pid_m = cdiv(SQ, BLOCK_M) - 1 - raw_pid_m` and the
complete kernel body unchanged. `256` M tiles divide `LAUNCH_GROUP_M=16`, and
`16 * 8 = 128` CTAs form a similarly weighted reverse-M scheduling wave.

The eager GQA descriptor launch must pass `BHKV=batch*hkv` and
`LAUNCH_GROUP_M=0`, preserving its current 3-D grid and behavior.

- [ ] **Step 3: Limit prepared swizzling to exact ID 5**

In `prepare_sdpa_forward_run`, define an exact predicate requiring all values
from Global Constraints, including full contiguous BHSD strides for q/k/v and
the already-contiguous fresh O/stats layouts.

For the exact predicate only:

```python
grid = (
    cdiv(SQ, BLOCK_M) * BHKV,
    cdiv(GROUP, BLOCK_H),
    1,
)
BHKV = batch * hkv
LAUNCH_GROUP_M = 16
```

For every other use of `_sdpa_fwd_gqa_causal_desc_kernel`, retain the existing
3-D grid and pass `LAUNCH_GROUP_M=0`. Update first-launch and cached-launch
argument builders in the same positional order as the kernel signature.
Do not change output factories, runtime tensor checks, descriptor creation,
or descriptor caching.

- [ ] **Step 4: Verify exact correctness and current-input semantics**

Run the exact ID 5 cases already covered by the standard forward suite:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa.py -k 'default_scale_with_stats and 4096'
```

If pytest IDs do not expose `4096` to `-k`, run the explicit ID 5 node IDs
reported by `--collect-only`. Both fp16 and bf16 must match cuDNN under the
existing tolerances. Also run the existing full forward suite before commit.

- [ ] **Step 5: Run the two-measurement retention gate**

Repeat Step 1 twice with new cache directories. Compare raw FlagDNN latency
against both pre-change runs. Retain only when both dtypes improve at least 2%
in both measurements. A single-dtype win, a noisy/non-repeatable win, or any
correctness regression requires complete rollback of this task.

If the swizzle reaches `>=0.9`, rerun with the benchmark's hard enforcement
wrapper. If it improves repeatably but remains below `0.9`, it may be retained
as a measured stepping stone for the explicit Hopper pipeline. If it misses
the 2% gate, do not try another group size in this task; record rejection and
move to the producer/consumer design.

- [ ] **Step 6: Verify and commit only a retained result**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q tests/test_sdpa.py
git diff --check
```

For a retained result, commit only the two standard forward source files with:

```bash
git commit -m "perf: swizzle long causal gqa scheduling"
```

For a rejected result, restore the task's production diff, leave `HEAD`
unchanged, verify the tracked worktree is clean, and preserve measurements only
in the ignored task report and optimization log.
