# Standard SDPA GQA Hopper Gluon Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development to execute this plan task by task.

**Goal:** Replace the serial dependency chain in standard SDPA forward GQA
shape ID 5 with an explicit Hopper TMA/WGMMA producer-consumer pipeline and
raise fp16/bf16 graph speedup to at least 0.9.

**Evidence:** The retained Triton kernel already emits TMA and WGMMA without
meaningful spill, but serializes K load, QK, online softmax, V load, and PV per
KV tile. Host descriptors, ordinary tiling, stages/warps, loop controls,
schedule swizzles, descriptor variants, masks, stats scheduling, and split-KV
are exhausted. Installed Triton 3.5.1 includes experimental Gluon, Hopper TMA,
mbarriers, WGMMA, and explicit warp specialization with no new dependency.
There is no runnable Hopper attention reference, so implementation is gated in
three independently reversible tasks.

## Global Constraints

- Modify standard `sdpa` forward only; do not modify backward or any FP8 file,
  test, config, import, or dispatch.
- Every invocation must compute output/stats from current q/k/v and allocate
  fresh output/stats. Only descriptor and compiled-launch metadata may persist.
- No output, stats, probability, prefix, accumulator, or result replay/cache.
- No cuDNN, PyTorch SDPA, cuBLAS, CUTLASS, or other implementation fallback.
- Add no dependency. Use only installed `triton.experimental.gluon` Hopper APIs.
- Exact route only: contiguous fp16/bf16 Q `(1,32,4096,128)`, K/V
  `(1,8,4096,128)`, top-left causal, no bias, stats requested, H100 capability
  9.x. All other cases retain the current standard Triton path.
- Preserve `BLOCK_M=16`, `BLOCK_H=4`, `BLOCK_N=64`, `D=128`, reverse-M
  traversal, online-softmax formulas, API, masks, tolerances, and fallback.
- Do not use the Triton-to-Gluon translator or Blackwell tcgen05 APIs.
- PTX/SASS tools are required. Reject any retained candidate with local spills,
  deadlock, resource overflow, numerical failure, or a one-dtype regression.
- Generated dumps, optimization logs, and benchmark data stay untracked.

## Installed References

- TMA: Triton v3.5.1 `python/tutorials/gluon/04-tma.py`.
- WGMMA: Triton v3.5.1 `python/tutorials/gluon/05-wgmma.py`.
- Explicit partitions: Triton v3.5.1
  `python/tutorials/gluon/08-warp-specialization.py`.
- Layout APIs: installed `triton/experimental/gluon/language/_layouts.py`.
- Hopper APIs: installed
  `triton/experimental/gluon/language/nvidia/hopper/`.

### Task 0: Serial Gluon Parity Prototype

**Files:**
- Create: `src/flag_dnn/ops/_sdpa_gqa_gluon.py`
- Modify: `tests/test_sdpa.py`

**Interfaces:**
- Produces a direct-only `_sdpa_fwd_gqa_id5_gluon_serial` kernel and test
  launcher. It is not imported by package initializers or production dispatch.

- [ ] **Step 1: Add the failing direct serial test**

Add a CUDA/Hopper-only test that lazily imports `run_gqa_id5_gluon_serial`
from the new module, constructs exact ID5 q/k/v, runs fp16 and bf16, and
compares output/stats with `_cudnn_sdpa` using existing tolerances.

The helper must be called three times per dtype: first inputs, new storage, and
same-storage in-place mutation. Keep all returned tensors alive and assert
fresh output and stats pointers between calls.

Run the test and record RED: import must fail because the module is absent.

- [ ] **Step 2: Create the isolated Gluon module**

Import only installed APIs:

```python
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    tma,
    warpgroup_mma,
    warpgroup_mma_init,
    warpgroup_mma_wait,
)

from flag_dnn.utils import libentry
```

Do not import this module anywhere else in Task 0.

Use fixed ID5 mapping:

- grid `(256, 8)`;
- one CTA owns one KV head, its four Q heads, and 16 query positions/head;
- flattened distributed row `r` maps to
  `q_head = kv_head * 4 + r // 16` and
  `q_pos = start_m + r % 16`;
- K/V descriptor metadata is `shape=[8*4096,128]`,
  `strides=[128,1]`, `block_shape=[64,128]`.

Define QK/PV distributed and operand layouts from the official WGMMA tutorial:

```python
qk_layout = gl.NVMMADistributedLayout(
    version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 16]
)
q_layout = gl.DotOperandLayout(operand_index=0, parent=qk_layout, k_width=2)
pv_layout = gl.NVMMADistributedLayout(
    version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 128, 16]
)
p_layout = gl.DotOperandLayout(operand_index=0, parent=pv_layout, k_width=2)
```

Derive the K/V `NVMMASharedLayout` with `get_default_for([64,128], dtype)`.
Use pointer-loaded Q in `q_layout`, shared K transposed for QK, P converted to
input dtype and `p_layout` for PV, shared V `[64,128]`, and fp32 m/l/acc.

- [ ] **Step 3: Implement serial TMA/WGMMA math**

Add:

```python
@libentry()
@gluon.jit
def _sdpa_fwd_gqa_id5_gluon_serial(...):
    ...
```

Launch `num_warps=4`, `maxnreg=128`. Allocate one K buffer, one V buffer, and
one ready mbarrier. For every causal KV tile:

1. set `mbarrier.expect` to exact K+V bytes;
2. issue both TMA loads with int32 flattened coordinates;
3. wait for the current phase;
4. QK WGMMA and wait;
5. apply causal mask with `start_m + r % 16`;
6. update fp32 m/l/acc using the retained exp2/log2 formulas;
7. convert P layout/dtype;
8. PV WGMMA and wait before accumulator rescale or buffer reuse.

Store fresh O and natural-log stats with the same formulas as current SDPA.
`run_gqa_id5_gluon_serial` validates exact shape/dtype/stride/device, allocates
fresh O/stats, constructs current K/V descriptors, launches, and returns them.

- [ ] **Step 4: Verify serial gates**

Run the direct test to GREEN. Dump TTGIR/PTX/SASS with fresh caches and verify:

- TMA plus both WGMMA sites are present;
- no PTX `ld.local/st.local` and no SASS `LDL/STL`;
- shared memory is below 232,448 bytes.

Measure preallocated direct launches with descriptors constructed outside the
timed region. Compare against the retained Triton kernel p50 evidence. Continue
only if both dtypes are within 10% of retained direct latency. Otherwise remove
the module/test, record rejection, and stop the Gluon plan.

- [ ] **Step 5: Commit serial parity**

Run Black, flake8, mypy, `git diff --check`, and the focused test. Commit:

```bash
git commit -m "perf: prototype serial gluon gqa attention"
```

### Task 1: Explicit One-Buffer And Two-Buffer Pipeline

**Files:**
- Modify: `src/flag_dnn/ops/_sdpa_gqa_gluon.py`
- Modify: `tests/test_sdpa.py`
- Modify: `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml`

- [ ] **Step 1: Add a direct pipeline test before implementation**

Extend the direct test to import and exercise `run_gqa_id5_gluon_ws`; record
RED because the helper is absent. It must reuse all Task 0 correctness,
rebinding, mutation, and fresh-output assertions.

- [ ] **Step 2: Add the warp-specialized entry**

Add `@libentry()` outermost, repository `@libtuner(...)`, then `@gluon.jit` for
`_sdpa_fwd_gqa_id5_gluon_ws`. Use the YAML key
`sdpa_gqa_id5_gluon_ws`, tuning key `ELEM_SIZE`, fixed 4 consumer warps,
one producer warp, one stage, and `maxnreg=128`.

Allocate NUM_BUFFERS K/V shared buffers plus ready/empty mbarriers. Use
`gl.warp_specialize`:

- default partition: four-warp QK/softmax/PV consumer;
- worker partition: one-warp TMA producer;
- worker registers 24.

Producer and consumer must compute identical causal tile counts/phases.
Producer waits for empty, issues K/V TMA, and TMA completion signals ready.
Consumer waits ready and signals empty only after PV WGMMA wait.

- [ ] **Step 3: Measure one buffer**

Add YAML with only `NUM_BUFFERS=1`, `ELEM_SIZE=2`, meta warps 4, launch warps
4, stages 1. Run correctness and inspect PTX/SASS. One-buffer must be within 5%
of serial for both dtypes and introduce no spill/deadlock.

- [ ] **Step 4: Change only one to two buffers**

Change only `NUM_BUFFERS=1` to `2`. Run correctness, PTX/SASS, and three
interleaved direct timing comparisons against the retained Triton kernel.
Continue only if both dtypes improve retained direct latency by at least 3%
in all three comparisons, async partitions remain distinct, and SASS has no
LDL/STL. Otherwise remove Task 1 pipeline/YAML/test changes and then remove
the unused Task 0 module/test in a cleanup commit.

- [ ] **Step 5: Commit a retained pipeline**

Keep one final NUM_BUFFERS value only, run hooks/tests, and commit:

```bash
git commit -m "perf: pipeline gluon gqa attention"
```

### Task 2: Exact Eager And Prepared Dispatch

**Files:**
- Modify: `src/flag_dnn/ops/sdpa.py`
- Modify: `src/flag_dnn/graph/prepared/sdpa_forward.py`
- Modify: `tests/test_sdpa.py`

- [ ] **Step 1: Add failing route/lifecycle tests**

Add exact ID5 eager and prepared tests for both dtypes. Monkeypatch or expose a
minimal route counter so RED proves the Gluon route is not yet selected.
Prepared coverage must compile once, run new storage, mutate same storage, keep
results alive, compare each run with fresh cuDNN references, and assert fresh O
and stats pointers. Add a noncontiguous ID5-shaped case that must use fallback.

- [ ] **Step 2: Add exact lazy dispatch**

In eager and prepared code, evaluate every exact shape/dtype/stride/mask/stats
predicate plus Hopper capability before lazily importing the Gluon module.
Catch only `ImportError` and `AttributeError`; otherwise propagate real errors.
If unavailable or ineligible, continue to the existing GQA descriptor path.

Eager creates current K/V descriptors per call. Prepared keeps separate K/V
descriptor metadata caches keyed by:

```python
(data_ptr, shape, stride, dtype, device.type, device.index)
```

Descriptors are runtime args, never static/captured launcher args. Output stays
on the existing fresh `make_stats_output` factory. Cached-call argument order
must match the Gluon kernel signature and returned tuning metadata exactly.

- [ ] **Step 3: Run route/lifecycle GREEN and full correctness**

Run focused exact/fallback tests, then full `tests/test_sdpa.py`. Existing small
GQA, non-ID5, noncontiguous, MHA, eager, and prepared cases must remain correct.

- [ ] **Step 4: Run final graph gate twice**

With independent fresh caches, 30 warmups and 300 iterations, run forward ID5
fp16/bf16. Both rows must be `>=0.9` in both runs and both FlagDNN latencies
must improve the pre-change baseline. Run canonical IDs 0-7 and reject any
non-target regression above 3%.

- [ ] **Step 5: Commit dispatch**

Run all hooks, static FP8/backward exclusion checks, and commit:

```bash
git commit -m "perf: dispatch long causal gqa to gluon"
```

If any task is rejected, preserve its measurements in ignored reports/logs and
ensure the net production diff contains no unused Gluon route.
