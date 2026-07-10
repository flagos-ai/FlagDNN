# Standard SDPA Long-Causal Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the eight dtype-specific fp16/bf16 long-causal `sdpa` and `sdpa_backward` benchmark rows for shape IDs 2 and 5 to repeatable speedup `>= 0.9` versus cuDNN on H100.

**Architecture:** Add prepared-only host K-descriptor forward kernels that still recompute the complete attention result on every call. Optimize the existing backward kernels first with bounded int32 program indexing; only if those rows remain below threshold, add a two-launch delta plus owner-compute backward path that directly stores dQ/dK/dV without atomics.

**Tech Stack:** Python 3.12, PyTorch, Triton 3.5.1, cuDNN frontend, pytest, H100 sm90, FlagDNN compiled graphs.

## Global Constraints

- Modify standard `sdpa` and `sdpa_backward` only; do not modify `sdpa_fp8` or `sdpa_fp8_backward`.
- Every invocation must recompute outputs and gradients from current input values.
- Descriptor/compiled-launch metadata may be cached; outputs, probabilities, prefixes, accumulators, and gradients may not be cached across calls.
- Do not call cuDNN, PyTorch SDPA, cuBLAS, or another fallback from implementation code.
- Do not add CUDA, CUTLASS, CuTe, or another build dependency.
- New fast paths are limited to contiguous fp16/bf16 D128 top-left causal shape IDs 2 and 5.
- Preserve public APIs, existing numerical tolerances, mask semantics, and fallback dispatch.
- Retain a performance change only when both fp16 and bf16 improve in two independent measurements.
- Final target runs use 30 warmups and 300 measured iterations; every target row must be `>=0.9`, with `>=0.91` as the implementation margin.
- Existing untracked benchmark/test outputs and `.kernel-generate` logs must not be staged or committed.

---

## File Map

- `src/flag_dnn/ops/sdpa.py`: forward Triton math and the two prepared-only host K-descriptor kernels.
- `src/flag_dnn/graph/prepared/sdpa_forward.py`: exact-shape dispatch, K descriptor metadata lifecycle, output allocation, and launcher binding.
- `src/flag_dnn/ops/sdpa_backward.py`: bounded-index changes and, conditionally, the owner-compute backward kernel.
- `src/flag_dnn/graph/prepared/sdpa_backward.py`: owner-path eligibility and two-step prepared pipeline.
- `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml`: narrow H100 configurations for new kernels.
- `tests/test_sdpa.py`: exact forward correctness, new-storage rebinding, in-place input update, and descriptor-only cache assertions.
- `tests/test_sdpa_backward.py`: exact backward correctness and conditional owner-path rebinding coverage.
- `.kernel-generate/optimization-log-sdpa.md`: append-only local measurements; never stage.
- `.kernel-generate/optimization-log-sdpa_backward.md`: append-only local measurements; never stage.

### Task 0: Capture Canonical Pre-Change Evidence

**Files:**
- Verify only: `benchmark/test_sdpa.py`
- Verify only: `benchmark/test_sdpa_backward.py`

**Interfaces:**
- Consumes: current `master` kernels before any implementation edit.
- Produces: target and unaffected-row baselines used by every later retention gate.

- [ ] **Step 1: Confirm the starting worktree**

```bash
git status --short --branch
git rev-parse HEAD
```

Expected: only the known untracked generated logs/data are present; record the
starting commit and do not stage those files.

- [ ] **Step 2: Measure canonical forward and backward IDs 0-7**

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-canonical-before.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=25 \
  FLAGDNN_CUDNN_PERF_REPEAT=100 \
  FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS=0,1,2,3,4,5,6,7 \
  FLAGDNN_CUDNN_SDPA_BACKWARD_PERF_SHAPE_IDS=0,1,2,3,4,5,6,7 \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  'benchmark/test_sdpa.py::test_sdpa[dtype0]' \
  'benchmark/test_sdpa.py::test_sdpa[dtype1]' \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype0]' \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype1]'
```

Expected: four timing tables. Keep the tables in task notes for the final 3%
regression comparison; do not write them into tracked benchmark logs.

### Task 1: Prepared Host K-Descriptor Forward Paths

**Files:**
- Modify: `tests/test_sdpa.py:103-132`
- Modify: `tests/test_sdpa.py:212-226`
- Modify: `src/flag_dnn/ops/sdpa.py:592-632`
- Modify: `src/flag_dnn/ops/sdpa.py:811-1084`
- Modify: `src/flag_dnn/graph/prepared/sdpa_forward.py:1-19`
- Modify: `src/flag_dnn/graph/prepared/sdpa_forward.py:340-526`
- Modify: `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml:343-379`

**Interfaces:**
- Consumes: current `_sdpa_fwd_gqa_causal_kdesc_inner` online-softmax semantics and prepared single-kernel launcher APIs.
- Produces: `_sdpa_fwd_mha_causal_hostdesc_kernel`, `_sdpa_fwd_gqa_causal_hostdesc_kernel`, and a prepared runtime that supplies a host `TensorDescriptor` for K only.

- [ ] **Step 1: Record the failing forward performance baseline**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-fwd-before.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=30 \
  FLAGDNN_CUDNN_PERF_REPEAT=300 \
  FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS=2,5 \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  'benchmark/test_sdpa.py::test_sdpa[dtype0]' \
  'benchmark/test_sdpa.py::test_sdpa[dtype1]'
```

Expected: all four rows print valid timings; at least shape ID 5 is below `0.9`. Save the console output in the task notes, not in a tracked repository file.

- [ ] **Step 2: Add the failing host-descriptor lifecycle test**

Add this exact test after `test_sdpa_gqa_causal_d128_low_precision`:

```python
@pytest.mark.sdpa
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape",
    (
        (2, 16, 16, 2048, 2048, 128),
        (1, 32, 8, 4096, 4096, 128),
    ),
    ids=("mha_b2_h16_s2048", "gqa_b1_h32_hkv8_s4096"),
)
def test_sdpa_long_causal_host_k_descriptor_rebinds_storage(
    cudnn_handle, monkeypatch, dtype, shape
):
    from triton.tools import tensor_descriptor as descriptor_module

    real_descriptor = descriptor_module.TensorDescriptor
    descriptor_ptrs = []

    def recording_descriptor(base, desc_shape, strides, block_shape):
        descriptor_ptrs.append(base.data_ptr())
        return real_descriptor(base, desc_shape, strides, block_shape)

    monkeypatch.setattr(
        descriptor_module, "TensorDescriptor", recording_descriptor
    )
    torch.manual_seed(17)
    q_a, k_a, v_a = _make_qkv(shape, dtype)

    @flag_dnn.graph
    def fn(q, k, v):
        return flag_dnn.sdpa(
            q,
            k,
            v,
            diagonal_band_right_bound=0,
            generate_stats=True,
            name="sdpa",
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(q_a, "q"),
            flag_dnn.TensorSpec.from_tensor(k_a, "k"),
            flag_dnn.TensorSpec.from_tensor(v_a, "v"),
        ],
        options={"cache": None},
    )

    def check(q, k, v):
        expected = _cudnn_sdpa(
            q, k, v, cudnn_handle, right_bound=0, generate_stats=True
        )
        actual = compiled.run(q, k, v)
        _assert_sdpa_close(actual, expected, dtype)
        return actual

    out_a = check(q_a, k_a, v_a)
    q_b, k_b, v_b = _make_qkv(shape, dtype)
    out_b = check(q_b, k_b, v_b)
    assert descriptor_ptrs == [k_a.data_ptr(), k_b.data_ptr()]
    assert out_a[0].data_ptr() != out_b[0].data_ptr()

    q_b.normal_()
    k_b.normal_()
    v_b.normal_()
    out_c = check(q_b, k_b, v_b)
    assert descriptor_ptrs == [k_a.data_ptr(), k_b.data_ptr()]
    assert out_b[0].data_ptr() != out_c[0].data_ptr()
```

- [ ] **Step 3: Run the test to verify the new route is absent**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa.py::test_sdpa_long_causal_host_k_descriptor_rebinds_storage
```

Expected: numerical comparisons pass through the current device-created K descriptor, then `descriptor_ptrs == []` fails.

- [ ] **Step 4: Add the host K-descriptor online-softmax helper**

Add a new helper next to `_sdpa_fwd_gqa_causal_kdesc_inner`. Its body must remain mathematically identical to the retained K-descriptor/V-pointer helper; only the descriptor row base changes.

```python
@triton.jit
def _sdpa_fwd_causal_host_kdesc_inner(
    acc,
    l_i,
    m_i,
    q,
    k_desc,
    k_row,
    v_base,
    qk_scale: tl.constexpr,
    offs_m,
    offs_dv,
    lo,
    hi,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED: tl.constexpr,
):
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        start_n_i32 = start_n.to(tl.int32)
        offs_n = start_n_i32 + tl.arange(0, BLOCK_N)
        k = tl.trans(k_desc.load([k_row + start_n_i32, 0]))
        score = tl.dot(q, k).to(tl.float32) * qk_scale
        if MASKED:
            score = tl.where(
                offs_n[None, :] <= offs_m[:, None],
                score,
                float("-inf"),
            )
        m_new = tl.maximum(m_i, tl.max(score, 1))
        p = tl.exp2(score - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(
            v_base
            + offs_n[:, None] * stride_vn
            + offs_dv[None, :] * stride_vd
        )
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new
    return acc, l_i, m_i
```

- [ ] **Step 5: Add the two prepared-only forward kernels**

Add the following MHA entry point:

```python
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_mha_causal_hostdesc"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE"],
    strategy=["log", "log", "default", "default", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def _sdpa_fwd_mha_causal_hostdesc_kernel(
    q_ptr,
    k_desc,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HQ: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    raw_pid_m = tl.program_id(1)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    off_b = pid_bh // HQ
    off_h = pid_bh % HQ
    start_m = pid_m * BLOCK_M
    offs_m = tl.max_contiguous(
        start_m + tl.arange(0, BLOCK_M), BLOCK_M
    )
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
    v_base = v_ptr + off_b * stride_vb + off_h * stride_vh
    q = tl.load(
        q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    )
    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    hi = start_m + BLOCK_M
    full_hi = tl.minimum((start_m // BLOCK_N) * BLOCK_N, hi)
    k_row = pid_bh * SKV
    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc, l_i, m_i, q, k_desc, k_row, v_base, qk_scale,
            offs_m, offs_dv, 0, full_hi, stride_vn, stride_vd,
            BLOCK_N=BLOCK_N, MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc, l_i, m_i, q, k_desc, k_row, v_base, qk_scale,
            offs_m, offs_dv, full_hi, hi, stride_vn, stride_vd,
            BLOCK_N=BLOCK_N, MASKED=True,
        )
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr + off_b * stride_ob + off_h * stride_oh
        + offs_m[:, None] * stride_om + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
    )
    tl.store(
        stats_ptr + off_b * stride_sb + off_h * stride_sh
        + offs_m * stride_sm,
        m_i / _LOG2E_KERNEL + tl.log(l_safe),
    )
```

Add the complete GQA entry point:

```python
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sdpa_gqa_causal_hostdesc"),
    key=["SQ", "SKV", "HEAD_DIM", "V_DIM", "ELEM_SIZE", "GROUP"],
    strategy=["log", "log", "default", "default", "default", "default"],
    warmup=5,
    rep=10,
)
@triton.jit
def _sdpa_fwd_gqa_causal_hostdesc_kernel(
    q_ptr,
    k_desc,
    v_ptr,
    o_ptr,
    stats_ptr,
    qk_scale: tl.constexpr,
    HKV: tl.constexpr,
    SQ: tl.constexpr,
    SKV: tl.constexpr,
    GROUP: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_om: tl.constexpr,
    stride_od: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    ELEM_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    raw_pid_m = tl.program_id(0)
    pid_m = tl.cdiv(SQ, BLOCK_M) - 1 - raw_pid_m
    pid_bkv = tl.program_id(1)
    pid_hg = tl.program_id(2)
    off_b = pid_bkv // HKV
    off_kh = pid_bkv % HKV
    start_m = pid_m * BLOCK_M
    offs_mh = tl.arange(0, BLOCK_M * BLOCK_H)
    offs_h = pid_hg * BLOCK_H + offs_mh // BLOCK_M
    offs_m = start_m + offs_mh % BLOCK_M
    q_head = off_kh * GROUP + offs_h
    row_mask = (offs_h < GROUP) & (offs_m < SQ)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    q = tl.load(
        q_ptr
        + off_b * stride_qb
        + q_head[:, None] * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd,
        mask=row_mask[:, None],
        other=0.0,
    )
    v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
    k_row = pid_bkv * SKV
    acc = tl.zeros(
        (BLOCK_M * BLOCK_H, BLOCK_DV), dtype=tl.float32
    )
    l_i = tl.zeros((BLOCK_M * BLOCK_H,), dtype=tl.float32)
    m_i = tl.full(
        (BLOCK_M * BLOCK_H,), float("-inf"), dtype=tl.float32
    )
    hi = tl.minimum(start_m + BLOCK_M, SKV)
    full_hi = ((start_m + 1) // BLOCK_N) * BLOCK_N
    full_hi = tl.minimum(full_hi, hi)
    if 0 < full_hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            k_row,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            0,
            full_hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=False,
        )
    if full_hi < hi:
        acc, l_i, m_i = _sdpa_fwd_causal_host_kdesc_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc,
            k_row,
            v_base,
            qk_scale,
            offs_m,
            offs_dv,
            full_hi,
            hi,
            stride_vn,
            stride_vd,
            BLOCK_N=BLOCK_N,
            MASKED=True,
        )
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / l_safe[:, None]
    tl.store(
        o_ptr
        + off_b * stride_ob
        + q_head[:, None] * stride_oh
        + offs_m[:, None] * stride_om
        + offs_dv[None, :] * stride_od,
        acc.to(o_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )
    tl.store(
        stats_ptr
        + off_b * stride_sb
        + q_head * stride_sh
        + offs_m * stride_sm,
        m_i / _LOG2E_KERNEL + tl.log(l_safe),
        mask=row_mask,
    )
```

- [ ] **Step 6: Bind host descriptors in the prepared runner**

Import `TensorDescriptor`, both new kernels, and leave the eager path
unchanged. For each exact branch, create this metadata-only cache:

```python
descriptor_key = None
descriptor = None
descriptor_shape = (batch * k_shape[1] * skv, head_dim)
descriptor_stride = (head_dim, 1)
descriptor_block = [64, head_dim]

def get_k_descriptor(k: torch.Tensor):
    nonlocal descriptor_key, descriptor
    key = (
        k.data_ptr(),
        tuple(k.shape),
        tuple(k.stride()),
        k.dtype,
        k.device.type,
        k.device.index,
    )
    if descriptor is None or descriptor_key != key:
        descriptor = TensorDescriptor(
            k,
            list(descriptor_shape),
            list(descriptor_stride),
            descriptor_block,
        )
        descriptor_key = key
    return descriptor
```

The exact MHA predicate additionally requires `B=2,HQ=HKV=16,S=2048`.
The exact GQA predicate additionally requires `B=1,HQ=32,HKV=8,S=4096`.
Both require exact contiguous BHSD strides. Supply runtime args as:

```python
def hostdesc_runtime_args(inputs, output):
    return (
        inputs[0],
        get_k_descriptor(inputs[1]),
        inputs[2],
        output[0],
        output[1],
    )
```

For MHA, bind:

```python
mha_host_tail = (
    attn_scale * _LOG2E,
    heads,
    sq,
    skv,
    *q_stride,
    *v_stride,
    *o_stride,
    *stats_stride,
)
mha_host_constexpr = {
    "HEAD_DIM": head_dim,
    "V_DIM": v_dim,
    "ELEM_SIZE": out_dtype.itemsize,
    "BLOCK_D": head_dim,
    "BLOCK_DV": v_dim,
}

def mha_host_grid(meta):
    return (batch_heads, triton.cdiv(sq, meta["BLOCK_M"]))

def build_mha_host_cached_call(meta):
    block_m = int(meta["BLOCK_M"])
    block_n = int(meta["BLOCK_N"])
    return (
        batch_heads,
        triton.cdiv(sq, block_m),
        1,
    ), mha_host_tail + (
        head_dim,
        v_dim,
        out_dtype.itemsize,
        block_m,
        block_n,
        head_dim,
        v_dim,
    )
```

For GQA, bind:

```python
gqa_host_tail = (
    attn_scale * _LOG2E,
    hkv,
    sq,
    skv,
    group,
    *q_stride,
    *v_stride,
    *o_stride,
    *stats_stride,
)
gqa_host_constexpr = {
    "HEAD_DIM": head_dim,
    "V_DIM": v_dim,
    "ELEM_SIZE": out_dtype.itemsize,
    "BLOCK_D": head_dim,
    "BLOCK_DV": v_dim,
}

def gqa_host_grid(meta):
    return (
        triton.cdiv(sq, meta["BLOCK_M"]),
        batch * hkv,
        triton.cdiv(group, meta["BLOCK_H"]),
    )

def build_gqa_host_cached_call(meta):
    block_m = int(meta["BLOCK_M"])
    block_h = int(meta["BLOCK_H"])
    block_n = int(meta["BLOCK_N"])
    return (
        triton.cdiv(sq, block_m),
        batch * hkv,
        triton.cdiv(group, block_h),
    ), gqa_host_tail + (
        head_dim,
        v_dim,
        out_dtype.itemsize,
        block_m,
        block_h,
        block_n,
        head_dim,
        v_dim,
    )
```

Return the selected exact branch with:

```python
return make_single_kernel_run_fn(
    PreparedSingleKernelRunSpec(
        kernel=PreparedSingleKernelSpec(
            kernel=host_kernel,
            grid=host_grid,
            static_args=host_tail,
            constexpr_kwargs=host_constexpr,
            build_cached_call=build_host_cached_call,
        ),
        input_checks=qkv_input_checks,
        output_factory=make_stats_output,
        runtime_args=hostdesc_runtime_args,
        pre_launch=_ensure_triton_tma_allocator,
    ),
    default_run_fn,
)
```

Set `host_kernel`, `host_grid`, `host_tail`,
`host_constexpr`, and `build_host_cached_call` from the MHA or GQA block
above before returning. Keep `make_stats_output`, which allocates fresh
O/stats tensors on every call. Non-exact cases continue to the existing
device-descriptor branches.

- [ ] **Step 7: Add narrow host-descriptor tune configs**

Add:

```yaml
sdpa_mha_causal_hostdesc:
  - gen: true
    param_map:
      META:
        BLOCK_M: block_m
        BLOCK_N: block_n
      num_warps: warps
      num_stages: stages
    block_m: [64]
    block_n: [64]
    warps: [4]
    stages: [2, 3]

sdpa_gqa_causal_hostdesc:
  - gen: true
    param_map:
      META:
        BLOCK_M: block_m
        BLOCK_H: block_h
        BLOCK_N: block_n
      num_warps: warps
      num_stages: stages
    block_m: [16]
    block_h: [4]
    block_n: [64]
    warps: [4]
    stages: [2, 3]
```

- [ ] **Step 8: Run targeted correctness and the hard performance gate**

Run the lifecycle test from Step 3. Expected: PASS for four parametrized cases.

Then run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-fwd-gate.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=30 \
  FLAGDNN_CUDNN_PERF_REPEAT=300 \
  FLAGDNN_CUDNN_PERF_MIN_SPEEDUP=0.9 \
  FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS=2,5 \
  /home/wangbingjie/py312/bin/python3 -c \
  'import pytest; from benchmark.test_sdpa import SdpaBenchmark; SdpaBenchmark.enforce_min_speedup=True; raise SystemExit(pytest.main(["-q","-s","benchmark/test_sdpa.py::test_sdpa[dtype0]","benchmark/test_sdpa.py::test_sdpa[dtype1]"]))'
```

Expected: PASS and every printed row `>=0.9`. Repeat with another
`mktemp` cache. If one stage value wins only one dtype, remove it; retain one
configuration only after both dtypes pass twice.

- [ ] **Step 9: Run full forward correctness**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q tests/test_sdpa.py
```

Expected: PASS with only pre-existing skips.

- [ ] **Step 10: Commit the retained forward work**

```bash
git add tests/test_sdpa.py \
  src/flag_dnn/ops/sdpa.py \
  src/flag_dnn/graph/prepared/sdpa_forward.py \
  src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml
git commit -m "perf: add host descriptors for long causal sdpa"
```

### Task 2: Bounded Program IDs In Existing Backward Kernels

**Files:**
- Modify: `tests/test_sdpa_backward.py:15-30`
- Modify: `tests/test_sdpa_backward.py:320-370`
- Modify: `src/flag_dnn/ops/sdpa_backward.py:2313-2452`
- Modify: `src/flag_dnn/ops/sdpa_backward.py:2697-2932`

**Interfaces:**
- Consumes: current MHA m-loop and GQA split prepared dispatch.
- Produces: exact large-shape regression coverage and retained int32/codegen improvements without changing the backward algorithm.

- [ ] **Step 1: Add exact long-causal backward correctness cases**

Add:

```python
SDPA_BACKWARD_LONG_CAUSAL_D128_CASES = (
    (2, 16, 16, 2048, 2048, 128),
    (1, 32, 8, 4096, 4096, 128),
)


@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape",
    SDPA_BACKWARD_LONG_CAUSAL_D128_CASES,
    ids=("mha_b2_h16_s2048", "gqa_b1_h32_hkv8_s4096"),
)
def test_sdpa_backward_long_causal_d128(cudnn_handle, dtype, shape):
    torch.manual_seed(23)
    q, k, v = _make_qkv(shape, dtype)
    dO = torch.randn_like(q)
    o, stats = _cudnn_sdpa_forward(
        q, k, v, cudnn_handle, right_bound=0
    )
    expected = _cudnn_sdpa_backward(
        q, k, v, o, dO, stats, cudnn_handle, right_bound=0
    )
    actual = _run_flag_dnn_sdpa_backward_graph(
        q, k, v, o, dO, stats, right_bound=0
    )
    _assert_grads_close(actual, expected, dtype)
```

- [ ] **Step 2: Run characterization correctness**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa_backward.py::test_sdpa_backward_long_causal_d128
```

Expected before the optimization: PASS. This locks correctness; the red signal
is the performance gate in Step 3.

- [ ] **Step 3: Run the failing backward performance gate**

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-bwd-before.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=30 \
  FLAGDNN_CUDNN_PERF_REPEAT=300 \
  FLAGDNN_CUDNN_PERF_MIN_SPEEDUP=0.9 \
  FLAGDNN_CUDNN_SDPA_BACKWARD_PERF_SHAPE_IDS=2,5 \
  /home/wangbingjie/py312/bin/python3 -c \
  'import pytest; from benchmark.test_sdpa_backward import SdpaBackwardBenchmark; SdpaBackwardBenchmark.enforce_min_speedup=True; raise SystemExit(pytest.main(["-q","-s","benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype0]","benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype1]"]))'
```

Expected: FAIL and report one or more rows below `0.9`.

- [ ] **Step 4: Replace only the six hot-path program-ID promotions**

In each of these three kernels:

- `_sdpa_bwd_mloop_causal_d128_kernel`
- `_sdpa_bwd_gqa_dq_store_delta_causal_d128_kernel`
- `_sdpa_bwd_gqa_dkdv_atomic_causal_d128_kernel`

replace:

```python
pid_tile = tle.program_id(0)
pid_bh = tle.program_id(1)
```

with the corresponding native names:

```python
pid_tile = tl.program_id(0)
pid_bh = tl.program_id(1)
```

Use the existing local names (`pid_m` or `pid_n`) rather than literally
`pid_tile`. Do not change grids, atomics, or tile configs. Native program IDs
and all derived head/tile indices remain int32 until pointer arithmetic.

- [ ] **Step 5: Run correctness and measure the isolated change**

Run the correctness command from Step 2, then the hard gate from Step 3 twice
with distinct fresh caches.

Expected: correctness PASS. Retain the program-ID change only if both dtypes
improve in both runs. If all four rows are `>=0.9`, commit and skip Task 3.

- [ ] **Step 6: If needed, test compiler hints one at a time**

Try these in order, reverting a candidate before trying the next:

1. Apply `tl.max_contiguous(rows, BLOCK_M)` to row-owned loops and
   `tl.max_contiguous(cols, BLOCK_N)` to the GQA N-owned loop.
2. Add `disable_licm=True` to the three long-causal `tl.range` loops.

After each candidate, run targeted correctness and two fresh-cache hard gates.
Retain a candidate only if both dtypes improve twice. Do not combine neutral
candidates and do not retry tile, eviction, or atomic experiments rejected in
the existing optimization log.

- [ ] **Step 7: Commit backward coverage and retained Stage 1 changes**

```bash
git add tests/test_sdpa_backward.py src/flag_dnn/ops/sdpa_backward.py
git commit -m "perf: use bounded ids in long causal sdpa backward"
```

If no codegen change survives, commit the test by itself:

```bash
git add tests/test_sdpa_backward.py
git commit -m "test: cover long causal sdpa backward targets"
```

### Task 3: Conditional Owner-Compute Backward Path

Execute this task only if any backward target remains below `0.9` after
Task 2.

**Files:**
- Modify: `tests/test_sdpa_backward.py`
- Modify: `src/flag_dnn/ops/sdpa_backward.py:38-92`
- Modify: `src/flag_dnn/ops/sdpa_backward.py:2933`
- Modify: `src/flag_dnn/graph/prepared/sdpa_backward.py:19-268`
- Modify: `src/flag_dnn/graph/prepared/sdpa_backward.py:449-930`
- Modify: `src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml:583-602`

**Interfaces:**
- Consumes: `_sdpa_bwd_delta_kernel`, existing prepared pipeline APIs, exact long-causal tests from Task 2.
- Produces: `_is_owner_compute_causal_d128` and `_sdpa_bwd_owner_causal_d128_kernel`; a two-step delta/owner prepared path with no output clear and no gradient atomics.

- [ ] **Step 1: Add a failing owner-path eligibility test**

Add a pure helper test:

```python
def test_owner_compute_causal_d128_eligibility():
    from flag_dnn.graph.prepared.sdpa_backward import (
        _is_owner_compute_causal_d128,
    )

    contiguous = lambda b, h, s, d: (h * s * d, s * d, d, 1)
    stats_stride = lambda h, s: (h * s, s, 1, 1)
    for shape in SDPA_BACKWARD_LONG_CAUSAL_D128_CASES:
        b, hq, hkv, sq, skv, d = shape
        q_shape = (b, hq, sq, d)
        kv_shape = (b, hkv, skv, d)
        s_shape = (b, hq, sq, 1)
        assert _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float16,
            causal_top_left=True,
        )
        assert not _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float32,
            causal_top_left=True,
        )
        assert not _is_owner_compute_causal_d128(
            q_shape,
            kv_shape,
            kv_shape,
            q_shape,
            q_shape,
            s_shape,
            contiguous(*q_shape),
            contiguous(*kv_shape),
            contiguous(*kv_shape),
            contiguous(*q_shape),
            contiguous(*q_shape),
            stats_stride(hq, sq),
            torch.float16,
            causal_top_left=False,
        )
```

Run:

```bash
PYTHONPATH=src:. /home/wangbingjie/py312/bin/python3 -m pytest -q \
  tests/test_sdpa_backward.py::test_owner_compute_causal_d128_eligibility
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 2: Add the exact eligibility helper**

Add it above `_prepare_sdpa_backward`. It returns true only for the two exact
shape tuples, fp16/bf16, exact contiguous BHSD/stat strides, D=DV=128, and
pure top-left causal:

```python
def _is_owner_compute_causal_d128(
    q_shape,
    k_shape,
    v_shape,
    o_shape,
    do_shape,
    stats_shape,
    q_stride,
    k_stride,
    v_stride,
    o_stride,
    do_stride,
    stats_stride,
    out_dtype,
    *,
    causal_top_left,
):
    if not causal_top_left or out_dtype not in (
        torch.float16,
        torch.bfloat16,
    ):
        return False
    exact_shapes = {
        (
            (2, 16, 2048, 128),
            (2, 16, 2048, 128),
        ),
        (
            (1, 32, 4096, 128),
            (1, 8, 4096, 128),
        ),
    }
    if (q_shape, k_shape) not in exact_shapes:
        return False
    if v_shape != k_shape or o_shape != q_shape or do_shape != q_shape:
        return False
    if stats_shape != (*q_shape[:3], 1):
        return False

    def bhsd_stride(shape):
        _, heads, seq, dim = shape
        return (heads * seq * dim, seq * dim, dim, 1)

    return (
        q_stride == bhsd_stride(q_shape)
        and k_stride == bhsd_stride(k_shape)
        and v_stride == bhsd_stride(v_shape)
        and o_stride == bhsd_stride(o_shape)
        and do_stride == bhsd_stride(do_shape)
        and stats_stride
        == (q_shape[1] * q_shape[2], q_shape[2], 1, 1)
    )
```

Run the eligibility test. Expected: PASS.

- [ ] **Step 3: Reuse and specialize the delta-only kernel**

Change `_sdpa_bwd_delta_kernel` to use native program IDs and make `SQ`
compile-time for the exact owner path:

```python
pid_m = tl.program_id(0)
pid_bh = tl.program_id(1)
```

Change its signature from `SQ` to `SQ: tl.constexpr`. Existing eager code
does not launch this currently unused kernel, so the prepared owner path is its
only consumer.

- [ ] **Step 4: Add the owner-compute kernel**

Add:

```python
@triton.jit
def _sdpa_bwd_owner_causal_d128_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    do_ptr,
    stats_ptr,
    delta_ptr,
    dq_ptr,
    dk_ptr,
    dv_ptr,
    attn_scale,
    HQ: tl.constexpr,
    Q_PER: tl.constexpr,
    SQ: tl.constexpr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qm: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_dob: tl.constexpr,
    stride_doh: tl.constexpr,
    stride_dom: tl.constexpr,
    stride_dod: tl.constexpr,
    stride_sb: tl.constexpr,
    stride_sh: tl.constexpr,
    stride_sm: tl.constexpr,
    stride_delta_b: tl.constexpr,
    stride_delta_h: tl.constexpr,
    stride_delta_m: tl.constexpr,
    stride_dqb: tl.constexpr,
    stride_dqh: tl.constexpr,
    stride_dqm: tl.constexpr,
    stride_dqd: tl.constexpr,
    stride_dkb: tl.constexpr,
    stride_dkh: tl.constexpr,
    stride_dkn: tl.constexpr,
    stride_dkd: tl.constexpr,
    stride_dvb: tl.constexpr,
    stride_dvh: tl.constexpr,
    stride_dvn: tl.constexpr,
    stride_dvd: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_M_DKDV: tl.constexpr,
    BLOCK_N_DKDV: tl.constexpr,
    BLOCK_M_DQ: tl.constexpr,
    BLOCK_N_DQ: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(BLOCK_M_DKDV == BLOCK_N_DKDV)
    tl.static_assert(BLOCK_M_DQ == BLOCK_N_DQ)
    pid = tl.program_id(0)
    off_b = tl.program_id(1)
    off_kh = tl.program_id(2)
    offs_d = tl.arange(0, BLOCK_D)

    if pid < NUM_N_BLOCKS:
        start_n = pid * BLOCK_N_DKDV
        cols = start_n + tl.arange(0, BLOCK_N_DKDV)
        k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
        v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
        k_tile = tl.load(
            k_base
            + cols[:, None] * stride_kn
            + offs_d[None, :] * stride_kd,
            eviction_policy="evict_last",
        )
        v_tile = tl.load(
            v_base
            + cols[:, None] * stride_vn
            + offs_d[None, :] * stride_vd,
            eviction_policy="evict_last",
        )
        dk = tl.zeros(
            (BLOCK_N_DKDV, BLOCK_D), dtype=tl.float32
        )
        dv = tl.zeros(
            (BLOCK_N_DKDV, BLOCK_D), dtype=tl.float32
        )
        rows_base = tl.arange(0, BLOCK_M_DKDV)
        for off_g in range(0, Q_PER):
            off_h = off_kh * Q_PER + off_g
            q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
            do_base = (
                do_ptr + off_b * stride_dob + off_h * stride_doh
            )
            stats_base = (
                stats_ptr + off_b * stride_sb + off_h * stride_sh
            )
            delta_base = (
                delta_ptr
                + off_b * stride_delta_b
                + off_h * stride_delta_h
            )
            for start_m in tl.range(
                start_n, SQ, BLOCK_M_DKDV
            ):
                rows = start_m + rows_base
                q_tile = tl.load(
                    q_base
                    + rows[:, None] * stride_qm
                    + offs_d[None, :] * stride_qd,
                    eviction_policy="evict_last",
                )
                do_tile = tl.load(
                    do_base
                    + rows[:, None] * stride_dom
                    + offs_d[None, :] * stride_dod,
                    eviction_policy="evict_last",
                )
                stats = tl.load(
                    stats_base + rows * stride_sm,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                delta = tl.load(
                    delta_base + rows * stride_delta_m,
                    eviction_policy="evict_last",
                ).to(tl.float32)
                score = tl.dot(
                    q_tile, tl.trans(k_tile)
                ).to(tl.float32) * (
                    attn_scale * 1.4426950408889634
                )
                if start_m == start_n:
                    valid = cols[None, :] <= rows[:, None]
                    p = tl.where(
                        valid,
                        tl.exp2(score - stats[:, None] * 1.4426950408889634),
                        0.0,
                    )
                else:
                    p = tl.exp2(
                        score - stats[:, None] * 1.4426950408889634
                    )
                dp = tl.dot(
                    do_tile, tl.trans(v_tile)
                ).to(tl.float32)
                ds = p * (dp - delta[:, None])
                dk += tl.dot(
                    tl.trans(ds).to(q_tile.dtype), q_tile
                )
                dv += tl.dot(
                    tl.trans(p).to(do_tile.dtype), do_tile
                )
        tl.store(
            dk_ptr
            + off_b * stride_dkb
            + off_kh * stride_dkh
            + cols[:, None] * stride_dkn
            + offs_d[None, :] * stride_dkd,
            (dk * attn_scale).to(dk_ptr.dtype.element_ty),
        )
        tl.store(
            dv_ptr
            + off_b * stride_dvb
            + off_kh * stride_dvh
            + cols[:, None] * stride_dvn
            + offs_d[None, :] * stride_dvd,
            dv.to(dv_ptr.dtype.element_ty),
        )
    else:
        query_pid = pid - NUM_N_BLOCKS
        off_g = query_pid // NUM_M_BLOCKS
        pid_m = query_pid % NUM_M_BLOCKS
        off_h = off_kh * Q_PER + off_g
        start_m = pid_m * BLOCK_M_DQ
        rows = start_m + tl.arange(0, BLOCK_M_DQ)
        q_base = q_ptr + off_b * stride_qb + off_h * stride_qh
        k_base = k_ptr + off_b * stride_kb + off_kh * stride_kh
        v_base = v_ptr + off_b * stride_vb + off_kh * stride_vh
        do_base = do_ptr + off_b * stride_dob + off_h * stride_doh
        stats_base = (
            stats_ptr + off_b * stride_sb + off_h * stride_sh
        )
        delta_base = (
            delta_ptr
            + off_b * stride_delta_b
            + off_h * stride_delta_h
        )
        q_tile = tl.load(
            q_base
            + rows[:, None] * stride_qm
            + offs_d[None, :] * stride_qd,
            eviction_policy="evict_last",
        )
        do_tile = tl.load(
            do_base
            + rows[:, None] * stride_dom
            + offs_d[None, :] * stride_dod,
            eviction_policy="evict_last",
        )
        stats = tl.load(
            stats_base + rows * stride_sm,
            eviction_policy="evict_last",
        ).to(tl.float32)
        delta = tl.load(
            delta_base + rows * stride_delta_m,
            eviction_policy="evict_last",
        ).to(tl.float32)
        dq = tl.zeros((BLOCK_M_DQ, BLOCK_D), dtype=tl.float32)
        cols_base = tl.arange(0, BLOCK_N_DQ)
        for start_n in tl.range(
            0, start_m + BLOCK_M_DQ, BLOCK_N_DQ
        ):
            cols = start_n + cols_base
            k_tile = tl.load(
                k_base
                + cols[:, None] * stride_kn
                + offs_d[None, :] * stride_kd,
                eviction_policy="evict_last",
            )
            v_tile = tl.load(
                v_base
                + cols[:, None] * stride_vn
                + offs_d[None, :] * stride_vd,
                eviction_policy="evict_last",
            )
            score = tl.dot(
                q_tile, tl.trans(k_tile)
            ).to(tl.float32) * (
                attn_scale * 1.4426950408889634
            )
            if start_n + BLOCK_N_DQ <= start_m:
                p = tl.exp2(
                    score - stats[:, None] * 1.4426950408889634
                )
            else:
                valid = cols[None, :] <= rows[:, None]
                p = tl.where(
                    valid,
                    tl.exp2(
                        score
                        - stats[:, None] * 1.4426950408889634
                    ),
                    0.0,
                )
            dp = tl.dot(
                do_tile, tl.trans(v_tile)
            ).to(tl.float32)
            ds = p * (dp - delta[:, None])
            dq += tl.dot(ds.to(k_tile.dtype), k_tile)
        tl.store(
            dq_ptr
            + off_b * stride_dqb
            + off_h * stride_dqh
            + rows[:, None] * stride_dqm
            + offs_d[None, :] * stride_dqd,
            (dq * attn_scale).to(dq_ptr.dtype.element_ty),
        )
```

- [ ] **Step 5: Add owner configs**

Add singleton starting configs:

```yaml
sdpa_backward_owner_mha_causal_d128:
  - gen: true
    param_map:
      META:
        BLOCK_M_DKDV: block_m_dkdv
        BLOCK_N_DKDV: block_n_dkdv
        BLOCK_M_DQ: block_m_dq
        BLOCK_N_DQ: block_n_dq
        BLOCK_D: block_d
      num_warps: warps
      num_stages: stages
    block_m_dkdv: [64]
    block_n_dkdv: [64]
    block_m_dq: [64]
    block_n_dq: [64]
    block_d: [128]
    warps: [4]
    stages: [2]

sdpa_backward_owner_gqa_causal_d128:
  - gen: true
    param_map:
      META:
        BLOCK_M_DKDV: block_m_dkdv
        BLOCK_N_DKDV: block_n_dkdv
        BLOCK_M_DQ: block_m_dq
        BLOCK_N_DQ: block_n_dq
        BLOCK_D: block_d
      num_warps: warps
      num_stages: stages
    block_m_dkdv: [64]
    block_n_dkdv: [64]
    block_m_dq: [64]
    block_n_dq: [64]
    block_d: [128]
    warps: [4]
    stages: [2]
```

- [ ] **Step 6: Wire the prepared two-step pipeline**

Compute `owner_kind` with `_is_owner_compute_causal_d128`. Load the MHA or
GQA config using `_single_tuned_config_kwargs`. Extend
`make_bwd_context` so owner mode allocates fp32 delta.

Return before the current decode/mloop/GQA branches with:

```python
num_n_blocks = triton.cdiv(skv, owner_config["BLOCK_N_DKDV"])
num_m_blocks = triton.cdiv(sq, owner_config["BLOCK_M_DQ"])
owner_grid = (
    num_n_blocks + q_per_k * num_m_blocks,
    batch,
    kv_heads,
)
delta_grid = (
    triton.cdiv(sq, 64),
    batch * heads,
    1,
)
```

Define the exact argument tails and runtime closures:

```python
delta_tail = (
    heads,
    sq,
    *o_stride,
    *do_stride,
    *delta_stride,
    128,
    64,
    128,
)
owner_tail = (
    attn_scale,
    heads,
    q_per_k,
    sq,
    *q_stride,
    *k_stride,
    *v_stride,
    *do_stride,
    stats_stride[0],
    stats_stride[1],
    stats_stride[2],
    *delta_stride,
    *q_stride,
    *k_stride,
    *v_stride,
    num_n_blocks,
    num_m_blocks,
    owner_config["BLOCK_M_DKDV"],
    owner_config["BLOCK_N_DKDV"],
    owner_config["BLOCK_M_DQ"],
    owner_config["BLOCK_N_DQ"],
    owner_config["BLOCK_D"],
)

def owner_delta_runtime_args(inputs, context):
    delta = context[3]
    assert isinstance(delta, torch.Tensor)
    return inputs[3], inputs[4], delta

def owner_runtime_args(inputs, context):
    delta = context[3]
    assert isinstance(delta, torch.Tensor)
    return (
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[4],
        inputs[5],
        delta,
        context[0],
        context[1],
        context[2],
    )

delta_step = PreparedPipelineStepSpec(
    kernel=_sdpa_bwd_delta_kernel,
    grid=delta_grid,
    runtime_args=owner_delta_runtime_args,
    static_args=delta_tail[:-3],
    constexpr_kwargs={
        "V_DIM": 128,
        "BLOCK_M": 64,
        "BLOCK_DV": 128,
        "num_warps": 4,
        "num_stages": 2,
    },
    build_cached_call=make_static_cached_call(delta_grid, delta_tail),
)
owner_step = PreparedPipelineStepSpec(
    kernel=_sdpa_bwd_owner_causal_d128_kernel,
    grid=owner_grid,
    runtime_args=owner_runtime_args,
    static_args=owner_tail[:-5],
    constexpr_kwargs=owner_config,
    build_cached_call=make_static_cached_call(owner_grid, owner_tail),
)
return make_kernel_pipeline_run_fn(
    PreparedKernelPipelineSpec(
        steps=(delta_step, owner_step),
        input_checks=bwd_input_checks,
        context_factory=make_bwd_context,
        result=bwd_result,
    ),
    default_run_fn,
)
```

Do not include `zero_step`. Keep eager `sdpa_backward()` unchanged. Enable
MHA and GQA routing independently only when each route beats its retained
Task 2 path.

- [ ] **Step 7: Run correctness and tune narrowly**

Run:

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  tests/test_sdpa_backward.py -k 'owner_compute or long_causal_d128'
```

Expected: PASS.

Run the hard performance gate from Task 2 twice. If the singleton does not
pass, test these configurations one at a time:

1. stages 3;
2. 8 warps, stages 3;
3. dK/dV tiles 64 and dQ tiles 32;
4. dK/dV tiles 32 and dQ tiles 64.

Retain one config per routed shape. Remove owner routing for a shape if no
correct candidate beats the Task 2 path and clears `0.9`.

- [ ] **Step 8: Add compiled-graph new-storage coverage if owner routing survives**

Add this no-bias compile helper:

```python
def _compile_flag_dnn_sdpa_backward_graph(
    q, k, v, o, dO, stats, *, right_bound
):
    @flag_dnn.graph
    def fn(q, k, v, o, dO, stats):
        return flag_dnn.sdpa_backward(
            q,
            k,
            v,
            o,
            dO,
            stats,
            diagonal_alignment="TOP_LEFT",
            diagonal_band_right_bound=right_bound,
            name="sdpa_backward",
        )

    compiled = flag_dnn.compile(
        fn,
        inputs=[
            flag_dnn.TensorSpec.from_tensor(q, "q"),
            flag_dnn.TensorSpec.from_tensor(k, "k"),
            flag_dnn.TensorSpec.from_tensor(v, "v"),
            flag_dnn.TensorSpec.from_tensor(o, "o"),
            flag_dnn.TensorSpec.from_tensor(dO, "dO"),
            flag_dnn.TensorSpec.from_tensor(stats, "stats"),
        ],
        options={"cache": None},
    )
    assert [node.op_type for node in compiled.graph.nodes] == [
        "sdpa_backward"
    ]
    return compiled
```

Add:

```python
@pytest.mark.sdpa_backward
@pytest.mark.graph
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "shape",
    SDPA_BACKWARD_LONG_CAUSAL_D128_CASES,
    ids=("mha_b2_h16_s2048", "gqa_b1_h32_hkv8_s4096"),
)
def test_sdpa_backward_long_causal_rebinds_storage(
    cudnn_handle, dtype, shape
):
    torch.manual_seed(29)
    q_a, k_a, v_a = _make_qkv(shape, dtype)
    dO_a = torch.randn_like(q_a)
    o_a, stats_a = _cudnn_sdpa_forward(
        q_a, k_a, v_a, cudnn_handle, right_bound=0
    )
    compiled = _compile_flag_dnn_sdpa_backward_graph(
        q_a, k_a, v_a, o_a, dO_a, stats_a, right_bound=0
    )

    def check(q, k, v, o, dO, stats):
        expected = _cudnn_sdpa_backward(
            q, k, v, o, dO, stats, cudnn_handle, right_bound=0
        )
        actual = compiled.run(q, k, v, o, dO, stats)
        _assert_grads_close(actual, expected, dtype)
        return actual

    grads_a = check(q_a, k_a, v_a, o_a, dO_a, stats_a)
    q_b, k_b, v_b = _make_qkv(shape, dtype)
    dO_b = torch.randn_like(q_b)
    o_b, stats_b = _cudnn_sdpa_forward(
        q_b, k_b, v_b, cudnn_handle, right_bound=0
    )
    grads_b = check(q_b, k_b, v_b, o_b, dO_b, stats_b)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(grads_a, grads_b)
    )

    q_b.normal_()
    k_b.normal_()
    v_b.normal_()
    dO_b.normal_()
    o_c, stats_c = _cudnn_sdpa_forward(
        q_b, k_b, v_b, cudnn_handle, right_bound=0
    )
    grads_c = check(q_b, k_b, v_b, o_c, dO_b, stats_c)
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(grads_b, grads_c)
    )
```

Run the targeted owner tests. Expected: PASS.

- [ ] **Step 9: Commit the retained owner path**

```bash
git add tests/test_sdpa_backward.py \
  src/flag_dnn/ops/sdpa_backward.py \
  src/flag_dnn/graph/prepared/sdpa_backward.py \
  src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml
git commit -m "perf: add owner compute causal d128 backward"
```

### Task 4: Final Regression And Performance Evidence

**Files:**
- Verify: all files changed in Tasks 1-3
- Append locally only: `.kernel-generate/optimization-log-sdpa.md`
- Append locally only: `.kernel-generate/optimization-log-sdpa_backward.md`

**Interfaces:**
- Consumes: retained forward and backward implementations.
- Produces: final correctness, static, target-performance, and canonical-regression evidence.

- [ ] **Step 1: Run static checks**

```bash
/home/wangbingjie/py312/bin/python3 -m py_compile \
  src/flag_dnn/ops/sdpa.py \
  src/flag_dnn/graph/prepared/sdpa_forward.py \
  src/flag_dnn/ops/sdpa_backward.py \
  src/flag_dnn/graph/prepared/sdpa_backward.py \
  tests/test_sdpa.py \
  tests/test_sdpa_backward.py
git diff --check
```

Expected: both commands exit 0.

- [ ] **Step 2: Run complete standard SDPA correctness**

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-tests.XXXXXX)" \
  /home/wangbingjie/py312/bin/python3 -m pytest -q \
  tests/test_sdpa.py tests/test_sdpa_backward.py
```

Expected: PASS with only documented pre-existing skips.

- [ ] **Step 3: Run two independent final target gates**

For run A and run B, create a new `mktemp` cache and run both hard-gated
commands from Tasks 1 and 2 with warmup 30/repeat 300.

Expected in both runs:

- forward ID 2 fp16/bf16 `>=0.9`;
- forward ID 5 fp16/bf16 `>=0.9`;
- backward ID 2 fp16/bf16 `>=0.9`;
- backward ID 5 fp16/bf16 `>=0.9`.

- [ ] **Step 4: Run canonical IDs 0-7**

```bash
env LD_LIBRARY_PATH=/root/micromamba-root/envs/toolchain/lib: \
  PYTHONPATH=src:. \
  TRITON_CACHE_DIR="$(mktemp -d /tmp/flagdnn-sdpa-canonical.XXXXXX)" \
  FLAGDNN_CUDNN_PERF_WARMUP=25 \
  FLAGDNN_CUDNN_PERF_REPEAT=100 \
  FLAGDNN_CUDNN_SDPA_PERF_SHAPE_IDS=0,1,2,3,4,5,6,7 \
  FLAGDNN_CUDNN_SDPA_BACKWARD_PERF_SHAPE_IDS=0,1,2,3,4,5,6,7 \
  /home/wangbingjie/py312/bin/python3 -m pytest -q -s \
  'benchmark/test_sdpa.py::test_sdpa[dtype0]' \
  'benchmark/test_sdpa.py::test_sdpa[dtype1]' \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype0]' \
  'benchmark/test_sdpa_backward.py::test_sdpa_backward[dtype1]'
```

Expected: no unaffected canonical row regresses by more than 3% versus the
recorded baseline, and no previously passing row falls below `0.9`.

- [ ] **Step 5: Record evidence without staging generated logs**

Append the retained/rejected experiment, exact commands, correctness results,
and both final timing tables to the two existing optimization logs. Then verify:

```bash
git status --short
git diff --cached --name-only
```

Expected: `.kernel-generate`, benchmark logs, and test logs remain untracked
or unstaged. Only intentional source/test/config commits are present.

- [ ] **Step 6: Review final commits**

```bash
git log --oneline --decorate -6
git diff origin/master...HEAD -- \
  src/flag_dnn/ops/sdpa.py \
  src/flag_dnn/graph/prepared/sdpa_forward.py \
  src/flag_dnn/ops/sdpa_backward.py \
  src/flag_dnn/graph/prepared/sdpa_backward.py \
  src/flag_dnn/runtime/backend/_nvidia/tune_configs.yaml \
  tests/test_sdpa.py \
  tests/test_sdpa_backward.py
```

Expected: the diff contains no FP8 source changes, no fallback calls, no
computed replay caches, and no unrelated refactors.
