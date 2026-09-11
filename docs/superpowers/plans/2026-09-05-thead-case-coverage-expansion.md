# THead Case Coverage Expansion Implementation Plan

> **Status:** Complete on 2026-09-06. All feasible public cases are qualified against acDNN;
> residual unsupported cases are closed with structured SDK evidence.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The user explicitly prohibited subagents and commits, so execution is inline in the current checkout.

**Goal:** Expand every feasible public THead functional and benchmark case from structured skip to a real FlagDNN Graph versus acDNN execution, while preserving explicit audited skips for cases that the installed acDNN SDK cannot represent reliably.

**Architecture:** Keep all changes under `backends/thead/**` and extend the existing fail-closed compiler, validation reference, tensor I/O, capability catalog, and paired benchmark pipeline. Qualify a case only after the production Triton path and the acDNN-only reference path both execute on PPU with the same typed, strided data and pass numerical comparison; benchmark pairing is enabled only after functional qualification of the same semantic slice.

**Tech Stack:** C++20, Python 3.12, CMake/Ninja, FlagDNN Frontend Graph IR v3, system THead-customized Triton, libtriton_jit CUDA-compatibility backend, PPU CUDA Driver API, acDNN.

**Spec:** `backends/thead/adaptation-recommendations.md`

## Global Constraints

- Modify only `backends/thead/**` and this THead plan document; do not modify common code, another backend, FlagTree, or libtriton_jit.
- Functional and performance reference execution uses acDNN only; do not introduce BLAS, PyTorch, CPU numerical or self-reference oracles.
- Production `flagdnn_backend_thead` must not link acDNN; acDNN remains validation-only.
- Unknown compiler, JIT, Graph, driver, artifact, and acDNN errors fail; only a catalogued SDK limitation may skip.
- Use `/usr/local/lib/python3.12/site-packages/triton`; keep `TRITON_JIT_BACKEND=CUDA` for libtriton_jit compatibility.
- Preserve the public manifests and account for every case exactly once as executed/paired or structured skip.
- Do not change `tests/core/run_tests_contract.py` and do not create commits.

---

### Task 1: Typed validation tensor I/O

**Files:**
- Create: `backends/thead/validation/numeric_types.hpp`
- Create: `backends/thead/validation/numeric_types.cpp`
- Modify: `backends/thead/validation/functional/add_runner.cpp`
- Modify: `backends/thead/validation/benchmark/runner.cpp`
- Modify: `backends/thead/validation/CMakeLists.txt`
- Test: `backends/thead/validation/validation_contract.cpp`

**Interfaces:**
- Produces: `element_size(flagdnnDataType_t)`, `encode_floating(...)`, and `decode_floating(...)` for FP32, FP16, BF16 and boolean storage without an external numerical library.
- Consumes: public `flagdnnDataType_t` values and byte spans used by all THead validation runners.

- [x] Add contract assertions for IEEE round-to-nearest-even FP16/BF16 encoding, signed zero, finite values, infinities, NaN, byte counts, and unsupported FP8 rejection.
- [x] Build and run `integration.thead.validation_contract`; confirm the new assertions fail because typed helpers do not exist.
- [x] Implement the minimal endian-stable typed conversion helpers and replace FP32-only allocation/upload/download logic in the shared functional and benchmark buffer paths.
- [x] Rebuild and run the validation contract plus one existing FP32 functional and benchmark suite; confirm existing behavior remains green.

### Task 2: Dense FP16/BF16 pointwise qualification

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/validation/pointwise_reference.cpp`
- Modify: `backends/thead/validation/backend_pointwise_reference.cpp`
- Modify: `backends/thead/validation/functional/add_runner.cpp`
- Modify: `backends/thead/validation/benchmark/runner.cpp`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: `backends/thead/validation/compiler_contract.py`
- Test: real `functional.thead.*` and `benchmark.thead.*` CTest cases selected by per-case environment filters.

**Interfaces:**
- Produces: same-dtype FP16/BF16 binary/unary Graph lowering with FP32 compute, matching acDNN tensor descriptors, and dtype-aware tolerances.
- Consumes: typed tensor I/O from Task 1 and existing common contiguous Triton kernels without editing them.

- [x] Add compiler contract cases that request dense FP16 and BF16 unary/binary graphs and assert typed pointer signatures, storage sizes, and FP32 compute specialization; confirm they fail at the current FP32-only validator.
- [x] Generalize only the THead compiler validators and add FP16/BF16 mappings to both primitive and backend-descriptor acDNN reference builders.
- [x] Promote a single small FP16 and BF16 case per reference family to `probe_required`, run it on PPU with `FLAGDNN_THEAD_QUALIFY_PROBES=1`, and classify any exact acDNN rejection with its observed status.
- [x] Promote all dense same-shape/default-attribute cases in families whose representatives pass, execute every promoted functional case, and retain structured SDK-specific skips for families whose representatives fail.
- [x] Add the matching public benchmark cases, run every new pair, and verify each case produces exactly one FlagDNN and one acDNN timing record.

### Task 3: Strided/NHWC pointwise qualification

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/validation/compiler_contract.py`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: selected real pointwise functional and benchmark CTest cases.

**Interfaces:**
- Produces: rank-1-through-8 pointwise physical-offset metadata and selection of existing `binary_strided_kernel` / `unary_pointwise_strided_kernel` when any participating tensor is non-dense.
- Consumes: non-overlapping tensor strides and identical logical shapes; broadcast remains a separate decision.

- [x] Add compiler contract RED cases for unary and binary NHWC/explicit-strided tensors, including overlapping-stride and mismatched-shape negative cases.
- [x] Implement padded dimension/stride constants, strided kernel signatures, registry validation, and launch manifests in the THead compiler.
- [x] Probe representative FP32, FP16, and BF16 NHWC cases against acDNN on PPU; then promote and execute every same-shape strided case supported by both paths.
- [x] Pair all matching benchmark cases and verify output padding remains unchanged for both providers.

### Task 4: Attributes, broadcast, composite and layout coverage

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/validation/pointwise_reference.cpp`
- Modify: `backends/thead/validation/acdnn_composite_reference.cpp`
- Modify: `backends/thead/validation/acdnn_layout_reference.cpp`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: compiler contracts and real Add, Sub, ReLU, identity, AddSquare, reshape, transpose and slice suites.

**Interfaces:**
- Produces: exact alpha/slope/default-attribute lowering, supported NumPy trailing broadcast offsets, typed AddSquare and typed/strided layout copies.
- Consumes: typed/strided infrastructure from Tasks 1-3 and only exact acDNN primitive/DAG semantics.

- [x] Add RED contracts for Add alpha, ReLU slope, binary broadcast, typed AddSquare, and typed layout transforms.
- [x] Implement only combinations expressible identically by the production kernel and acDNN reference; keep semantic mismatches fail-closed.
- [x] Run every promoted functional case on PPU, including sentinel checks for binding offsets and padding.
- [x] Pair every functionally qualified public benchmark case and account for remaining attribute/broadcast cases with observed reason codes.

### Task 5: Reduction and normalization coverage

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/kernels/normalization.py`
- Modify: `backends/thead/validation/acdnn_reduction_reference.cpp`
- Modify: `backends/thead/validation/acdnn_normalization_reference.cpp`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: compiler contracts and real reduction, BatchNorm, BatchNormInference, LayerNorm and RMSNorm suites.

**Interfaces:**
- Produces: qualified dtypes, axes, normalized extents, layouts and statistics with explicit FP32 accumulation where required.
- Consumes: existing reduction and normalization Graph builders and acDNN primitive/DAG reference plans.

- [x] Add RED compiler contracts for each public dtype/axis/shape class currently skipped.
- [x] Remove arbitrary normalization extent allowlists where the generated launch is safe, and generalize validation-only acDNN descriptors to the dtype/layout combinations accepted by the installed runtime.
- [x] Probe one representative for each new dtype/axis/layout class, then run all promoted functional cases.
- [x] Pair the corresponding benchmark cases and preserve skips for acDNN modes such as unsupported reduction products or layouts only when observed on PPU.

### Task 6: MatMul coverage

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/validation/acdnn_matmul_reference.cpp`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: compiler contracts and real MatMul functional/benchmark suites.

**Interfaces:**
- Produces: FP16/BF16/FP32 batched and broadcast MatMul using the existing Triton strided kernel and acDNN backend MATMUL descriptor, with dtype-appropriate accumulation and tolerances.

- [x] Add RED contracts for FP16/BF16, rank-2, batched, broadcast, and legal non-overlapping stride forms.
- [x] Generalize the THead MatMul validator, signatures and acDNN tensor descriptors without adding a BLAS dependency.
- [x] Probe small representatives before large public cases; run all feasible functional cases and classify exact SDK/runtime failures.
- [x] Run all feasible benchmark pairs with bounded serial execution and validate record pairing.

### Task 7: Convolution coverage

**Files:**
- Modify: `backends/thead/compiler.py`
- Modify: `backends/thead/kernels/convolution.py`
- Modify: `backends/thead/validation/acdnn_convolution_reference.cpp`
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Test: compiler contracts and real ConvFprop, ConvDgrad, ConvWgrad and ConvBiasRelu suites.

**Interfaces:**
- Produces: all feasible public dimensionality, dtype, stride, padding and dilation combinations, while retaining acDNN algorithm selection as the sole reference.

- [x] Add RED compiler contracts for 1D/2D/3D, FP16/BF16 and representative asymmetric/dilated geometry.
- [x] Generalize only THead kernels, compiler metadata and acDNN descriptors needed by public cases; validate storage spans and int32 launch bounds.
- [x] Probe small cases before YOLO and large spatial cases; execute feasible functional cases serially and record observed acDNN algorithm limitations.
- [x] Pair feasible benchmarks serially, preserving deterministic skips for memory, runtime or algorithm limits demonstrated by the installed SDK.

### Task 8: Unsupported-reference audit and final gates

**Files:**
- Modify: `backends/thead/validation/capability.json`
- Modify: `backends/thead/validation/benchmark/comparable_cases.json`
- Modify: `backends/thead/adaptation-recommendations.md`
- Modify: `docs/superpowers/plans/2026-09-03-thead-operator-coverage.md`
- Modify: this plan document.
- Test: the complete THead build, CTest label, compiler contract and `tools/run_tests.py` workflow.

**Interfaces:**
- Produces: final exhaustive executed/skip accounting and documented evidence for every remaining skip.

- [x] Re-probe only exact acDNN candidates for Mod, LeakyReLU, logical operators, BinarySelect, Erf and SDPA; do not substitute a non-acDNN oracle.
- [x] Validate catalog closure, stable skip schema, benchmark pairing, dependency boundaries, Graph/JIT/autotune/install contracts and warnings-as-errors compilation.
- [x] Run `cmake --build build/thead -j2` and `ctest --test-dir build/thead --output-on-failure -L thead -j1`.
- [x] Run `tools/run_tests.py --platform thead --build-dir build/thead --ops all --suites all --no-preflight`
  and inspect every failed, skipped and paired record. The final invocation used an isolated reflink of the
  CTest metadata to avoid log-file collision while executing the same final `build/thead` binaries.
- [x] Run independent real Triton compilation, ELF dependency checks, `git diff --check`, forbidden-path diff checks, and update both THead documents with fresh counts and residual SDK blockers.

**Final evidence (2026-09-06):** 247 CTest entries close as 231 passed plus 16 expected skip after
the corrected 900-second exhaustive-compile recheck; the 118-task runner closes as 102 passed plus
16 skip. Functional coverage is `1052/1142` executed with 90 audited skip, benchmark coverage is
`1125/1182` paired with 57 audited skip, and both catalogs have zero pending probes. All 1125 required
pairs and 2250 provider records are present, 147 skip records pass schema validation, and all dependency,
accounting and diff-hygiene error counts are zero. The recorded ratio has no threshold, so it is evidence
that the paired performance path works rather than a blanket performance-target claim.
