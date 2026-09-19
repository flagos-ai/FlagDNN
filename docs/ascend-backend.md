# Ascend backend

The backend owns dispatch, source assets, launch generation and validation.
It does not import another backend at runtime. Production execution uses
AscendCL and the pinned NPU libtriton_jit; ACLNN dependencies belong to the
validation targets only.

## Compiler layout

- `compiler.py` validates the request identity and orchestrates compilation.
- `dispatch/` validates tensors and attributes, selects operator plans, and
  assigns graph workspace. Existing persistent plans retain their tuning tables.
- All operator planners live directly in `dispatch/`, including exact storage
  operations and mixed input/output dtypes. `dispatch/extended.py` expands
  concatenate and attention backward into ordered stages. Kernel variants live
  together by operator family directly in `kernels/`, following NVIDIA's layout.
- `codegen/` materializes immutable source assets and emits signatures,
  arguments, launch grids and candidates. Persistent programs use schema 3;
  extended programs use schema 4 and a fixed candidate. `capabilities.json`
  retains the persistent-program contract; extended dtype/operator validation
  lives in the corresponding operator dispatch modules. Schema 4 does not claim
  to autotune kernels.
- `artifact.cpp` and `extended_artifact.cpp` validate artifacts before execution.
  Private SDPA delta storage is reconstructed from the request. Kernel source
  hashes are pinned at build time; compiler identity covers dispatch, codegen,
  kernels and the code generation environment.

Extended kernels compile through the pinned standalone compiler before their
first launch. FlagDNN owns their compiler scratch allocations and reuses them
for execution; this avoids relying on libtriton_jit's raw-launch cleanup
callbacks, which require an application callback-report thread.

Contiguous convolution weight gradients use a separate tiled kernel. Dispatch
retains 16-by-16 tiles for narrow stems and selects 32-by-32 tiles for large
filters. Inactive gather lanes receive in-range addresses as well as load masks,
avoiding invalid addresses in CANN's gather lowering.
For low-channel stride-2 input gradients, dispatch separates even and odd
coordinates and accumulates in FP32, avoiding channel padding for matrix tiles.

The older `add_plan.py` remains a compatibility export for existing integration
contracts. New implementation belongs in the dispatch/codegen packages.

`engines/` follows NVIDIA's two-file structure: `engine.hpp` declares one concrete
`ExecutionEngine`, and `libtriton_jit.cpp` owns its private implementation. There
is no extra engine factory or virtual dispatch layer. Process lifetime Python
ownership and private runtime discovery belong to `context.cpp`; compiler stdout
containment and raw launch arguments remain private to the JIT implementation.

## Validation

Functional workloads come from `tests/common`; performance workloads come from
`benchmark/common` or the shared native operator catalogs, following the NVIDIA
suite registration. Ascend does not maintain reduced copies of those catalogs.
Integer and raw-storage supplements run in separate CTest processes and compare
logical bytes exactly. Floating comparisons preserve the shared tolerances and
check output padding and repeated execution.

`validation/functional/aclnn_plan.hpp` owns native descriptors, repeatable
executors and scratch storage. Where ACLNN has no matching fused primitive,
references compose ACLNN operations (for example attention, normalization
backward and grouped matmul). Their benchmark measures the complete native
composition. Saved forward attention state is prepared before timing a backward
operator. Native executors are prepared before all steady-state measurements.

Both providers report stream, host submission and end-to-end timings. The
Ascend runner adapter maps stream microseconds into the repository's legacy
performance summary and retains all native metrics. Capability gaps are explicit
case skips or CTest skip code 77; mixed supported/skipped dtype suites retain
their successful cases. Native sidecars retain explicit skip reasons.

Compiler, artifact and native-plan contracts live directly in `validation/`,
alongside the functional and benchmark suites, as in the NVIDIA backend.

Run the same public entry point as other backends:

```sh
python3 tools/run_tests.py --platform ascend --device 0 --ops all --suites all
```

`--ops all` includes the dtype/precision supplements and extended benchmarks.
The supported hardware/dtype limits and skip reasons are recorded in
[`operator-support.md`](operator-support.md).

Large convolution and reduction catalogs can take substantially longer than
pointwise tests because each shape/dtype builds an independent graph and kernel.
The Ascend adapter allows 7200 seconds per operator invocation; CTest also keeps
individual suite limits, including 7200 seconds for convolution-gradient
benchmarks. Do not reduce the shared workload catalog to shorten qualification.
