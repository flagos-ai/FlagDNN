# THead Backend Comprehensive Audit and Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to execute this plan
> task-by-task. The user prohibits subagents, worktree migration and commits for this audit, so all work
> is performed inline in the current checkout and every edit remains uncommitted.

**Goal:** Prove that the current THead backend is architecturally sound, contains no serious known bug
or unused repository file, does not regress common or other-platform behavior, and completes build,
install, functional, performance and `tools/run_tests.py` workflows; repair any issue found.

**Architecture:** Audit production and validation code separately. Use `backends/nvidia` as the reference
for ambiguous backend ABI, Graph lifecycle, engine loading, artifact and installation behavior, while
preserving the THead-specific system Triton plus CUDA-compatible `libtriton_jit` execution model and the
acDNN-only validation oracle. Keep fixes under `backends/thead/**` unless a demonstrated common-code bug
cannot be fixed locally without violating an existing public contract.

**Tech Stack:** C++20, Python 3.12, CMake/Ninja, FlagDNN Frontend Graph IR v3, THead Triton
`3.5.0+ppu2.0.0.oe`, CUDA-compatible `libtriton_jit`, PPU CUDA Driver API and acDNN
`2.0.0-715aa1`/runtime ABI 1400.

**Spec:** `backends/thead/adaptation-recommendations.md`

## Global Constraints

- Do not use subagents and do not create commits.
- Do not modify `libtriton_jit`, FlagTree, `backends/nvidia/**` or another platform backend.
- Functional and performance references must execute acDNN only; no BLAS, PyTorch, CPU or self oracle.
- Preserve structured skip only for an exact catalogued acDNN limitation; unknown failures must fail.
- Treat `tests/core/run_tests_contract.py` as a previously deferred common issue unless this audit proves
  that the THead adapter introduced its failure.
- Use `apply_patch` for source edits and preserve unrelated user changes.

---

### Task 1: Change boundary, repository hygiene and file reachability

**Files:**
- Inspect: all paths reported by `git status --short`, including untracked THead and plan files.
- Inspect: `backends/thead/CMakeLists.txt`
- Inspect: `backends/thead/validation/CMakeLists.txt`
- Inspect: install manifests under `build/thead`
- Modify only if needed: unused or generated files under `backends/thead/**`

**Interfaces:**
- Consumes: current dirty checkout at HEAD `86604480d0e77436a185d0398f68b3b964f0a0a7`.
- Produces: an exact allowed-path list and proof that every retained source/data file is referenced by
  CMake, Python import, runtime registry, capability catalog, documentation or an explicit contract.

- [x] Enumerate tracked, untracked, ignored and staged changes; fail if any unexpected common or other
  backend file is modified.
- [x] Classify every THead file as production source, installed runtime asset, validation-only source,
  contract, capability data or documentation.
- [x] Search for build products, cache files, temporary logs, duplicate sources, stale manifests,
  `__pycache__`, `.pyc`, editor files, TODO placeholders and unreachable files.
- [x] Cross-check CMake target source lists, install rules, Python imports and JSON registry paths; remove
  only files proven redundant.

### Task 2: Production ABI, lifecycle and NVIDIA-reference architecture audit

**Files:**
- Inspect: `backends/thead/backend.cpp`, `context.*`, `error.*`, `artifact.*`
- Inspect: `backends/thead/engines/**`
- Compare: corresponding interfaces and lifecycle choices under `backends/nvidia/**`
- Test: THead integration contracts and ABI/dependency inspection

**Interfaces:**
- Consumes: FlagDNN backend plugin ABI, Graph lifecycle and engine interfaces.
- Produces: a fail-closed THead plugin with correct ownership, error propagation, stream behavior,
  artifact validation, concurrency and Graph capture semantics.

- [x] Compare exported backend ABI, context creation/destruction, stream ownership and error mapping with
  NVIDIA; document intentional THead differences.
- [x] Audit every allocation, file descriptor, Python object, driver handle, `libtriton_jit` object and
  Graph executable lifetime for leak, double-free, use-after-free and exception-boundary hazards.
- [x] Audit integer conversions, shape/stride arithmetic, workspace sizing and launch argument packing for
  overflow, aliasing, alignment and empty-tensor errors.
- [x] Audit artifact cache locking, atomic publication, symlink/path traversal resistance, identity
  invalidation and concurrent reader/writer behavior.
- [x] Audit Graph capture, non-default streams, event ordering, repeated execute and autotune candidate
  compatibility; add a regression contract before any behavior fix.

### Task 3: Compiler, kernel and tuning correctness audit

**Files:**
- Inspect: `backends/thead/compiler.py`, `compiler_identity.py`, `python_environment_identity.py`
- Inspect: `backends/thead/kernels/**`, `backends/thead/tuning/**`
- Compare: NVIDIA code generation and engine contracts where semantics overlap.
- Test: `compiler_contract.py`, Triton identity and real compilation contracts

**Interfaces:**
- Consumes: Graph IR v3 and registry metadata.
- Produces: deterministic Triton source/signature/launch metadata and real PPU cubin artifacts for every
  qualified operator family.

- [x] Parse every Python source with the current interpreter and run repository compiler contracts.
- [x] Audit dtype/layout/stride/broadcast/reduction/normalization/MatMul/convolution lowering against the
  public Graph semantics and validation case catalog.
- [x] Audit generated source escaping, signature construction, pointer/scalar ABI, grid calculations,
  autotune metadata and launch bounds.
- [x] Verify compiler identity contains all semantic/toolchain inputs without unstable or irrelevant
  process state; verify system Triton is used and FlagTree is not imported.
- [x] Run exhaustive real compilation under the corrected timeout and repair any reproducible failure.

### Task 4: acDNN-only functional and benchmark oracle audit

**Files:**
- Inspect: `backends/thead/validation/acdnn_*`, `backend_pointwise_reference.*`,
  `pointwise_reference.*`, `numeric_types.*`, `tensor_io.*`, `ppu_driver.hpp`
- Inspect: `backends/thead/validation/functional/**`
- Inspect: `backends/thead/validation/benchmark/**`
- Inspect: `capability.*`, `capability.json`, `benchmark/comparable_cases.json`

**Interfaces:**
- Consumes: public functional and benchmark manifests plus installed acDNN API.
- Produces: exact FlagDNN Graph versus acDNN numerical comparison and same-case paired timing with complete
  accounting and deterministic unsupported records.

- [x] Prove reference dependency closure excludes BLAS/cuDNN/PyTorch/CPU/self-oracle paths.
- [x] Audit tensor initialization, dtype conversion, explicit strides, storage span, padding, output
  readback, tolerance policy and NaN/Inf comparison.
- [x] Audit every acDNN descriptor/handle/workspace lifetime and status check, especially asymmetric and
  1D/2D/3D convolution staging plus Dgrad/Wgrad subprocess isolation.
- [x] Audit timing fairness: identical inputs, warmup/sample counts, synchronization boundaries, retained
  lifetimes, median/p90 units and provider pairing.
- [x] Reconcile all public cases exactly once across supported/comparable/unsupported states and prove zero
  pending probes, duplicate records, missing pairs or unstructured skips.

### Task 5: CMake, dependency discovery, installation and cross-platform isolation

**Files:**
- Inspect: `backends/thead/CMakeLists.txt`, `backends/thead/cmake/**`
- Inspect: `backends/thead/validation/InstalledConsumerContract.cmake`
- Compare: `backends/nvidia/CMakeLists.txt` and NVIDIA package/install conventions.
- Test: clean `/tmp` configure/build/install/consumer trees

**Interfaces:**
- Consumes: top-level generic backend discovery and `find_package(FlagDNN)` export contracts.
- Produces: relocatable THead installation with deterministic dependency resolution and no side effects
  when another backend is selected.

- [x] Audit all CMake cache variables, imported targets, RPATH/RUNPATH, version checks, install components,
  generated environment metadata and source-tree leakage.
- [x] Configure THead from a clean build tree with warnings as errors, build and install to a clean prefix,
  then compile/run the installed consumer without development-tree compiler overrides.
- [x] Configure a non-THead/common-only path far enough to prove the new backend directory is inert unless
  selected; run available common contracts that do not require another vendor SDK.
- [x] Inspect production and validation ELF dependencies and dynamic symbols; reject acDNN/BLAS from the
  production plugin and BLAS/cuDNN from validation references.

### Task 6: Test adapter and repository tools audit

**Files:**
- Inspect: `backends/thead/validation/run_tests_adapter.py`
- Inspect: `backends/thead/validation/run_tests_adapter_contract.py`
- Inspect: `tools/run_tests.py` without modifying it unless a THead-caused common defect is proven.
- Test: CLI discovery/error paths, adapter contract and schema-v2 report invariants

**Interfaces:**
- Consumes: common `tools/run_tests.py` platform adapter protocol and CTest registrations.
- Produces: deterministic 118-task THead execution and a complete machine-readable report.

- [x] Compare THead adapter behavior with NVIDIA adapter conventions: selection, subprocess invocation,
  timeout, status normalization, output parsing and exit code.
- [x] Exercise help/list/dry error paths and malformed/missing result handling without a device.
- [x] Run adapter unit contract and the applicable common tool tests; distinguish the previously deferred
  common contract issue from THead regressions.
- [x] Run the complete `tools/run_tests.py --platform thead --ops all --suites all --no-preflight` workflow
  and assert task, case, pair, provider, skip, schema and performance-accounting invariants.

### Task 7: Repair cycle and final qualification

**Files:**
- Modify: only files implicated by demonstrated findings.
- Update: `backends/thead/adaptation-recommendations.md`
- Update: this plan with completed checks and final evidence.

**Interfaces:**
- Consumes: findings from Tasks 1-6.
- Produces: reviewed source plus fresh end-to-end evidence for the final exact tree.

- [x] For each defect, capture a failing focused regression first, implement the smallest local fix, then
  demonstrate red/green behavior and run the affected broader suite.
- [x] Rebuild with warnings as errors and run all 247 THead CTest entries on real PPU, including install,
  JIT, Graph, autotune and exhaustive compile.
- [x] Run complete functional and paired benchmark certification against acDNN only and validate the final
  JSON report with exact invariants.
- [x] Run JSON/Python syntax, catalog closure, dependency, forbidden-reference, diff/scope, trailing-space,
  cache-file and unused-file checks.
- [x] Record unresolved SDK limitations separately from code defects and report every repair with evidence.

## Final Evidence (2026-09-07)

- Allowed change scope: 92 retained files under `backends/thead/**` plus this THead documentation; no
  common, NVIDIA or other-backend source modification.
- Clean THead warnings-as-errors build: 374 Ninja steps; clean install: 40 manifest entries / 37 files;
  installed consumer: 37 scenarios passed on PPU in 497.09 seconds.
- Common-only isolation build: 24 Ninja steps passed without selecting or discovering THead dependencies.
- Applicable common CTest: 13/13 passed. The excluded architecture test was rerun separately and named only
  the explicitly deferred `tests/core/run_tests_contract.py` as its failure; no THead path was implicated.
- Complete CTest coverage: 118 functional/benchmark entries executed by the final runner plus the remaining
  129/129 integration entries, covering all 247 registered THead tests. Equivalent result: 231 passed and
  16 exact capability skips.
- Final runner JSON: schema v2, overall passed, exit 0, 102 tasks passed / 16 skipped;
  functional `1142 = 1052 + 90`, benchmark `1182 = 1125 + 57`, 1125/1125 required pairs,
  2250 provider records, 147 validated skip records and zero coverage/accounting errors.
- Exhaustive system-Triton compilation to real PPU cubin: 399.40 seconds, passed. Production/reference ELF
  dependency contracts and all JIT, Graph, capture, autotune and runtime tests passed.
- Known exception: the pre-existing common `tests/core/run_tests_contract.py` issue remains explicitly
  deferred by user direction and consequently also trips `core.test_architecture`; neither failure is
  introduced by the THead adapter, whose own adapter contract passes.
- The common core ELF intentionally retains its configure-time compiler path as a build-tree fallback.
  Installed resolution was proven to select the relocatable `../share/flagdnn/compiler` resource first;
  installed text/CMake metadata, plugin RPATH and the THead plugin contain no active source-tree dependency.
