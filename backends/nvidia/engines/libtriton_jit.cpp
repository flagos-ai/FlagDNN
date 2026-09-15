/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/engines/engine.hpp"

#include "backends/nvidia/error.hpp"
#include "backends/nvidia/cuda_launch.hpp"

#include <Python.h>

#include "backends/autotune_policy.hpp"

#include <triton_jit/triton_jit_function.h>
#include <triton_jit/triton_kernel.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef FLAGDNN_LIBTRITON_JIT_PYTHONPATH
#define FLAGDNN_LIBTRITON_JIT_PYTHONPATH ""
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY
#define FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY "unknown"
#endif

namespace flagdnn::cuda {
namespace {

std::mutex libtriton_jit_mutex;
std::atomic<bool> environment_prepared{false};
std::atomic<bool> owns_python_interpreter{false};
std::once_flag python_path_once;
std::once_flag python_runtime_once;
std::once_flag jit_library_once;
void* python_global_handle = nullptr;
void* jit_library_handle = nullptr;

void pin_jit_library() {
  std::call_once(jit_library_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(&pin_jit_library));
    if (dladdr(address, &information) == 0 ||
        information.dli_fname == nullptr) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot locate the NVIDIA JIT plugin");
    }
    // Dependency caches and Python exit callbacks outlive individual handles.
    // The backend owns this lifetime, including when loaded by an older core.
    jit_library_handle = dlopen(
        information.dli_fname, RTLD_NOW | RTLD_NOLOAD | RTLD_NODELETE);
    if (jit_library_handle == nullptr) {
      const char* detail = dlerror();
      throw CudaError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot retain the NVIDIA JIT plugin until process exit: " +
              std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
  });
}

bool logging_enabled() noexcept {
  const char* value = std::getenv("FLAGDNN_PRINT_AUTOTUNING");
  return value != nullptr && value[0] != '\0' &&
         std::string_view(value) != "0";
}

void promote_python_runtime() {
  std::call_once(python_runtime_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(&Py_IsInitialized));
    if (dladdr(address, &information) == 0 ||
        information.dli_fname == nullptr) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot locate the embedded Python runtime");
    }
    python_global_handle = dlopen(
        information.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (python_global_handle == nullptr) {
      python_global_handle =
          dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (python_global_handle == nullptr) {
      const char* detail = dlerror();
      throw CudaError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot expose embedded Python symbols to Triton extensions: " +
              std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
  });
}

void configure_python_path() {
  std::call_once(python_path_once, [] {
    const std::string required = FLAGDNN_LIBTRITON_JIT_PYTHONPATH;
    if (required.empty()) {
      return;
    }
    const char* current_value = std::getenv("PYTHONPATH");
    const std::string current =
        current_value == nullptr ? std::string{} : current_value;
    std::string_view remaining(current);
    while (!remaining.empty()) {
      const std::size_t separator = remaining.find(':');
      if (remaining.substr(0, separator) == required) {
        return;
      }
      if (separator == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(separator + 1);
    }
    const std::string updated =
        current.empty() ? required : required + ":" + current;
    if (setenv("PYTHONPATH", updated.c_str(), 1) != 0) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot configure PYTHONPATH for libtriton_jit");
    }
  });
}

// libtriton_jit may initialize CPython on this build thread. Release only the
// initial GIL that we created; never release a Python caller's pre-existing GIL.
void acquire_python_gil_at_shutdown() noexcept {
  if (Py_IsInitialized()) {
    // Native-process dependency destructors contain pybind11 objects. Keep the
    // GIL for the remainder of teardown, without finalizing someone else's Python.
    (void)PyGILState_Ensure();
  }
}

class PythonInitializationGuard {
 public:
  PythonInitializationGuard()
      : owns_initialization_(!Py_IsInitialized()),
        initialization_attempt_(
            !environment_prepared.load(std::memory_order_acquire)) {}
  ~PythonInitializationGuard() {
    if (owns_initialization_ && Py_IsInitialized()) {
      owns_python_interpreter.store(true, std::memory_order_release);
    }
    if (initialization_attempt_ &&
        owns_python_interpreter.load(std::memory_order_acquire) &&
        Py_IsInitialized()) {
      // Register after imports, including partially failed attempts, so the GIL
      // is acquired before dependency destructors registered by those imports.
      // The JIT library pin keeps this callback valid until process exit.
      (void)std::atexit(acquire_python_gil_at_shutdown);
    }
    if (owns_initialization_ && Py_IsInitialized() && PyGILState_Check()) {
      (void)PyEval_SaveThread();
    }
  }
  PythonInitializationGuard(const PythonInitializationGuard&) = delete;
  PythonInitializationGuard& operator=(const PythonInitializationGuard&) = delete;

 private:
  bool owns_initialization_;
  bool initialization_attempt_;
};

class JitTuningResources {
 public:
  JitTuningResources(const CudaKernelArtifact& kernel,
                     std::size_t workspace_size)
      : JitTuningResources(
            std::vector<const CudaKernelArtifact*>{&kernel},
            workspace_size) {}

  JitTuningResources(
      const std::vector<const CudaKernelArtifact*>& kernels,
      std::size_t workspace_size) {
    for (const CudaKernelArtifact* kernel : kernels) {
      require(kernel != nullptr,
              "libtriton_jit autotune kernel is null",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      for (const ArgumentSpec& argument : kernel->arguments) {
        if (argument.kind != ArgumentKind::kTensor &&
            argument.kind != ArgumentKind::kTensorMap) {
          continue;
        }
        auto existing = std::find_if(
            allocations_.begin(),
            allocations_.end(),
            [&](const TuningAllocation& allocation) {
              return allocation.uid == argument.uid;
            });
        if (existing == allocations_.end()) {
          allocations_.push_back(
              {argument.uid, argument.storage_size, 0});
        } else {
          existing->size = std::max(existing->size, argument.storage_size);
        }
      }
    }
    try {
      for (TuningAllocation& allocation : allocations_) {
        check_cuda(cuMemAlloc(&allocation.pointer, allocation.size),
                   "cuMemAlloc(libtriton_jit autotune tensor)");
      }
      if (workspace_size != 0) {
        check_cuda(cuMemAlloc(&workspace_, workspace_size),
                   "cuMemAlloc(libtriton_jit autotune workspace)");
      }
      check_cuda(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate(libtriton_jit autotune)");
      check_cuda(cuEventCreate(&start_, CU_EVENT_DEFAULT),
                 "cuEventCreate(libtriton_jit autotune start)");
      check_cuda(cuEventCreate(&stop_, CU_EVENT_DEFAULT),
                 "cuEventCreate(libtriton_jit autotune stop)");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~JitTuningResources() { cleanup(); }

  JitTuningResources(const JitTuningResources&) = delete;
  JitTuningResources& operator=(const JitTuningResources&) = delete;

  [[nodiscard]] const std::vector<TuningAllocation>& allocations() const {
    return allocations_;
  }
  [[nodiscard]] CUdeviceptr workspace() const noexcept { return workspace_; }
  [[nodiscard]] CUstream stream() const noexcept { return stream_; }
  [[nodiscard]] CUevent start() const noexcept { return start_; }
  [[nodiscard]] CUevent stop() const noexcept { return stop_; }

 private:
  void cleanup() noexcept {
    if (start_ != nullptr) {
      (void)cuEventDestroy(start_);
      start_ = nullptr;
    }
    if (stop_ != nullptr) {
      (void)cuEventDestroy(stop_);
      stop_ = nullptr;
    }
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (workspace_ != 0) {
      (void)cuMemFree(workspace_);
      workspace_ = 0;
    }
    for (TuningAllocation& allocation : allocations_) {
      if (allocation.pointer != 0) {
        (void)cuMemFree(allocation.pointer);
        allocation.pointer = 0;
      }
    }
  }

  std::vector<TuningAllocation> allocations_;
  CUdeviceptr workspace_ = 0;
  CUstream stream_ = nullptr;
  CUevent start_ = nullptr;
  CUevent stop_ = nullptr;
};

class CapturedLaunchBatch {
 public:
  template <typename Function>
  CapturedLaunchBatch(CUstream stream,
                      unsigned int execution_count,
                      Function&& launch)
      : execution_count_(execution_count) {
    require(execution_count_ != 0,
            "libtriton_jit autotune batch cannot be empty",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    check_cuda(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
               "cuStreamBeginCapture(libtriton_jit autotune)");
    try {
      for (unsigned int iteration = 0;
           iteration < execution_count_;
           ++iteration) {
        launch();
      }
    } catch (...) {
      CUgraph abandoned = nullptr;
      if (cuStreamEndCapture(stream, &abandoned) == CUDA_SUCCESS &&
          abandoned != nullptr) {
        (void)cuGraphDestroy(abandoned);
      }
      throw;
    }

    CUgraph graph = nullptr;
    check_cuda(cuStreamEndCapture(stream, &graph),
               "cuStreamEndCapture(libtriton_jit autotune)");
    try {
      check_cuda(cuGraphInstantiate(&executable_, graph, 0),
                 "cuGraphInstantiate(libtriton_jit autotune)");
      check_cuda(cuGraphDestroy(graph),
                 "cuGraphDestroy(libtriton_jit autotune source)");
    } catch (...) {
      if (graph != nullptr) {
        (void)cuGraphDestroy(graph);
      }
      cleanup();
      throw;
    }
  }

  ~CapturedLaunchBatch() { cleanup(); }

  CapturedLaunchBatch(const CapturedLaunchBatch&) = delete;
  CapturedLaunchBatch& operator=(const CapturedLaunchBatch&) = delete;

  void launch(CUstream stream) const {
    check_cuda(cuGraphLaunch(executable_, stream),
               "cuGraphLaunch(libtriton_jit autotune)");
  }

  [[nodiscard]] unsigned int execution_count() const noexcept {
    return execution_count_;
  }

 private:
  void cleanup() noexcept {
    if (executable_ != nullptr) {
      (void)cuGraphExecDestroy(executable_);
      executable_ = nullptr;
    }
  }

  CUgraphExec executable_ = nullptr;
  unsigned int execution_count_ = 0;
};

using RawArguments = KernelArguments;

using JitFunction = triton_jit::TritonJITFunction;

// Both forms are public libtriton_jit APIs. Raw signatures remain unchanged
// for ordinary kernels; descriptor/Gluon kernels use artifact-owned caches.
// This object lives with the prepared stage; selection happens only at build.
class JitStage {
 public:
  explicit JitStage(const CudaStageArtifact& stage) {
    // The cache loader itself does not initialize Python. Always initialize
    // the public JIT function under the build-environment writer lock first,
    // so a later raw stage cannot unexpectedly mutate the process environment.
    const bool cached = !stage.variants.front().compiled_cache.empty();
    const auto& function = JitFunction::get_instance(
        stage.source.string(), cached ? "_flagdnn_jit_initialize" : stage.function_name);
    if (!cached) {
      function_ = &function;
    } else {
      for (const auto& variant : stage.variants) {
        cached_.emplace_back(variant.variant_id, std::make_unique<CachedKernel>(
            variant.compiled_cache.string(), stage.function_name));
      }
    }
  }

  void launch(const CudaKernelArtifact& kernel, CUstream stream,
              RawArguments& arguments) const {
    if (function_ != nullptr) {
      function_->launch_with_raw_args(stream, kernel.grid[0], kernel.grid[1], kernel.grid[2],
                                     kernel.num_warps, kernel.num_stages, kernel.full_signature,
                                     arguments.data(), arguments.size());
      return;
    }
    const auto cached = std::find_if(cached_.begin(), cached_.end(), [&](const auto& entry) {
      return entry.first == kernel.variant_id;
    });
    require(cached != cached_.end(), "compiled JIT variant is missing",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    // The cache API does not parse the raw comma-separated signature.
    cached->second->launch_with_signature(
        kernel.grid[0], kernel.grid[1], kernel.grid[2],
        static_cast<int>(kernel.num_warps), stream, arguments.data(),
        kernel.full_signature, arguments.size());
  }

 private:
  using CachedKernel = triton_jit::TritonKernelImpl<triton_jit::CudaBackend>;
  const JitFunction* function_ = nullptr;
  std::vector<std::pair<std::string, std::unique_ptr<CachedKernel>>> cached_;
};

void launch_jit(const JitStage& function,
                const CudaKernelArtifact& kernel,
                CUstream stream,
                RawArguments& arguments) {
  function.launch(kernel, stream, arguments);
}

struct PreparedCudaLaunch {
  CUfunction function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

std::size_t jit_global_scratch_size(const CudaArtifact& artifact) {
  std::size_t size = 0;
  for (const auto& stage : artifact.stages) {
    for (const auto& variant : stage.variants) {
      size = std::max(size, variant.global_scratch_size);
    }
  }
  require(size != 0 && size <= artifact.workspace_size,
          "libtriton_jit global scratch exceeds workspace",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  return size;
}

void prepare_jit_candidate(const JitStage& function,
                           const CudaKernelArtifact& kernel,
                           std::size_t workspace_size,
                           std::size_t global_scratch_offset) {
  JitTuningResources resources(kernel, workspace_size);
  RawArguments arguments(kernel, resources.allocations(), resources.workspace(),
                         resources.workspace() + global_scratch_offset);
  launch_jit(function, kernel, resources.stream(), arguments);
  check_cuda(cuStreamSynchronize(resources.stream()),
             "cuStreamSynchronize(libtriton_jit prepare)");
}

PreparedCudaLaunch prepare_cuda_launch(
    const JitStage& function,
    const CudaKernelArtifact& kernel,
    std::size_t workspace_size,
    std::size_t global_scratch_offset) {
  JitTuningResources resources(kernel, workspace_size);
  RawArguments arguments(
      kernel,
      resources.allocations(),
      resources.workspace(),
      resources.workspace() + global_scratch_offset);

  CUgraph graph = nullptr;
  bool capture_active = false;
  try {
    check_cuda(
        cuStreamBeginCapture(
            resources.stream(), CU_STREAM_CAPTURE_MODE_RELAXED),
        "cuStreamBeginCapture(libtriton_jit prepared launch)");
    capture_active = true;
    launch_jit(function, kernel, resources.stream(), arguments);
    check_cuda(cuStreamEndCapture(resources.stream(), &graph),
               "cuStreamEndCapture(libtriton_jit prepared launch)");
    capture_active = false;

    std::size_t node_count = 0;
    check_cuda(cuGraphGetNodes(graph, nullptr, &node_count),
               "cuGraphGetNodes(libtriton_jit prepared launch count)");
    std::vector<CUgraphNode> nodes(node_count);
    check_cuda(cuGraphGetNodes(graph, nodes.data(), &node_count),
               "cuGraphGetNodes(libtriton_jit prepared launch)");

    CUgraphNode kernel_node = nullptr;
    for (const CUgraphNode node : nodes) {
      CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
      check_cuda(cuGraphNodeGetType(node, &type),
                 "cuGraphNodeGetType(libtriton_jit prepared launch)");
      if (type != CU_GRAPH_NODE_TYPE_KERNEL) {
        continue;
      }
      require(kernel_node == nullptr,
              "libtriton_jit launch captured more than one CUDA kernel",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      kernel_node = node;
    }
    require(kernel_node != nullptr,
            "libtriton_jit launch did not capture a CUDA kernel",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    CUDA_KERNEL_NODE_PARAMS parameters{};
    check_cuda(cuGraphKernelNodeGetParams(kernel_node, &parameters),
               "cuGraphKernelNodeGetParams(libtriton_jit prepared launch)");
    require(parameters.func != nullptr,
            "libtriton_jit captured kernel has no CUfunction",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    PreparedCudaLaunch prepared{
        parameters.func,
        {parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ},
        {parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ},
        parameters.sharedMemBytes};
    check_cuda(cuGraphDestroy(graph),
               "cuGraphDestroy(libtriton_jit prepared launch)");
    graph = nullptr;
    return prepared;
  } catch (...) {
    if (capture_active) {
      CUgraph abandoned = nullptr;
      if (cuStreamEndCapture(resources.stream(), &abandoned) ==
              CUDA_SUCCESS &&
          abandoned != nullptr) {
        (void)cuGraphDestroy(abandoned);
      }
    } else if (graph != nullptr) {
      (void)cuGraphDestroy(graph);
    }
    throw;
  }
}

void launch_prepared_cuda(const PreparedCudaLaunch& prepared,
                          CUstream stream,
                          RawArguments& arguments) {
  check_cuda(cuLaunchKernel(prepared.function,
                            prepared.grid[0],
                            prepared.grid[1],
                            prepared.grid[2],
                            prepared.block[0],
                            prepared.block[1],
                            prepared.block[2],
                            prepared.shared_memory,
                            stream,
                            arguments.data(),
                            nullptr),
             "cuLaunchKernel(libtriton_jit prepared launch)");
}

struct LoadedJitKernel {
  CudaKernelArtifact specification;
  PreparedCudaLaunch prepared;
  std::unique_ptr<JitStage> source;
};

}  // namespace

class ExecutionEngine::Impl {
 public:
  Impl(const EngineBuildContext& context, CudaArtifact artifact)
      : context_(context),
        binding_uids_(std::move(artifact.binding_uids)),
        workspace_size_(artifact.workspace_size) {
    global_scratch_offset_ = workspace_size_ - jit_global_scratch_size(artifact);

    CUcontext retained_context = nullptr;
    check_cuda(cuDevicePrimaryCtxRetain(&retained_context, context_.device),
               "cuDevicePrimaryCtxRetain(libtriton_jit engine)");
    if (retained_context != context_.context) {
      (void)cuDevicePrimaryCtxRelease(context_.device);
      throw CudaError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "CUDA primary context changed while building JIT executable");
    }
    retained_ = true;

    try {
      std::unique_lock lock(libtriton_jit_mutex);
      pin_jit_library();
      PythonInitializationGuard python_initialization;
      promote_python_runtime();
      configure_python_path();
      ContextGuard guard(context_.context);
      kernels_.reserve(artifact.stages.size());
      for (const CudaStageArtifact& stage : artifact.stages) {
        if (stage.source.empty() || stage.function_name.empty() ||
            stage.variants.empty()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "libtriton_jit stage is incomplete");
        }
        auto source = std::make_unique<JitStage>(stage);
        const JitStage& function = *source;
        const std::size_t selected =
            stage.autotune ? select_candidate(function, stage) : 0;
        if (selected >= stage.variants.size()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                          "libtriton_jit autotune selected an invalid candidate");
        }
        if (!stage.autotune) {
          prepare_candidate(function, stage.variants[selected]);
        }
        CudaKernelArtifact specification = stage.variants[selected];
        PreparedCudaLaunch prepared = prepare_cuda_launch(
            function,
            specification,
            workspace_size_,
            global_scratch_offset_);
        kernels_.push_back(
            {std::move(specification), prepared, std::move(source)});
      }
      environment_prepared.store(true, std::memory_order_release);
    } catch (const CudaError&) {
      release_context();
      throw;
    } catch (const std::exception& error) {
      release_context();
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "libtriton_jit executable build failed: " +
                          std::string(error.what()));
    } catch (...) {
      release_context();
      throw;
    }
  }

  ~Impl() { release_context(); }

  Impl(const Impl&) = delete;
  Impl& operator=(const Impl&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept {
    return workspace_size_;
  }

  void execute(CUstream stream,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const {
    require(workspace_size >= workspace_size_,
            "workspace is smaller than CUDA executable requirement");
    require(workspace_size_ == 0 || workspace != nullptr,
            "CUDA executable workspace is null");
    require(
        workspace_size_ == 0 ||
            (reinterpret_cast<std::uintptr_t>(workspace) & 0x0fU) == 0U,
        "libtriton_jit workspace must be at least 16-byte aligned");
    require(binding_count == binding_uids_.size(),
            "binding count does not match CUDA executable");
    require(binding_count == 0 || bindings != nullptr,
            "binding array is null");

    try {
      // Default-stream handles resolve against the calling thread's context.
      // Explicit streams already carry their context and keep the fast path.
      ContextGuard guard(context_.context,
                         stream == nullptr || stream == CU_STREAM_LEGACY ||
                             stream == CU_STREAM_PER_THREAD);
      for (const LoadedJitKernel& kernel : kernels_) {
        RawArguments arguments(kernel.specification,
                               bindings,
                               binding_count,
                               workspace,
                               scratch_pointer(workspace));
        launch_prepared_cuda(kernel.prepared, stream, arguments);
      }
    } catch (const CudaError&) {
      throw;
    } catch (const std::exception& error) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "libtriton_jit kernel launch failed: " +
                          std::string(error.what()));
    }
  }

 private:
  [[nodiscard]] std::size_t select_candidate(
      const JitStage& function,
      const CudaStageArtifact& stage) const {
    require(stage.autotune && stage.variants.size() >= 2,
            "invalid libtriton_jit autotune stage",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    backend::autotune::SelectionRequest full_request;
    full_request.candidate_identity = stage.candidate_identity;
    full_request.device_identity = context_.device_identity;
    full_request.measurement_identity =
        "nvidia-libtriton-jit-stage-cuda-graph-v3-build-" +
        std::string(FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY) + "-stage-" +
        std::to_string(kernels_.size());
    full_request.cache_path = stage.selection_cache;
    full_request.warmup_milliseconds = stage.warmup;
    full_request.benchmark_milliseconds = stage.repetitions;
    full_request.candidate_ids.reserve(stage.variants.size());
    for (const CudaKernelArtifact& variant : stage.variants) {
      full_request.candidate_ids.push_back(variant.variant_id);
    }

    if (const auto cached =
            backend::autotune::find_cached_candidate(full_request)) {
      try {
        prepare_candidate(function, stage.variants[*cached]);
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] cache hit "
                    << stage.candidate_identity.substr(0, 12) << " -> "
                    << stage.variants[*cached].variant_id << '\n';
        }
        return *cached;
      } catch (const std::exception& error) {
        backend::autotune::discard_cached_candidate(full_request);
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] discarded cached "
                    << stage.variants[*cached].variant_id << ": "
                    << error.what() << '\n';
        }
      }
    }

    // Triton tuning spaces can legitimately contain configurations that do
    // not fit a particular device (for example, a convolution tile whose
    // shared-memory requirement is too large). libtriton_jit currently
    // reports such failures while lazily compiling on the first launch. A
    // single device-incompatible configuration must not reject the whole
    // graph, so establish the runnable subset before invoking the shared
    // timing/cache policy.
    std::vector<std::size_t> runnable_indices;
    runnable_indices.reserve(stage.variants.size());
    std::string rejected_candidates;
    for (std::size_t index = 0; index < stage.variants.size(); ++index) {
      try {
        prepare_candidate(function, stage.variants[index]);
        runnable_indices.push_back(index);
      } catch (const std::exception& error) {
        if (!rejected_candidates.empty()) {
          rejected_candidates += "; ";
        }
        rejected_candidates += stage.variants[index].variant_id + ": " +
                               std::string(error.what());
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] skipped "
                    << stage.variants[index].variant_id << ": "
                    << error.what() << '\n';
        }
      }
    }
    if (runnable_indices.empty()) {
      throw CudaError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "libtriton_jit autotune has no runnable candidates" +
              (rejected_candidates.empty()
                   ? std::string{}
                   : std::string(": ") + rejected_candidates));
    }
    if (runnable_indices.size() == 1) {
      auto singleton_request = full_request;
      singleton_request.candidate_ids = {
          stage.variants[runnable_indices.front()].variant_id};
      (void)backend::autotune::select_best_candidate(singleton_request, {}, {});
      if (logging_enabled()) {
        std::cerr << "[FlagDNN autotune/JIT] only runnable candidate "
                  << stage.variants[runnable_indices.front()].variant_id
                  << '\n';
      }
      return runnable_indices.front();
    }

    backend::autotune::SelectionRequest request;
    request.candidate_identity = stage.candidate_identity;
    request.device_identity = context_.device_identity;
    request.measurement_identity = full_request.measurement_identity;
    request.cache_path = stage.selection_cache;
    request.warmup_milliseconds = stage.warmup;
    request.benchmark_milliseconds = stage.repetitions;
    request.candidate_ids.reserve(runnable_indices.size());
    for (const std::size_t index : runnable_indices) {
      request.candidate_ids.push_back(stage.variants[index].variant_id);
    }

    std::vector<const CudaKernelArtifact*> tuning_kernels;
    tuning_kernels.reserve(kernels_.size() + 1);
    for (const LoadedJitKernel& kernel : kernels_) {
      tuning_kernels.push_back(&kernel.specification);
    }
    tuning_kernels.push_back(
        &stage.variants[runnable_indices.front()]);
    JitTuningResources resources(tuning_kernels, workspace_size_);
    const auto initialize_candidate_inputs = [&]() {
      for (const LoadedJitKernel& prefix : kernels_) {
        RawArguments prefix_arguments(
            prefix.specification,
            resources.allocations(),
            resources.workspace(),
            scratch_pointer(resources.workspace()));
        launch_prepared_cuda(
            prefix.prepared, resources.stream(), prefix_arguments);
      }
    };
    const auto launch_candidate = [&](std::size_t index) {
      const CudaKernelArtifact& variant =
          stage.variants[runnable_indices[index]];
      RawArguments arguments(variant,
                             resources.allocations(),
                             resources.workspace(),
                             scratch_pointer(resources.workspace()));
      launch_jit(function, variant, resources.stream(), arguments);
    };
    constexpr unsigned int kGraphBatchSize = 32;
    std::unique_ptr<CapturedLaunchBatch> captured_batch;
    std::size_t captured_candidate = runnable_indices.size();
    const auto batch_for = [&](std::size_t index)
        -> CapturedLaunchBatch& {
      if (captured_batch == nullptr || captured_candidate != index) {
        captured_batch.reset();
        captured_batch = std::make_unique<CapturedLaunchBatch>(
            resources.stream(), kGraphBatchSize, [&] {
              launch_candidate(index);
            });
        captured_candidate = index;
      }
      return *captured_batch;
    };
    const auto replay_count = [](unsigned int requested,
                                 unsigned int batch_size) {
      return (requested + batch_size - 1U) / batch_size;
    };

    const backend::autotune::SelectionResult result =
        backend::autotune::select_best_candidate(
            request,
            [&](std::size_t index, unsigned int iterations) {
              CapturedLaunchBatch& batch = batch_for(index);
              initialize_candidate_inputs();
              const unsigned int replays =
                  replay_count(iterations, batch.execution_count());
              for (unsigned int replay = 0; replay < replays; ++replay) {
                batch.launch(resources.stream());
              }
              check_cuda(cuStreamSynchronize(resources.stream()),
                         "cuStreamSynchronize(libtriton_jit autotune warmup)");
            },
            [&](std::size_t index, unsigned int iterations) {
              CapturedLaunchBatch& batch = batch_for(index);
              initialize_candidate_inputs();
              const unsigned int replays =
                  replay_count(iterations, batch.execution_count());
              check_cuda(cuEventRecord(resources.start(), resources.stream()),
                         "cuEventRecord(libtriton_jit autotune start)");
              for (unsigned int replay = 0; replay < replays; ++replay) {
                batch.launch(resources.stream());
              }
              check_cuda(cuEventRecord(resources.stop(), resources.stream()),
                         "cuEventRecord(libtriton_jit autotune stop)");
              check_cuda(cuEventSynchronize(resources.stop()),
                         "cuEventSynchronize(libtriton_jit autotune)");
              float milliseconds = 0.0F;
              check_cuda(
                  cuEventElapsedTime(
                      &milliseconds, resources.start(), resources.stop()),
                  "cuEventElapsedTime(libtriton_jit autotune)");
              const unsigned int measured_iterations =
                  replays * batch.execution_count();
              return milliseconds /
                     static_cast<float>(measured_iterations);
            });

    if (logging_enabled()) {
      const std::string& selected =
          request.candidate_ids[result.candidate_index];
      if (result.cache_hit) {
        std::cerr << "[FlagDNN autotune/JIT] cache hit "
                  << stage.candidate_identity.substr(0, 12) << " -> "
                  << selected << '\n';
      } else {
        for (std::size_t index = 0;
             index < result.median_milliseconds.size();
             ++index) {
          std::cerr << "[FlagDNN autotune/JIT] "
                    << request.candidate_ids[index] << " median_ms="
                    << result.median_milliseconds[index] << '\n';
        }
        std::cerr << "[FlagDNN autotune/JIT] selected " << selected << '\n';
      }
    }
    return runnable_indices[result.candidate_index];
  }

  void prepare_candidate(const JitStage& function,
                         const CudaKernelArtifact& kernel) const {
    prepare_jit_candidate(function, kernel, workspace_size_,
                          global_scratch_offset_);
  }

  void release_context() noexcept {
    if (retained_) {
      (void)cuDevicePrimaryCtxRelease(context_.device);
      retained_ = false;
    }
  }

  [[nodiscard]] CUdeviceptr scratch_pointer(
      CUdeviceptr workspace) const noexcept {
    return workspace + global_scratch_offset_;
  }

  [[nodiscard]] CUdeviceptr scratch_pointer(void* workspace) const noexcept {
    return static_cast<CUdeviceptr>(
               reinterpret_cast<std::uintptr_t>(workspace)) +
           global_scratch_offset_;
  }

  EngineBuildContext context_;
  std::vector<std::int64_t> binding_uids_;
  std::vector<LoadedJitKernel> kernels_;
  std::size_t workspace_size_ = 0;
  std::size_t global_scratch_offset_ = 0;
  bool retained_ = false;
};

bool libtriton_jit_environment_prepared() noexcept {
  return environment_prepared.load(std::memory_order_acquire);
}

void prepare_libtriton_jit_environment(
    const EngineBuildContext& context, const CudaArtifact& artifact) {
  std::unique_lock lock(libtriton_jit_mutex);
  if (environment_prepared.load(std::memory_order_acquire)) {
    return;
  }
  require(!artifact.stages.empty(),
          "environment preparation requires a nonempty JIT program",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  pin_jit_library();
  PythonInitializationGuard python_initialization;
  promote_python_runtime();
  configure_python_path();
  ContextGuard guard(context.context);
  const auto& stage = artifact.stages.front();
  const auto scratch_offset =
      artifact.workspace_size - jit_global_scratch_size(artifact);
  try {
    const JitStage function(stage);
    std::string failures;
    // The current dependency has no standalone initialization API. Prepare a
    // real variant, so its Python environment writes and lazy CUDA initialization
    // happen under the core's exclusive lock. Subsequent builds reuse this cache.
    for (const auto& variant : stage.variants) {
      try {
        prepare_jit_candidate(function, variant, artifact.workspace_size,
                              scratch_offset);
        environment_prepared.store(true, std::memory_order_release);
        return;
      } catch (const std::exception& error) {
        failures += variant.variant_id + ": " + error.what() + "\n";
      }
    }
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "no JIT variant could initialize the environment: " + failures);
  } catch (const CudaError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "libtriton_jit initialization failed: " +
                        std::string(error.what()));
  }
}

ExecutionEngine::ExecutionEngine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input)
    : impl_(std::make_unique<Impl>(context, parse_cuda_artifact(context, input))) {}

ExecutionEngine::~ExecutionEngine() = default;

std::size_t ExecutionEngine::workspace_size() const noexcept {
  return impl_->workspace_size();
}

void ExecutionEngine::execute(
    CUstream stream, const flagdnnBackendBindingV2 bindings[],
    std::size_t binding_count, void* workspace,
    std::size_t workspace_size) const {
  impl_->execute(stream, bindings, binding_count, workspace, workspace_size);
}

std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  return std::make_unique<ExecutionEngine>(context, input);
}

}  // namespace flagdnn::cuda
