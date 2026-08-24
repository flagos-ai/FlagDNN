/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/engines/libtriton_jit.hpp"

#include "backends/autotune_policy.hpp"
#include "backends/iluvatar/artifact.hpp"
#include "backends/iluvatar/error.hpp"

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/ix_backend.h>
#include <triton_jit/triton_jit_function.h>

#include <Python.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_IX
#error "The Iluvatar execution engine requires an IX libtriton_jit build"
#endif

#ifndef FLAGDNN_ILUVATAR_PYTHONPATH
#define FLAGDNN_ILUVATAR_PYTHONPATH ""
#endif

#ifndef FLAGDNN_ILUVATAR_PYTHON_LIBRARY_SONAME
#error "The Iluvatar execution engine requires a Python shared-library SONAME"
#endif

static_assert(triton_jit::IxBackend::WARP_SIZE == 64);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::IxBackend>);

namespace flagdnn::iluvatar {
namespace {

using JitFunction = triton_jit::TritonJITFunctionImpl<triton_jit::IxBackend>;

std::mutex jit_build_mutex;
std::once_flag python_path_once;
void *python_runtime_handle = nullptr;

[[noreturn]] void compilation_error(std::string message) {
  throw IluvatarError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      std::move(message));
}

void check_preparation(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    compilation_error(corex_error(result, operation));
  }
}

void configure_python_path() {
  std::call_once(python_path_once, [] {
    if (Py_GetVersion() == nullptr) {
      compilation_error("cannot resolve the embedded Python runtime");
    }
    python_runtime_handle =
        dlopen(FLAGDNN_ILUVATAR_PYTHON_LIBRARY_SONAME, RTLD_NOW | RTLD_GLOBAL);
    if (python_runtime_handle == nullptr) {
      const char *detail = dlerror();
      compilation_error(
          "cannot expose embedded Python symbols to JIT extensions: " +
          std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
    const std::string required = FLAGDNN_ILUVATAR_PYTHONPATH;
    if (required.empty()) {
      return;
    }
    const char *configured = std::getenv("PYTHONPATH");
    const std::string current =
        configured == nullptr ? std::string{} : configured;
    const std::string updated =
        current.empty() ? required : required + ':' + current;
    if (setenv("PYTHONPATH", updated.c_str(), 1) != 0) {
      compilation_error("cannot configure PYTHONPATH for IX libtriton_jit");
    }
  });
}

struct TemporaryAllocation {
  std::int64_t uid = 0;
  CUdeviceptr pointer = 0;
};

class PreparationResources final {
public:
  explicit PreparationResources(const ExecutionProgramArtifact &artifact) {
    try {
      check_preparation(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                        "cuStreamCreate(IX JIT preparation)");
      allocations_.reserve(artifact.tensors.size());
      for (const TensorArtifact &tensor : artifact.tensors) {
        if (tensor.is_virtual) {
          continue;
        }
        TemporaryAllocation allocation{tensor.uid, 0};
        check_preparation(cuMemAlloc(&allocation.pointer, tensor.storage_size),
                          "cuMemAlloc(IX JIT preparation tensor)");
        check_preparation(cuMemsetD8Async(allocation.pointer, 0,
                                          tensor.storage_size, stream_),
                          "cuMemsetD8Async(IX JIT preparation tensor)");
        allocations_.push_back(allocation);
      }
      if (artifact.workspace_size != 0) {
        check_preparation(cuMemAlloc(&workspace_, artifact.workspace_size),
                          "cuMemAlloc(IX JIT preparation workspace)");
        check_preparation(
            cuMemsetD8Async(workspace_, 0, artifact.workspace_size, stream_),
            "cuMemsetD8Async(IX JIT preparation workspace)");
      }
      check_preparation(cuStreamSynchronize(stream_),
                        "cuStreamSynchronize(IX JIT preparation init)");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~PreparationResources() { cleanup(); }

  PreparationResources(const PreparationResources &) = delete;
  PreparationResources &operator=(const PreparationResources &) = delete;

  [[nodiscard]] CUdeviceptr tensor(std::int64_t uid) const {
    const auto found =
        std::find_if(allocations_.begin(), allocations_.end(),
                     [uid](const TemporaryAllocation &allocation) {
                       return allocation.uid == uid;
                     });
    if (found == allocations_.end()) {
      compilation_error("IX JIT preparation is missing a tensor allocation");
    }
    return found->pointer;
  }

  [[nodiscard]] CUdeviceptr workspace() const noexcept { return workspace_; }

  [[nodiscard]] CUstream stream() const noexcept { return stream_; }

private:
  void cleanup() noexcept {
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (workspace_ != 0) {
      (void)cuMemFree(workspace_);
      workspace_ = 0;
    }
    for (TemporaryAllocation &allocation : allocations_) {
      if (allocation.pointer != 0) {
        (void)cuMemFree(allocation.pointer);
        allocation.pointer = 0;
      }
    }
  }

  std::vector<TemporaryAllocation> allocations_;
  CUdeviceptr workspace_ = 0;
  CUstream stream_ = nullptr;
};

struct ArgumentValue {
  CUdeviceptr pointer = 0;
  std::int32_t scalar_i32 = 0;
  float scalar_f32 = 0.0F;
};

class LaunchArguments final {
public:
  template <typename PointerResolver>
  LaunchArguments(const KernelVariant &variant,
                  PointerResolver &&resolve_pointer) {
    const std::size_t visible_count = variant.arguments.size();
    require(visible_count <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
            "IX kernel has too many runtime arguments",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    parameter_count_ = visible_count + 2;

    for (std::size_t index = 0; index < visible_count; ++index) {
      const KernelArgument &argument = variant.arguments[index];
      if (argument.kind == ArgumentKind::kTensor ||
          argument.kind == ArgumentKind::kWorkspaceTensor) {
        values_[index].pointer = resolve_pointer(argument);
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kScalarI32) {
        values_[index].scalar_i32 = argument.scalar_i32;
        parameters_[index] = &values_[index].scalar_i32;
      } else if (argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_f32 = argument.scalar_f32;
        parameters_[index] = &values_[index].scalar_f32;
      } else {
        compilation_error("IX kernel argument kind is unsupported");
      }
    }
    parameters_[visible_count] = &global_scratch_;
    parameters_[visible_count + 1] = &profile_scratch_;
  }

  [[nodiscard]] void **data() noexcept { return parameters_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return parameter_count_; }

private:
  std::array<ArgumentValue, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> values_{};
  std::array<void *, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2> parameters_{};
  std::size_t parameter_count_ = 0;
  CUdeviceptr global_scratch_ = 0;
  CUdeviceptr profile_scratch_ = 0;
};

std::vector<std::string_view> signature_tokens(std::string_view signature) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  unsigned int nesting = 0;
  for (std::size_t index = 0; index <= signature.size(); ++index) {
    const bool at_end = index == signature.size();
    const char character = at_end ? ',' : signature[index];
    if (!at_end && character == '(') {
      ++nesting;
    } else if (!at_end && character == ')') {
      if (nesting == 0) {
        compilation_error("IX JIT signature has unbalanced parentheses");
      }
      --nesting;
    }
    if (character == ',' && nesting == 0) {
      const std::string_view token = signature.substr(start, index - start);
      if (token.empty()) {
        compilation_error("IX JIT signature contains an empty token");
      }
      result.push_back(token);
      start = index + 1;
    }
  }
  if (nesting != 0) {
    compilation_error("IX JIT signature has unbalanced parentheses");
  }
  return result;
}

void validate_jit_abi(const JitFunction &function,
                      const KernelVariant &variant) {
  const triton_jit::StaticSignature &static_signature =
      function.get_static_sig();
  const std::vector<std::string_view> tokens =
      signature_tokens(variant.full_signature);
  if (static_signature.num_args < 0 ||
      static_cast<std::size_t>(static_signature.num_args) != tokens.size() ||
      static_signature.arg_type.size() != tokens.size()) {
    compilation_error(
        "IX JIT static signature differs from the artifact signature");
  }

  std::size_t runtime_arguments = 0;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const triton_jit::ArgType kind = static_signature.arg_type[index];
    if (kind == triton_jit::ArgType::CONSTEXPR) {
      continue;
    }
    const bool specialized_one =
        tokens[index].front() != '*' && tokens[index].ends_with(":1") &&
        (kind == triton_jit::ArgType::SPECIALIZED ||
         kind == triton_jit::ArgType::SPECIALIZED_NO_ALIGNMENT);
    if (!specialized_one) {
      ++runtime_arguments;
    }
  }
  if (runtime_arguments != variant.arguments.size()) {
    compilation_error(
        "IX JIT runtime argument count differs from the artifact ABI");
  }
}

void launch_jit(const JitFunction &function, const KernelVariant &variant,
                CUstream stream, LaunchArguments &arguments) {
  function.launch_with_raw_args(stream, variant.grid[0], variant.grid[1],
                                variant.grid[2], variant.num_warps,
                                variant.num_stages, variant.full_signature,
                                arguments.data(), arguments.size());
}

struct PreparedLaunch {
  CUfunction function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

PreparedLaunch capture_prepared_launch(const JitFunction &function,
                                       const KernelVariant &variant,
                                       PreparationResources &resources) {
  LaunchArguments arguments(
      variant, [&](const KernelArgument &argument) -> CUdeviceptr {
        if (argument.kind == ArgumentKind::kTensor) {
          return resources.tensor(argument.uid);
        }
        if (resources.workspace() == 0) {
          compilation_error("IX JIT preparation is missing artifact workspace");
        }
        return resources.workspace() + argument.workspace_offset;
      });

  launch_jit(function, variant, resources.stream(), arguments);
  check_preparation(cuStreamSynchronize(resources.stream()),
                    "cuStreamSynchronize(IX JIT compile check)");

  CUgraph graph = nullptr;
  bool capture_active = false;
  try {
    check_preparation(cuStreamBeginCapture(resources.stream(),
                                           CU_STREAM_CAPTURE_MODE_RELAXED),
                      "cuStreamBeginCapture(IX prepared launch)");
    capture_active = true;
    launch_jit(function, variant, resources.stream(), arguments);
    check_preparation(cuStreamEndCapture(resources.stream(), &graph),
                      "cuStreamEndCapture(IX prepared launch)");
    capture_active = false;

    std::size_t node_count = 0;
    check_preparation(cuGraphGetNodes(graph, nullptr, &node_count),
                      "cuGraphGetNodes(IX prepared launch count)");
    if (node_count != 1) {
      compilation_error(
          "one IX execution stage must capture exactly one kernel node");
    }
    CUgraphNode node = nullptr;
    check_preparation(cuGraphGetNodes(graph, &node, &node_count),
                      "cuGraphGetNodes(IX prepared launch)");
    CUgraphNodeType node_type = CU_GRAPH_NODE_TYPE_EMPTY;
    check_preparation(cuGraphNodeGetType(node, &node_type),
                      "cuGraphNodeGetType(IX prepared launch)");
    if (node_type != CU_GRAPH_NODE_TYPE_KERNEL) {
      compilation_error("IX JIT did not capture a kernel launch");
    }

    CUDA_KERNEL_NODE_PARAMS parameters{};
    check_preparation(cuGraphKernelNodeGetParams(node, &parameters),
                      "cuGraphKernelNodeGetParams(IX prepared launch)");
    const PreparedLaunch result{
        parameters.func,
        {parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ},
        {parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ},
        parameters.sharedMemBytes};
    if (result.function == nullptr || result.grid != variant.grid ||
        result.block != variant.block) {
      compilation_error(
          "captured IX grid/block metadata differs from the artifact");
    }
    check_preparation(cuGraphDestroy(graph),
                      "cuGraphDestroy(IX prepared launch)");
    graph = nullptr;
    return result;
  } catch (...) {
    if (capture_active) {
      CUgraph abandoned = nullptr;
      if (cuStreamEndCapture(resources.stream(), &abandoned) == CUDA_SUCCESS &&
          abandoned != nullptr) {
        (void)cuGraphDestroy(abandoned);
      }
    } else if (graph != nullptr) {
      (void)cuGraphDestroy(graph);
    }
    throw;
  }
}

struct BindingRequirement {
  std::int64_t uid = 0;
  std::size_t alignment = 1;
};

struct PreparedStage {
  const JitFunction *function = nullptr;
  KernelVariant variant;
  PreparedLaunch launch;
};

class TuningEvents final {
public:
  TuningEvents() {
    check_preparation(cuEventCreate(&start_, CU_EVENT_DEFAULT),
                      "cuEventCreate(IX autotune start)");
    try {
      check_preparation(cuEventCreate(&stop_, CU_EVENT_DEFAULT),
                        "cuEventCreate(IX autotune stop)");
    } catch (...) {
      (void)cuEventDestroy(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~TuningEvents() {
    if (stop_ != nullptr) {
      (void)cuEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cuEventDestroy(start_);
    }
  }

  TuningEvents(const TuningEvents &) = delete;
  TuningEvents &operator=(const TuningEvents &) = delete;

  [[nodiscard]] CUevent start() const noexcept { return start_; }
  [[nodiscard]] CUevent stop() const noexcept { return stop_; }

private:
  CUevent start_ = nullptr;
  CUevent stop_ = nullptr;
};

void launch_prepared_for_tuning(const PreparedLaunch &launch,
                                const KernelVariant &variant,
                                PreparationResources &resources) {
  LaunchArguments arguments(
      variant, [&](const KernelArgument &argument) -> CUdeviceptr {
        if (argument.kind == ArgumentKind::kTensor) {
          return resources.tensor(argument.uid);
        }
        if (resources.workspace() == 0) {
          compilation_error("IX autotune is missing artifact workspace");
        }
        return resources.workspace() + argument.workspace_offset;
      });
  check_preparation(
      cuLaunchKernel(launch.function, launch.grid[0], launch.grid[1],
                     launch.grid[2], launch.block[0], launch.block[1],
                     launch.block[2], launch.shared_memory, resources.stream(),
                     arguments.data(), nullptr),
      "cuLaunchKernel(IX autotune)");
}

PreparedStage prepare_stage(const JitFunction &function,
                            const ExecutionStage &stage,
                            PreparationResources &resources,
                            const EngineBuildContext &context) {
  require(!stage.variants.empty(), "Iluvatar IX stage has no variants",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  if (!stage.autotune) {
    require(stage.variants.size() == 1,
            "fixed Iluvatar IX stage must have one variant",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    validate_jit_abi(function, stage.variants.front());
    PreparedLaunch launch =
        capture_prepared_launch(function, stage.variants.front(), resources);
    return {&function, stage.variants.front(), launch};
  }

  require(stage.variants.size() >= 2,
          "Iluvatar IX autotune stage needs at least two variants",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  backend::autotune::SelectionRequest request;
  request.candidate_identity = stage.candidate_identity;
  request.device_identity = context.device_identity;
  request.measurement_identity =
      "iluvatar-libtriton-jit-corex-event-v1-stage-" +
      std::to_string(stage.stage_id);
  request.cache_path = stage.selection_cache;
  request.warmup_milliseconds = stage.warmup;
  request.benchmark_milliseconds = stage.repetitions;
  request.candidate_ids.reserve(stage.variants.size());
  for (const KernelVariant &variant : stage.variants) {
    request.candidate_ids.push_back(variant.variant_id);
  }

  if (const auto cached = backend::autotune::find_cached_candidate(request)) {
    try {
      validate_jit_abi(function, stage.variants[*cached]);
      PreparedLaunch launch =
          capture_prepared_launch(function, stage.variants[*cached], resources);
      return {&function, stage.variants[*cached], launch};
    } catch (const std::exception &) {
      backend::autotune::discard_cached_candidate(request);
    }
  }

  std::vector<PreparedLaunch> launches;
  launches.reserve(stage.variants.size());
  for (const KernelVariant &variant : stage.variants) {
    validate_jit_abi(function, variant);
    launches.push_back(capture_prepared_launch(function, variant, resources));
  }

  TuningEvents events;
  const auto launch = [&](std::size_t index) {
    launch_prepared_for_tuning(launches[index], stage.variants[index],
                               resources);
  };
  const backend::autotune::SelectionResult selected =
      backend::autotune::select_best_candidate(
          request,
          [&](std::size_t index, unsigned int iterations) {
            for (unsigned int iteration = 0; iteration < iterations;
                 ++iteration) {
              launch(index);
            }
            check_preparation(cuStreamSynchronize(resources.stream()),
                              "cuStreamSynchronize(IX autotune warmup)");
          },
          [&](std::size_t index, unsigned int iterations) {
            check_preparation(cuEventRecord(events.start(), resources.stream()),
                              "cuEventRecord(IX autotune start)");
            for (unsigned int iteration = 0; iteration < iterations;
                 ++iteration) {
              launch(index);
            }
            check_preparation(cuEventRecord(events.stop(), resources.stream()),
                              "cuEventRecord(IX autotune stop)");
            check_preparation(cuEventSynchronize(events.stop()),
                              "cuEventSynchronize(IX autotune stop)");
            float milliseconds = 0.0F;
            check_preparation(cuEventElapsedTime(&milliseconds, events.start(),
                                                 events.stop()),
                              "cuEventElapsedTime(IX autotune)");
            return milliseconds / static_cast<float>(iterations);
          });
  if (selected.candidate_index >= stage.variants.size()) {
    compilation_error("IX autotune selected an invalid variant");
  }
  return {&function, stage.variants[selected.candidate_index],
          launches[selected.candidate_index]};
}

class LibTritonJitExecutable final : public IluvatarExecutable {
public:
  LibTritonJitExecutable(const EngineBuildContext &context,
                         ExecutionProgramArtifact artifact)
      : context_(context), workspace_size_(artifact.workspace_size),
        workspace_alignment_(artifact.workspace_alignment) {
    require(context_.context != nullptr,
            "Iluvatar executable received a null CoreX context",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    require(artifact.backend == "iluvatar" && artifact.target == "corex_71" &&
                artifact.engine == "libtriton_jit",
            "Iluvatar executable received an incompatible artifact",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    binding_requirements_.reserve(artifact.external_uids.size());
    for (const std::int64_t uid : artifact.external_uids) {
      const auto tensor =
          std::find_if(artifact.tensors.begin(), artifact.tensors.end(),
                       [uid](const TensorArtifact &candidate) {
                         return candidate.uid == uid;
                       });
      if (tensor == artifact.tensors.end() || tensor->is_virtual) {
        compilation_error(
            "Iluvatar external binding has no tensor specification");
      }
      binding_requirements_.push_back({uid, tensor->alignment});
    }

    try {
      std::lock_guard<std::mutex> lock(jit_build_mutex);
      configure_python_path();
      ContextGuard guard(context_.context);
      PreparationResources resources(artifact);
      stages_.reserve(artifact.stages.size());
      for (const ExecutionStage &stage : artifact.stages) {
        if (stage.source.empty() || stage.function_name.empty() ||
            stage.variants.empty()) {
          compilation_error("Iluvatar IX execution stage is incomplete");
        }
        const JitFunction &function = JitFunction::get_instance(
            stage.source.string(), stage.function_name);
        stages_.push_back(prepare_stage(function, stage, resources, context_));
      }
    } catch (const std::bad_alloc &) {
      throw;
    } catch (const IluvatarError &) {
      throw;
    } catch (const std::exception &error) {
      compilation_error("IX libtriton_jit executable build failed: " +
                        std::string(error.what()));
    }
  }

  LibTritonJitExecutable(const LibTritonJitExecutable &) = delete;
  LibTritonJitExecutable &operator=(const LibTritonJitExecutable &) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(CUstream stream, const flagdnnBackendBindingV2 *bindings,
               std::size_t binding_count, void *workspace,
               std::size_t workspace_size) override {
    validate_execution_arguments(bindings, binding_count, workspace,
                                 workspace_size);
    try {
      ContextGuard guard(context_.context);
      for (const PreparedStage &stage : stages_) {
        LaunchArguments arguments(
            stage.variant, [&](const KernelArgument &argument) -> CUdeviceptr {
              if (argument.kind == ArgumentKind::kTensor) {
                return binding_pointer(argument.uid, bindings, binding_count);
              }
              const std::uintptr_t base =
                  reinterpret_cast<std::uintptr_t>(workspace);
              if (argument.workspace_offset >
                  std::numeric_limits<std::uintptr_t>::max() - base) {
                throw IluvatarError(
                    FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                    "Iluvatar workspace pointer arithmetic overflowed");
              }
              return static_cast<CUdeviceptr>(base + argument.workspace_offset);
            });
        check_corex(cuLaunchKernel(stage.launch.function, stage.launch.grid[0],
                                   stage.launch.grid[1], stage.launch.grid[2],
                                   stage.launch.block[0], stage.launch.block[1],
                                   stage.launch.block[2],
                                   stage.launch.shared_memory, stream,
                                   arguments.data(), nullptr),
                    "cuLaunchKernel(Iluvatar prepared stage)");
      }
    } catch (const IluvatarError &) {
      throw;
    } catch (const std::exception &error) {
      throw IluvatarError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                          "Iluvatar prepared execution failed: " +
                              std::string(error.what()));
    }
  }

private:
  void validate_execution_arguments(const flagdnnBackendBindingV2 *bindings,
                                    std::size_t binding_count, void *workspace,
                                    std::size_t workspace_size) const {
    require(binding_count == binding_requirements_.size(),
            "binding count does not match Iluvatar executable");
    require(binding_count == 0 || bindings != nullptr,
            "Iluvatar binding array is null");
    require(workspace_size >= workspace_size_,
            "workspace is smaller than Iluvatar executable requirement");
    require(workspace_size_ == 0 || workspace != nullptr,
            "Iluvatar executable workspace is null");
    require(workspace_size_ == 0 ||
                reinterpret_cast<std::uintptr_t>(workspace) %
                        workspace_alignment_ ==
                    0,
            "Iluvatar workspace does not satisfy artifact alignment");

    for (std::size_t index = 0; index < binding_count; ++index) {
      require(bindings[index].uid > 0, "Iluvatar binding UID must be positive");
      require(bindings[index].device_pointer != nullptr,
              "Iluvatar binding device pointer is null");
      for (std::size_t previous = 0; previous < index; ++previous) {
        require(bindings[previous].uid != bindings[index].uid,
                "Iluvatar binding UID is duplicated");
      }
      const auto expected = std::find_if(
          binding_requirements_.begin(), binding_requirements_.end(),
          [&](const BindingRequirement &requirement) {
            return requirement.uid == bindings[index].uid;
          });
      require(expected != binding_requirements_.end(),
              "Iluvatar binding UID is not required by the executable");
      require(reinterpret_cast<std::uintptr_t>(bindings[index].device_pointer) %
                      expected->alignment ==
                  0,
              "Iluvatar tensor binding does not satisfy alignment");
    }
  }

  static CUdeviceptr binding_pointer(std::int64_t uid,
                                     const flagdnnBackendBindingV2 *bindings,
                                     std::size_t binding_count) {
    for (std::size_t index = 0; index < binding_count; ++index) {
      if (bindings[index].uid == uid) {
        return static_cast<CUdeviceptr>(
            reinterpret_cast<std::uintptr_t>(bindings[index].device_pointer));
      }
    }
    throw IluvatarError(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                        "a required Iluvatar tensor binding is missing");
  }

  EngineBuildContext context_;
  std::vector<BindingRequirement> binding_requirements_;
  std::vector<PreparedStage> stages_;
  std::size_t workspace_size_ = 0;
  std::size_t workspace_alignment_ = 1;
};

} // namespace

std::unique_ptr<IluvatarExecutable>
create_libtriton_jit_executable(const EngineBuildContext &context,
                                const flagdnnBackendBuildInputV2 &input) {
  ExecutionProgramArtifact artifact =
      load_and_validate_artifact(context, input);
  return std::make_unique<LibTritonJitExecutable>(context, std::move(artifact));
}

} // namespace flagdnn::iluvatar
