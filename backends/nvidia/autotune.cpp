/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/autotune.hpp"

#include "backends/autotune_policy.hpp"
#include "backends/nvidia/error.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::cuda {
namespace {

struct TuningKernel {
  std::string variant_id;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
  std::vector<ArgumentSpec> arguments;
  std::vector<char> binary;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

struct TuningAllocation {
  std::int64_t uid = 0;
  std::size_t size = 0;
  CUdeviceptr pointer = 0;
};

bool logging_enabled() noexcept {
  const char* value = std::getenv("FLAGDNN_PRINT_AUTOTUNING");
  return value != nullptr && value[0] != '\0' &&
         std::string_view(value) != "0";
}

void unload_kernel(TuningKernel& kernel) noexcept {
  if (kernel.module != nullptr) {
    (void)cuModuleUnload(kernel.module);
    kernel.module = nullptr;
    kernel.function = nullptr;
  }
}

TuningKernel load_kernel(const CudaKernelArtifact& specification) {
  TuningKernel loaded;
  loaded.variant_id = specification.variant_id;
  loaded.grid = specification.grid;
  loaded.block = specification.block;
  loaded.shared_memory = specification.shared_memory;
  loaded.arguments = specification.arguments;

  std::ifstream input(specification.binary, std::ios::binary);
  if (!input) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "cannot open validated CUDA artifact binary for autotune");
  }
  loaded.binary.assign(std::istreambuf_iterator<char>(input),
                       std::istreambuf_iterator<char>());
  if (input.bad() || loaded.binary.empty()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "cannot read validated CUDA artifact binary for autotune");
  }

  try {
    check_cuda(cuModuleLoadDataEx(&loaded.module,
                                  loaded.binary.data(),
                                  0,
                                  nullptr,
                                  nullptr),
               "cuModuleLoadDataEx(autotune)");
    check_cuda(cuModuleGetFunction(&loaded.function,
                                   loaded.module,
                                   specification.entry_symbol.c_str()),
               "cuModuleGetFunction(autotune)");
    if (loaded.shared_memory > 48U * 1024U) {
      check_cuda(cuFuncSetAttribute(
                     loaded.function,
                     CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                     static_cast<int>(loaded.shared_memory)),
                 "cuFuncSetAttribute(autotune dynamic shared memory)");
    }
  } catch (...) {
    unload_kernel(loaded);
    throw;
  }
  return loaded;
}

void launch_kernel(const TuningKernel& kernel,
                   const std::vector<TuningAllocation>& allocations,
                   CUdeviceptr workspace,
                   CUstream stream) {
  struct ArgumentValue {
    CUdeviceptr pointer = 0;
    std::int32_t scalar_i32 = 0;
    float scalar_f32 = 0.0F;
  };

  std::vector<ArgumentValue> values(kernel.arguments.size());
  std::vector<void*> parameters(kernel.arguments.size() + 2, nullptr);
  for (std::size_t index = 0; index < kernel.arguments.size(); ++index) {
    const ArgumentSpec& argument = kernel.arguments[index];
    if (argument.kind == ArgumentKind::kTensor) {
      const auto allocation = std::find_if(
          allocations.begin(),
          allocations.end(),
          [&](const TuningAllocation& value) {
            return value.uid == argument.uid;
          });
      if (allocation == allocations.end()) {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "autotune tensor allocation is missing");
      }
      values[index].pointer = allocation->pointer;
      parameters[index] = &values[index].pointer;
    } else if (argument.kind == ArgumentKind::kWorkspaceTensor) {
      if (workspace == 0) {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "autotune workspace allocation is missing");
      }
      values[index].pointer = workspace + argument.workspace_offset;
      parameters[index] = &values[index].pointer;
    } else if (argument.kind == ArgumentKind::kScalarI32) {
      values[index].scalar_i32 = argument.scalar_i32;
      parameters[index] = &values[index].scalar_i32;
    } else if (argument.kind == ArgumentKind::kScalarF32) {
      values[index].scalar_f32 = argument.scalar_f32;
      parameters[index] = &values[index].scalar_f32;
    } else {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "autotune argument kind is unsupported");
    }
  }

  CUdeviceptr global_scratch = 0;
  CUdeviceptr profile_scratch = 0;
  parameters[kernel.arguments.size()] = &global_scratch;
  parameters[kernel.arguments.size() + 1] = &profile_scratch;
  check_cuda(cuLaunchKernel(kernel.function,
                            kernel.grid[0],
                            kernel.grid[1],
                            kernel.grid[2],
                            kernel.block[0],
                            kernel.block[1],
                            kernel.block[2],
                            kernel.shared_memory,
                            stream,
                            parameters.data(),
                            nullptr),
             "cuLaunchKernel(autotune)");
}

}  // namespace

std::size_t select_autotune_candidate(
    const EngineBuildContext& context,
    std::size_t workspace_size,
    const CudaStageArtifact& stage) {
  require(stage.autotune && stage.variants.size() >= 2,
          "invalid CUDA autotune stage",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  backend::autotune::SelectionRequest request;
  request.candidate_identity = stage.candidate_identity;
  request.device_identity = context.device_identity;
  request.measurement_identity = "nvidia-stage-driver-events-v2";
  request.cache_path = stage.selection_cache;
  request.warmup_milliseconds = stage.warmup;
  request.benchmark_milliseconds = stage.repetitions;
  request.candidate_ids.reserve(stage.variants.size());
  for (const CudaKernelArtifact& variant : stage.variants) {
    request.candidate_ids.push_back(variant.variant_id);
  }

  std::unique_ptr<ContextGuard> guard;
  std::vector<TuningKernel> candidates;
  std::vector<TuningAllocation> allocations;
  CUdeviceptr workspace = 0;
  CUstream stream = nullptr;
  CUevent start = nullptr;
  CUevent stop = nullptr;

  const auto cleanup = [&]() noexcept {
    if (start != nullptr) {
      (void)cuEventDestroy(start);
      start = nullptr;
    }
    if (stop != nullptr) {
      (void)cuEventDestroy(stop);
      stop = nullptr;
    }
    if (stream != nullptr) {
      (void)cuStreamDestroy(stream);
      stream = nullptr;
    }
    if (workspace != 0) {
      (void)cuMemFree(workspace);
      workspace = 0;
    }
    for (TuningAllocation& allocation : allocations) {
      if (allocation.pointer != 0) {
        (void)cuMemFree(allocation.pointer);
        allocation.pointer = 0;
      }
    }
    for (TuningKernel& candidate : candidates) {
      unload_kernel(candidate);
    }
  };

  const auto initialize_resources = [&]() {
    if (guard != nullptr) {
      return;
    }
    guard = std::make_unique<ContextGuard>(context.context);
    candidates.reserve(stage.variants.size());
    for (const CudaKernelArtifact& variant : stage.variants) {
      candidates.push_back(load_kernel(variant));
    }

    for (const ArgumentSpec& argument : candidates.front().arguments) {
      if (argument.kind != ArgumentKind::kTensor) {
        continue;
      }
      const auto existing = std::find_if(
          allocations.begin(),
          allocations.end(),
          [&](const TuningAllocation& allocation) {
            return allocation.uid == argument.uid;
          });
      if (existing == allocations.end()) {
        allocations.push_back({argument.uid, argument.storage_size, 0});
      } else {
        existing->size = std::max(existing->size, argument.storage_size);
      }
    }
    for (TuningAllocation& allocation : allocations) {
      check_cuda(cuMemAlloc(&allocation.pointer, allocation.size),
                 "cuMemAlloc(autotune tensor)");
    }
    if (workspace_size != 0) {
      check_cuda(cuMemAlloc(&workspace, workspace_size),
                 "cuMemAlloc(autotune workspace)");
    }
    check_cuda(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate(autotune)");
    check_cuda(cuEventCreate(&start, CU_EVENT_DEFAULT),
               "cuEventCreate(autotune start)");
    check_cuda(cuEventCreate(&stop, CU_EVENT_DEFAULT),
               "cuEventCreate(autotune stop)");
  };

  try {
    const backend::autotune::SelectionResult result =
        backend::autotune::select_best_candidate(
            request,
            [&](std::size_t candidate_index, unsigned int iterations) {
              initialize_resources();
              for (unsigned int iteration = 0;
                   iteration < iterations;
                   ++iteration) {
                launch_kernel(candidates[candidate_index],
                              allocations,
                              workspace,
                              stream);
              }
              check_cuda(cuStreamSynchronize(stream),
                         "cuStreamSynchronize(autotune warmup)");
            },
            [&](std::size_t candidate_index, unsigned int iterations) {
              initialize_resources();
              const TuningKernel& candidate = candidates[candidate_index];
              check_cuda(cuEventRecord(start, stream),
                         "cuEventRecord(autotune start)");
              for (unsigned int iteration = 0;
                   iteration < iterations;
                   ++iteration) {
                launch_kernel(candidate, allocations, workspace, stream);
              }
              check_cuda(cuEventRecord(stop, stream),
                         "cuEventRecord(autotune stop)");
              check_cuda(cuEventSynchronize(stop),
                         "cuEventSynchronize(autotune)");
              float milliseconds = 0.0F;
              check_cuda(cuEventElapsedTime(&milliseconds, start, stop),
                         "cuEventElapsedTime(autotune)");
              return milliseconds / static_cast<float>(iterations);
            });

    const std::string& selected_id =
        request.candidate_ids[result.candidate_index];
    if (logging_enabled()) {
      if (result.cache_hit) {
        std::cerr << "[FlagDNN autotune] cache hit "
                  << stage.candidate_identity.substr(0, 12) << " -> "
                  << selected_id << '\n';
      } else {
        for (std::size_t index = 0;
             index < result.median_milliseconds.size();
             ++index) {
          std::cerr << "[FlagDNN autotune] " << request.candidate_ids[index]
                    << " median_ms="
                    << result.median_milliseconds[index] << '\n';
        }
        std::cerr << "[FlagDNN autotune] selected " << selected_id
                  << " median_ms="
                  << result.median_milliseconds[result.candidate_index]
                  << '\n';
      }
    }
    cleanup();
    guard.reset();
    return result.candidate_index;
  } catch (...) {
    cleanup();
    guard.reset();
    throw;
  }
}

}  // namespace flagdnn::cuda
