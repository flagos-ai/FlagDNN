/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_CUDA_LAUNCH_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_CUDA_LAUNCH_HPP_

#include "backends/nvidia/artifact.hpp"
#include "backends/nvidia/context.hpp"
#include "backends/nvidia/error.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace flagdnn::cuda {

struct TuningAllocation {
  std::int64_t uid = 0;
  std::size_t size = 0;
  CUdeviceptr pointer = 0;
};

// Per-call, stack-owned ABI storage shared by JIT execution and autotuning.
// Only live slots are initialized; no heap allocation or mutable shared state.
class KernelArguments {
 public:
  KernelArguments(std::span<const ArgumentSpec> arguments,
                  const flagdnnBackendBindingV2* bindings,
                  std::size_t binding_count, void* workspace,
                  CUdeviceptr global_scratch = 0) {
    pack(arguments, [&](const ArgumentSpec& argument) {
      for (std::size_t i = 0; i < binding_count; ++i) {
        if (bindings[i].uid == argument.uid) {
          const auto pointer = static_cast<CUdeviceptr>(
              reinterpret_cast<std::uintptr_t>(bindings[i].device_pointer));
          require(argument.alignment != 0 &&
                      pointer % argument.alignment == 0,
                  "a tensor binding does not satisfy its declared alignment");
          return pointer;
        }
      }
      throw CudaError(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                      "a required tensor UID is missing from bindings");
    }, static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(workspace)),
    global_scratch);
  }

  KernelArguments(std::span<const ArgumentSpec> arguments,
                  const std::vector<TuningAllocation>& allocations,
                  CUdeviceptr workspace, CUdeviceptr global_scratch = 0) {
    pack(arguments, [&](const ArgumentSpec& argument) {
      const auto allocation = std::find_if(
          allocations.begin(), allocations.end(),
          [&](const TuningAllocation& value) { return value.uid == argument.uid; });
      require(allocation != allocations.end(),
              "autotune tensor allocation is missing",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      return allocation->pointer;
    }, workspace, global_scratch);
  }

  KernelArguments(const CudaKernelArtifact& kernel,
                  const flagdnnBackendBindingV2* bindings,
                  std::size_t binding_count, void* workspace,
                  CUdeviceptr global_scratch)
      : KernelArguments(kernel.arguments, bindings, binding_count,
                        workspace, global_scratch) {}

  KernelArguments(const CudaKernelArtifact& kernel,
                  const std::vector<TuningAllocation>& allocations,
                  CUdeviceptr workspace, CUdeviceptr global_scratch)
      : KernelArguments(kernel.arguments, allocations, workspace, global_scratch) {}

  KernelArguments(const KernelArguments&) = delete;
  KernelArguments& operator=(const KernelArguments&) = delete;

  [[nodiscard]] void** data() noexcept { return parameters_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return parameter_count_; }

 private:
  union Value {
    CUdeviceptr pointer;
    std::int32_t scalar_i32;
    std::int64_t scalar_i64;
    float scalar_f32;
  };

  template <typename ResolveTensor>
  void pack(std::span<const ArgumentSpec> arguments,
            ResolveTensor&& resolve, CUdeviceptr workspace,
            CUdeviceptr global_scratch) {
    require(arguments.size() <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
            "CUDA kernel has too many arguments",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::size_t i = 0;
    std::size_t map_index = 0;
    for (const auto& argument : arguments) {
      const std::size_t slots = argument.kind == ArgumentKind::kTensorMap ? 5 : 1;
      require(i + slots <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
              "expanded CUDA argument ABI exceeds capacity",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      switch (argument.kind) {
        case ArgumentKind::kTensorMap: {
          require(map_index < maps_.size(), "too many CUDA TensorMaps",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          auto& descriptor = maps_[map_index++];
          const auto& spec = argument.tensor_map;
          const cuuint64_t dimensions[] = {
              static_cast<cuuint64_t>(spec.shape[1]),
              static_cast<cuuint64_t>(spec.shape[0])};
          const cuuint64_t strides[] = {dimensions[0] * sizeof(float)};
          // Triton splits the logical innermost tile into 128-byte boxes.
          const cuuint32_t box[] = {32, spec.block_shape[0]};
          const cuuint32_t element_strides[] = {1, 1};
          check_cuda(cuTensorMapEncodeTiled(
                         &descriptor, CU_TENSOR_MAP_DATA_TYPE_TFLOAT32, 2,
                         reinterpret_cast<void*>(static_cast<std::uintptr_t>(resolve(argument))),
                         dimensions, strides, box, element_strides,
                         CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                         CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                     "cuTensorMapEncodeTiled(TF32-RNE)");
          parameters_[i] = &descriptor;
          for (std::size_t axis = 0; axis < 2; ++axis) {
            values_[i + 1 + axis].scalar_i32 = spec.shape[axis];
            parameters_[i + 1 + axis] = &values_[i + 1 + axis].scalar_i32;
            values_[i + 3 + axis].scalar_i64 = axis == 0 ? spec.shape[1] : 1;
            parameters_[i + 3 + axis] = &values_[i + 3 + axis].scalar_i64;
          }
          break;
        }
        case ArgumentKind::kTensor:
          values_[i].pointer = resolve(argument);
          parameters_[i] = &values_[i].pointer;
          break;
        case ArgumentKind::kWorkspaceTensor:
          require(workspace != 0, "CUDA kernel workspace is missing");
          values_[i].pointer = workspace + argument.workspace_offset;
          parameters_[i] = &values_[i].pointer;
          break;
        case ArgumentKind::kScalarI32:
          values_[i].scalar_i32 = argument.scalar_i32;
          parameters_[i] = &values_[i].scalar_i32;
          break;
        case ArgumentKind::kScalarF32:
          values_[i].scalar_f32 = argument.scalar_f32;
          parameters_[i] = &values_[i].scalar_f32;
          break;
        default:
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "CUDA kernel argument kind is unsupported");
      }
      i += slots;
    }
    global_scratch_ = global_scratch;
    parameters_[i] = &global_scratch_;
    parameters_[i + 1] = &profile_scratch_;
    parameter_count_ = i + 2;
  }

  std::array<Value, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> values_;
  alignas(128) std::array<CUtensorMap, kMaximumTensorMaps> maps_;
  std::array<void*, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2> parameters_;
  std::size_t parameter_count_ = 0;
  CUdeviceptr global_scratch_ = 0;
  CUdeviceptr profile_scratch_ = 0;
};

}  // namespace flagdnn::cuda
#endif  // FLAGDNN_BACKENDS_NVIDIA_CUDA_LAUNCH_HPP_
