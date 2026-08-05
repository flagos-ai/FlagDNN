/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/engines/engine.hpp"

#include "backends/nvidia/artifact.hpp"
#include "backends/nvidia/autotune.hpp"
#include "backends/nvidia/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::cuda {
namespace {

struct LoadedKernel {
  std::string variant_id;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
  std::vector<ArgumentSpec> arguments;
  std::vector<char> binary;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

class ExternalArtifactEngine final : public ExecutionEngine {
 public:
  ExternalArtifactEngine(const EngineBuildContext& context,
                         CudaArtifact artifact)
      : context_(context) {
    workspace_size_ = artifact.workspace_size;
    binding_uids_ = std::move(artifact.binding_uids);

    CUcontext retained_context = nullptr;
    check_cuda(cuDevicePrimaryCtxRetain(&retained_context, context_.device),
               "cuDevicePrimaryCtxRetain(external artifact engine)");
    if (retained_context != context_.context) {
      (void)cuDevicePrimaryCtxRelease(context_.device);
      throw CudaError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "CUDA primary context changed while building executable");
    }
    retained_ = true;

    try {
      kernels_.reserve(artifact.stages.size());
      for (const CudaStageArtifact& stage : artifact.stages) {
        if (stage.variants.empty()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "CUDA execution stage has no kernel variants");
        }
        const std::size_t selected =
            stage.autotune
                ? select_autotune_candidate(
                      context_, workspace_size_, stage)
                : 0;
        if (selected >= stage.variants.size()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                          "CUDA autotune selected an invalid candidate");
        }
        ContextGuard guard(context_.context);
        kernels_.push_back(load_kernel(stage.variants[selected]));
      }
    } catch (...) {
      unload_all();
      (void)cuDevicePrimaryCtxRelease(context_.device);
      retained_ = false;
      throw;
    }
  }

  ~ExternalArtifactEngine() override {
    unload_all();
    if (retained_) {
      (void)cuDevicePrimaryCtxRelease(context_.device);
    }
  }

  ExternalArtifactEngine(const ExternalArtifactEngine&) = delete;
  ExternalArtifactEngine& operator=(const ExternalArtifactEngine&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(CUstream stream,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const override {
    require(workspace_size >= workspace_size_,
            "workspace is smaller than CUDA executable requirement");
    require(workspace_size_ == 0 || workspace != nullptr,
            "CUDA executable workspace is null");
    require(binding_count == binding_uids_.size(),
            "binding count does not match CUDA executable");
    require(binding_count == 0 || bindings != nullptr,
            "binding array is null");

    ContextGuard guard(context_.context);
    for (const LoadedKernel& kernel : kernels_) {
      launch_kernel(kernel,
                    stream,
                    bindings,
                    binding_count,
                    workspace);
    }
  }

 private:
  static void unload_kernel(LoadedKernel& kernel) noexcept {
    if (kernel.module != nullptr) {
      (void)cuModuleUnload(kernel.module);
      kernel.module = nullptr;
      kernel.function = nullptr;
    }
  }

  void unload_all() noexcept {
    if (kernels_.empty()) {
      return;
    }
    try {
      ContextGuard guard(context_.context);
      for (LoadedKernel& kernel : kernels_) {
        unload_kernel(kernel);
      }
    } catch (...) {
    }
    kernels_.clear();
  }

  [[nodiscard]] static LoadedKernel load_kernel(
      const CudaKernelArtifact& specification) {
    LoadedKernel loaded;
    loaded.variant_id = specification.variant_id;
    loaded.grid = specification.grid;
    loaded.block = specification.block;
    loaded.shared_memory = specification.shared_memory;
    loaded.arguments = specification.arguments;

    std::ifstream input(specification.binary, std::ios::binary);
    if (!input) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot open validated CUDA artifact binary");
    }
    loaded.binary.assign(std::istreambuf_iterator<char>(input),
                         std::istreambuf_iterator<char>());
    if (input.bad() || loaded.binary.empty()) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "cannot read validated CUDA artifact binary");
    }

    try {
      check_cuda(cuModuleLoadDataEx(&loaded.module,
                                    loaded.binary.data(),
                                    0,
                                    nullptr,
                                    nullptr),
                 "cuModuleLoadDataEx");
      check_cuda(cuModuleGetFunction(&loaded.function,
                                     loaded.module,
                                     specification.entry_symbol.c_str()),
                 "cuModuleGetFunction");
      if (loaded.shared_memory > 48U * 1024U) {
        check_cuda(cuFuncSetAttribute(
                       loaded.function,
                       CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                       static_cast<int>(loaded.shared_memory)),
                   "cuFuncSetAttribute(dynamic shared memory)");
      }
    } catch (...) {
      unload_kernel(loaded);
      throw;
    }
    return loaded;
  }

  static void launch_kernel(
      const LoadedKernel& kernel,
      CUstream stream,
      const flagdnnBackendBindingV2 bindings[],
      std::size_t binding_count,
      void* workspace) {
    struct ArgumentValue {
      CUdeviceptr pointer = 0;
      std::int32_t scalar_i32 = 0;
      float scalar_f32 = 0.0F;
    };

    std::array<ArgumentValue, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> values{};
    std::array<void*, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2> parameters{};
    for (std::size_t index = 0; index < kernel.arguments.size(); ++index) {
      const ArgumentSpec& argument = kernel.arguments[index];
      if (argument.kind == ArgumentKind::kTensor) {
        bool found = false;
        for (std::size_t supplied = 0;
             supplied < binding_count;
             ++supplied) {
          if (bindings[supplied].uid == argument.uid) {
            values[index].pointer = static_cast<CUdeviceptr>(
                reinterpret_cast<std::uintptr_t>(
                    bindings[supplied].device_pointer));
            found = true;
            break;
          }
        }
        require(found, "a required tensor UID is missing from bindings");
        require(values[index].pointer % argument.alignment == 0,
                "a tensor binding does not satisfy its declared alignment");
        parameters[index] = &values[index].pointer;
      } else if (argument.kind == ArgumentKind::kWorkspaceTensor) {
        values[index].pointer = static_cast<CUdeviceptr>(
            reinterpret_cast<std::uintptr_t>(workspace) +
            argument.workspace_offset);
        parameters[index] = &values[index].pointer;
      } else if (argument.kind == ArgumentKind::kScalarI32) {
        values[index].scalar_i32 = argument.scalar_i32;
        parameters[index] = &values[index].scalar_i32;
      } else if (argument.kind == ArgumentKind::kScalarF32) {
        values[index].scalar_f32 = argument.scalar_f32;
        parameters[index] = &values[index].scalar_f32;
      } else {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "external artifact argument kind is unsupported");
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
               "cuLaunchKernel");
  }

  EngineBuildContext context_;
  std::vector<std::int64_t> binding_uids_;
  std::vector<LoadedKernel> kernels_;
  std::size_t workspace_size_ = 0;
  bool retained_ = false;
};

}  // namespace

std::unique_ptr<ExecutionEngine> create_external_artifact_engine(
    const EngineBuildContext& context,
    CudaArtifact artifact) {
  return std::make_unique<ExternalArtifactEngine>(context,
                                                   std::move(artifact));
}

}  // namespace flagdnn::cuda
