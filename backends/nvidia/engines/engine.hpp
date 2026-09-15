/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_
#define FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_

#include "backends/nvidia/artifact.hpp"

#include <cstddef>
#include <memory>

namespace flagdnn::cuda {

// NVIDIA has one execution path. Hide JIT/Python dependencies behind the
// implementation without a second engine factory or a virtual dispatch layer.
class ExecutionEngine final {
 public:
  ExecutionEngine(const EngineBuildContext& context,
                  const flagdnnBackendBuildInputV2& input);
  ~ExecutionEngine();
  ExecutionEngine(const ExecutionEngine&) = delete;
  ExecutionEngine& operator=(const ExecutionEngine&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept;
  void execute(
      CUstream stream,
      const flagdnnBackendBindingV2 bindings[],
      std::size_t binding_count,
      void* workspace,
      std::size_t workspace_size) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input);

[[nodiscard]] bool libtriton_jit_environment_prepared() noexcept;
void prepare_libtriton_jit_environment(
    const EngineBuildContext& context, const CudaArtifact& artifact);

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_
