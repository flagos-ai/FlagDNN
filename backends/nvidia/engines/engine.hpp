/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_
#define FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_

#include "backends/nvidia/artifact.hpp"

#include <cstddef>
#include <memory>

namespace flagdnn::cuda {

class ExecutionEngine {
 public:
  virtual ~ExecutionEngine() = default;

  [[nodiscard]] virtual std::size_t workspace_size() const noexcept = 0;
  virtual void execute(
      CUstream stream,
      const flagdnnBackendBindingV2 bindings[],
      std::size_t binding_count,
      void* workspace,
      std::size_t workspace_size) const = 0;
};

[[nodiscard]] std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input);

[[nodiscard]] std::unique_ptr<ExecutionEngine>
create_external_artifact_engine(
    const EngineBuildContext& context,
    CudaArtifact artifact);

[[nodiscard]] std::unique_ptr<ExecutionEngine>
create_libtriton_jit_engine(
    const EngineBuildContext& context,
    CudaArtifact artifact);

[[nodiscard]] bool libtriton_jit_engine_available() noexcept;

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_ENGINES_ENGINE_HPP_
