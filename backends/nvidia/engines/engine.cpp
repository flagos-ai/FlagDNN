/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/engines/engine.hpp"

#include "backends/nvidia/error.hpp"

#include <utility>

namespace flagdnn::cuda {

std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  CudaArtifact artifact = parse_cuda_artifact(context, input);
  switch (artifact.engine) {
    case EngineKind::kExternalArtifact:
      return create_external_artifact_engine(context, std::move(artifact));
    case EngineKind::kLibTritonJit:
      return create_libtriton_jit_engine(context, std::move(artifact));
  }
  throw CudaError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                  "unknown CUDA execution engine");
}

}  // namespace flagdnn::cuda
