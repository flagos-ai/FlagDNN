// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/engines/engine.hpp"

#include "backends/thead/error.hpp"

#include <utility>

namespace flagdnn::thead {

std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    ExecutionProgramArtifact artifact) {
  require(artifact.engine == "libtriton_jit",
          "THead execution artifact names an unsupported engine",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  return create_libtriton_jit_engine(context, std::move(artifact));
}

}  // namespace flagdnn::thead
