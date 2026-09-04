/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/engines/engine.hpp"

#include <utility>

namespace flagdnn::mthreads {

std::unique_ptr<ExecutionEngine> create_execution_engine(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  MthreadsArtifact artifact = parse_mthreads_artifact(context, input);
  return create_libtriton_jit_engine(context, std::move(artifact));
}

}  // namespace flagdnn::mthreads
