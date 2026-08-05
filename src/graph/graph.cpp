/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/graph.hpp"

#include "error.hpp"
#include "graph/ir.hpp"
#include "graph/lowering/lowering.hpp"
#include "graph/validation.hpp"
#include "runtime/artifact.hpp"
#include "runtime/context.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::native {

void validate_graph_structure(const GraphSpec& graph) {
  (void)validate_graph(graph);
  for (const OperationSpec& operation : graph.operations) {
    (void)lower_operation(operation);
  }
}

std::unique_ptr<Executable> build_graph_executable(
    RuntimeContext& context,
    const GraphSpec& graph,
    const flagdnnBuildOptions_t& options) {
  if (!graph.finalized) {
    throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                   "graph must be finalized before it is built");
  }
  if (graph.operations.empty()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "cannot build an empty graph");
  }
  if (graph.operations.size() > 1024) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "graph operation count exceeds the executable limit");
  }
  if ((options.flags & ~FLAGDNN_BUILD_OPTION_FLAGS_ALL) != 0) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "requested build option flags are not supported");
  }

  const ValidatedGraph validated = validate_graph(graph);
  std::vector<LoweredOperation> lowered;
  lowered.reserve(graph.operations.size());
  for (const OperationSpec& operation : graph.operations) {
    lowered.push_back(lower_operation(operation));
  }
  const std::string graph_ir =
      make_graph_ir(context, graph, options, lowered, validated);
  ArtifactPackage artifact = prepare_artifact_package(context, graph_ir);
  std::unique_ptr<BackendExecutable> backend_executable;
  try {
    backend_executable = context.create_executable(artifact);
  } catch (const ApiError& error) {
    if (!artifact.cache_hit ||
        error.status() != FLAGDNN_STATUS_COMPILATION_FAILED) {
      throw;
    }
    invalidate_cached_artifact(artifact);
    artifact = prepare_artifact_package(context, graph_ir);
    backend_executable = context.create_executable(artifact);
  }
  return std::make_unique<Executable>(
      std::move(backend_executable),
      validated.external_binding_uids,
      graph.operations.size());
}

}  // namespace flagdnn::native
