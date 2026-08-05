/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_GRAPH_HPP_
#define FLAGDNN_GRAPH_GRAPH_HPP_

#include <flagdnn/flagdnn.h>

#include "graph/types.hpp"

#include <memory>

namespace flagdnn::native {

class Executable;
class RuntimeContext;

std::unique_ptr<Executable> build_graph_executable(
    RuntimeContext& context,
    const GraphSpec& graph,
    const flagdnnBuildOptions_t& options);

void validate_graph_structure(const GraphSpec& graph);

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_GRAPH_HPP_
