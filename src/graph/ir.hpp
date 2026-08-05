/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_IR_HPP_
#define FLAGDNN_GRAPH_IR_HPP_

#include <flagdnn/flagdnn.h>

#include "graph/lowering/lowering.hpp"
#include "graph/types.hpp"
#include "graph/validation.hpp"

#include <string>
#include <vector>

namespace flagdnn::native {

class RuntimeContext;

[[nodiscard]] std::string make_graph_ir(
    const RuntimeContext& context,
    const GraphSpec& graph,
    const flagdnnBuildOptions_t& options,
    const std::vector<LoweredOperation>& lowered,
    const ValidatedGraph& validated);

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_IR_HPP_
