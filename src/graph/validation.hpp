/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_VALIDATION_HPP_
#define FLAGDNN_GRAPH_VALIDATION_HPP_

#include "graph/types.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flagdnn::native {

struct ValidatedGraph {
  std::vector<std::size_t> execution_order;
  std::vector<TensorSpec> tensors;
  std::vector<std::int64_t> external_binding_uids;
};

[[nodiscard]] ValidatedGraph validate_graph(const GraphSpec& graph);

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_VALIDATION_HPP_
