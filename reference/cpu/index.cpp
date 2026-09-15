/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/index.hpp"

#include <stdexcept>
namespace flagdnn::reference::cpu {
std::pair<std::size_t, std::size_t> index_source(
    const IndexParameters& parameters, std::size_t output_index) {
  const auto rank = parameters.output_shape.size();
  const auto axis = static_cast<std::size_t>(
      parameters.axis < 0 ? parameters.axis + static_cast<std::int64_t>(rank)
                          : parameters.axis);
  std::vector<std::size_t> coordinates(rank);
  for (std::size_t dim = rank; dim != 0; --dim) {
    coordinates[dim - 1] = output_index % parameters.output_shape[dim - 1];
    output_index /= parameters.output_shape[dim - 1];
  }
  if (parameters.operation == "gen_index") return {0, coordinates[axis]};
  for (std::size_t input = 0; input < parameters.input_shapes.size(); ++input) {
    const auto& shape = parameters.input_shapes[input];
    if (coordinates[axis] < static_cast<std::size_t>(shape[axis])) {
      std::size_t source = 0;
      for (std::size_t dim = 0; dim < rank; ++dim)
        source = source * shape[dim] + coordinates[dim];
      return {input, source};
    }
    coordinates[axis] -= shape[axis];
  }
  throw std::invalid_argument("concatenate oracle coordinate outside inputs");
}
}  // namespace flagdnn::reference::cpu
