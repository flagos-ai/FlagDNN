/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/index.hpp"
#include "reference/cpu/shape.hpp"

#include <algorithm>
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

std::vector<std::uint8_t> evaluate_concatenate_bytes(
    const IndexParameters& p,
    const std::vector<std::vector<std::uint8_t>>& inputs, std::size_t width) {
  const auto size = detail::byte_count(p.output_shape, width);
  const auto rank = static_cast<std::int64_t>(p.output_shape.size());
  const auto axis = p.axis < 0 ? p.axis + rank : p.axis;
  if (p.operation != "concatenate" || inputs.empty() ||
      inputs.size() != p.input_shapes.size() || axis < 0 || axis >= rank)
    throw std::invalid_argument("CPU concatenate inputs/axis are invalid");
  std::int64_t remaining = p.output_shape[axis];
  for (std::size_t k = 0; k < inputs.size(); ++k) {
    const auto& shape = p.input_shapes[k];
    if (shape.size() != p.output_shape.size() ||
        detail::byte_count(shape, width) != inputs[k].size())
      throw std::invalid_argument("CPU concatenate input byte count differs");
    for (std::int64_t dim = 0; dim < rank; ++dim)
      if (dim != axis && shape[dim] != p.output_shape[dim])
        throw std::invalid_argument("CPU concatenate non-axis shape differs");
    if (shape[axis] > remaining)
      throw std::invalid_argument("CPU concatenate axis sum differs");
    remaining -= shape[axis];
  }
  if (remaining != 0)
    throw std::invalid_argument("CPU concatenate axis sum differs");
  std::vector<std::uint8_t> output(size);
  for (std::size_t index = 0; index < size / width; ++index) {
    const auto [tensor, source] = index_source(p, index);
    std::copy_n(inputs[tensor].begin() + source * width, width,
                output.begin() + index * width);
  }
  return output;
}
}  // namespace flagdnn::reference::cpu
