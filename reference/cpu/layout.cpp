/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/layout.hpp"
#include "reference/cpu/shape.hpp"

#include <algorithm>
#include <numeric>
namespace flagdnn::reference::cpu {
std::size_t layout_source_index(const LayoutParameters& test_case,
                                std::size_t output_index) {
  if (test_case.operation == "reshape") return output_index;
  std::vector<std::size_t> coordinates(test_case.input_dimensions.size());
  for (std::size_t i = test_case.output_dimensions.size(); i > 0; --i) {
    const auto axis = i - 1;
    const auto coordinate = output_index % test_case.output_dimensions[axis];
    output_index /= test_case.output_dimensions[axis];
    if (test_case.operation == "transpose")
      coordinates[static_cast<std::size_t>(test_case.permutation[axis])] =
          coordinate;
    else
      coordinates[axis] =
          static_cast<std::size_t>(test_case.slices[axis].first) +
          coordinate * test_case.slice_strides[axis];
  }
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < coordinates.size(); ++axis)
    result = result * test_case.input_dimensions[axis] + coordinates[axis];
  return result;
}

std::vector<float> evaluate_layout(const LayoutParameters& parameters,
                                   std::span<const float> input) {
  const auto count = std::accumulate(parameters.output_dimensions.begin(),
                                     parameters.output_dimensions.end(),
                                     std::size_t{1}, std::multiplies<>());
  std::vector<float> output(count);
  for (std::size_t index = 0; index < count; ++index)
    output[index] = input[layout_source_index(parameters, index)];
  return output;
}

std::vector<std::uint8_t> evaluate_layout_bytes(
    const LayoutParameters& p, std::span<const std::uint8_t> input,
    std::size_t width) {
  const auto input_size = detail::byte_count(p.input_dimensions, width);
  const auto output_size = detail::byte_count(p.output_dimensions, width);
  if (input.size() != input_size)
    throw std::invalid_argument("CPU layout input byte count differs");
  const auto rank = p.input_dimensions.size();
  if (p.operation == "reshape") {
    if (input_size != output_size)
      throw std::invalid_argument("CPU reshape element count differs");
  } else if (p.operation == "transpose") {
    if (p.output_dimensions.size() != rank || p.permutation.size() != rank)
      throw std::invalid_argument("CPU transpose rank differs");
    std::vector<bool> used(rank);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto source = p.permutation[axis];
      if (source < 0 || static_cast<std::size_t>(source) >= rank || used[source] ||
          p.output_dimensions[axis] != p.input_dimensions[source])
        throw std::invalid_argument("CPU transpose permutation/shape differs");
      used[source] = true;
    }
  } else if (p.operation == "slice") {
    if (p.output_dimensions.size() != rank || p.slices.size() != rank ||
        p.slice_strides.size() != rank)
      throw std::invalid_argument("CPU slice rank differs");
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto [start, end] = p.slices[axis];
      const auto step = p.slice_strides[axis];
      if (start < 0 || end <= start || end > p.input_dimensions[axis] ||
          step <= 0 || p.output_dimensions[axis] != 1 + (end - start - 1) / step)
        throw std::invalid_argument("CPU slice bounds/shape differ");
    }
  } else {
    throw std::invalid_argument("CPU layout operation is unsupported");
  }
  std::vector<std::uint8_t> output(output_size);
  for (std::size_t index = 0; index < output_size / width; ++index) {
    const auto source = layout_source_index(p, index);
    std::copy_n(input.begin() + source * width, width,
                output.begin() + index * width);
  }
  return output;
}
}  // namespace flagdnn::reference::cpu
