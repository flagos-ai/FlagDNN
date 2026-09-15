/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/layout.hpp"

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
}  // namespace flagdnn::reference::cpu
