/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "reference/cpu/pointwise.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::reference::cpu {
namespace {

std::size_t checked_element_count(std::span<const std::int64_t> dimensions,
                                  std::string_view role) {
  if (dimensions.empty()) {
    throw std::invalid_argument(std::string(role) +
                                " dimensions must not be empty");
  }
  std::size_t count = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(std::string(role) +
                                  " dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (count > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error(std::string(role) +
                                " element count overflows");
    }
    count *= value;
  }
  return count;
}

void validate_input(std::span<const float> values,
                    std::span<const std::int64_t> dimensions,
                    std::span<const std::int64_t> output_dimensions,
                    std::string_view role) {
  if (values.size() != checked_element_count(dimensions, role)) {
    throw std::invalid_argument(std::string(role) +
                                " value count does not match dimensions");
  }
  if (dimensions.size() > output_dimensions.size()) {
    throw std::invalid_argument(std::string(role) +
                                " rank exceeds output rank");
  }
  const std::size_t leading = output_dimensions.size() - dimensions.size();
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    const std::int64_t input_dimension = dimensions[axis];
    const std::int64_t output_dimension = output_dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      throw std::invalid_argument(std::string(role) +
                                  " dimensions do not broadcast to output");
    }
  }
}

std::size_t broadcast_index(std::size_t output_index,
                            std::span<const std::int64_t> input_dimensions,
                            std::span<const std::int64_t> output_dimensions) {
  const std::size_t leading =
      output_dimensions.size() - input_dimensions.size();
  std::size_t input_index = 0;
  std::size_t input_stride = 1;
  for (std::size_t axis = output_dimensions.size(); axis != 0; --axis) {
    const std::size_t output_axis = axis - 1;
    const std::size_t output_dimension =
        static_cast<std::size_t>(output_dimensions[output_axis]);
    const std::size_t coordinate = output_index % output_dimension;
    output_index /= output_dimension;
    if (output_axis < leading) {
      continue;
    }
    const std::size_t input_axis = output_axis - leading;
    if (input_dimensions[input_axis] != 1) {
      input_index += coordinate * input_stride;
    }
    input_stride *= static_cast<std::size_t>(input_dimensions[input_axis]);
  }
  return input_index;
}

} // namespace

bool supports_binary_pointwise(flagdnnPointwiseMode_t mode) noexcept {
  switch (mode) {
  case FLAGDNN_POINTWISE_DIV:
  case FLAGDNN_POINTWISE_POW:
  case FLAGDNN_POINTWISE_MOD:
  case FLAGDNN_POINTWISE_CMP_EQ:
    return true;
  default:
    return false;
  }
}

std::vector<float> evaluate_binary_pointwise(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions) {
  if (!supports_binary_pointwise(mode)) {
    throw std::invalid_argument(
        "CPU reference does not support this pointwise mode");
  }
  const std::size_t output_count =
      checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");

  std::vector<float> output(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    const float left_value =
        left[broadcast_index(index, left_dimensions, output_dimensions)];
    const float right_value =
        right[broadcast_index(index, right_dimensions, output_dimensions)];
    switch (mode) {
    case FLAGDNN_POINTWISE_DIV:
      output[index] = left_value / right_value;
      break;
    case FLAGDNN_POINTWISE_POW:
      output[index] =
          static_cast<float>(std::pow(left_value, right_value));
      break;
    case FLAGDNN_POINTWISE_MOD:
      output[index] = std::fmod(left_value, right_value);
      break;
    case FLAGDNN_POINTWISE_CMP_EQ:
      output[index] = left_value == right_value ? 1.0F : 0.0F;
      break;
    default:
      throw std::logic_error("validated CPU pointwise mode became invalid");
    }
  }
  return output;
}

} // namespace flagdnn::reference::cpu
