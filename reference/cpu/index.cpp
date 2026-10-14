/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/index.hpp"
#include "reference/cpu/shape.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace flagdnn::reference::cpu {
namespace {
struct Geometry {
  std::size_t axis;
  std::size_t output_count;
  std::size_t inner_count;
  std::vector<std::size_t> input_counts;
};

Geometry validate(const IndexParameters& parameters) {
  const auto count = detail::byte_count(parameters.output_shape);
  const auto rank = static_cast<std::int64_t>(parameters.output_shape.size());
  if (parameters.axis < -rank || parameters.axis >= rank)
    throw std::invalid_argument("index oracle axis is outside output rank");
  const auto axis = static_cast<std::size_t>(
      parameters.axis < 0 ? parameters.axis + rank : parameters.axis);
  Geometry geometry{axis, count, 1, {}};
  for (std::size_t dim = axis + 1; dim < parameters.output_shape.size(); ++dim)
    geometry.inner_count *= static_cast<std::size_t>(parameters.output_shape[dim]);
  if (parameters.operation == "gen_index") {
    if (!parameters.input_shapes.empty())
      throw std::invalid_argument("gen_index oracle does not accept inputs");
    return geometry;
  }
  if (parameters.operation != "concatenate")
    throw std::invalid_argument("unsupported index oracle operation");
  if (parameters.input_shapes.empty())
    throw std::invalid_argument("concatenate oracle requires inputs");
  std::size_t axis_sum = 0;
  for (const auto& shape : parameters.input_shapes) {
    if (shape.size() != parameters.output_shape.size())
      throw std::invalid_argument("concatenate oracle input rank differs");
    geometry.input_counts.push_back(detail::byte_count(shape));
    for (std::size_t dim = 0; dim < shape.size(); ++dim)
      if (dim != axis && shape[dim] != parameters.output_shape[dim])
        throw std::invalid_argument("concatenate oracle non-axis dimensions differ");
    const auto extent = static_cast<std::size_t>(shape[axis]);
    if (axis_sum > std::numeric_limits<std::size_t>::max() - extent)
      throw std::overflow_error("concatenate oracle axis sum overflows");
    axis_sum += extent;
  }
  if (axis_sum != static_cast<std::size_t>(parameters.output_shape[axis]))
    throw std::invalid_argument("concatenate oracle axis sum differs from output");
  return geometry;
}

std::pair<std::size_t, std::size_t> source(
    const IndexParameters& parameters, const Geometry& geometry,
    std::size_t output_index) {
  const auto extent = static_cast<std::size_t>(parameters.output_shape[geometry.axis]);
  const auto axis_index = output_index / geometry.inner_count;
  auto coordinate = axis_index % extent;
  if (parameters.operation == "gen_index") return {0, coordinate};
  const auto outer = axis_index / extent;
  const auto inner = output_index % geometry.inner_count;
  for (std::size_t input = 0; input < parameters.input_shapes.size(); ++input) {
    const auto input_extent = static_cast<std::size_t>(
        parameters.input_shapes[input][geometry.axis]);
    if (coordinate < input_extent)
      return {input, (outer * input_extent + coordinate) * geometry.inner_count + inner};
    coordinate -= input_extent;
  }
  throw std::logic_error("validated concatenate coordinate is outside inputs");
}

std::size_t storage_width(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32:
    case FLAGDNN_DATA_INT32: return 4;
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16: return 2;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
    case FLAGDNN_DATA_FP8_E8M0: return 1;
  }
  throw std::invalid_argument("unsupported index oracle data type");
}

std::size_t byte_count(std::size_t count, std::size_t width) {
  if (count > std::numeric_limits<std::size_t>::max() / width)
    throw std::overflow_error("index oracle byte count overflows");
  return count * width;
}
}  // namespace

std::pair<std::size_t, std::size_t> index_source(
    const IndexParameters& parameters, std::size_t output_index) {
  const auto geometry = validate(parameters);
  if (output_index >= geometry.output_count)
    throw std::out_of_range("index oracle output index is outside shape");
  return source(parameters, geometry, output_index);
}

std::vector<std::byte> evaluate_index_bytes(
    const IndexParameters& parameters,
    std::span<const std::vector<std::byte>> inputs,
    flagdnnDataType_t data_type) {
  const auto geometry = validate(parameters);
  const auto width = storage_width(data_type);
  if (inputs.size() != geometry.input_counts.size())
    throw std::invalid_argument("index oracle input buffer count differs");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].size() != byte_count(geometry.input_counts[i], width))
      throw std::invalid_argument("index oracle input byte count differs from shape");
  if (parameters.operation == "gen_index") {
    if (data_type != FLAGDNN_DATA_FLOAT32 && data_type != FLAGDNN_DATA_INT32)
      throw std::invalid_argument("gen_index oracle requires FP32 or INT32 output");
    if (data_type == FLAGDNN_DATA_INT32 &&
        parameters.output_shape[geometry.axis] - 1 >
            std::numeric_limits<std::int32_t>::max())
      throw std::overflow_error("gen_index oracle value exceeds INT32");
  }
  std::vector<std::byte> output(byte_count(geometry.output_count, width));
  for (std::size_t i = 0; i < geometry.output_count; ++i) {
    const auto [input, index] = source(parameters, geometry, i);
    auto* destination = output.data() + i * width;
    if (parameters.operation == "concatenate") {
      std::memcpy(destination, inputs[input].data() + index * width, width);
    } else if (data_type == FLAGDNN_DATA_INT32) {
      const auto value = static_cast<std::int32_t>(index);
      std::memcpy(destination, &value, sizeof(value));
    } else {
      const auto value = static_cast<float>(index);
      std::memcpy(destination, &value, sizeof(value));
    }
  }
  return output;
}

std::vector<std::uint8_t> evaluate_concatenate_bytes(
    const IndexParameters& parameters,
    const std::vector<std::vector<std::uint8_t>>& inputs, std::size_t width) {
  const auto geometry = validate(parameters);
  if (parameters.operation != "concatenate" ||
      inputs.size() != geometry.input_counts.size())
    throw std::invalid_argument("CPU concatenate inputs/operation are invalid");
  const auto size = detail::byte_count(parameters.output_shape, width);
  for (std::size_t input = 0; input < inputs.size(); ++input)
    if (inputs[input].size() != byte_count(geometry.input_counts[input], width))
      throw std::invalid_argument("CPU concatenate input byte count differs");
  std::vector<std::uint8_t> output(size);
  for (std::size_t index = 0; index < geometry.output_count; ++index) {
    const auto [tensor, source_index] = source(parameters, geometry, index);
    std::copy_n(inputs[tensor].begin() + source_index * width, width,
                output.begin() + index * width);
  }
  return output;
}

}  // namespace flagdnn::reference::cpu
