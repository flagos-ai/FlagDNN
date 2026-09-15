/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/resample.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_resample(
    const ResampleParameters& parameters,
    const std::vector<std::vector<float>>& inputs) {
  const auto& shape = parameters.input_shape;
  const auto& output_shape = parameters.output_shape;
  const auto spatial = shape.size() - 2;
  const auto count = std::accumulate(output_shape.begin(), output_shape.end(),
                                     std::size_t{1}, std::multiplies<>());
  std::vector<std::vector<float>> result(parameters.output_count,
                                         std::vector<float>(count));
  const auto flat = [](const std::vector<std::int64_t>& coordinates,
                       const std::vector<std::int64_t>& dimensions) {
    std::size_t result = 0;
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis)
      result = result * dimensions[axis] + coordinates[axis];
    return result;
  };
  for (std::size_t index = 0; index < count; ++index) {
    std::vector<std::int64_t> coordinates(output_shape.size());
    auto remaining = index;
    for (std::size_t axis = coordinates.size(); axis > 0; --axis) {
      coordinates[axis - 1] = remaining % output_shape[axis - 1];
      remaining /= output_shape[axis - 1];
    }
    auto source = coordinates;
    if (parameters.mode == 4) {
      for (std::size_t axis = 2; axis < shape.size(); ++axis)
        source[axis] = coordinates[axis] * shape[axis] / output_shape[axis];
      result[0][index] = inputs[0][flat(source, shape)];
      continue;
    }
    if (parameters.mode == 3) {
      std::array<double, 2> position;
      for (std::size_t axis = 0; axis < 2; ++axis) {
        const auto input_extent = shape[axis + 2],
                   output_extent = output_shape[axis + 2];
        const double value =
            parameters.align_corners
                ? (output_extent > 1
                       ? static_cast<double>(coordinates[axis + 2]) *
                             (input_extent - 1) / (output_extent - 1)
                       : 0.0)
                : (coordinates[axis + 2] + 0.5) * input_extent / output_extent -
                      0.5;
        position[axis] =
            std::clamp(value, 0.0, static_cast<double>(input_extent - 1));
      }
      double value = 0.0;
      for (int neighbor = 0; neighbor < 4; ++neighbor) {
        double weight = 1.0;
        for (std::size_t axis = 0; axis < 2; ++axis) {
          const auto lower =
              static_cast<std::int64_t>(std::floor(position[axis]));
          const auto high = (neighbor >> axis) & 1;
          source[axis + 2] = std::min(lower + high, shape[axis + 2] - 1);
          const double fraction = position[axis] - lower;
          weight *= high ? fraction : 1.0 - fraction;
        }
        value += weight * inputs[0][flat(source, shape)];
      }
      result[0][index] = static_cast<float>(value);
      continue;
    }
    const auto volume =
        std::accumulate(parameters.window.begin(), parameters.window.end(),
                        std::size_t{1}, std::multiplies<>());
    double value =
        parameters.mode == 5 ? -std::numeric_limits<double>::infinity() : 0.0;
    std::int64_t best = -1;
    std::size_t valid_count = 0;
    for (std::size_t window = 0; window < volume; ++window) {
      auto window_index = window;
      bool valid = true;
      for (std::size_t axis = spatial; axis > 0; --axis) {
        source[axis + 1] = coordinates[axis + 1] * parameters.stride[axis - 1] -
                           parameters.pre[axis - 1] +
                           window_index % parameters.window[axis - 1];
        window_index /= parameters.window[axis - 1];
        valid &= source[axis + 1] >= 0 && source[axis + 1] < shape[axis + 1];
      }
      if (valid) ++valid_count;
      double sample = parameters.padding == 2
                          ? -std::numeric_limits<double>::infinity()
                          : 0.0;
      if (parameters.padding == 1)
        for (std::size_t axis = 2; axis < shape.size(); ++axis)
          source[axis] =
              std::clamp<std::int64_t>(source[axis], 0, shape[axis] - 1);
      if (valid || parameters.padding == 1)
        sample = inputs[0][flat(source, shape)];
      if (parameters.mode == 5) {
        if (best == -1 || sample > value) {
          value = sample;
          best = static_cast<std::int64_t>(window);
        }
      } else
        value += sample;
    }
    if (parameters.mode == 1) value = valid_count ? value / valid_count : 0.0;
    if (parameters.mode == 2) value /= volume;
    result[0][index] = static_cast<float>(value);
    if (result.size() == 2) result[1][index] = static_cast<float>(best);
  }
  return result;
}
}  // namespace flagdnn::reference::cpu
