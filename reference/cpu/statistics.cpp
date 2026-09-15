/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/statistics.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_statistics(
    std::string_view operation, std::span<const std::int64_t> input_shape,
    std::size_t output_count, double epsilon, double accum_count,
    double momentum, const std::vector<std::vector<float>>& inputs) {
  if (operation == "bn_finalize") {
    std::vector<std::vector<float>> result(
        output_count, std::vector<float>(inputs[0].size()));
    for (std::size_t index = 0; index < inputs[0].size(); ++index) {
      const double mean = static_cast<double>(inputs[0][index]) / accum_count;
      const double variance = std::max(
          static_cast<double>(inputs[1][index]) / accum_count - mean * mean,
          0.0);
      const double inverse = 1.0 / std::sqrt(variance + epsilon);
      result[0][index] = static_cast<float>(inputs[2][index] * inverse);
      result[1][index] = static_cast<float>(inputs[3][index] -
                                            mean * inputs[2][index] * inverse);
      result[2][index] = mean;
      result[3][index] = static_cast<float>(inverse);
      if (result.size() == 6) {
        const double unbiased =
            accum_count > 1 ? variance * accum_count / (accum_count - 1) : 0.0;
        result[4][index] = static_cast<float>(
            (1 - momentum) * inputs[4][index] + momentum * mean);
        result[5][index] = static_cast<float>(
            (1 - momentum) * inputs[5][index] + momentum * unbiased);
      }
    }
    return result;
  }
  const auto& shape = input_shape;
  const auto channels = static_cast<std::size_t>(shape[1]);
  const auto spatial = std::accumulate(shape.begin() + 2, shape.end(),
                                       std::size_t{1}, std::multiplies<>());
  std::vector<double> sum(channels), square(channels);
  for (std::size_t index = 0; index < inputs.at(0).size(); ++index) {
    const auto channel = (index / spatial) % channels;
    const double x = inputs[0][index];
    sum[channel] += x;
    square[channel] += x * x;
  }
  return {std::vector<float>(sum.begin(), sum.end()),
          std::vector<float>(square.begin(), square.end())};
}
}  // namespace flagdnn::reference::cpu
