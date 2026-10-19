/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/statistics.hpp"

#include <algorithm>
#include <cmath>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_bn_finalize(
    std::size_t output_count, double epsilon, double accum_count,
    double momentum, const std::vector<std::vector<float>>& inputs) {

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
}  // namespace flagdnn::reference::cpu
