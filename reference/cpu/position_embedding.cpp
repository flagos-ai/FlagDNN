/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/position_embedding.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_rope(
    std::span<const std::int64_t> shape, std::int64_t rope_dim,
    float output_scale, bool backward,
    const std::vector<std::vector<float>>& inputs) {
  const auto d = static_cast<std::size_t>(shape[3]),
             sequence = static_cast<std::size_t>(shape[2]);
  const auto width = rope_dim ? static_cast<std::size_t>(rope_dim) : d;
  std::vector<float> output(inputs[0].size());
  for (std::size_t i = 0; i < output.size(); ++i) {
    const auto channel = i % d;
    if (channel < d - width) {
      output[i] = inputs[0][i] * output_scale;
      continue;
    }
    const auto relative = channel - (d - width),
               partner = relative < width / 2 ? relative + width / 2
                                              : relative - width / 2;
    const auto position = i / d % sequence;
    const double angle = inputs[1][position * width + relative],
                 paired_angle = inputs[1][position * width + partner];
    const double sign = (relative < width / 2) == backward ? 1.0 : -1.0;
    output[i] = static_cast<float>(
        (inputs[0][i] * std::cos(angle) +
         sign * inputs[0][i - channel + d - width + partner] *
             std::sin(backward ? paired_angle : angle)) *
        output_scale);
  }
  return {output};
}
}  // namespace flagdnn::reference::cpu
