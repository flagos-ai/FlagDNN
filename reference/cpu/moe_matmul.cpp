/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/moe_matmul.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
namespace flagdnn::reference::cpu {
namespace {
using Shape = std::vector<std::int64_t>;
std::int64_t elements(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                         std::multiplies<>());
}
}  // namespace
std::vector<std::vector<float>> evaluate_moe_matmul(
    const MoeMatmulParameters& parameters,
    const std::vector<std::vector<float>>& inputs) {
  const auto& td = parameters.input_shapes[parameters.backward ? 1 : 0];
  const auto& md = parameters.input_shapes[parameters.backward ? 0 : 1];
  const auto k = td[2], n = md[2],
             routed = static_cast<std::int64_t>(parameters.token_index.size());
  std::vector<double> output(elements(parameters.output_shape));
  for (std::size_t expert = 0; expert < parameters.offsets.size(); ++expert) {
    const auto begin = parameters.offsets[expert];
    const auto end = expert + 1 < parameters.offsets.size()
                         ? parameters.offsets[expert + 1]
                         : routed;
    for (auto slot = begin; slot < end; ++slot) {
      if (parameters.backward) {
        for (std::int64_t row = 0; row < k; ++row)
          for (std::int64_t col = 0; col < n; ++col)
            output[(expert * k + row) * n + col] +=
                static_cast<double>(inputs[1][slot * k + row]) *
                inputs[0][slot * n + col];
      } else {
        const auto source =
            parameters.mode == 1 ? parameters.token_index[slot] : slot;
        const auto target =
            parameters.mode == 2
                ? parameters.token_index[slot] * parameters.top_k +
                      parameters.token_ks[slot]
                : slot;
        for (std::int64_t col = 0; col < n; ++col)
          for (std::int64_t inner = 0; inner < k; ++inner)
            output[target * n + col] +=
                static_cast<double>(inputs[0][source * k + inner]) *
                inputs[1][(expert * k + inner) * n + col];
      }
    }
  }
  return {std::vector<float>(output.begin(), output.end())};
}
}  // namespace flagdnn::reference::cpu
