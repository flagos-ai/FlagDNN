/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/fp8_matmul.hpp"

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
std::int64_t batch_index(std::int64_t batch, const Shape& out,
                         const Shape& input) {
  std::int64_t result = 0, stride = 1;
  for (std::size_t trailing = 2; trailing < out.size(); ++trailing) {
    const auto coord = batch % out[out.size() - 1 - trailing];
    batch /= out[out.size() - 1 - trailing];
    if (trailing < input.size()) {
      const auto extent = input[input.size() - 1 - trailing];
      if (extent != 1) result += coord * stride;
      stride *= extent;
    }
  }
  return result;
}
}  // namespace
std::vector<std::vector<float>> evaluate_fp8_matmul(
    const Fp8MatmulParameters& parameters,
    const std::vector<std::vector<float>>& inputs) {
  const auto& a = parameters.input_shapes[0];
  const auto& b = parameters.input_shapes[1];
  const auto& c = parameters.output_shape;
  const auto m = a[a.size() - 2], k = a.back(), n = b.back();
  const auto blocks = (k + 31) / 32;
  std::vector<float> output(elements(c));
  for (std::int64_t batch = 0; batch < elements(c) / (m * n); ++batch) {
    const auto ba = batch_index(batch, c, a), bb = batch_index(batch, c, b);
    for (std::int64_t row = 0; row < m; ++row) {
      for (std::int64_t col = 0; col < n; ++col) {
        double value = 0.0;
        for (std::int64_t inner = 0; inner < k; ++inner) {
          double av = inputs[0][(ba * m + row) * k + inner],
                 bv = inputs[1][(bb * k + inner) * n + col];
          if (parameters.scale_mode == 2) {
            av *= inputs[2][(ba * m + row) * blocks + inner / 32];
            bv *= inputs[3][(bb * blocks + inner / 32) * n + col];
          }
          value += av * bv;
        }
        if (parameters.scale_mode == 1)
          value *= static_cast<double>(inputs[2][0]) * inputs[3][0];
        output[(batch * m + row) * n + col] = static_cast<float>(value);
      }
    }
  }
  return {output};
}
}  // namespace flagdnn::reference::cpu
