/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/causal_convolution.hpp"

#include <bit>
#include <cmath>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_causal_convolution(
    const CausalConvolutionParameters& test_case,
    const std::vector<std::vector<float>>& inputs) {
  const auto& shape = test_case.dimensions;
  const auto channels = shape[1], length = shape[2],
             width = test_case.kernel_size;
  std::vector<float> output(inputs[0].size());
  const auto tf32 = [](float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return std::bit_cast<float>((bits + 0xFFFU + ((bits >> 13) & 1U)) &
                                0xFFFFE000U);
  };
  for (std::int64_t batch = 0; batch < shape[0]; ++batch) {
    for (std::int64_t channel = 0; channel < channels; ++channel) {
      for (std::int64_t position = 0; position < length; ++position) {
        double value = 0.0;
        for (std::int64_t tap = 0; tap < width; ++tap) {
          const auto source = position - (width - 1 - tap) * test_case.dilation;
          if (source < 0) continue;
          float x = inputs[0][(batch * channels + channel) * length + source],
                w = inputs[1][channel * width + tap];
          if (test_case.precision == 2) {
            x = tf32(x);
            w = tf32(w);
          }
          value += static_cast<double>(x) * w;
        }
        if (inputs.size() == 3) value += inputs[2][channel];
        if (test_case.silu) value /= 1.0 + std::exp(-value);
        output[(batch * channels + channel) * length + position] =
            static_cast<float>(value);
      }
    }
  }
  return {output};
}
}  // namespace flagdnn::reference::cpu
