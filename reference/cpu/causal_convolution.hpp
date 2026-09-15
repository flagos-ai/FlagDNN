/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_CAUSAL_CONVOLUTION_HPP_
#define FLAGDNN_REFERENCE_CPU_CAUSAL_CONVOLUTION_HPP_
#include <cstdint>
#include <vector>
namespace flagdnn::reference::cpu {
struct CausalConvolutionParameters {
  std::vector<std::int64_t> dimensions;
  std::int64_t kernel_size = 1, dilation = 1;
  int precision = 0;
  bool silu = false;
};
std::vector<std::vector<float>> evaluate_causal_convolution(
    const CausalConvolutionParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
