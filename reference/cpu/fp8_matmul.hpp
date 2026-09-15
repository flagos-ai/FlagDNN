/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_FP8_MATMUL_HPP_
#define FLAGDNN_REFERENCE_CPU_FP8_MATMUL_HPP_
#include <cstddef>
#include <cstdint>
#include <vector>
namespace flagdnn::reference::cpu {
struct Fp8MatmulParameters {
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::int64_t> output_shape;
  int scale_mode = 0;
};
std::vector<std::vector<float>> evaluate_fp8_matmul(
    const Fp8MatmulParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
