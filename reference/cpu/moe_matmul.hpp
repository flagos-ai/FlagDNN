/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_MOE_MATMUL_HPP_
#define FLAGDNN_REFERENCE_CPU_MOE_MATMUL_HPP_
#include <cstddef>
#include <cstdint>
#include <vector>
namespace flagdnn::reference::cpu {
struct MoeMatmulParameters {
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::int64_t> output_shape;
  std::vector<std::int32_t> offsets, token_index, token_ks;
  int mode = 0, top_k = 1;
  bool backward = false;
};
std::vector<std::vector<float>> evaluate_moe_matmul(
    const MoeMatmulParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
