/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_RESAMPLE_HPP_
#define FLAGDNN_REFERENCE_CPU_RESAMPLE_HPP_
#include <cstddef>
#include <cstdint>
#include <vector>
namespace flagdnn::reference::cpu {
struct ResampleParameters {
  std::vector<std::int64_t> input_shape, output_shape;
  std::vector<std::int64_t> window, stride, pre, post;
  std::size_t output_count = 1;
  int mode = 5, padding = 3;
  bool align_corners = false;
};
std::vector<std::vector<float>> evaluate_resample(
    const ResampleParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
