/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_NORMALIZATION_HPP_
#define FLAGDNN_REFERENCE_CPU_NORMALIZATION_HPP_
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace flagdnn::reference::cpu {
struct NormalizationParameters {
  std::string operation;
  std::vector<std::int64_t> dimensions, parameter_dimensions;
  std::vector<std::size_t> axes;
  double epsilon = 1.0e-5;
};
void populate_normalization_statistics(
    const NormalizationParameters& parameters,
    std::vector<std::vector<float>>& inputs);
std::vector<std::vector<float>> evaluate_normalization(
    const NormalizationParameters& parameters,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
