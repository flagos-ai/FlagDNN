/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_NORMALIZATION_HPP_
#define FLAGDNN_REFERENCE_CPU_NORMALIZATION_HPP_
#include <cstddef>
#include <cstdint>
#include <string>
#include <span>
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
// Logical row-major inputs; results are output, mean, inverse standard deviation.
std::vector<std::vector<float>> evaluate_row_normalization(
    bool rms, std::span<const float> input, std::span<const float> scale,
    std::span<const float> bias, double epsilon);
std::vector<float> evaluate_batchnorm_inference(
    std::span<const float> input, std::size_t spatial,
    std::span<const float> mean, std::span<const float> inverse_std,
    std::span<const float> scale, std::span<const float> bias);
// output, mean, inverse standard deviation, running mean, running variance.
// Saved statistics are FP32; running variance uses the unbiased estimate.
std::vector<std::vector<float>> evaluate_batchnorm_training(
    std::span<const float> input, std::size_t spatial,
    std::span<const float> scale, std::span<const float> bias,
    std::span<const float> previous_mean, std::span<const float> previous_variance,
    double epsilon, double momentum);
void variance_to_inverse_standard_deviation(std::span<float> variance,
                                            double epsilon);
}  // namespace flagdnn::reference::cpu
#endif
