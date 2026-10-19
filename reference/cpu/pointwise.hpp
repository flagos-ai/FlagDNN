/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
#define FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_

#include <flagdnn/flagdnn.h>

#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace flagdnn::reference::cpu {

// Storage-preserving identity. No numeric conversion, including FP8 NaNs.
[[nodiscard]] std::vector<std::uint8_t> evaluate_identity_bytes(
    std::span<const std::uint8_t> input);

[[nodiscard]] float gelu_exact(float value) noexcept;
[[nodiscard]] float gelu_approximate_tanh(float value) noexcept;
[[nodiscard]] float gelu_accurate_logit(float value) noexcept;
[[nodiscard]] float sigmoid(float logit) noexcept;
[[nodiscard]] float sigmoid_backward(float gradient, float logit) noexcept;

[[nodiscard]] bool supports_binary_pointwise(
    flagdnnPointwiseMode_t mode) noexcept;

// Inputs are logical, contiguous values after input-dtype quantization. The
// caller owns physical-layout gather/scatter and output-dtype quantization.
// Broadcasting is right-aligned against output_dimensions.
[[nodiscard]] std::vector<float> evaluate_binary_pointwise(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions, std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions);

// Keep the original entry point above unchanged for existing callers.
// Alpha scales the right operand of ADD/SUB only.
[[nodiscard]] std::vector<float> evaluate_binary_pointwise_with_alpha(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions, std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, double alpha);

// Activation gradients use logical, equally sized DY and X values, after
// input-dtype quantization. The caller quantizes the returned DX to its dtype.
[[nodiscard]] bool supports_activation_backward(
    flagdnnPointwiseMode_t mode) noexcept;

[[nodiscard]] std::vector<float> evaluate_activation_backward(
    flagdnnPointwiseMode_t mode, std::span<const float> dy,
    std::span<const float> x, const flagdnnPointwiseAttributes_t& attributes);

// AddSquare computes left + quantize_virtual(right * right). The callback
// models the graph's virtual storage dtype; output quantization remains with
// the caller, as for the other logical floating pointwise evaluators.
using PointwiseQuantizer =
    std::function<std::vector<float>(std::span<const float>)>;
[[nodiscard]] std::vector<float> evaluate_add_square(
    std::span<const float> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions,
    const PointwiseQuantizer& quantize_virtual);

}  // namespace flagdnn::reference::cpu

#endif  // FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
