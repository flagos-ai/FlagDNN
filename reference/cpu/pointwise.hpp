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

// Integer values stay exact; arithmetic follows pointwise_integer_reference.
// Comparisons return 0 or 1, which the caller can encode as Boolean output.
[[nodiscard]] std::vector<std::int32_t> evaluate_binary_pointwise_int32(
    flagdnnPointwiseMode_t mode, std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, std::int32_t alpha = 1);

// Activation gradients use logical, equally sized DY and X values, after
// input-dtype quantization. The caller quantizes the returned DX to its dtype.
[[nodiscard]] bool supports_activation_backward(
    flagdnnPointwiseMode_t mode) noexcept;

[[nodiscard]] std::vector<float> evaluate_activation_backward(
    flagdnnPointwiseMode_t mode, std::span<const float> dy,
    std::span<const float> x, const flagdnnPointwiseAttributes_t& attributes);

// Select preserves exact INT32 values and independently broadcasts all three
// logical input tensors. Any nonzero Boolean predicate selects the left input.
[[nodiscard]] std::vector<std::int32_t> evaluate_binary_select_int32(
    std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::uint8_t> predicate,
    std::span<const std::int64_t> predicate_dimensions,
    std::span<const std::int64_t> output_dimensions);

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

[[nodiscard]] std::vector<std::int32_t> evaluate_add_square_int32(
    std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions);

[[nodiscard]] std::int32_t pointwise_integer_reference(
    flagdnnPointwiseMode_t mode, std::int32_t left, std::int32_t right,
    bool predicate, std::int32_t alpha);

}  // namespace flagdnn::reference::cpu

#endif  // FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
