/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
#define FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_

#include <flagdnn/flagdnn.h>

#include <cstdint>
#include <span>
#include <vector>

namespace flagdnn::reference::cpu {

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
// CMP_EQ returns 0 or 1, which the caller can encode as Boolean output.
[[nodiscard]] std::vector<std::int32_t> evaluate_binary_pointwise_int32(
    flagdnnPointwiseMode_t mode, std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, std::int32_t alpha = 1);

[[nodiscard]] std::int32_t pointwise_integer_reference(
    flagdnnPointwiseMode_t mode, std::int32_t left, std::int32_t right,
    bool predicate, std::int32_t alpha);

}  // namespace flagdnn::reference::cpu

#endif  // FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
