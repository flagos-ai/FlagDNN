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
    std::span<const std::int64_t> left_dimensions,
    std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions);

} // namespace flagdnn::reference::cpu

#endif // FLAGDNN_REFERENCE_CPU_POINTWISE_HPP_
