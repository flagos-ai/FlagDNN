/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_ATTENTION_HPP_
#define FLAGDNN_REFERENCE_CPU_ATTENTION_HPP_
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
namespace flagdnn::reference::cpu {
struct AttentionTensor {
  std::vector<std::int64_t> dimensions;
};
enum class AttentionDiagonalAlignment { kTopLeft, kBottomRight };
struct AttentionOptions {
  std::optional<float> attention_scale;
  std::optional<std::int64_t> diagonal_band_left_bound;
  std::optional<std::int64_t> diagonal_band_right_bound;
  AttentionDiagonalAlignment diagonal_alignment =
      AttentionDiagonalAlignment::kTopLeft;
};
struct AttentionParameters {
  AttentionTensor q, k, v, output, doutput;
  std::optional<AttentionTensor> bias, dbias;
  AttentionOptions options;
};
struct AttentionForwardResult {
  std::vector<float> output, stats;
};
struct AttentionBackwardResult {
  std::vector<float> dq, dk, dv, dbias;
};
AttentionForwardResult evaluate_attention_forward(
    const AttentionParameters& parameters, std::span<const float> q,
    std::span<const float> k, std::span<const float> v,
    std::span<const float> bias);
AttentionBackwardResult evaluate_attention_backward(
    const AttentionParameters& parameters, std::span<const float> q,
    std::span<const float> k, std::span<const float> v,
    std::span<const float> doutput, std::span<const float> bias,
    const AttentionForwardResult& primal);
}  // namespace flagdnn::reference::cpu
#endif
