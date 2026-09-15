/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_POSITION_EMBEDDING_HPP_
#define FLAGDNN_REFERENCE_CPU_POSITION_EMBEDDING_HPP_
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_rope(
    std::span<const std::int64_t> shape, std::int64_t rope_dim,
    float output_scale, bool backward,
    const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
