/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_STATISTICS_HPP_
#define FLAGDNN_REFERENCE_CPU_STATISTICS_HPP_
#include <cstddef>
#include <vector>
namespace flagdnn::reference::cpu {
std::vector<std::vector<float>> evaluate_bn_finalize(
    std::size_t output_count, double epsilon, double accum_count,
    double momentum, const std::vector<std::vector<float>>& inputs);
}  // namespace flagdnn::reference::cpu
#endif
