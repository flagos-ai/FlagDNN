/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_LAYOUT_HPP_
#define FLAGDNN_REFERENCE_CPU_LAYOUT_HPP_
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>
namespace flagdnn::reference::cpu {
struct LayoutParameters {
  std::string operation;
  std::vector<std::int64_t> input_dimensions, output_dimensions, permutation;
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::vector<std::int64_t> slice_strides;
};
std::size_t layout_source_index(const LayoutParameters& parameters,
                                std::size_t output_index);
std::vector<float> evaluate_layout(const LayoutParameters& parameters,
                                   std::span<const float> input);
}  // namespace flagdnn::reference::cpu
#endif
