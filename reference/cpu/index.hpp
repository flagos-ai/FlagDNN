/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_INDEX_HPP_
#define FLAGDNN_REFERENCE_CPU_INDEX_HPP_
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
namespace flagdnn::reference::cpu {
struct IndexParameters {
  std::string operation;
  std::vector<std::vector<std::int64_t>> input_shapes;
  std::vector<std::int64_t> output_shape;
  std::int64_t axis = 0;
};
// Returns (source tensor, source logical index); gen_index returns its value
// in the second component.
std::pair<std::size_t, std::size_t> index_source(
    const IndexParameters& parameters, std::size_t output_index);
}  // namespace flagdnn::reference::cpu
#endif
