/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_INDEX_HPP_
#define FLAGDNN_REFERENCE_CPU_INDEX_HPP_
#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <span>
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
std::vector<std::uint8_t> evaluate_concatenate_bytes(
    const IndexParameters& parameters,
    const std::vector<std::vector<std::uint8_t>>& inputs,
    std::size_t element_bytes = 1);

// Evaluate the complete logical output. Inputs contain packed logical values;
// device bindings, offsets, strides and padding are owned by the caller.
// Concatenate copies representations exactly, including INT32 and FP8 bits.
// GenIndex supports FP32/INT32 and never converts integer values through float.
std::vector<std::byte> evaluate_index_bytes(
    const IndexParameters& parameters,
    std::span<const std::vector<std::byte>> inputs,
    flagdnnDataType_t data_type);
}  // namespace flagdnn::reference::cpu
#endif
