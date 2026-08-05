/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cstdint>
#include <vector>

namespace flagdnn::native {

flagdnnReductionMode_t reduction_mode(const OperationSpec& operation) {
  return static_cast<flagdnnReductionMode_t>(
      integer_attribute(operation, "mode"));
}

LoweredOperation lower_reduction(const OperationSpec& operation) {
  require_port_count(operation, 1, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "input");
  require_non_overlapping_tensor(output, "output");
  require_same_data_type(
      input, output, "Reduction input/output data types must match");

  const std::int64_t rank =
      static_cast<std::int64_t>(input.dimensions.size());
  if (rank == 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "Reduction input must have positive rank");
  }
  std::int64_t axis = integer_attribute(operation, "axis");
  const bool keep_dimensions =
      boolean_attribute(operation, "keep_dimensions");
  if (axis < -rank || axis >= rank) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "Reduction axis is out of range");
  }
  if (axis < 0) {
    axis += rank;
  }

  std::vector<std::int64_t> expected = input.dimensions;
  if (keep_dimensions) {
    expected[static_cast<std::size_t>(axis)] = 1;
  } else {
    expected.erase(expected.begin() + axis);
  }
  if (output.dimensions != expected) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "Reduction output shape is incorrect");
  }

  const std::int64_t extent =
      input.dimensions[static_cast<std::size_t>(axis)];
  std::int64_t outer = 1;
  for (std::int64_t index = 0; index < axis; ++index) {
    outer = checked_multiply(
        outer,
        input.dimensions[static_cast<std::size_t>(index)],
        "Reduction outer extent overflows");
  }
  std::int64_t inner = 1;
  for (std::int64_t index = axis + 1; index < rank; ++index) {
    inner = checked_multiply(
        inner,
        input.dimensions[static_cast<std::size_t>(index)],
        "Reduction inner extent overflows");
  }
  const std::int64_t output_elements = checked_multiply(
      outer, inner, "Reduction output extent overflows");
  return {{{"outer", outer},
           {"reduction", extent},
           {"inner", inner},
           {"output_elements", output_elements},
           {"axis", axis},
           {"keep_dimensions", keep_dimensions ? 1 : 0}},
          {},
          {}};
}

}  // namespace flagdnn::native
