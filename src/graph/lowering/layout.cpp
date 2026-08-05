/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cstdint>
#include <vector>

namespace flagdnn::native {

LoweredOperation lower_reshape(const OperationSpec& operation) {
  require_port_count(operation, 1, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "reshape input");
  require_non_overlapping_tensor(output, "reshape output");
  require_same_data_type(
      input, output, "reshape input/output data types must match");
  if (input.dimensions.size() > 8 || output.dimensions.size() > 8) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "reshape tensor ranks must be in [0, 8]");
  }
  if (input.element_count() != output.element_count()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "reshape input/output element counts must match");
  }
  const std::int64_t mode = integer_attribute(operation, "reshape_mode");
  if (mode != 1 && mode != 2) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "reshape mode must be VIEW_ONLY or LOGICAL");
  }
  if (mode == 1) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "reshape VIEW_ONLY requires graph tensor aliasing; use LOGICAL "
        "for a materialized output");
  }
  return {{{"n_elements", output.element_count()},
           {"input_rank",
            static_cast<std::int64_t>(input.dimensions.size())},
           {"output_rank",
            static_cast<std::int64_t>(output.dimensions.size())},
           {"reshape_mode", mode}},
          {},
          {{"input_dimensions", input.dimensions},
           {"input_strides", input.strides},
           {"output_dimensions", output.dimensions},
           {"output_strides", output.strides}}};
}

LoweredOperation lower_transpose(const OperationSpec& operation) {
  require_port_count(operation, 1, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "transpose input");
  require_non_overlapping_tensor(output, "transpose output");
  require_same_data_type(
      input, output, "transpose input/output data types must match");
  const std::size_t rank = input.dimensions.size();
  if (rank == 0 || rank > 8 || output.dimensions.size() != rank) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "transpose tensor ranks must match and be in [1, 8]");
  }
  const std::vector<std::int64_t>& permutation =
      integer_array_attribute(operation, "permutation");
  if (permutation.size() != rank) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "transpose permutation rank does not match input");
  }
  std::vector<bool> seen(rank, false);
  for (std::size_t axis = 0; axis < rank; ++axis) {
    const std::int64_t source_axis = permutation[axis];
    if (source_axis < 0 ||
        source_axis >= static_cast<std::int64_t>(rank) ||
        seen[static_cast<std::size_t>(source_axis)]) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "transpose permutation must contain each axis once");
    }
    seen[static_cast<std::size_t>(source_axis)] = true;
    if (output.dimensions[axis] !=
        input.dimensions[static_cast<std::size_t>(source_axis)]) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "transpose output shape does not match permutation");
    }
  }
  return {{{"n_elements", output.element_count()},
           {"rank", static_cast<std::int64_t>(rank)}},
          {},
          {{"permutation", permutation},
           {"input_dimensions", input.dimensions},
           {"input_strides", input.strides},
           {"output_dimensions", output.dimensions},
           {"output_strides", output.strides}}};
}

LoweredOperation lower_slice(const OperationSpec& operation) {
  require_port_count(operation, 1, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "slice input");
  require_non_overlapping_tensor(output, "slice output");
  require_same_data_type(
      input, output, "slice input/output data types must match");
  const std::size_t rank = input.dimensions.size();
  if (rank == 0 || rank > 8 || output.dimensions.size() != rank) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "slice tensor ranks must match and be in [1, 8]");
  }
  const std::vector<std::int64_t>& starts =
      integer_array_attribute(operation, "starts");
  const std::vector<std::int64_t>& limits =
      integer_array_attribute(operation, "limits");
  const std::vector<std::int64_t>& slice_strides =
      integer_array_attribute(operation, "slice_strides");
  if (starts.size() != rank || limits.size() != rank ||
      slice_strides.size() != rank) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "slice attributes must match input rank");
  }
  for (std::size_t axis = 0; axis < rank; ++axis) {
    if (starts[axis] < 0 || limits[axis] <= starts[axis] ||
        limits[axis] > input.dimensions[axis] ||
        slice_strides[axis] <= 0) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "slice range or stride is invalid");
    }
    const std::int64_t expected =
        (limits[axis] - starts[axis] + slice_strides[axis] - 1) /
        slice_strides[axis];
    if (output.dimensions[axis] != expected) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "slice output shape does not match slice attributes");
    }
  }
  return {{{"n_elements", output.element_count()},
           {"rank", static_cast<std::int64_t>(rank)}},
          {},
          {{"starts", starts},
           {"limits", limits},
           {"slice_strides", slice_strides},
           {"input_dimensions", input.dimensions},
           {"input_strides", input.strides},
           {"output_dimensions", output.dimensions},
           {"output_strides", output.strides}}};
}

}  // namespace flagdnn::native
