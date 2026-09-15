/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "graph/lowering/lowering.hpp"
#include "graph/lowering/helpers.hpp"
#include <cmath>
namespace flagdnn::native {
LoweredOperation lower_rope(const OperationSpec& operation) {
  const bool backward = operation.custom_operation_name == "rope_backward";
  require_port_count(operation, 2, 1);
  const auto& x =
      require_port(operation.inputs, backward ? "dy" : "input", "input");
  const auto& freqs = require_port(operation.inputs, "freqs", "input");
  const auto& y =
      require_port(operation.outputs, backward ? "dx" : "output", "output");
  for (const auto* tensor : {&x, &freqs, &y})
    require_non_overlapping_tensor(*tensor, "RoPE tensor");
  require_floating_data_type(x, "RoPE input must be floating");
  require_same_data_type(x, y, "RoPE output type must match input");
  if (x.dimensions.size() != 4 || y.dimensions != x.dimensions)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RoPE requires BHSD input and matching output");
  auto width = integer_attribute(operation, "rope_dim");
  if (width == 0) width = x.dimensions[3];
  if (width < 2 || width % 2 || width > x.dimensions[3])
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "RoPE rotation width must be even, positive and no larger than D");
  if (freqs.data_type != FLAGDNN_DATA_FLOAT32 ||
      freqs.dimensions !=
          std::vector<std::int64_t>{x.dimensions[2], 1, 1, width})
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RoPE frequencies must be FP32 [S,1,1,rope_dim]");
  const auto scale = real_attribute(operation, "output_scale");
  if (!std::isfinite(scale))
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "RoPE output scale must be finite");
  return {{{"rope_dim", width}, {"n_elements", x.element_count()}},
          {{"output_scale", scale}},
          {}};
}
}  // namespace flagdnn::native
