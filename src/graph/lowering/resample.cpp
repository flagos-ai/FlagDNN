/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "graph/lowering/lowering.hpp"
#include "graph/lowering/helpers.hpp"
#include <limits>
namespace flagdnn::native {
LoweredOperation lower_resample(const OperationSpec& operation) {
  const auto index = integer_attribute(operation, "generate_index"),
             mode = integer_attribute(operation, "mode");
  const auto padding = integer_attribute(operation, "padding"),
             align = integer_attribute(operation, "align_corners");
  if ((index != 0 && index != 1) || (align != 0 && align != 1) || mode < 1 ||
      mode > 5 || padding < 1 || padding > 3)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "resample mode flags are invalid");
  if ((index && mode != 5) || (align && mode != 3))
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "resample index requires maxpool and align_corners requires bilinear");
  require_port_count(operation, 1, index ? 2 : 1);
  const auto& x = require_port(operation.inputs, "input", "input");
  const auto& y = require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(x, "resample input");
  require_non_overlapping_tensor(y, "resample output");
  require_floating_data_type(x, "resample input must be floating");
  require_same_data_type(x, y, "resample input/output types must match");
  if (x.dimensions.size() < 3 || x.dimensions.size() > 5 ||
      y.dimensions.size() != x.dimensions.size())
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "resample requires 1..3 spatial dimensions");
  const auto spatial = x.dimensions.size() - 2;
  const auto& window = integer_array_attribute(operation, "window");
  const auto& stride = integer_array_attribute(operation, "stride");
  const auto& pre = integer_array_attribute(operation, "pre_padding");
  const auto& post = integer_array_attribute(operation, "post_padding");
  if (window.size() != spatial || stride.size() != spatial ||
      pre.size() != spatial || post.size() != spatial ||
      y.dimensions[0] != x.dimensions[0] || y.dimensions[1] != x.dimensions[1])
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "resample spatial metadata is inconsistent");
  const bool resize = mode == 3 || mode == 4;
  if ((mode == 3 && spatial != 2) || (resize && padding != 1) ||
      (mode == 1 && padding != 3) || (mode == 2 && padding == 2))
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "resample interpolation/padding combination is invalid");
  std::int64_t volume = 1;
  for (std::size_t axis = 0; axis < spatial; ++axis) {
    if (window[axis] <= 0 || stride[axis] <= 0 || pre[axis] < 0 ||
        post[axis] < 0)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "resample window, stride or padding is invalid");
    volume = checked_multiply(volume, window[axis],
                              "resample window volume overflows");
    if (!resize) {
      const auto dim = x.dimensions[axis + 2],
                 max = std::numeric_limits<std::int64_t>::max();
      if (pre[axis] > max - dim || post[axis] > max - dim - pre[axis] ||
          dim + pre[axis] + post[axis] < window[axis] ||
          y.dimensions[axis + 2] !=
              (dim + pre[axis] + post[axis] - window[axis]) / stride[axis] + 1)
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "resample output dimensions are invalid");
    } else if (window[axis] != 1 || stride[axis] != 1 || pre[axis] != 0 ||
               post[axis] != 0)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "resize cannot use pooling windows or padding");
  }
  if (index) {
    const auto& indices = require_port(operation.outputs, "index", "output");
    require_non_overlapping_tensor(indices, "resample index");
    if (indices.data_type != FLAGDNN_DATA_INT32 ||
        indices.dimensions != y.dimensions || volume > 2147483647)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "resample index requires INT32 window offsets");
  }
  return {{{"mode", mode},
           {"padding", padding},
           {"generate_index", index},
           {"align_corners", align},
           {"n_elements", y.element_count()}},
          {},
          {{"window", window},
           {"stride", stride},
           {"pre_padding", pre},
           {"post_padding", post}}};
}
}  // namespace flagdnn::native
