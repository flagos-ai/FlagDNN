/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace flagdnn::native {
namespace {

std::vector<std::int64_t> matmul_batch_dimensions(
    const std::vector<std::int64_t>& left,
    const std::vector<std::int64_t>& right) {
  const std::size_t rank = std::max(left.size(), right.size());
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t left_dimension =
        trailing < left.size() ? left[left.size() - 1 - trailing] : 1;
    const std::int64_t right_dimension =
        trailing < right.size() ? right[right.size() - 1 - trailing] : 1;
    if (left_dimension != right_dimension && left_dimension != 1 &&
        right_dimension != 1) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "MatMul batch dimensions are not broadcast-compatible");
    }
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
  }
  return result;
}

}  // namespace

LoweredOperation lower_matmul(const OperationSpec& operation) {
  require_port_count(operation, 2, 1);
  const TensorSpec& a = require_port(operation.inputs, "a", "input");
  const TensorSpec& b = require_port(operation.inputs, "b", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  if (a.data_type == FLAGDNN_DATA_FP8_E4M3 ||
      a.data_type == FLAGDNN_DATA_FP8_E5M2) {
    (void)requested_input_precision(operation, a);
    auto scaled = operation;
    scaled.attributes["scale_mode"] = std::int64_t{0};
    return lower_matmul_fp8(scaled);
  }
  require_non_overlapping_tensor(a, "A");
  require_non_overlapping_tensor(b, "B");
  require_non_overlapping_tensor(output, "output");
  require_same_data_type(a, b, "MatMul input data types must match");
  require_same_data_type(a, output,
                         "MatMul input/output data types must match");
  require_floating_data_type(
      a, "MatMul tensors must use a floating data type");
  if (a.dimensions.size() < 2 || b.dimensions.size() < 2 ||
      a.dimensions.size() > 8 || b.dimensions.size() > 8) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "MatMul input ranks must be in [2, 8]");
  }

  const std::int64_t m = a.dimensions[a.dimensions.size() - 2];
  const std::int64_t k = a.dimensions.back();
  const std::int64_t b_k = b.dimensions[b.dimensions.size() - 2];
  const std::int64_t n = b.dimensions.back();
  if (k != b_k) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MatMul contraction dimensions do not match");
  }
  const std::vector<std::int64_t> a_batch(
      a.dimensions.begin(), a.dimensions.end() - 2);
  const std::vector<std::int64_t> b_batch(
      b.dimensions.begin(), b.dimensions.end() - 2);
  std::vector<std::int64_t> expected =
      matmul_batch_dimensions(a_batch, b_batch);
  expected.push_back(m);
  expected.push_back(n);
  if (output.dimensions != expected) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MatMul output shape is incorrect");
  }

  std::int64_t batch_count = 1;
  for (std::size_t axis = 0; axis + 2 < expected.size(); ++axis) {
    batch_count = checked_multiply(
        batch_count, expected[axis], "MatMul batch extent overflows");
  }
  return {{{"input_precision", requested_input_precision(operation, a)},
           {"batch", batch_count},
           {"m", m},
           {"n", n},
           {"k", k}},
          {},
          {}};
}

LoweredOperation lower_matmul_fp8(const OperationSpec& operation) {
  const auto mode = integer_attribute(operation, "scale_mode");
  if (mode < 0 || mode > 2)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "invalid FP8 matmul scale mode");
  require_port_count(operation, mode ? 4 : 2, 1);
  const auto& a = require_port(operation.inputs, "a", "input");
  const auto& b = require_port(operation.inputs, "b", "input");
  const auto& output = require_port(operation.outputs, "output", "output");
  const auto fp8 = [](flagdnnDataType_t type) {
    return type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2;
  };
  if (!fp8(a.data_type) || !fp8(b.data_type))
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "FP8 MatMul requires E4M3 or E5M2 inputs");
  require_floating_data_type(output,
                             "FP8 MatMul output must be FP32, FP16 or BF16");
  // Reuse the shared batch, contraction, output-shape and stride validation.
  auto shape_operation = operation;
  shape_operation.inputs = {{"a", a}, {"b", b}};
  for (auto& port : shape_operation.inputs)
    port.tensor.data_type = output.data_type;
  auto result = lower_matmul(shape_operation);
  result.parameters.emplace_back("scale_mode", mode);
  if (mode) {
    for (const auto& item :
         {std::pair{"descale_a", &a}, std::pair{"descale_b", &b}}) {
      const auto& scale = require_port(operation.inputs, item.first, "input");
      require_non_overlapping_tensor(scale, "FP8 scale");
      if (mode == 1) {
        if (scale.data_type != FLAGDNN_DATA_FLOAT32 ||
            scale.element_count() != 1)
          throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                         "FP8 descales must be FP32 scalars");
      } else {
        auto expected = item.second->dimensions;
        const auto axis =
            item.second == &a ? expected.size() - 1 : expected.size() - 2;
        expected[axis] = (expected[axis] - 1) / 32 + 1;
        if (scale.data_type != FLAGDNN_DATA_FP8_E8M0 ||
            scale.dimensions != expected)
          throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                         "MXFP8 E8M0 scales must replace the contraction "
                         "extent with ceil(K/32)");
      }
    }
  }
  return result;
}

}  // namespace flagdnn::native
