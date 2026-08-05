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
  return {{{"batch", batch_count}, {"m", m}, {"n", n}, {"k", k}},
          {},
          {}};
}

}  // namespace flagdnn::native
