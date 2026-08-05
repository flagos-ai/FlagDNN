/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/types.hpp"

#include "error.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace flagdnn::native {
namespace {

std::int64_t checked_multiply(std::int64_t left,
                              std::int64_t right,
                              const char* message) {
  if (left < 0 || right < 0 ||
      (right != 0 &&
       left > std::numeric_limits<std::int64_t>::max() / right)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
  return left * right;
}

void require_configured(const TensorSpec& tensor, const char* name) {
  if (!tensor.configured) {
    throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                   std::string(name) + " tensor descriptor is not configured");
  }
}

}  // namespace

std::size_t TensorSpec::element_size() const {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return 4;
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16:
      return 2;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      return 1;
  }
  throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, "unknown tensor data type");
}

std::int64_t TensorSpec::element_count() const {
  require_configured(*this, "requested");
  std::int64_t result = 1;
  for (const std::int64_t dimension : dimensions) {
    result = checked_multiply(
        result, dimension, "tensor element count overflows");
  }
  return result;
}

std::size_t TensorSpec::storage_size_in_bytes() const {
  require_configured(*this, "requested");
  std::uint64_t maximum_offset = 0;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    const std::uint64_t extent =
        static_cast<std::uint64_t>(dimensions[index] - 1);
    const std::uint64_t stride = static_cast<std::uint64_t>(strides[index]);
    if (extent != 0 &&
        stride > (std::numeric_limits<std::uint64_t>::max() - maximum_offset) /
                     extent) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "tensor storage extent overflows");
    }
    maximum_offset += extent * stride;
  }
  if (maximum_offset == std::numeric_limits<std::uint64_t>::max()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "tensor storage extent overflows");
  }
  const std::uint64_t elements = maximum_offset + 1;
  const std::uint64_t bytes_per_element = element_size();
  if (elements > std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "tensor storage size exceeds size_t");
  }
  return static_cast<std::size_t>(elements * bytes_per_element);
}

bool TensorSpec::is_contiguous() const {
  if (!configured || dimensions.size() != strides.size()) {
    return false;
  }
  std::int64_t expected = 1;
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    const std::size_t current = index - 1;
    if (strides[current] != expected) {
      return false;
    }
    if (dimensions[current] != 0 &&
        expected > std::numeric_limits<std::int64_t>::max() /
                       dimensions[current]) {
      return false;
    }
    expected *= dimensions[current];
  }
  return true;
}

bool TensorSpec::has_non_overlapping_strides() const {
  if (!configured || dimensions.size() != strides.size()) {
    return false;
  }
  std::vector<std::size_t> axes;
  axes.reserve(dimensions.size());
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] > 1) {
      axes.push_back(axis);
    }
  }
  std::sort(axes.begin(), axes.end(), [&](std::size_t left, std::size_t right) {
    return strides[left] < strides[right];
  });
  std::uint64_t required_span = 1;
  for (const std::size_t axis : axes) {
    const std::uint64_t stride = static_cast<std::uint64_t>(strides[axis]);
    if (stride < required_span) {
      return false;
    }
    const std::uint64_t extent =
        static_cast<std::uint64_t>(dimensions[axis] - 1);
    if (extent != 0 &&
        stride > (std::numeric_limits<std::uint64_t>::max() - required_span) /
                     extent) {
      return false;
    }
    required_span += extent * stride;
  }
  return true;
}

}  // namespace flagdnn::native
