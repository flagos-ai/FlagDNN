/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_LOWERING_HELPERS_HPP_
#define FLAGDNN_GRAPH_LOWERING_HELPERS_HPP_

#include "error.hpp"
#include "graph/types.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace flagdnn::native {

inline const AttributeValue& require_attribute(
    const OperationSpec& operation,
    std::string_view name) {
  const auto iterator = operation.attributes.find(name);
  if (iterator == operation.attributes.end()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured operation is missing attribute: " +
                       std::string(name));
  }
  return iterator->second;
}

inline std::int64_t integer_attribute(const OperationSpec& operation,
                                      std::string_view name) {
  const auto* value =
      std::get_if<std::int64_t>(&require_attribute(operation, name));
  if (value == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured operation integer attribute has wrong type: " +
                       std::string(name));
  }
  return *value;
}

inline double real_attribute(const OperationSpec& operation,
                             std::string_view name) {
  const auto* value = std::get_if<double>(&require_attribute(operation, name));
  if (value == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured operation real attribute has wrong type: " +
                       std::string(name));
  }
  return *value;
}

inline bool boolean_attribute(const OperationSpec& operation,
                              std::string_view name) {
  const auto* value = std::get_if<bool>(&require_attribute(operation, name));
  if (value == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured operation boolean attribute has wrong type: " +
                       std::string(name));
  }
  return *value;
}

inline const std::vector<std::int64_t>& integer_array_attribute(
    const OperationSpec& operation,
    std::string_view name) {
  const auto* value = std::get_if<std::vector<std::int64_t>>(
      &require_attribute(operation, name));
  if (value == nullptr) {
    throw ApiError(
        FLAGDNN_STATUS_INTERNAL_ERROR,
        "configured operation integer-array attribute has wrong type: " +
            std::string(name));
  }
  return *value;
}

inline const TensorSpec& require_port(
    const std::vector<OperationPort>& ports,
    std::string_view name,
    const char* direction) {
  const OperationPort* result = nullptr;
  for (const OperationPort& port : ports) {
    if (port.name == name) {
      if (result != nullptr) {
        throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                       std::string("configured operation has duplicate ") +
                           direction + " port: " + std::string(name));
      }
      result = &port;
    }
  }
  if (result == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   std::string("configured operation is missing ") +
                       direction + " port: " + std::string(name));
  }
  return result->tensor;
}

inline void require_port_count(const OperationSpec& operation,
                               std::size_t input_count,
                               std::size_t output_count) {
  if (operation.inputs.size() != input_count ||
      operation.outputs.size() != output_count) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "configured operation has an invalid port count");
  }
}

inline std::int64_t checked_multiply(std::int64_t left,
                                     std::int64_t right,
                                     const char* message) {
  if (left < 0 || right < 0 ||
      (right != 0 &&
       left > std::numeric_limits<std::int64_t>::max() / right)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
  return left * right;
}

inline std::int64_t checked_add(std::int64_t left,
                                std::int64_t right,
                                const char* message) {
  if (right > 0 &&
      left > std::numeric_limits<std::int64_t>::max() - right) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
  return left + right;
}

inline void require_configured(const TensorSpec& tensor,
                               const char* name) {
  if (!tensor.configured) {
    throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                   std::string(name) +
                       " tensor descriptor is not configured");
  }
}

inline void require_non_overlapping_tensor(const TensorSpec& tensor,
                                           const char* name) {
  require_configured(tensor, name);
  if (!tensor.has_non_overlapping_strides()) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        std::string(name) + " must have non-overlapping element strides");
  }
}

inline void require_same_shape(const TensorSpec& left,
                               const TensorSpec& right,
                               const char* message) {
  if (left.dimensions != right.dimensions) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
}

inline std::vector<std::int64_t> broadcast_dimensions(
    const TensorSpec& left,
    const TensorSpec& right) {
  const std::size_t rank =
      std::max(left.dimensions.size(), right.dimensions.size());
  std::vector<std::int64_t> result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t left_dimension =
        trailing < left.dimensions.size()
            ? left.dimensions[left.dimensions.size() - 1 - trailing]
            : 1;
    const std::int64_t right_dimension =
        trailing < right.dimensions.size()
            ? right.dimensions[right.dimensions.size() - 1 - trailing]
            : 1;
    if (left_dimension != right_dimension && left_dimension != 1 &&
        right_dimension != 1) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "binary pointwise input shapes are not "
                     "broadcast-compatible");
    }
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
  }
  return result;
}

inline void require_same_data_type(const TensorSpec& left,
                                   const TensorSpec& right,
                                   const char* message) {
  if (left.data_type != right.data_type) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
}

inline bool is_floating_data_type(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FLOAT32 ||
         data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16;
}

inline void require_floating_data_type(const TensorSpec& tensor,
                                       const char* message) {
  if (!is_floating_data_type(tensor.data_type)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
}

inline void require_boolean_data_type(const TensorSpec& tensor,
                                      const char* message) {
  if (tensor.data_type != FLAGDNN_DATA_BOOLEAN) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
}

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_LOWERING_HELPERS_HPP_
