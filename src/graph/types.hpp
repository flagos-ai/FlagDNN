/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_TYPES_HPP_
#define FLAGDNN_GRAPH_TYPES_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace flagdnn::native {

struct TensorSpec {
  bool configured = false;
  bool is_virtual = false;
  std::int64_t uid = 0;
  std::int64_t alignment = 16;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;

  [[nodiscard]] std::size_t element_size() const;
  [[nodiscard]] std::int64_t element_count() const;
  [[nodiscard]] std::size_t storage_size_in_bytes() const;
  [[nodiscard]] bool is_contiguous() const;
  [[nodiscard]] bool has_non_overlapping_strides() const;
};

struct OperationPort {
  std::string name;
  TensorSpec tensor;
  bool optional = false;
};

using AttributeValue =
    std::variant<std::int64_t,
                 double,
                 bool,
                 std::string,
                 std::vector<std::int64_t>>;
using AttributeMap = std::map<std::string, AttributeValue, std::less<>>;

struct OperationSpec {
  explicit OperationSpec(flagdnnOperation_t value) : operation(value) {}

  flagdnnOperation_t operation = FLAGDNN_OPERATION_RELU;
  bool configured = false;
  std::string name;
  std::string custom_operation_name;
  bool has_compute_data_type = false;
  flagdnnDataType_t compute_data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<OperationPort> inputs;
  std::vector<OperationPort> outputs;
  AttributeMap attributes;
};

struct GraphSpec {
  bool finalized = false;
  std::string name;
  std::vector<OperationSpec> operations;
};

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_TYPES_HPP_
