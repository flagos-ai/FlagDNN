/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/validation.hpp"

#include "error.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::native {
namespace {

struct GraphTensorState {
  TensorSpec specification;
  std::optional<std::size_t> producer;
};

void require_configured(const TensorSpec& tensor, const char* name) {
  if (!tensor.configured) {
    throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                   std::string(name) +
                       " tensor descriptor is not configured");
  }
}

bool same_tensor_metadata(const TensorSpec& left,
                          const TensorSpec& right) {
  return left.is_virtual == right.is_virtual &&
         left.alignment == right.alignment &&
         left.data_type == right.data_type &&
         left.dimensions == right.dimensions &&
         left.strides == right.strides;
}

GraphTensorState& register_graph_tensor(
    std::vector<GraphTensorState>& tensors,
    const TensorSpec& tensor) {
  for (GraphTensorState& existing : tensors) {
    if (existing.specification.uid == tensor.uid) {
      if (!same_tensor_metadata(existing.specification, tensor)) {
        throw ApiError(
            FLAGDNN_STATUS_INVALID_VALUE,
            "tensors sharing a graph UID must have identical metadata");
      }
      return existing;
    }
  }
  tensors.push_back({tensor, std::nullopt});
  return tensors.back();
}

GraphTensorState& find_graph_tensor(std::vector<GraphTensorState>& tensors,
                                    std::int64_t uid) {
  const auto iterator = std::find_if(
      tensors.begin(), tensors.end(), [uid](const GraphTensorState& tensor) {
        return tensor.specification.uid == uid;
      });
  if (iterator == tensors.end()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "graph tensor registry is inconsistent");
  }
  return *iterator;
}

void append_unique_uid(std::vector<std::int64_t>& result,
                       std::int64_t uid) {
  if (std::find(result.begin(), result.end(), uid) == result.end()) {
    result.push_back(uid);
  }
}

void validate_ports(const std::vector<OperationPort>& ports,
                    const char* direction) {
  for (std::size_t index = 0; index < ports.size(); ++index) {
    const OperationPort& port = ports[index];
    if (port.name.empty()) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     std::string("operation ") + direction +
                         " port name must not be empty");
    }
    require_configured(port.tensor, port.name.c_str());
    if (port.tensor.uid <= 0) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "tensor UID must be greater than zero");
    }
    for (std::size_t previous = 0; previous < index; ++previous) {
      if (ports[previous].name == port.name) {
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       std::string("operation has duplicate ") + direction +
                           " port name: " + port.name);
      }
    }
  }
}

}  // namespace

ValidatedGraph validate_graph(const GraphSpec& graph) {
  if (graph.operations.empty()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "cannot validate an empty graph");
  }

  std::vector<GraphTensorState> tensors;
  std::vector<std::int64_t> external_binding_uids;
  bool has_external_output = false;

  for (std::size_t operation_index = 0;
       operation_index < graph.operations.size();
       ++operation_index) {
    const OperationSpec& operation = graph.operations[operation_index];
    if (!operation.configured) {
      throw ApiError(FLAGDNN_STATUS_NOT_INITIALIZED,
                     "operation descriptor is not configured");
    }
    if (operation.outputs.empty()) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "graph operation has no output ports");
    }
    validate_ports(operation.inputs, "input");
    validate_ports(operation.outputs, "output");

    for (const OperationPort& port : operation.inputs) {
      register_graph_tensor(tensors, port.tensor);
      if (!port.tensor.is_virtual) {
        append_unique_uid(external_binding_uids, port.tensor.uid);
      }
    }
    for (const OperationPort& port : operation.outputs) {
      GraphTensorState& state = register_graph_tensor(tensors, port.tensor);
      if (state.producer.has_value()) {
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "a graph tensor may be produced by only one operation");
      }
      state.producer = operation_index;
      if (!port.tensor.is_virtual) {
        append_unique_uid(external_binding_uids, port.tensor.uid);
        has_external_output = true;
      }
    }
  }

  std::vector<std::vector<std::size_t>> consumers(graph.operations.size());
  std::vector<std::size_t> indegree(graph.operations.size(), 0);
  for (std::size_t consumer = 0; consumer < graph.operations.size();
       ++consumer) {
    for (const OperationPort& port : graph.operations[consumer].inputs) {
      GraphTensorState& state = find_graph_tensor(tensors, port.tensor.uid);
      if (port.tensor.is_virtual && !state.producer.has_value() &&
          !port.optional) {
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "a virtual tensor input has no producer");
      }
      if (!state.producer.has_value()) {
        continue;
      }
      const std::size_t producer = *state.producer;
      if (producer == consumer) {
        throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                       "in-place tensor aliases are not supported yet");
      }
      auto& producer_consumers = consumers[producer];
      if (std::find(producer_consumers.begin(),
                    producer_consumers.end(),
                    consumer) == producer_consumers.end()) {
        producer_consumers.push_back(consumer);
        ++indegree[consumer];
      }
    }
  }

  if (external_binding_uids.empty()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "graph has no externally bound tensors");
  }
  if (!has_external_output) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "graph has no non-virtual output tensor");
  }

  std::priority_queue<std::size_t,
                      std::vector<std::size_t>,
                      std::greater<std::size_t>> ready;
  for (std::size_t index = 0; index < indegree.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push(index);
    }
  }

  std::vector<std::size_t> execution_order;
  execution_order.reserve(graph.operations.size());
  while (!ready.empty()) {
    const std::size_t producer = ready.top();
    ready.pop();
    execution_order.push_back(producer);
    for (const std::size_t consumer : consumers[producer]) {
      if (--indegree[consumer] == 0) {
        ready.push(consumer);
      }
    }
  }
  if (execution_order.size() != graph.operations.size()) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "graph contains a tensor dependency cycle");
  }

  ValidatedGraph result;
  result.execution_order = std::move(execution_order);
  result.external_binding_uids = std::move(external_binding_uids);
  result.tensors.reserve(tensors.size());
  for (GraphTensorState& tensor : tensors) {
    result.tensors.push_back(std::move(tensor.specification));
  }
  return result;
}

}  // namespace flagdnn::native
