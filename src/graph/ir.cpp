/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/ir.hpp"

#include "error.hpp"
#include "runtime/context.hpp"

#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace flagdnn::native {
namespace {

std::string_view data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return "float32";
    case FLAGDNN_DATA_FLOAT16:
      return "float16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bfloat16";
    case FLAGDNN_DATA_BOOLEAN:
      return "boolean";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR, "unknown tensor data type");
}

void append_json_string(std::ostringstream& output, std::string_view value) {
  output.put(static_cast<char>(34));
  for (const unsigned char character : value) {
    switch (character) {
      case 34:
        output << "\\\"";
        break;
      case 92:
        output << "\\\\";
        break;
      case 8:
        output << "\\b";
        break;
      case 12:
        output << "\\f";
        break;
      case 10:
        output << "\\n";
        break;
      case 13:
        output << "\\r";
        break;
      case 9:
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          const auto flags = output.flags();
          const char fill = output.fill();
          output << "\\u" << std::hex << std::setw(4)
                 << std::setfill(static_cast<char>(48))
                 << static_cast<unsigned int>(character);
          output.flags(flags);
          output.fill(fill);
        } else {
          output.put(static_cast<char>(character));
        }
        break;
    }
  }
  output.put(static_cast<char>(34));
}

void append_integer_array(std::ostringstream& output,
                          const std::vector<std::int64_t>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    output << values[index];
  }
  output << ']';
}

void append_attribute_value(std::ostringstream& output,
                            const AttributeValue& attribute) {
  if (const auto* value = std::get_if<std::int64_t>(&attribute)) {
    output << *value;
    return;
  }
  if (const auto* value = std::get_if<double>(&attribute)) {
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << *value;
    return;
  }
  if (const auto* value = std::get_if<bool>(&attribute)) {
    output << (*value ? "true" : "false");
    return;
  }
  if (const auto* value = std::get_if<std::string>(&attribute)) {
    append_json_string(output, *value);
    return;
  }
  if (const auto* value =
          std::get_if<std::vector<std::int64_t>>(&attribute)) {
    append_integer_array(output, *value);
    return;
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                 "unknown operation attribute type");
}

AttributeMap graph_ir_attributes(const OperationSpec& operation,
                                 const LoweredOperation& lowered) {
  AttributeMap result = operation.attributes;
  for (const auto& [name, value] : lowered.parameters) {
    result.insert_or_assign(name, value);
  }
  for (const auto& [name, value] : lowered.real_parameters) {
    result.insert_or_assign(name, value);
  }
  for (const auto& [name, value] : lowered.integer_array_parameters) {
    result.insert_or_assign(name, value);
  }
  return result;
}

void append_tensor(std::ostringstream& output, const TensorSpec& tensor) {
  output << "{\"uid\":" << tensor.uid;
  output << ",\"data_type\":\"" << data_type_name(tensor.data_type)
         << "\",\"dimensions\":";
  append_integer_array(output, tensor.dimensions);
  output << ",\"strides\":";
  append_integer_array(output, tensor.strides);
  output << ",\"alignment\":" << tensor.alignment;
  output << ",\"virtual\":" << (tensor.is_virtual ? "true" : "false");
  output << '}';
}

void append_ports(std::ostringstream& output,
                  const std::vector<OperationPort>& ports) {
  output << '[';
  for (std::size_t index = 0; index < ports.size(); ++index) {
    if (index != 0) {
      output << ',';
    }
    const OperationPort& port = ports[index];
    output << "{\"name\":";
    append_json_string(output, port.name);
    output << ",\"uid\":" << port.tensor.uid;
    if (port.optional) {
      output << ",\"optional\":true";
    }
    output << '}';
  }
  output << ']';
}

}  // namespace

std::string make_graph_ir(const RuntimeContext& context,
                          const GraphSpec& graph,
                          const flagdnnBuildOptions_t& options,
                          const std::vector<LoweredOperation>& lowered,
                          const ValidatedGraph& validated) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "{\"schema_version\":3,\"flagdnn_version\":\""
         << FLAGDNN_VERSION_STRING << "\",";
  output << "\"backend\":\"" << context.backend_name() << "\",";
  output << "\"target\":\"" << context.target_fingerprint() << "\",";
  const std::uint64_t requested_heuristic_flags =
      options.flags & (FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A |
                       FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_FALLBACK);
  const std::uint64_t heuristic_flags =
      requested_heuristic_flags == 0
          ? FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A
          : requested_heuristic_flags;
  output << "\"build_options\":{\"heuristic_modes\":[";
  bool has_heuristic_mode = false;
  if ((heuristic_flags & FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_A) != 0) {
    output << "\"A\"";
    has_heuristic_mode = true;
  }
  if ((heuristic_flags &
       FLAGDNN_BUILD_OPTION_HEURISTIC_MODE_FALLBACK) != 0) {
    if (has_heuristic_mode) {
      output << ',';
    }
    output << "\"FALLBACK\"";
  }
  output << "],\"autotune\":"
         << ((options.flags & FLAGDNN_BUILD_OPTION_AUTOTUNE) != 0
                 ? "true"
                 : "false")
         << "},";
  output << "\"graph\":{\"name\":";
  append_json_string(output, graph.name);
  output << ",\"tensor_count\":" << validated.tensors.size()
         << ",\"tensors\":[";
  for (std::size_t tensor_index = 0;
       tensor_index < validated.tensors.size();
       ++tensor_index) {
    if (tensor_index != 0) {
      output << ',';
    }
    append_tensor(output, validated.tensors[tensor_index]);
  }
  output << "],\"node_count\":" << graph.operations.size()
         << ",\"nodes\":[";
  for (std::size_t order_index = 0;
       order_index < validated.execution_order.size();
       ++order_index) {
    if (order_index != 0) {
      output << ',';
    }
    const std::size_t operation_index =
        validated.execution_order[order_index];
    const OperationSpec& operation = graph.operations[operation_index];
    const LoweredOperation& current = lowered[operation_index];
    output << "{\"id\":" << operation_index << ",\"type\":\""
           << operation_name(operation) << "\",\"name\":";
    append_json_string(output, operation.name);
    output << ",\"compute_data_type\":";
    append_json_string(
        output, data_type_name(operation_compute_data_type(operation)));
    output << ",\"inputs\":";
    append_ports(output, operation.inputs);
    output << ",\"outputs\":";
    append_ports(output, operation.outputs);
    output << ",\"attributes\":{";
    const AttributeMap attributes = graph_ir_attributes(operation, current);
    bool has_attribute = false;
    for (const auto& [name, value] : attributes) {
      if (has_attribute) {
        output.put(static_cast<char>(44));
      }
      append_json_string(output, name);
      output.put(static_cast<char>(58));
      append_attribute_value(output, value);
      has_attribute = true;
    }
    output << "}}";
  }
  output << "]}}";
  return output.str();
}

}  // namespace flagdnn::native
