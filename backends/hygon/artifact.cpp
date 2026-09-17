/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/artifact.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "backends/hygon/error.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

namespace flagdnn::hygon {
namespace {

constexpr std::int64_t kArtifactSchemaVersion = 5;
constexpr std::int64_t kExecutionProgramVersion = 2;
constexpr std::size_t kMaximumAutotuneCandidates = 1024;
constexpr std::size_t kWorkspaceAlignment = 256;

struct RequestTensorMetadata {
  bool is_virtual = false;
  std::string pointer_token;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t element_count = 1;
  std::size_t storage_elements = 1;
  std::size_t storage_size = 0;
  std::size_t alignment = 1;
  std::size_t workspace_offset = 0;
};

struct RequestNodeMetadata {
  std::string operation;
  std::set<std::int64_t> tensor_uids;
  std::map<std::string, std::int64_t> inputs;
  std::map<std::string, std::int64_t> outputs;
  std::map<std::string, std::int64_t> integer_attributes;
  std::map<std::string, double> numeric_attributes;
  std::optional<std::int64_t> n_elements;
  std::optional<std::int64_t> pointwise_mode;
  std::optional<double> alpha;
};

struct RequestGraphMetadata {
  std::map<std::int64_t, RequestTensorMetadata> tensors;
  std::map<std::size_t, RequestNodeMetadata> nodes;
  std::set<std::int64_t> external_binding_uids;
  std::int64_t maximum_tensor_uid = 0;
  std::size_t graph_workspace_size = 0;
  std::size_t workspace_alignment = kWorkspaceAlignment;
  bool autotune = false;
};

std::size_t checked_size(std::int64_t value, const char *field);

[[noreturn]] void invalid_artifact(const char *message) {
  throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, message);
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        const char *field) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact field overflows: ") + field);
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char *field) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact field overflows: ") + field);
  }
  return left * right;
}

std::size_t align_up(std::size_t value, std::size_t alignment,
                     const char *field) {
  if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact alignment: ") + field);
  }
  return checked_add(value, alignment - 1, field) & ~(alignment - 1);
}

std::string pointer_token_for(std::string_view data_type) {
  if (data_type == "int32")
    return "*i32";
  if (data_type == "fp8_e8m0")
    return "*u8";
  if (data_type == "float32") {
    return "*fp32";
  }
  if (data_type == "float16") {
    return "*fp16";
  }
  if (data_type == "bfloat16") {
    return "*bf16";
  }
  if (data_type == "boolean") {
    return "*i8";
  }
  if (data_type == "fp8_e4m3") {
    return "*fp8e4nv";
  }
  if (data_type == "fp8_e5m2") {
    return "*fp8e5";
  }
  invalid_artifact("request tensor data type is unsupported");
}

std::size_t element_size_for(std::string_view data_type) {
  if (data_type == "float32" || data_type == "int32") {
    return 4;
  }
  if (data_type == "float16" || data_type == "bfloat16") {
    return 2;
  }
  if (data_type == "fp8_e8m0" || data_type == "boolean" ||
      data_type == "fp8_e4m3" || data_type == "fp8_e5m2") {
    return 1;
  }
  invalid_artifact("request tensor data type is unsupported");
}

RequestGraphMetadata
parse_request_graph(const flagdnn::native::json::Value &request_root) {
  RequestGraphMetadata result;
  result.autotune = request_root.at("build_options").at("autotune").as_bool();
  const auto &graph = request_root.at("graph");
  const auto &tensor_values = graph.at("tensors").as_array();
  if (tensor_values.empty() ||
      checked_size(graph.at("tensor_count").as_int(), "graph.tensor_count") !=
          tensor_values.size()) {
    invalid_artifact("request graph tensor table is invalid");
  }

  std::size_t workspace_cursor = 0;
  for (const auto &tensor_value : tensor_values) {
    const std::int64_t uid = tensor_value.at("uid").as_int();
    const std::size_t alignment = checked_size(
        tensor_value.at("alignment").as_int(), "graph.tensor.alignment");
    if (uid <= 0 || alignment == 0 || alignment > (1ULL << 31) ||
        (alignment & (alignment - 1)) != 0 || result.tensors.contains(uid)) {
      invalid_artifact("request graph tensor identity is invalid");
    }
    const auto &dimensions = tensor_value.at("dimensions").as_array();
    const auto &strides = tensor_value.at("strides").as_array();
    if (dimensions.size() > 8 || dimensions.size() != strides.size()) {
      invalid_artifact("request graph tensor rank is invalid");
    }
    std::size_t storage_elements = 1;
    std::size_t element_count = 1;
    std::vector<std::int64_t> parsed_dimensions;
    std::vector<std::int64_t> parsed_strides;
    parsed_dimensions.reserve(dimensions.size());
    parsed_strides.reserve(strides.size());
    for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
      const std::int64_t dimension = dimensions[axis].as_int();
      const std::int64_t stride = strides[axis].as_int();
      if (dimension <= 0 ||
          dimension > std::numeric_limits<std::int32_t>::max() || stride <= 0 ||
          stride > std::numeric_limits<std::int32_t>::max()) {
        invalid_artifact("request graph tensor shape is invalid");
      }
      const std::size_t contribution = checked_multiply(
          static_cast<std::size_t>(dimension - 1),
          static_cast<std::size_t>(stride), "graph.tensor.storage_elements");
      storage_elements = checked_add(storage_elements, contribution,
                                     "graph.tensor.storage_elements");
      element_count =
          checked_multiply(element_count, static_cast<std::size_t>(dimension),
                           "graph.tensor.element_count");
      parsed_dimensions.push_back(dimension);
      parsed_strides.push_back(stride);
    }
    const std::string &data_type = tensor_value.at("data_type").as_string();
    RequestTensorMetadata metadata;
    metadata.is_virtual = tensor_value.at("virtual").as_bool();
    metadata.pointer_token = pointer_token_for(data_type);
    metadata.dimensions = std::move(parsed_dimensions);
    metadata.strides = std::move(parsed_strides);
    metadata.element_count = element_count;
    metadata.storage_elements = storage_elements;
    metadata.storage_size =
        checked_multiply(storage_elements, element_size_for(data_type),
                         "graph.tensor.storage_size");
    metadata.alignment = alignment;
    if (metadata.is_virtual) {
      const std::size_t workspace_alignment =
          std::max(kWorkspaceAlignment, alignment);
      result.workspace_alignment =
          std::max(result.workspace_alignment, workspace_alignment);
      workspace_cursor =
          align_up(workspace_cursor, workspace_alignment, "graph.workspace");
      metadata.workspace_offset = workspace_cursor;
      workspace_cursor = checked_add(workspace_cursor, metadata.storage_size,
                                     "graph.workspace");
    } else {
      result.external_binding_uids.insert(uid);
    }
    result.maximum_tensor_uid = std::max(result.maximum_tensor_uid, uid);
    result.tensors.emplace(uid, std::move(metadata));
  }
  if (workspace_cursor != 0) {
    workspace_cursor =
        align_up(workspace_cursor, kWorkspaceAlignment, "graph.workspace");
  }
  result.graph_workspace_size = workspace_cursor;

  const auto &node_values = graph.at("nodes").as_array();
  const std::size_t node_count =
      checked_size(graph.at("node_count").as_int(), "graph.node_count");
  if (node_count == 0 || node_count > 1024 ||
      node_values.size() != node_count) {
    invalid_artifact("request graph node table is invalid");
  }
  for (const auto &node_value : node_values) {
    const std::size_t node_id =
        checked_size(node_value.at("id").as_int(), "graph.node.id");
    RequestNodeMetadata metadata;
    metadata.operation = node_value.at("type").as_string();
    if (node_id >= node_count || metadata.operation.empty() ||
        result.nodes.contains(node_id)) {
      invalid_artifact("request graph node identity is invalid");
    }
    const auto add_ports = [&](const char *field,
                               std::map<std::string, std::int64_t> &ports) {
      for (const auto &port : node_value.at(field).as_array()) {
        const std::int64_t uid = port.at("uid").as_int();
        const std::string &name = port.at("name").as_string();
        if (!result.tensors.contains(uid)) {
          invalid_artifact("request graph node references an unknown tensor");
        }
        if (name.empty() || !ports.emplace(name, uid).second) {
          invalid_artifact("request graph node port identity is invalid");
        }
        metadata.tensor_uids.insert(uid);
      }
    };
    add_ports("inputs", metadata.inputs);
    add_ports("outputs", metadata.outputs);
    if (metadata.tensor_uids.empty()) {
      invalid_artifact("request graph node has no tensor ports");
    }
    const auto &attributes = node_value.at("attributes").as_object();
    for (const auto &[name, value] : attributes) {
      try {
        const std::int64_t integer = value.as_int();
        metadata.integer_attributes.emplace(name, integer);
        metadata.numeric_attributes.emplace(name, static_cast<double>(integer));
        continue;
      } catch (const std::exception &) {
      }
      try {
        metadata.numeric_attributes.emplace(name, value.as_double());
      } catch (const std::exception &) {
      }
    }
    if (const auto entry = attributes.find("n_elements");
        entry != attributes.end()) {
      metadata.n_elements = entry->second.as_int();
    }
    if (const auto entry = attributes.find("pointwise_mode");
        entry != attributes.end()) {
      metadata.pointwise_mode = entry->second.as_int();
    }
    if (const auto entry = attributes.find("alpha");
        entry != attributes.end()) {
      metadata.alpha = entry->second.as_double();
    }
    result.nodes.emplace(node_id, std::move(metadata));
  }
  return result;
}

std::string read_text_file(const std::filesystem::path &path,
                           std::size_t maximum_size) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > maximum_size) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot stat artifact metadata or it exceeds size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot open artifact metadata");
  }
  std::string result(static_cast<std::size_t>(file_size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot read artifact metadata");
  }
  return result;
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(value.begin(), value.end(),
                     [](const unsigned char character) {
                       return std::isxdigit(character) != 0;
                     });
}

unsigned int checked_positive_unsigned(std::int64_t value, const char *field) {
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        std::numeric_limits<unsigned int>::max()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact field: ") + field);
  }
  return static_cast<unsigned int>(value);
}

unsigned int checked_nonnegative_unsigned(std::int64_t value,
                                          const char *field) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<unsigned int>::max()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact field: ") + field);
  }
  return static_cast<unsigned int>(value);
}

std::size_t checked_size(std::int64_t value, const char *field) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact field: ") + field);
  }
  return static_cast<std::size_t>(value);
}

std::int32_t checked_i32(std::int64_t value, const char *field) {
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact field: ") + field);
  }
  return static_cast<std::int32_t>(value);
}

float checked_f32(double value, const char *field) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("invalid artifact field: ") + field);
  }
  return static_cast<float>(value);
}

std::array<unsigned int, 3>
parse_triplet(const flagdnn::native::json::Value &value, const char *field) {
  const auto &array = value.as_array();
  if (array.size() != 3) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact ") + field +
                         " must contain three integers");
  }
  return {checked_positive_unsigned(array[0].as_int(), field),
          checked_positive_unsigned(array[1].as_int(), field),
          checked_positive_unsigned(array[2].as_int(), field)};
}

bool is_safe_basename(std::string_view value) {
  return !value.empty() && value != "." && value != ".." &&
         std::filesystem::path(value).filename().string() == value;
}

std::filesystem::path
validate_file(const std::filesystem::path &directory,
              const flagdnn::native::json::Value &descriptor,
              std::size_t maximum_size, const char *label) {
  const std::string name = descriptor.at("file").as_string();
  if (!is_safe_basename(name)) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact ") + label + " path is unsafe");
  }
  const std::filesystem::path path = directory / name;
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact ") + label +
                         " is missing or not a regular file");
  }
  const std::size_t expected_size =
      checked_size(descriptor.at("size").as_int(), label);
  const std::uintmax_t actual_size = std::filesystem::file_size(path, error);
  if (error || expected_size == 0 || expected_size > maximum_size ||
      actual_size != expected_size) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact ") + label +
                         " size does not match manifest");
  }
  const std::string expected_hash = descriptor.at("sha256").as_string();
  if (!is_sha256(expected_hash) ||
      flagdnn::native::sha256_file(path) != expected_hash) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string("artifact ") + label +
                         " SHA-256 does not match manifest");
  }
  return path;
}

bool is_identifier(std::string_view value) {
  if (value.empty() || value.size() > 256 ||
      (std::isalpha(static_cast<unsigned char>(value.front())) == 0 &&
       value.front() != '_')) {
    return false;
  }
  return std::all_of(value.begin() + 1, value.end(),
                     [](const unsigned char character) {
                       return std::isalnum(character) != 0 || character == '_';
                     });
}

std::vector<ArgumentSpec>
parse_argument_abi(const flagdnn::native::json::Value &value,
                   std::size_t workspace_size,
                   std::vector<std::int64_t> &binding_uids) {
  const auto &abi = value.as_array();
  if (abi.size() < 3 || abi.size() > FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "artifact argument ABI count is invalid");
  }
  if (abi[abi.size() - 2].at("kind").as_string() != "global_scratch_pointer" ||
      abi[abi.size() - 1].at("kind").as_string() != "profile_scratch_pointer") {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "artifact hidden scratch ABI is incompatible");
  }

  std::vector<ArgumentSpec> result;
  result.reserve(abi.size() - 2);
  for (std::size_t index = 0; index + 2 < abi.size(); ++index) {
    const std::string &kind = abi[index].at("kind").as_string();
    if (kind == "tensor") {
      const auto &argument_object = abi[index].as_object();
      const std::int64_t uid = abi[index].at("uid").as_int();
      const std::size_t size =
          checked_size(abi[index].at("size").as_int(), "argument_abi.size");
      const auto alignment_entry = argument_object.find("alignment");
      const std::size_t alignment =
          alignment_entry == argument_object.end()
              ? 1
              : checked_size(alignment_entry->second.as_int(),
                             "argument_abi.alignment");
      const std::string &role = abi[index].at("role").as_string();
      if (uid <= 0 || size == 0 || alignment == 0 ||
          (alignment & (alignment - 1)) != 0 || !is_identifier(role)) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "artifact tensor argument metadata is invalid");
      }
      result.push_back(
          {ArgumentKind::kTensor, uid, 0, 0.0F, 0, size, alignment, role});
      if (std::find(binding_uids.begin(), binding_uids.end(), uid) ==
          binding_uids.end()) {
        binding_uids.push_back(uid);
      }
    } else if (kind == "workspace_tensor") {
      const auto &argument_object = abi[index].as_object();
      const std::int64_t uid = abi[index].at("uid").as_int();
      const std::size_t offset =
          checked_size(abi[index].at("offset").as_int(), "argument_abi.offset");
      const std::size_t size =
          checked_size(abi[index].at("size").as_int(), "argument_abi.size");
      const auto alignment_entry = argument_object.find("alignment");
      const std::size_t alignment =
          alignment_entry == argument_object.end()
              ? 256
              : checked_size(alignment_entry->second.as_int(),
                             "argument_abi.alignment");
      const std::string &role = abi[index].at("role").as_string();
      if (uid <= 0 || size == 0 || alignment == 0 ||
          (alignment & (alignment - 1)) != 0 || offset % alignment != 0 ||
          offset > workspace_size || size > workspace_size - offset ||
          !is_identifier(role)) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "artifact workspace tensor range is invalid");
      }
      result.push_back({ArgumentKind::kWorkspaceTensor, uid, 0, 0.0F, offset,
                        size, alignment, role});
    } else if (kind == "scalar_i32") {
      const std::string &name = abi[index].at("name").as_string();
      if (!is_identifier(name)) {
        invalid_artifact("artifact scalar argument name is invalid");
      }
      result.push_back(
          {ArgumentKind::kScalarI32, 0,
           checked_i32(abi[index].at("value").as_int(), "argument_abi.value"),
           0.0F, 0, 0, 1, name});
    } else if (kind == "scalar_f32") {
      const std::string &name = abi[index].at("name").as_string();
      if (!is_identifier(name)) {
        invalid_artifact("artifact scalar argument name is invalid");
      }
      result.push_back({ArgumentKind::kScalarF32, 0, 0,
                        checked_f32(abi[index].at("value").as_double(),
                                    "argument_abi.value"),
                        0, 0, 1, name});
    } else {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact contains an unsupported argument kind");
    }
  }
  return result;
}

std::vector<std::string_view>
full_signature_tokens(std::string_view full_signature) {
  std::vector<std::string_view> result;
  std::size_t offset = 0;
  while (offset <= full_signature.size()) {
    const std::size_t separator = full_signature.find(',', offset);
    const std::size_t length = separator == std::string_view::npos
                                   ? full_signature.size() - offset
                                   : separator - offset;
    const std::string_view token = full_signature.substr(offset, length);
    if (token.empty()) {
      invalid_artifact("libtriton_jit full signature contains an empty token");
    }
    result.push_back(token);
    if (separator == std::string_view::npos) {
      break;
    }
    offset = separator + 1;
  }
  return result;
}

std::vector<std::string_view>
runtime_signature_tokens(std::string_view full_signature) {
  std::vector<std::string_view> result;
  for (const std::string_view token : full_signature_tokens(full_signature)) {
    if (token.starts_with('*') || token == "i32" || token == "fp32") {
      result.push_back(token);
    }
  }
  return result;
}

std::optional<std::int64_t> binary_pointwise_mode(std::string_view operation) {
  static constexpr std::array<std::pair<std::string_view, std::int64_t>, 24>
      modes = {{{"add", 1},
                {"sub", 17},
                {"mul", 18},
                {"div", 19},
                {"min", 20},
                {"max", 21},
                {"mod", 22},
                {"pow", 23},
                {"cmp_eq", 25},
                {"cmp_neq", 26},
                {"cmp_gt", 27},
                {"cmp_ge", 28},
                {"cmp_lt", 29},
                {"cmp_le", 30},
                {"logical_and", 31},
                {"logical_or", 32},
                {"sigmoid_backward", 40},
                {"relu_backward", 42},
                {"tanh_backward", 43},
                {"elu_backward", 44},
                {"gelu_backward", 45},
                {"softplus_backward", 46},
                {"swish_backward", 47},
                {"gelu_approx_tanh_backward", 48}}};
  const auto entry =
      std::find_if(modes.begin(), modes.end(), [&](const auto &candidate) {
        return candidate.first == operation;
      });
  if (entry == modes.end()) {
    return std::nullopt;
  }
  return entry->second;
}

bool contains_string(std::span<const std::string_view> values,
                     std::string_view value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

bool is_unary_pointwise_operation(std::string_view operation) {
  static constexpr std::array<std::string_view, 23> operations = {
      "relu",       "sqrt",
      "erf",        "identity",
      "exp",        "log",
      "neg",        "abs",
      "ceil",       "cos",
      "floor",      "rsqrt",
      "sin",        "tan",
      "reciprocal", "sigmoid",
      "tanh",       "elu",
      "gelu",       "softplus",
      "swish",      "gelu_approx_tanh",
      "logical_not"};
  return contains_string(operations, operation);
}

bool kernel_function_allowed(std::string_view operation,
                             std::string_view function) {
  static constexpr std::array<std::string_view, 3> unary_functions = {
      "unary_pointwise_contiguous_kernel", "unary_pointwise_strided_kernel",
      "identity_packed_contiguous_kernel"};
  static constexpr std::array<std::string_view, 4> reduction_functions = {
      "reduction_2d_kernel", "reduction_3d_small_extent_kernel",
      "reduction_3d_kernel", "reduction_strided_kernel"};
  static constexpr std::array<std::string_view, 9> conv_fprop_functions = {
      "conv1d_gemm_kernel",
      "conv2d_spatial_nchw_kernel",
      "conv3d_spatial_ncdhw_m_kernel",
      "hygon_conv2d_im2col_nchw_kernel",
      "hygon_conv2d_fprop_stride2_low_ci_nchw_kernel",
      "hygon_conv2d_fprop_standard_3x3_nchw_kernel",
      "hygon_conv2d_fprop_im2col_kernel",
      "hygon_conv2d_fprop_yolo_x_p5_gemm_kernel",
      "hygon_conv2d_fprop_1x1_nchw_kernel"};
  static constexpr std::array<std::string_view, 9> conv_dgrad_functions = {
      "conv_dgrad_nd_kernel",
      "hygon_conv_dgrad2d_1x1_nchw_kernel",
      "hygon_conv_dgrad2d_stride1_kernel",
      "hygon_conv_dgrad2d_stride2_contribution_gemm_kernel",
      "hygon_conv_dgrad2d_stride2_block_pointer_gemm_kernel",
      "hygon_conv_dgrad2d_stride2_contribution_col2im_kernel",
      "hygon_conv_dgrad2d_stride2_parity_kernel",
      "hygon_conv_dgrad2d_exact_3x3_s1_gemm_kernel",
      "hygon_conv_dgrad2d_exact_3x3_s1_col2im_kernel"};
  static constexpr std::array<std::string_view, 12> conv_wgrad_functions = {
      "conv_wgrad_nd_kernel",
      "hygon_conv2d_im2col_nchw_kernel",
      "hygon_conv_wgrad2d_im2row_kernel",
      "hygon_conv_wgrad2d_rowmajor_kernel",
      "hygon_conv_wgrad2d_p5_block_ptr_kernel",
      "hygon_conv_wgrad2d_im2col_kernel",
      "hygon_conv_wgrad2d_im2col_split_kernel",
      "hygon_conv_wgrad2d_stem_split_kernel",
      "hygon_conv_wgrad2d_multirow_split_kernel",
      "hygon_conv_wgrad2d_direct_split_kernel",
      "hygon_conv_wgrad2d_1x1_split_kernel",
      "hygon_conv_wgrad2d_reduce_kernel"};
  static constexpr std::array<std::string_view, 5> sdpa_backward_functions = {
      "_zero_contiguous_kernel", "_sdpa_bwd_dq_dbias_kernel",
      "_sdpa_bwd_dkdv_kernel", "_sdpa_bwd_dk_kernel", "_sdpa_bwd_dv_kernel"};
  static constexpr std::array<std::string_view, 2> sdpa_fp8_functions = {
      "_zero_sdpa_fp8_fwd_amax_kernel", "_sdpa_fp8_fwd_kernel"};
  static constexpr std::array<std::string_view, 3> sdpa_fp8_backward_functions =
      {"_zero_sdpa_fp8_bwd_amax_kernel", "_sdpa_fp8_bwd_dq_kernel",
       "_sdpa_fp8_bwd_dkdv_kernel"};

  if (operation.ends_with("_backward") &&
      binary_pointwise_mode(operation).has_value()) {
    return function == "activation_backward_contiguous_kernel" ||
           function == "activation_backward_strided_kernel";
  }
  if (binary_pointwise_mode(operation).has_value()) {
    return function == "binary_contiguous_kernel" ||
           function == "binary_strided_kernel";
  }
  if (is_unary_pointwise_operation(operation)) {
    return contains_string(unary_functions, function);
  }
  if (operation == "binary_select") {
    return function == "binary_select_tensor_kernel" ||
           function == "binary_select_strided_kernel";
  }
  if (operation == "reshape" || operation == "transpose" ||
      operation == "slice") {
    return function == "layout_copy_kernel";
  }
  if (operation == "reduction_sum" || operation == "reduction_avg" ||
      operation == "reduction_mul") {
    return contains_string(reduction_functions, function);
  }
  if (operation == "matmul") {
    return function == "matmul_strided_kernel" ||
           function == "fp8_matmul_kernel";
  }
  if (operation == "conv2d_fprop" || operation == "convolution_fprop") {
    return contains_string(conv_fprop_functions, function);
  }
  if (operation == "convolution_dgrad") {
    return contains_string(conv_dgrad_functions, function);
  }
  if (operation == "convolution_wgrad") {
    return contains_string(conv_wgrad_functions, function);
  }
  if (operation == "layernorm") {
    return function == "layer_norm_kernel";
  }
  if (operation == "rmsnorm") {
    return function == "rms_norm_kernel";
  }
  if (operation == "batchnorm") {
    return function == "batch_norm_nchw_kernel" ||
           function == "batch_norm_kernel";
  }
  if (operation == "batchnorm_inference") {
    return function == "batch_norm_inference_nchw_kernel" ||
           function == "batch_norm_inference_kernel";
  }
  if (operation == "sdpa") {
    return function == "_sdpa_fwd_kernel";
  }
  if (operation == "sdpa_backward") {
    return contains_string(sdpa_backward_functions, function);
  }
  if (operation == "sdpa_fp8") {
    return contains_string(sdpa_fp8_functions, function);
  }
  if (operation == "sdpa_fp8_backward") {
    return contains_string(sdpa_fp8_backward_functions, function);
  }
  if (operation == "gen_index")
    return function == "gen_index_kernel";
  if (operation == "concatenate")
    return function == "concatenate_copy_kernel";
  if (operation == "genstats")
    return function == "genstats_kernel";
  if (operation == "bn_finalize")
    return function == "bn_finalize_kernel";
  if (operation == "rng")
    return function == "rng_kernel";
  if (operation == "rope" || operation == "rope_backward")
    return function == "rope_kernel";
  if (operation == "resample")
    return function == "resample_kernel";
  if (operation == "causal_conv1d")
    return function == "causal_conv1d_kernel";
  if (operation == "matmul_fp8")
    return function == "fp8_matmul_kernel";
  if (operation == "moe_grouped_matmul" ||
      operation == "moe_grouped_matmul_bwd")
    return function == "moe_matmul_kernel";
  if (operation == "instancenorm" || operation == "adalayernorm")
    return function == "extended_normalization_forward" ||
           function == "compact_normalization_forward";
  if (operation == "instancenorm_backward" ||
      operation == "adalayernorm_backward" ||
      operation == "layernorm_backward" || operation == "rmsnorm_backward" ||
      operation == "batchnorm_backward")
    return function == "extended_normalization_backward" ||
           function == "compact_batchnorm_backward";
  return false;
}

std::int64_t signature_integer(std::string_view token,
                               const char *error_message) {
  std::int64_t result = 0;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), result);
  if (error != std::errc{} || end != token.data() + token.size()) {
    invalid_artifact(error_message);
  }
  return result;
}

double signature_double(std::string_view token, const char *error_message) {
  double result = 0.0;
  const auto [end, error] =
      std::from_chars(token.data(), token.data() + token.size(), result,
                      std::chars_format::general);
  if (error != std::errc{} || end != token.data() + token.size() ||
      !std::isfinite(result)) {
    invalid_artifact(error_message);
  }
  return result;
}

const RequestTensorMetadata &
pointwise_tensor(const RequestGraphMetadata &graph,
                 const std::map<std::string, std::int64_t> &ports,
                 std::string_view port_name) {
  const auto port = ports.find(std::string(port_name));
  if (port == ports.end()) {
    invalid_artifact("pointwise request Graph IR port schema is invalid");
  }
  const auto tensor = graph.tensors.find(port->second);
  if (tensor == graph.tensors.end()) {
    invalid_artifact("pointwise request Graph IR tensor is missing");
  }
  return tensor->second;
}

bool has_non_overlapping_strides(const RequestTensorMetadata &tensor) {
  std::vector<std::pair<std::int64_t, std::int64_t>> axes;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] > 1) {
      axes.emplace_back(tensor.strides[axis], tensor.dimensions[axis]);
    }
  }
  std::sort(axes.begin(), axes.end());
  std::size_t required_span = 1;
  for (const auto &[stride, dimension] : axes) {
    if (static_cast<std::size_t>(stride) < required_span) {
      return false;
    }
    required_span =
        checked_add(required_span,
                    checked_multiply(static_cast<std::size_t>(dimension - 1),
                                     static_cast<std::size_t>(stride),
                                     "pointwise.tensor.strides"),
                    "pointwise.tensor.strides");
  }
  return true;
}

bool is_physically_dense(const RequestTensorMetadata &tensor) {
  return has_non_overlapping_strides(tensor) &&
         tensor.storage_elements == tensor.element_count;
}

std::array<std::int64_t, 8>
pointwise_dimensions(const RequestTensorMetadata &output) {
  std::array<std::int64_t, 8> result{};
  result.fill(1);
  const std::size_t leading = result.size() - output.dimensions.size();
  std::copy(output.dimensions.begin(), output.dimensions.end(),
            result.begin() + static_cast<std::ptrdiff_t>(leading));
  return result;
}

std::array<std::int64_t, 8>
pointwise_input_strides(const RequestTensorMetadata &input) {
  std::array<std::int64_t, 8> result{};
  const std::size_t leading = result.size() - input.dimensions.size();
  for (std::size_t axis = 0; axis < input.dimensions.size(); ++axis) {
    result[leading + axis] =
        input.dimensions[axis] == 1 ? 0 : input.strides[axis];
  }
  return result;
}

std::array<std::int64_t, 8>
pointwise_output_strides(const RequestTensorMetadata &output) {
  std::array<std::int64_t, 8> result{};
  const std::size_t leading = result.size() - output.strides.size();
  std::copy(output.strides.begin(), output.strides.end(),
            result.begin() + static_cast<std::ptrdiff_t>(leading));
  return result;
}

void validate_pointwise_constants(const HygonKernelArtifact &kernel,
                                  std::string_view function_name,
                                  const RequestGraphMetadata &graph,
                                  const RequestNodeMetadata &node,
                                  const flagdnn::native::json::Value &entry) {
  const std::optional<std::int64_t> expected_mode =
      binary_pointwise_mode(node.operation);
  if (!expected_mode.has_value()) {
    return;
  }
  if (node.inputs.size() != 2 || node.outputs.size() != 1 ||
      !node.n_elements.has_value() || !node.pointwise_mode.has_value() ||
      !node.alpha.has_value() || *node.pointwise_mode != *expected_mode) {
    invalid_artifact("pointwise request Graph IR semantics are invalid");
  }

  const RequestTensorMetadata &left =
      pointwise_tensor(graph, node.inputs, "left");
  const RequestTensorMetadata &right =
      pointwise_tensor(graph, node.inputs, "right");
  const RequestTensorMetadata &output =
      pointwise_tensor(graph, node.outputs, "output");
  if (*node.n_elements <= 0 ||
      static_cast<std::uint64_t>(*node.n_elements) != output.element_count ||
      *node.n_elements > std::numeric_limits<std::int32_t>::max()) {
    invalid_artifact("pointwise request n_elements is inconsistent");
  }

  const bool dense =
      left.dimensions == output.dimensions &&
      right.dimensions == output.dimensions && left.strides == output.strides &&
      right.strides == output.strides && is_physically_dense(left) &&
      is_physically_dense(right) && is_physically_dense(output);
  const bool activation_backward = node.operation.ends_with("_backward");
  const std::string_view expected_function =
      activation_backward
          ? (dense ? "activation_backward_contiguous_kernel"
                   : "activation_backward_strided_kernel")
          : (dense ? "binary_contiguous_kernel" : "binary_strided_kernel");
  if (function_name != expected_function) {
    invalid_artifact(
        "pointwise kernel function does not match request Graph IR layout");
  }

  if (kernel.arguments.size() != 4 ||
      kernel.arguments[0].uid != node.inputs.at("left") ||
      kernel.arguments[1].uid != node.inputs.at("right") ||
      kernel.arguments[2].uid != node.outputs.at("output") ||
      kernel.arguments[3].kind != ArgumentKind::kScalarI32 ||
      kernel.arguments[3].semantic_name != "n_elements" ||
      kernel.arguments[3].scalar_i32 != *node.n_elements) {
    invalid_artifact(
        "pointwise argument ABI does not match request Graph IR semantics");
  }

  const std::vector<std::string_view> tokens =
      full_signature_tokens(kernel.full_signature);
  const std::size_t constant_offset = dense ? 4 : 36;
  const std::size_t expected_token_count =
      (dense ? 7 : 39) + (activation_backward ? (dense ? 8 : 7) : 0);
  if (tokens.size() != expected_token_count) {
    invalid_artifact(
        "pointwise full signature does not match request Graph IR layout");
  }

  if (!dense) {
    const auto dimensions = pointwise_dimensions(output);
    const auto left_strides = pointwise_input_strides(left);
    const auto right_strides = pointwise_input_strides(right);
    const auto output_strides = pointwise_output_strides(output);
    for (std::size_t axis = 0; axis < 8; ++axis) {
      if (signature_integer(tokens[4 + axis],
                            "pointwise DIM signature is invalid") !=
              dimensions[axis] ||
          signature_integer(tokens[12 + axis],
                            "pointwise LEFT_STRIDE signature is invalid") !=
              left_strides[axis] ||
          signature_integer(tokens[20 + axis],
                            "pointwise RIGHT_STRIDE signature is invalid") !=
              right_strides[axis] ||
          signature_integer(tokens[28 + axis],
                            "pointwise OUTPUT_STRIDE signature is invalid") !=
              output_strides[axis]) {
        invalid_artifact(
            "pointwise DIM/STRIDE constants do not match request Graph IR");
      }
    }
  }

  if (signature_integer(tokens[constant_offset],
                        "pointwise OP_KIND signature is invalid") !=
      *expected_mode) {
    invalid_artifact("pointwise OP_KIND does not match request Graph IR");
  }
  const double alpha = signature_double(tokens[constant_offset + 1],
                                        "pointwise ALPHA signature is invalid");
  if (std::bit_cast<std::uint64_t>(alpha) !=
      std::bit_cast<std::uint64_t>(*node.alpha)) {
    invalid_artifact("pointwise ALPHA does not match request Graph IR");
  }
  const std::int64_t block_size = signature_integer(
      tokens[constant_offset + 2], "pointwise BLOCK_SIZE signature is invalid");
  if (block_size <= 0 || block_size > 65536 ||
      (block_size & (block_size - 1)) != 0 ||
      (!graph.autotune && block_size != 256)) {
    invalid_artifact(
        "pointwise BLOCK_SIZE does not match compiler tuning contract");
  }
  if (activation_backward) {
    if (left.dimensions != output.dimensions ||
        right.dimensions != output.dimensions) {
      invalid_artifact("activation backward tensor shapes must match");
    }
    const std::array<std::pair<std::string_view, double>, 7> attributes = {
        {{"relu_lower_clip_slope", 0.0},
         {"relu_lower_clip", 0.0},
         {"relu_upper_clip", 0.0},
         {"has_upper_clip", 0.0},
         {"swish_beta", 1.0},
         {"elu_alpha", 1.0},
         {"softplus_beta", 1.0}}};
    for (std::size_t index = 0; index < attributes.size(); ++index) {
      const auto [name, default_value] = attributes[index];
      const auto found = node.numeric_attributes.find(std::string(name));
      const double expected = found == node.numeric_attributes.end()
                                  ? default_value
                                  : found->second;
      const auto token = tokens[constant_offset + 3 + index];
      if (name == "has_upper_clip") {
        if ((expected != 0.0 && expected != 1.0) ||
            token != (expected == 1.0 ? "true" : "false")) {
          invalid_artifact("activation clip flag does not match Graph IR");
        }
      } else if (signature_double(token, "invalid activation attribute") !=
                 expected) {
        invalid_artifact("activation attribute does not match Graph IR");
      }
    }
    if (dense && tokens.back() != "true") {
      invalid_artifact("activation tail mask does not match compiler contract");
    }
  }
  if (graph.autotune) {
    const auto config = entry.as_object().find("config");
    if (config == entry.as_object().end()) {
      invalid_artifact("pointwise autotune variant has no config metadata");
    }
    const auto &config_object = config->second.as_object();
    const auto &meta = config->second.at("META").as_object();
    if (meta.size() != 1 || !meta.contains("BLOCK_SIZE") ||
        meta.at("BLOCK_SIZE").as_int() != block_size ||
        config_object.at("num_warps").as_int() != kernel.num_warps ||
        config_object.at("num_stages").as_int() != kernel.num_stages) {
      invalid_artifact(
          "pointwise autotune config does not match full signature");
    }
  }
  const std::uint64_t expected_grid =
      (static_cast<std::uint64_t>(*node.n_elements) +
       static_cast<std::uint64_t>(block_size) - 1) /
      static_cast<std::uint64_t>(block_size);
  if (kernel.grid != std::array<unsigned int, 3>{
                         static_cast<unsigned int>(expected_grid), 1, 1}) {
    invalid_artifact("pointwise launch grid does not match request Graph IR");
  }
}

void validate_kernel_semantics(const HygonKernelArtifact &kernel,
                               std::string_view function_name,
                               const RequestGraphMetadata &graph,
                               const RequestNodeMetadata &node,
                               const flagdnn::native::json::Value &entry) {
  if (!kernel_function_allowed(node.operation, function_name)) {
    invalid_artifact(
        "kernel function is not allowed for the request Graph operation");
  }

  std::set<std::string> scalar_names;
  for (const ArgumentSpec &argument : kernel.arguments) {
    if (argument.kind == ArgumentKind::kScalarI32) {
      if (!scalar_names.insert(argument.semantic_name).second) {
        invalid_artifact("kernel scalar argument names must be unique");
      }
      const auto expected =
          node.integer_attributes.find(argument.semantic_name);
      if (expected != node.integer_attributes.end() &&
          argument.scalar_i32 != expected->second) {
        invalid_artifact(
            "kernel int32 argument does not match request Graph attribute");
      }
      continue;
    }
    if (argument.kind == ArgumentKind::kScalarF32) {
      if (!scalar_names.insert(argument.semantic_name).second) {
        invalid_artifact("kernel scalar argument names must be unique");
      }
      const auto expected =
          node.numeric_attributes.find(argument.semantic_name);
      if (expected != node.numeric_attributes.end() &&
          std::bit_cast<std::uint32_t>(argument.scalar_f32) !=
              std::bit_cast<std::uint32_t>(
                  checked_f32(expected->second, "request Graph attribute"))) {
        invalid_artifact(
            "kernel float32 argument does not match request Graph attribute");
      }
      continue;
    }

    const auto input = node.inputs.find(argument.semantic_name);
    const auto output = node.outputs.find(argument.semantic_name);
    const auto tensor = graph.tensors.find(argument.uid);
    if (tensor != graph.tensors.end()) {
      const bool input_matches =
          input != node.inputs.end() && input->second == argument.uid;
      const bool output_matches =
          output != node.outputs.end() && output->second == argument.uid;
      if (!input_matches && !output_matches) {
        invalid_artifact(
            "kernel tensor role/UID does not match request Graph port");
      }
    } else if (input != node.inputs.end() || output != node.outputs.end()) {
      invalid_artifact(
          "kernel internal workspace role shadows a request Graph port");
    }
  }

  if (binary_pointwise_mode(node.operation).has_value()) {
    validate_pointwise_constants(kernel, function_name, graph, node, entry);
    return;
  }

  const bool unary = node.inputs.size() == 1 && node.inputs.contains("input") &&
                     node.outputs.size() == 1 &&
                     node.outputs.contains("output") &&
                     is_unary_pointwise_operation(node.operation);
  if (unary) {
    const RequestTensorMetadata &input =
        pointwise_tensor(graph, node.inputs, "input");
    const RequestTensorMetadata &output =
        pointwise_tensor(graph, node.outputs, "output");
    const bool dense = input.dimensions == output.dimensions &&
                       input.strides == output.strides &&
                       is_physically_dense(input) &&
                       is_physically_dense(output);
    std::string_view expected_function =
        dense ? "unary_pointwise_contiguous_kernel"
              : "unary_pointwise_strided_kernel";
    const std::size_t pack_factor =
        input.pointer_token == "*fp32" ? 2
        : (input.pointer_token == "*fp16" || input.pointer_token == "*bf16")
            ? 4
            : 1;
    if (node.operation == "identity" && pack_factor > 1 && dense &&
        output.element_count >= 4096 &&
        output.element_count % pack_factor == 0 && input.alignment >= 8 &&
        output.alignment >= 8) {
      expected_function = "identity_packed_contiguous_kernel";
    }
    if (function_name != expected_function || kernel.arguments.size() != 3 ||
        kernel.arguments[0].semantic_name != "input" ||
        kernel.arguments[1].semantic_name != "output" ||
        kernel.arguments[2].kind != ArgumentKind::kScalarI32 ||
        kernel.arguments[2].semantic_name != "n_elements" ||
        kernel.arguments[2].scalar_i32 !=
            static_cast<std::int64_t>(output.element_count)) {
      invalid_artifact(
          "unary pointwise ABI does not match request Graph semantics");
    }
  } else if (node.operation == "binary_select") {
    if (kernel.arguments.size() != 5 ||
        kernel.arguments[0].semantic_name != "a" ||
        kernel.arguments[1].semantic_name != "b" ||
        kernel.arguments[2].semantic_name != "t" ||
        kernel.arguments[3].semantic_name != "output" ||
        kernel.arguments[4].kind != ArgumentKind::kScalarI32 ||
        kernel.arguments[4].semantic_name != "n_elements") {
      invalid_artifact(
          "ternary pointwise ABI does not match request Graph semantics");
    }
  }
}

bool storage_pointer_matches(std::string_view actual, std::string_view expected,
                             std::string_view function) {
  if (actual == expected)
    return true;
  if (function != "concatenate_copy_kernel" && function != "layout_copy_kernel")
    return false;
  return (actual == "*i32" && expected == "*fp32") ||
         (actual == "*u16" && (expected == "*fp16" || expected == "*bf16")) ||
         (actual == "*u8" && (expected == "*i8" || expected == "*fp8e4nv" ||
                              expected == "*fp8e5"));
}

void validate_argument_abi(const HygonKernelArtifact &kernel,
                           const RequestGraphMetadata &graph,
                           const std::set<std::int64_t> &source_tensor_uids,
                           std::string_view function_name) {
  const std::vector<std::string_view> signature_tokens =
      runtime_signature_tokens(kernel.full_signature);
  if (signature_tokens.size() != kernel.arguments.size()) {
    invalid_artifact("libtriton_jit full signature and argument ABI disagree");
  }

  for (std::size_t index = 0; index < kernel.arguments.size(); ++index) {
    const ArgumentSpec &argument = kernel.arguments[index];
    const std::string_view signature_token = signature_tokens[index];
    if (argument.kind == ArgumentKind::kScalarI32) {
      if (signature_token != "i32") {
        invalid_artifact("libtriton_jit int32 argument signature is invalid");
      }
      continue;
    }
    if (argument.kind == ArgumentKind::kScalarF32) {
      if (signature_token != "fp32") {
        invalid_artifact("libtriton_jit float32 argument signature is invalid");
      }
      continue;
    }
    if (!signature_token.starts_with('*')) {
      invalid_artifact("libtriton_jit pointer argument signature is invalid");
    }
    const std::size_t annotation = signature_token.find(':');
    const std::string_view specialization =
        annotation == std::string_view::npos
            ? std::string_view{}
            : signature_token.substr(annotation);
    const bool aligned_signature =
        specialization == ":16" || specialization == ":16S";
    const bool range32_signature =
        specialization == ":S" || specialization == ":16S";
    if (specialization != std::string_view{} &&specialization != ":16" &&
        specialization != ":S" && specialization != ":16S") {
      invalid_artifact("libtriton_jit pointer specialization is invalid");
    }
    if ((argument.alignment >= 16) != aligned_signature) {
      invalid_artifact("libtriton_jit pointer alignment signature is invalid");
    }
    const bool storage_fits_range32 =
        argument.storage_size != 0 &&
        argument.storage_size <= static_cast<std::uint64_t>(
                                     std::numeric_limits<std::int32_t>::max());
    // The pointer-range specialization is an optional optimization. HCU
    // Triton 3.1/3.2 cannot encode it, while 3.3 and newer can. A present S
    // marker is therefore required to be truthful, but its absence must not
    // make an otherwise identical artifact invalid.
    if (range32_signature && !storage_fits_range32) {
      invalid_artifact("libtriton_jit pointer range signature is invalid");
    }
    const std::string_view pointer_token =
        signature_token.substr(0, annotation);

    const auto tensor = graph.tensors.find(argument.uid);
    if (argument.kind == ArgumentKind::kTensor) {
      if (tensor == graph.tensors.end() || tensor->second.is_virtual ||
          !source_tensor_uids.contains(argument.uid) ||
          argument.storage_size != tensor->second.storage_size ||
          argument.alignment != tensor->second.alignment ||
          !storage_pointer_matches(pointer_token, tensor->second.pointer_token,
                                   function_name)) {
        invalid_artifact(
            "artifact external tensor ABI does not match request Graph IR");
      }
      continue;
    }
    if (argument.kind != ArgumentKind::kWorkspaceTensor) {
      invalid_artifact("artifact argument kind is invalid");
    }
    if (tensor != graph.tensors.end()) {
      if (!tensor->second.is_virtual ||
          !source_tensor_uids.contains(argument.uid) ||
          argument.workspace_offset != tensor->second.workspace_offset ||
          argument.storage_size != tensor->second.storage_size ||
          argument.alignment !=
              std::max(kWorkspaceAlignment, tensor->second.alignment) ||
          !storage_pointer_matches(pointer_token, tensor->second.pointer_token,
                                   function_name)) {
        invalid_artifact(
            "artifact virtual tensor ABI does not match request Graph IR");
      }
    } else if (argument.uid <= graph.maximum_tensor_uid) {
      invalid_artifact("artifact internal workspace tensor UID is invalid");
    }
  }
}

HygonKernelArtifact parse_kernel(
    EngineKind engine, const flagdnn::native::json::Value &entry,
    const std::filesystem::path &artifact_directory, std::size_t workspace_size,
    std::string_view expected_source_hash, const RequestGraphMetadata &graph,
    const std::set<std::int64_t> &source_tensor_uids,
    std::string_view function_name, const RequestNodeMetadata *semantic_node) {
  const std::string kernel_source_hash = entry.at("source_sha256").as_string();
  if (!is_sha256(kernel_source_hash) ||
      kernel_source_hash != expected_source_hash) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "artifact kernel source identity does not match stage");
  }

  HygonKernelArtifact result;
  const auto &entry_object = entry.as_object();
  const auto variant_entry = entry_object.find("variant_id");
  if (variant_entry != entry_object.end()) {
    result.variant_id = variant_entry->second.as_string();
  }
  if (result.variant_id.empty() || result.variant_id.size() > 128 ||
      std::any_of(result.variant_id.begin(), result.variant_id.end(),
                  [](const unsigned char character) {
                    return std::isalnum(character) == 0 && character != '_' &&
                           character != '-';
                  })) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "artifact variant ID is invalid");
  }

  if (engine == EngineKind::kExternalArtifact) {
    result.binary = validate_file(artifact_directory, entry.at("binary"),
                                  1U << 30, "binary");
    result.entry_symbol = entry.at("entry_symbol").as_string();
    if (result.entry_symbol.empty() || result.entry_symbol.size() > 1024) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact entry symbol is invalid");
    }
  } else {
    result.full_signature = entry.at("full_signature").as_string();
    if (result.full_signature.empty() ||
        result.full_signature.size() > (64U << 10) ||
        std::any_of(result.full_signature.begin(), result.full_signature.end(),
                    [](const unsigned char character) {
                      return character < 0x21U || character > 0x7eU;
                    })) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "libtriton_jit full signature is invalid");
    }
    const auto &options = entry.at("compile_options");
    result.num_warps = checked_positive_unsigned(
        options.at("num_warps").as_int(), "compile_options.num_warps");
    result.num_stages = checked_positive_unsigned(
        options.at("num_stages").as_int(), "compile_options.num_stages");
    if (result.num_warps > 32 || result.num_stages > 32) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "libtriton_jit compile options exceed safety limits");
    }
  }

  result.arguments = parse_argument_abi(entry.at("argument_abi"),
                                        workspace_size, result.binding_uids);
  const auto &launch = entry.at("launch");
  result.grid = parse_triplet(launch.at("grid"), "launch.grid");
  result.block = parse_triplet(launch.at("block"), "launch.block");
  const auto cluster = parse_triplet(launch.at("cluster"), "launch.cluster");
  const std::uint64_t block_threads =
      static_cast<std::uint64_t>(result.block[0]) * result.block[1] *
      result.block[2];
  if (block_threads > 1024) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "HIP block contains more than 1024 threads");
  }
  if (cluster != std::array<unsigned int, 3>{1, 1, 1} ||
      launch.at("num_ctas").as_int() != 1) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                     "Hygon backend v2 does not support cluster launch");
  }
  result.shared_memory = checked_nonnegative_unsigned(
      launch.at("shared_memory").as_int(), "launch.shared_memory");
  result.global_scratch_size = checked_size(
      launch.at("global_scratch_size").as_int(), "launch.global_scratch_size");
  result.profile_scratch_size =
      checked_size(launch.at("profile_scratch_size").as_int(),
                   "launch.profile_scratch_size");
  if (engine == EngineKind::kExternalArtifact &&
      (result.global_scratch_size != 0 || result.profile_scratch_size != 0)) {
    throw HygonError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "external Hygon artifacts do not support scratch buffers yet");
  }
  if (engine == EngineKind::kLibTritonJit &&
      (result.global_scratch_size == 0 ||
       result.global_scratch_size > workspace_size ||
       result.global_scratch_size % 256 != 0 ||
       result.profile_scratch_size != 0)) {
    throw HygonError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "libtriton_jit scratch metadata is incompatible with workspace");
  }
  const bool valid_jit_block = result.block[1] == 1U && result.block[2] == 1U &&
                               result.block[0] == result.num_warps * 64U;
  if (engine == EngineKind::kLibTritonJit &&
      (!valid_jit_block || result.shared_memory != 0)) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "libtriton_jit launch plan is inconsistent");
  }
  if (engine == EngineKind::kLibTritonJit) {
    validate_argument_abi(result, graph, source_tensor_uids, function_name);
    if (semantic_node != nullptr) {
      validate_kernel_semantics(result, function_name, graph, *semantic_node,
                                entry);
    }
  }
  return result;
}

bool same_argument_abi(const HygonKernelArtifact &left,
                       const HygonKernelArtifact &right) {
  if (left.binding_uids != right.binding_uids ||
      left.arguments.size() != right.arguments.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.arguments.size(); ++index) {
    const ArgumentSpec &a = left.arguments[index];
    const ArgumentSpec &b = right.arguments[index];
    if (a.kind != b.kind || a.uid != b.uid || a.scalar_i32 != b.scalar_i32 ||
        a.scalar_f32 != b.scalar_f32 ||
        a.workspace_offset != b.workspace_offset ||
        a.storage_size != b.storage_size || a.alignment != b.alignment) {
      return false;
    }
    if (a.semantic_name != b.semantic_name) {
      return false;
    }
  }
  return true;
}

void append_canonical_json_string(std::string &output, std::string_view value) {
  output.push_back('"');
  for (const unsigned char character : value) {
    if (character == '"' || character == '\\') {
      output.push_back('\\');
      output.push_back(static_cast<char>(character));
    } else if (character < 0x20U || character > 0x7eU) {
      invalid_artifact(
          "autotune identity contains a noncanonical string character");
    } else {
      output.push_back(static_cast<char>(character));
    }
  }
  output.push_back('"');
}

void append_canonical_triplet(std::string &output,
                              const std::array<unsigned int, 3> &values) {
  output.push_back('[');
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      output.push_back(',');
    }
    output += std::to_string(values[index]);
  }
  output.push_back(']');
}

std::string canonical_autotune_candidate_identity(
    std::string_view base_candidate_identity,
    std::string_view stage_source_hash,
    const std::vector<HygonKernelArtifact> &variants) {
  std::string output;
  output.reserve(1024 + variants.size() * 1024);
  output += "{\"base_candidate_identity\":";
  append_canonical_json_string(output, base_candidate_identity);
  output += ",\"engine\":\"libtriton_jit\",\"schema_version\":1,";
  output += "\"variants\":[";
  for (std::size_t index = 0; index < variants.size(); ++index) {
    if (index != 0) {
      output.push_back(',');
    }
    const HygonKernelArtifact &variant = variants[index];
    output += "{\"compile_options\":{\"num_stages\":";
    output += std::to_string(variant.num_stages);
    output += ",\"num_warps\":";
    output += std::to_string(variant.num_warps);
    output += "},\"full_signature\":";
    append_canonical_json_string(output, variant.full_signature);
    output += ",\"launch\":{\"block\":";
    append_canonical_triplet(output, variant.block);
    output += ",\"cluster\":[1,1,1],\"global_scratch_size\":";
    output += std::to_string(variant.global_scratch_size);
    output += ",\"grid\":";
    append_canonical_triplet(output, variant.grid);
    output += ",\"num_ctas\":1,\"profile_scratch_size\":";
    output += std::to_string(variant.profile_scratch_size);
    output += ",\"shared_memory\":";
    output += std::to_string(variant.shared_memory);
    output += "},\"source_sha256\":";
    append_canonical_json_string(output, stage_source_hash);
    output += ",\"variant_id\":";
    append_canonical_json_string(output, variant.variant_id);
    output.push_back('}');
  }
  output += "]}";
  return output;
}

EngineKind parse_engine(std::string_view value) {
  if (value == "external_artifact") {
    return EngineKind::kExternalArtifact;
  }
  if (value == "libtriton_jit") {
    return EngineKind::kLibTritonJit;
  }
  throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                   "artifact execution engine is invalid");
}

} // namespace

HygonArtifact parse_hygon_artifact(const EngineBuildContext &context,
                                   const flagdnnBackendBuildInputV2 &input) {
  try {
    require(input.graph_ir != nullptr && input.graph_ir_size != 0,
            "graph IR is empty");
    require(input.artifact_directory != nullptr, "artifact directory is null");
    require(input.request_sha256 != nullptr, "request SHA-256 is null");
    const std::string_view graph_ir(static_cast<const char *>(input.graph_ir),
                                    input.graph_ir_size);
    const std::string_view request_hash(input.request_sha256);
    require(is_sha256(request_hash) &&
                flagdnn::native::sha256(graph_ir) == request_hash,
            "graph IR SHA-256 does not match build input",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    const auto request_root = flagdnn::native::json::parse(graph_ir);
    const std::string compiler_identity =
        request_root.at("compiler_identity").as_string();
    if (request_root.at("schema_version").as_int() != 3 ||
        request_root.at("flagdnn_version").as_string() !=
            FLAGDNN_VERSION_STRING ||
        request_root.at("backend").as_string() != "hygon" ||
        request_root.at("target").as_string() != context.target_fingerprint ||
        !is_sha256(compiler_identity)) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "HIP build request identity is invalid");
    }
    const RequestGraphMetadata request_graph =
        parse_request_graph(request_root);

    const std::filesystem::path artifact_directory(input.artifact_directory);
    const auto root = flagdnn::native::json::parse(
        read_text_file(artifact_directory / "manifest.json", 16U << 20));
    if (root.at("schema_version").as_int() != kArtifactSchemaVersion ||
        root.at("artifact_kind").as_string() != "flagdnn_execution_program" ||
        root.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
        root.at("backend").as_string() != "hygon" ||
        root.at("target").as_string() != context.target_fingerprint ||
        root.at("request_sha256").as_string() != request_hash ||
        root.at("compiler").at("identity_sha256").as_string() !=
            compiler_identity) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact target or version does not match build input");
    }
    const std::string source_hash = root.at("source_sha256").as_string();
    if (!is_sha256(source_hash) ||
        root.at("compiler").at("provider").as_string().empty() ||
        root.at("compiler").at("triton_version").as_string().empty()) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact compiler/source identity is invalid");
    }

    HygonArtifact result;
    result.workspace_size =
        checked_size(root.at("workspace_size").as_int(), "workspace_size");
    result.workspace_alignment = checked_size(
        root.at("workspace_alignment").as_int(), "workspace_alignment");
    if (result.workspace_alignment < kWorkspaceAlignment ||
        result.workspace_alignment > (1ULL << 31) ||
        (result.workspace_alignment & (result.workspace_alignment - 1)) != 0) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact workspace alignment is invalid");
    }
    const std::size_t graph_node_count =
        checked_size(root.at("graph_node_count").as_int(), "graph_node_count");
    if (graph_node_count != request_graph.nodes.size()) {
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "artifact graph node count does not match request Graph IR");
    }
    const auto &program = root.at("program");
    if (program.at("schema_version").as_int() != kExecutionProgramVersion) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "HIP Execution Program version is unsupported");
    }
    const auto &stages = program.at("stages").as_array();
    const std::size_t stage_count =
        checked_size(program.at("stage_count").as_int(), "stage_count");
    if (stage_count == 0 ||
        stage_count > FLAGDNN_BACKEND_MAX_EXECUTION_STAGES ||
        stages.size() != stage_count) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact Execution Program stage count is invalid");
    }

    result.stages.reserve(stage_count);
    bool engine_initialized = false;
    std::set<std::size_t> covered_source_nodes;
    struct InternalWorkspaceContract {
      std::string role;
      std::size_t offset = 0;
      std::size_t size = 0;
      std::size_t alignment = 1;
    };
    std::map<std::int64_t, InternalWorkspaceContract> internal_workspaces;
    std::map<std::pair<std::size_t, std::string>, std::int64_t>
        internal_workspace_roles;
    std::vector<std::string> stage_source_hashes;
    stage_source_hashes.reserve(stage_count);
    for (std::size_t index = 0; index < stages.size(); ++index) {
      const auto &stage = stages[index];
      if (checked_size(stage.at("stage_id").as_int(), "stage_id") != index ||
          stage.at("kind").as_string() != "kernel") {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "artifact Execution Program stage is invalid");
      }
      const auto &source_nodes = stage.at("source_node_ids").as_array();
      if (source_nodes.empty()) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "execution stage has no source graph nodes");
      }
      std::vector<std::size_t> seen_source_nodes;
      std::set<std::int64_t> source_tensor_uids;
      const std::string &stage_operation = stage.at("operation").as_string();
      if (stage_operation.empty()) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "execution stage operation is empty");
      }
      for (const auto &source_node : source_nodes) {
        const std::size_t node_id =
            checked_size(source_node.as_int(), "source_node_id");
        const auto request_node = request_graph.nodes.find(node_id);
        if (request_node == request_graph.nodes.end() ||
            request_node->second.operation != stage_operation ||
            std::find(seen_source_nodes.begin(), seen_source_nodes.end(),
                      node_id) != seen_source_nodes.end()) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "execution stage source nodes do not match request Graph IR");
        }
        seen_source_nodes.push_back(node_id);
        covered_source_nodes.insert(node_id);
        source_tensor_uids.insert(request_node->second.tensor_uids.begin(),
                                  request_node->second.tensor_uids.end());
      }
      if (seen_source_nodes.size() != 1) {
        throw HygonError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "Hygon execution stage must cover exactly one request node");
      }
      const RequestNodeMetadata *semantic_node =
          &request_graph.nodes.at(seen_source_nodes.front());
      const auto &dependencies = stage.at("dependencies").as_array();
      std::vector<std::size_t> seen_dependencies;
      for (const auto &dependency : dependencies) {
        const std::size_t dependency_id =
            checked_size(dependency.as_int(), "stage dependency");
        if (dependency_id >= index ||
            std::find(seen_dependencies.begin(), seen_dependencies.end(),
                      dependency_id) != seen_dependencies.end()) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "execution stage dependency list is invalid");
        }
        seen_dependencies.push_back(dependency_id);
      }

      const EngineKind stage_engine =
          parse_engine(stage.at("engine").as_string());
      if (!engine_initialized) {
        result.engine = stage_engine;
        engine_initialized = true;
      } else if (result.engine != stage_engine) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                         "mixed execution engines are not supported yet");
      }

      HygonStageArtifact parsed_stage;
      parsed_stage.selection_cache =
          artifact_directory /
          (".flagdnn-autotune-v1-stage-" + std::to_string(index) + "-" +
           context.device_identity + ".json");
      const std::string stage_source_hash =
          stage.at("source_sha256").as_string();
      if (!is_sha256(stage_source_hash)) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "execution stage source identity is invalid");
      }
      stage_source_hashes.push_back(stage_source_hash);
      if (stage_engine == EngineKind::kLibTritonJit) {
        const auto &kernel = stage.at("kernel");
        parsed_stage.source =
            validate_file(artifact_directory, kernel.at("materialized_source"),
                          1U << 20, "JIT source");
        parsed_stage.materialized_source_sha256 =
            kernel.at("materialized_source").at("sha256").as_string();
        parsed_stage.materialized_source =
            read_text_file(parsed_stage.source, 1U << 20);
        if (flagdnn::native::sha256(parsed_stage.materialized_source) !=
            parsed_stage.materialized_source_sha256) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "JIT source changed after artifact validation");
        }
        if (parsed_stage.materialized_source_sha256 != stage_source_hash) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "JIT materialized source identity does not match execution "
              "stage");
        }
        parsed_stage.function_name = kernel.at("function").as_string();
        if (!is_identifier(parsed_stage.function_name)) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "libtriton_jit function name is invalid");
        }
      }

      const auto &stage_object = stage.as_object();
      const auto variants_entry = stage_object.find("variants");
      if (stage_engine == EngineKind::kLibTritonJit &&
          variants_entry != stage_object.end() && !request_graph.autotune) {
        throw HygonError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "artifact autotune plan does not match request Graph IR");
      }
      if (variants_entry == stage_object.end()) {
        parsed_stage.variants.push_back(parse_kernel(
            stage_engine, stage, artifact_directory, result.workspace_size,
            stage.at("source_sha256").as_string(), request_graph,
            source_tensor_uids, parsed_stage.function_name, semantic_node));
      } else {
        parsed_stage.autotune = true;
        const auto &variants = variants_entry->second.as_array();
        if (variants.size() < 2 ||
            variants.size() > kMaximumAutotuneCandidates) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "autotune stage candidate count must be in [2, 1024]");
        }
        const auto &tuning = stage.at("tuning");
        if (tuning.at("schema_version").as_int() != 1) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "autotune metadata schema is unsupported");
        }
        parsed_stage.warmup = checked_nonnegative_unsigned(
            tuning.at("warmup").as_int(), "tuning.warmup");
        parsed_stage.repetitions = checked_positive_unsigned(
            tuning.at("repetitions").as_int(), "tuning.repetitions");
        parsed_stage.candidate_identity =
            tuning.at("candidate_identity").as_string();
        const std::string base_candidate_identity =
            tuning.at("base_candidate_identity").as_string();
        const std::string tuning_source_hash =
            tuning.at("source_sha256").as_string();
        if (parsed_stage.warmup > 100 || parsed_stage.repetitions > 100 ||
            !is_sha256(parsed_stage.candidate_identity) ||
            !is_sha256(base_candidate_identity) ||
            !is_sha256(tuning_source_hash) ||
            tuning.at("key").as_string().empty() ||
            tuning.at("strategy").as_string().empty()) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "autotune metadata is invalid");
        }
        parsed_stage.variants.reserve(variants.size());
        for (const auto &variant : variants) {
          HygonKernelArtifact candidate = parse_kernel(
              stage_engine, variant, artifact_directory, result.workspace_size,
              stage.at("source_sha256").as_string(), request_graph,
              source_tensor_uids, parsed_stage.function_name, semantic_node);
          if (std::any_of(parsed_stage.variants.begin(),
                          parsed_stage.variants.end(),
                          [&](const HygonKernelArtifact &existing) {
                            return existing.variant_id == candidate.variant_id;
                          })) {
            throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                             "autotune variant IDs must be unique");
          }
          if (!parsed_stage.variants.empty() &&
              !same_argument_abi(candidate, parsed_stage.variants.front())) {
            throw HygonError(
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                "autotune variants have incompatible argument ABIs");
          }
          parsed_stage.variants.push_back(std::move(candidate));
        }
        const std::string rendered_identity =
            canonical_autotune_candidate_identity(base_candidate_identity,
                                                  stage_source_hash,
                                                  parsed_stage.variants);
        if (flagdnn::native::sha256(rendered_identity) !=
            parsed_stage.candidate_identity) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "autotune candidate identity does not match variants");
        }
      }

      for (const std::int64_t uid :
           parsed_stage.variants.front().binding_uids) {
        if (std::find(result.binding_uids.begin(), result.binding_uids.end(),
                      uid) == result.binding_uids.end()) {
          result.binding_uids.push_back(uid);
        }
      }
      for (const ArgumentSpec &argument :
           parsed_stage.variants.front().arguments) {
        if (argument.kind != ArgumentKind::kWorkspaceTensor ||
            request_graph.tensors.contains(argument.uid)) {
          continue;
        }
        const InternalWorkspaceContract contract{
            argument.semantic_name, argument.workspace_offset,
            argument.storage_size, argument.alignment};
        const auto [entry, inserted] =
            internal_workspaces.emplace(argument.uid, contract);
        if (!inserted && (entry->second.role != contract.role ||
                          entry->second.offset != contract.offset ||
                          entry->second.size != contract.size ||
                          entry->second.alignment != contract.alignment)) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "internal workspace tensor ABI differs across execution stages");
        }
        const auto [role_entry, role_inserted] =
            internal_workspace_roles.emplace(
                std::make_pair(seen_source_nodes.front(), contract.role),
                argument.uid);
        if (!role_inserted && role_entry->second != argument.uid) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "internal workspace role resolves to multiple tensor UIDs");
        }
      }
      result.stages.push_back(std::move(parsed_stage));
    }
    if (covered_source_nodes.size() != request_graph.nodes.size()) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact does not cover every request graph node");
    }
    std::string source_identity = "[";
    for (std::size_t index = 0; index < stage_source_hashes.size(); ++index) {
      if (index != 0) {
        source_identity += ',';
      }
      source_identity += '\"';
      source_identity += stage_source_hashes[index];
      source_identity += '\"';
    }
    source_identity += ']';
    if (stage_source_hashes.size() != result.stages.size() ||
        flagdnn::native::sha256(source_identity) != source_hash) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact root source identity is inconsistent");
    }
    const std::set<std::int64_t> artifact_bindings(result.binding_uids.begin(),
                                                   result.binding_uids.end());
    if (artifact_bindings.size() != result.binding_uids.size() ||
        artifact_bindings != request_graph.external_binding_uids) {
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "artifact bindings do not match external request tensors");
    }
    if (result.binding_uids.empty()) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "artifact has no external tensor bindings");
    }
    if (result.engine == EngineKind::kLibTritonJit) {
      std::size_t global_scratch_size = 0;
      std::size_t required_workspace_alignment =
          request_graph.workspace_alignment;
      for (const HygonStageArtifact &stage : result.stages) {
        for (const HygonKernelArtifact &variant : stage.variants) {
          global_scratch_size =
              std::max(global_scratch_size, variant.global_scratch_size);
          for (const ArgumentSpec &argument : variant.arguments) {
            if (argument.kind == ArgumentKind::kWorkspaceTensor) {
              required_workspace_alignment =
                  std::max(required_workspace_alignment, argument.alignment);
            }
          }
        }
      }
      if (result.workspace_alignment != required_workspace_alignment) {
        throw HygonError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "artifact workspace alignment does not match workspace ABI");
      }
      if (result.workspace_size == 0 ||
          result.workspace_size % kWorkspaceAlignment != 0 ||
          global_scratch_size == 0 ||
          global_scratch_size > result.workspace_size ||
          global_scratch_size % kWorkspaceAlignment != 0) {
        throw HygonError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "artifact workspace/global scratch alignment is invalid");
      }
      const std::size_t global_scratch_offset =
          result.workspace_size - global_scratch_size;
      if (global_scratch_offset % kWorkspaceAlignment != 0 ||
          global_scratch_offset < request_graph.graph_workspace_size) {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "artifact global scratch overlaps graph workspace");
      }
      for (const HygonStageArtifact &stage : result.stages) {
        for (const HygonKernelArtifact &variant : stage.variants) {
          for (const ArgumentSpec &argument : variant.arguments) {
            if (argument.kind == ArgumentKind::kWorkspaceTensor &&
                (argument.workspace_offset > global_scratch_offset ||
                 argument.storage_size >
                     global_scratch_offset - argument.workspace_offset)) {
              throw HygonError(
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                  "artifact workspace tensor overlaps global scratch");
            }
          }
        }
      }
    }
    return result;
  } catch (const HygonError &) {
    throw;
  } catch (const std::exception &error) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "invalid HIP artifact manifest: " +
                         std::string(error.what()));
  }
}

} // namespace flagdnn::hygon
