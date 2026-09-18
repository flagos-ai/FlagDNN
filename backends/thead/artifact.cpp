// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/artifact.hpp"

#include "backends/thead/error.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::thead {
namespace {

namespace json = flagdnn::native::json;

constexpr std::size_t kMaximumDocumentBytes = 64U * 1024U * 1024U;
constexpr std::size_t kMaximumTensors = 4096;
constexpr std::size_t kMaximumStages = 65536;
constexpr std::size_t kMaximumArguments = FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS;
constexpr std::size_t kMaximumVariants = 256;
constexpr std::size_t kMaximumSharedMemory = 1U << 30U;
constexpr std::array<std::string_view, 6> kApprovedPpuCompilerOptions = {
    "debug",
    "enable_fp_fusion",
    "enable_reflect_ftz",
    "instrumentation_mode",
    "ppu_llc_options",
    "sanitize_overflow",
};

[[noreturn]] void artifact_error(std::string message) {
  throw TheadError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                   "invalid THead artifact: " + std::move(message));
}

void require_artifact(bool condition, std::string message) {
  if (!condition) {
    artifact_error(std::move(message));
  }
}

bool is_identifier(std::string_view value) {
  if (value.empty() || value.size() > 127 || value.front() < 'a' ||
      value.front() > 'z') {
    return false;
  }
  return std::all_of(value.begin() + 1, value.end(), [](char character) {
    return (character >= 'a' && character <= 'z') ||
           (character >= '0' && character <= '9') || character == '_';
  });
}

bool is_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

void require_fields(const json::Value::Object& object,
                    std::initializer_list<std::string_view> required,
                    std::initializer_list<std::string_view> optional,
                    std::string_view description) {
  std::set<std::string_view> allowed;
  for (const std::string_view field : required) {
    allowed.insert(field);
    if (!object.contains(field)) {
      artifact_error(std::string(description) + " is missing field " +
                     std::string(field));
    }
  }
  allowed.insert(optional.begin(), optional.end());
  for (const auto& [field, value] : object) {
    (void)value;
    if (!allowed.contains(field)) {
      artifact_error(std::string(description) + " has unknown field " + field);
    }
  }
}

std::size_t to_size(std::int64_t value, std::string_view description) {
  if (value < 0 ||
      static_cast<std::uint64_t>(value) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    artifact_error(std::string(description) + " is outside size_t range");
  }
  return static_cast<std::size_t>(value);
}

unsigned int to_unsigned(std::int64_t value,
                         std::string_view description,
                         unsigned int maximum =
                             std::numeric_limits<unsigned int>::max()) {
  if (value < 0 || static_cast<std::uint64_t>(value) > maximum) {
    artifact_error(std::string(description) + " is outside supported range");
  }
  return static_cast<unsigned int>(value);
}

std::size_t checked_add(std::size_t left,
                        std::size_t right,
                        std::string_view description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    artifact_error(std::string(description) + " overflow");
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left,
                             std::size_t right,
                             std::string_view description) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    artifact_error(std::string(description) + " overflow");
  }
  return left * right;
}

bool power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::string read_file(const std::filesystem::path& path,
                      std::size_t maximum_size,
                      std::string_view description) {
  std::error_code error;
  const auto status = std::filesystem::status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    artifact_error(std::string(description) + " is not a regular file");
  }
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > maximum_size) {
    artifact_error(std::string(description) + " exceeds the size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    artifact_error("cannot open " + std::string(description));
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) {
    artifact_error("cannot read " + std::string(description));
  }
  return result;
}

bool below_root(const std::filesystem::path& root,
                const std::filesystem::path& path) {
  const std::filesystem::path relative = path.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() &&
         relative != ".." &&
         (relative.empty() || *relative.begin() != "..");
}

std::filesystem::path safe_relative_path(
    const std::filesystem::path& root,
    std::string_view value,
    std::string_view description,
    bool must_exist) {
  require_artifact(!value.empty() && value.size() <= 4096,
                   std::string(description) + " is empty or too long");
  const std::filesystem::path relative{std::string(value)};
  require_artifact(!relative.is_absolute() && !relative.has_root_name() &&
                       !relative.has_root_directory(),
                   "unsafe " + std::string(description));
  for (const auto& component : relative) {
    require_artifact(component != ".." && component != "." &&
                         !component.empty(),
                     "unsafe " + std::string(description));
  }
  std::error_code error;
  std::filesystem::path resolved;
  if (must_exist) {
    resolved = std::filesystem::canonical(root / relative, error);
  } else {
    resolved = std::filesystem::weakly_canonical(root / relative, error);
  }
  require_artifact(!error && below_root(root, resolved),
                   std::string(description) + " escaped artifact root");
  if (must_exist) {
    const auto status = std::filesystem::status(resolved, error);
    require_artifact(!error && std::filesystem::is_regular_file(status),
                     std::string(description) + " is not a regular file");
  }
  return resolved;
}

std::size_t data_type_size(std::string_view data_type) {
  if (data_type == "float32" || data_type == "int32") {
    return 4;
  }
  if (data_type == "float16" || data_type == "bfloat16") {
    return 2;
  }
  if (data_type == "boolean" || data_type == "fp8_e4m3" ||
      data_type == "fp8_e5m2" || data_type == "fp8_e8m0") {
    return 1;
  }
  artifact_error("tensor data_type is unsupported: " + std::string(data_type));
}

std::vector<std::int64_t> integer_array(const json::Value& value,
                                        std::string_view description,
                                        std::size_t maximum_count) {
  const auto& array = value.as_array();
  require_artifact(array.size() <= maximum_count,
                   std::string(description) + " is too large");
  std::vector<std::int64_t> result;
  result.reserve(array.size());
  for (const auto& entry : array) {
    result.push_back(entry.as_int());
  }
  return result;
}

std::size_t tensor_storage_size(std::string_view data_type,
                                const std::vector<std::int64_t>& dimensions,
                                const std::vector<std::int64_t>& strides) {
  require_artifact(dimensions.size() == strides.size(),
                   "tensor dimensions/strides rank mismatch");
  require_artifact(dimensions.size() <= 8, "tensor rank exceeds 8");
  std::size_t maximum_offset = 0;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    require_artifact(dimensions[index] > 0,
                     "tensor dimension must be positive");
    require_artifact(strides[index] >= 0,
                     "tensor stride must be nonnegative");
    const std::size_t extent = to_size(dimensions[index] - 1,
                                       "tensor dimension extent");
    const std::size_t stride = to_size(strides[index], "tensor stride");
    maximum_offset =
        checked_add(maximum_offset,
                    checked_multiply(extent, stride, "tensor storage"),
                    "tensor storage");
  }
  return checked_multiply(checked_add(maximum_offset, 1, "tensor storage"),
                          data_type_size(data_type),
                          "tensor storage");
}

struct GraphTensor {
  std::int64_t uid = 0;
  std::string data_type;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t alignment = 1;
  bool is_virtual = false;
  std::size_t storage_size = 0;
};

struct GraphNode {
  std::int64_t id = 0;
  std::string operation;
  std::vector<std::int64_t> inputs;
  std::vector<std::int64_t> outputs;
};

struct GraphDocument {
  std::string version;
  std::string target;
  std::string compiler_identity;
  std::vector<GraphTensor> tensors;
  std::vector<GraphNode> nodes;
  std::vector<std::int64_t> external_uids;
};

std::vector<std::int64_t> parse_ports(
    const json::Value& value,
    const std::map<std::int64_t, std::size_t>& tensor_indices,
    std::string_view description) {
  const auto& ports = value.as_array();
  require_artifact(!ports.empty(), std::string(description) + " is empty");
  std::set<std::string> names;
  std::vector<std::int64_t> result;
  for (const auto& port_value : ports) {
    const auto& port = port_value.as_object();
    require_fields(port, {"name", "uid"}, {"optional"}, description);
    const std::string& name = port.at("name").as_string();
    require_artifact(is_identifier(name),
                     std::string(description) + " name is invalid");
    require_artifact(names.insert(name).second,
                     std::string(description) + " names are duplicated");
    const std::int64_t uid = port.at("uid").as_int();
    require_artifact(tensor_indices.contains(uid),
                     std::string(description) + " references unknown tensor");
    if (port.contains("optional")) {
      (void)port.at("optional").as_bool();
    }
    result.push_back(uid);
  }
  return result;
}

GraphDocument parse_graph(std::string_view graph_ir) {
  const json::Value root_value = json::parse(graph_ir);
  const auto& root = root_value.as_object();
  require_fields(root,
                 {"schema_version", "flagdnn_version", "backend", "target",
                  "compiler_identity", "build_options", "graph"},
                 {},
                 "Graph IR");
  require_artifact(root.at("schema_version").as_int() == 3,
                   "Graph IR schema_version mismatch");
  require_artifact(root.at("backend").as_string() == "thead",
                   "Graph IR backend mismatch");
  GraphDocument result;
  result.version = root.at("flagdnn_version").as_string();
  result.target = root.at("target").as_string();
  result.compiler_identity = root.at("compiler_identity").as_string();
  require_artifact(is_sha256(result.compiler_identity),
                   "Graph IR compiler identity is invalid");

  const auto& build_options = root.at("build_options").as_object();
  require_fields(build_options, {"heuristic_modes", "autotune"}, {},
                 "Graph IR build_options");
  const auto& heuristic_modes =
      build_options.at("heuristic_modes").as_array();
  require_artifact(!heuristic_modes.empty(),
                   "Graph IR heuristic_modes is empty");
  for (const auto& mode : heuristic_modes) {
    const std::string& name = mode.as_string();
    require_artifact(name == "A" || name == "FALLBACK",
                     "Graph IR heuristic mode is invalid");
  }
  (void)build_options.at("autotune").as_bool();

  const auto& graph = root.at("graph").as_object();
  require_fields(graph,
                 {"name", "tensor_count", "tensors", "node_count", "nodes"},
                 {},
                 "Graph IR graph");
  (void)graph.at("name").as_string();
  const auto& tensors = graph.at("tensors").as_array();
  require_artifact(to_size(graph.at("tensor_count").as_int(),
                           "Graph IR tensor_count") == tensors.size() &&
                       !tensors.empty() && tensors.size() <= kMaximumTensors,
                   "Graph IR tensor_count mismatch");
  std::map<std::int64_t, std::size_t> tensor_indices;
  for (const auto& tensor_value : tensors) {
    const auto& tensor = tensor_value.as_object();
    require_fields(tensor,
                   {"uid", "data_type", "dimensions", "strides", "alignment",
                    "virtual"},
                   {},
                   "Graph IR tensor");
    GraphTensor parsed;
    parsed.uid = tensor.at("uid").as_int();
    require_artifact(parsed.uid >= 0 &&
                         tensor_indices.emplace(parsed.uid,
                                                result.tensors.size()).second,
                     "Graph IR tensor UID is invalid or duplicated");
    parsed.data_type = tensor.at("data_type").as_string();
    parsed.dimensions =
        integer_array(tensor.at("dimensions"), "tensor dimensions", 8);
    parsed.strides = integer_array(tensor.at("strides"), "tensor strides", 8);
    parsed.alignment = to_size(tensor.at("alignment").as_int(),
                               "tensor alignment");
    require_artifact(power_of_two(parsed.alignment) &&
                         parsed.alignment <= 4096,
                     "tensor alignment is invalid");
    parsed.is_virtual = tensor.at("virtual").as_bool();
    parsed.storage_size = tensor_storage_size(
        parsed.data_type, parsed.dimensions, parsed.strides);
    if (!parsed.is_virtual) {
      result.external_uids.push_back(parsed.uid);
    }
    result.tensors.push_back(std::move(parsed));
  }

  const auto& nodes = graph.at("nodes").as_array();
  require_artifact(to_size(graph.at("node_count").as_int(),
                           "Graph IR node_count") == nodes.size() &&
                       !nodes.empty() && nodes.size() <= 1024,
                   "Graph IR node_count mismatch");
  std::set<std::int64_t> node_ids;
  std::map<std::int64_t, std::int64_t> producers;
  result.nodes.resize(nodes.size());
  for (const auto& node_value : nodes) {
    const auto& node = node_value.as_object();
    require_fields(node,
                   {"id", "type", "name", "compute_data_type", "inputs",
                    "outputs", "attributes"},
                   {},
                   "Graph IR node");
    GraphNode parsed;
    parsed.id = node.at("id").as_int();
    require_artifact(parsed.id >= 0 &&
                         parsed.id < static_cast<std::int64_t>(nodes.size()) &&
                         node_ids.insert(parsed.id).second,
                     "Graph IR node ID is invalid or duplicated");
    parsed.operation = node.at("type").as_string();
    require_artifact(is_identifier(parsed.operation),
                     "Graph IR node type is invalid");
    (void)node.at("name").as_string();
    (void)node.at("compute_data_type").as_string();
    parsed.inputs = parse_ports(node.at("inputs"), tensor_indices,
                                "Graph IR node inputs");
    parsed.outputs = parse_ports(node.at("outputs"), tensor_indices,
                                 "Graph IR node outputs");
    (void)node.at("attributes").as_object();
    for (const std::int64_t uid : parsed.outputs) {
      require_artifact(producers.emplace(uid, parsed.id).second,
                       "Graph IR tensor has multiple producers");
    }
    result.nodes.at(static_cast<std::size_t>(parsed.id)) = std::move(parsed);
  }
  return result;
}

const GraphTensor& graph_tensor(
    const GraphDocument& graph,
    const std::map<std::int64_t, std::size_t>& indices,
    std::int64_t uid) {
  const auto iterator = indices.find(uid);
  if (iterator == indices.end()) {
    artifact_error("manifest references unknown tensor UID");
  }
  return graph.tensors.at(iterator->second);
}

std::string tensor_pointer_signature(const GraphTensor& tensor,
                                     bool fp8_storage_bytes) {
  std::string scalar;
  if (fp8_storage_bytes) {
    require_artifact(tensor.data_type == "fp8_e4m3" ||
                         tensor.data_type == "fp8_e5m2" ||
                         tensor.data_type == "fp8_e8m0",
                     "FP8 byte view requires an FP8 tensor");
    scalar = "i8";
  } else if (tensor.data_type == "float32") {
    scalar = "fp32";
  } else if (tensor.data_type == "float16") {
    scalar = "fp16";
  } else if (tensor.data_type == "bfloat16") {
    scalar = "bf16";
  } else if (tensor.data_type == "int32") {
    scalar = "i32";
  } else if (tensor.data_type == "boolean" || tensor.data_type == "fp8_e8m0") {
    scalar = "i8";
  } else if (tensor.data_type == "fp8_e4m3") {
    scalar = "fp8e4nv";
  } else if (tensor.data_type == "fp8_e5m2") {
    scalar = "fp8e5";
  } else {
    artifact_error("tensor pointer data_type is unsupported: " +
                   tensor.data_type);
  }
  return "*" + scalar + (tensor.alignment >= 16 ? ":16" : "");
}

std::array<unsigned int, 3> unsigned_triplet(const json::Value& value,
                                             std::string_view description,
                                             bool positive) {
  const auto& array = value.as_array();
  require_artifact(array.size() == 3,
                   std::string(description) + " must contain three values");
  std::array<unsigned int, 3> result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = to_unsigned(array[index].as_int(), description);
    if (positive) {
      require_artifact(result[index] != 0,
                       std::string(description) + " values must be positive");
    }
  }
  return result;
}

std::vector<std::string> signature_tokens(std::string_view signature) {
  require_artifact(!signature.empty() && signature.size() <= 65536,
                   "kernel full_signature is empty or too long");
  const std::string_view arguments = signature;
  std::vector<std::string> result;
  if (arguments.empty()) {
    return result;
  }
  std::size_t start = 0;
  while (start <= arguments.size()) {
    const std::size_t comma = arguments.find(',', start);
    std::string token(arguments.substr(
        start, comma == std::string_view::npos ? arguments.size() - start
                                               : comma - start));
    token.erase(std::remove_if(token.begin(), token.end(), [](char character) {
                  return character == ' ' || character == '\t';
                }),
                token.end());
    require_artifact(!token.empty(), "kernel signature contains an empty type");
    result.push_back(std::move(token));
    if (comma == std::string_view::npos) {
      break;
    }
    start = comma + 1;
  }
  return result;
}

KernelArgument parse_argument(
    const json::Value& value,
    const GraphDocument& graph,
    const std::vector<TensorArtifact>& artifact_tensors,
    const std::map<std::int64_t, std::size_t>& tensor_indices) {
  const auto& object = value.as_object();
  const std::string& kind = object.at("kind").as_string();
  KernelArgument result;
  if (kind == "tensor" || kind == "workspace_tensor") {
    const bool workspace = kind == "workspace_tensor";
    require_fields(object,
                   {"kind", "uid", "size", "alignment"},
                   workspace ? std::initializer_list<std::string_view>{
                                   "workspace_offset"}
                             : std::initializer_list<std::string_view>{
                                   "storage_view"},
                   "kernel tensor argument");
    if (workspace && !object.contains("workspace_offset")) {
      artifact_error("workspace tensor argument is missing workspace_offset");
    }
    result.kind = workspace ? ArgumentKind::kWorkspaceTensor
                            : ArgumentKind::kTensor;
    result.uid = object.at("uid").as_int();
    const GraphTensor& tensor = graph_tensor(graph, tensor_indices, result.uid);
    if (object.contains("storage_view")) {
      require_artifact(object.at("storage_view").as_string() == "fp8_bytes" &&
                           (tensor.data_type == "fp8_e4m3" ||
                            tensor.data_type == "fp8_e5m2" ||
                            tensor.data_type == "fp8_e8m0"),
                       "storage_view requires an explicit FP8 byte view");
      result.fp8_storage_bytes = true;
    }
    const TensorArtifact& artifact_tensor =
        artifact_tensors.at(tensor_indices.at(result.uid));
    require_artifact(tensor.is_virtual == workspace,
                     "kernel argument kind does not match tensor storage");
    result.storage_size = to_size(object.at("size").as_int(),
                                  "kernel argument size");
    result.alignment = to_size(object.at("alignment").as_int(),
                               "kernel argument alignment");
    require_artifact(result.storage_size == tensor.storage_size,
                     "kernel argument size does not match tensor");
    require_artifact(result.alignment == tensor.alignment,
                     "kernel argument alignment does not match tensor");
    if (workspace) {
      require_artifact(artifact_tensor.workspace_offset.has_value(),
                       "workspace tensor has no materialized storage");
      result.workspace_offset = to_size(
          object.at("workspace_offset").as_int(),
          "kernel argument workspace offset");
      require_artifact(
          result.workspace_offset == *artifact_tensor.workspace_offset,
          "kernel argument workspace offset does not match tensor");
    }
    return result;
  }
  if (kind == "scalar_i32") {
    require_fields(object, {"kind", "name", "value"}, {},
                   "kernel scalar_i32 argument");
    result.kind = ArgumentKind::kScalarI32;
    result.name = object.at("name").as_string();
    require_artifact(is_identifier(result.name),
                     "kernel scalar_i32 name is invalid");
    const std::int64_t value_i64 = object.at("value").as_int();
    require_artifact(value_i64 >= std::numeric_limits<std::int32_t>::min() &&
                         value_i64 <= std::numeric_limits<std::int32_t>::max(),
                     "kernel scalar_i32 value is outside int32");
    result.scalar_i32 = static_cast<std::int32_t>(value_i64);
    return result;
  }
  if (kind == "scalar_f32") {
    require_fields(object, {"kind", "name", "value"}, {},
                   "kernel scalar_f32 argument");
    result.kind = ArgumentKind::kScalarF32;
    result.name = object.at("name").as_string();
    require_artifact(is_identifier(result.name),
                     "kernel scalar_f32 name is invalid");
    const double value_f64 = object.at("value").as_double();
    require_artifact(std::isfinite(value_f64) &&
                         std::abs(value_f64) <=
                             std::numeric_limits<float>::max(),
                     "kernel scalar_f32 value is not finite float32");
    result.scalar_f32 = static_cast<float>(value_f64);
    return result;
  }
  artifact_error("kernel argument kind is unsupported: " + kind);
}

KernelVariant parse_variant(
    const json::Value& value,
    const GraphDocument& graph,
    const std::vector<TensorArtifact>& artifact_tensors,
    const std::map<std::int64_t, std::size_t>& tensor_indices) {
  const auto& object = value.as_object();
  require_fields(object,
                 {"variant_id", "full_signature", "argument_count", "arguments",
                  "compile_options", "launch"},
                 {},
                 "kernel variant");
  KernelVariant result;
  result.variant_id = object.at("variant_id").as_string();
  require_artifact(is_identifier(result.variant_id),
                   "kernel variant ID is invalid");
  result.full_signature = object.at("full_signature").as_string();
  const auto& arguments = object.at("arguments").as_array();
  require_artifact(arguments.size() <= kMaximumArguments &&
                       to_size(object.at("argument_count").as_int(),
                               "kernel argument_count") == arguments.size(),
                   "kernel argument_count mismatch or overflow");
  result.arguments.reserve(arguments.size());
  for (const auto& argument : arguments) {
    result.arguments.push_back(
        parse_argument(argument, graph, artifact_tensors, tensor_indices));
  }
  const std::vector<std::string> tokens = signature_tokens(result.full_signature);
  for (const std::string& token : tokens) {
    if (token.starts_with('*') || token == "i32" || token == "i32:16" || token == "fp32" ||
        token == "f32") {
      result.runtime_signature.push_back(token);
    }
  }
  require_artifact(result.runtime_signature.size() == result.arguments.size(),
                   "kernel signature argument count mismatch");
  for (std::size_t index = 0; index < result.runtime_signature.size(); ++index) {
    const ArgumentKind kind = result.arguments[index].kind;
    const std::string& token = result.runtime_signature[index];
    bool compatible = false;
    if (kind == ArgumentKind::kTensor ||
        kind == ArgumentKind::kWorkspaceTensor) {
      const GraphTensor& tensor = graph_tensor(
          graph, tensor_indices, result.arguments[index].uid);
      compatible = token == tensor_pointer_signature(
          tensor, result.arguments[index].fp8_storage_bytes);
    } else if (kind == ArgumentKind::kScalarI32) {
      compatible = token == "i32" ||
                   (token == "i32:16" && result.arguments[index].scalar_i32 % 16 == 0);
    } else if (kind == ArgumentKind::kScalarF32) {
      compatible = token == "fp32" || token == "f32";
    }
    require_artifact(compatible,
                     "kernel signature and argument kind mismatch");
  }

  const auto& compile = object.at("compile_options").as_object();
  require_fields(compile,
                 {"num_warps", "num_stages", "maxnreg",
                  "ppu_compiler_options"},
                 {},
                 "kernel compile_options");
  result.num_warps =
      to_unsigned(compile.at("num_warps").as_int(), "kernel num_warps", 32);
  require_artifact(result.num_warps != 0 && power_of_two(result.num_warps),
                   "kernel num_warps is invalid");
  result.num_stages =
      to_unsigned(compile.at("num_stages").as_int(), "kernel num_stages", 16);
  require_artifact(result.num_stages != 0, "kernel num_stages is invalid");
  const json::Value& maxnreg = compile.at("maxnreg");
  if (!maxnreg.is_null()) {
    try {
      const unsigned int parsed =
          to_unsigned(maxnreg.as_int(), "kernel maxnreg",
                      static_cast<std::uint64_t>(
                          std::numeric_limits<std::int32_t>::max()));
      require_artifact(parsed != 0, "kernel maxnreg is invalid");
      result.maxnreg = parsed;
    } catch (const TheadError&) {
      throw;
    } catch (const std::exception&) {
      artifact_error("kernel maxnreg must be null or a positive integer");
    }
  }
  const auto& ppu_options =
      compile.at("ppu_compiler_options").as_object();
  for (const auto& [name, value] : ppu_options) {
    if (std::find(kApprovedPpuCompilerOptions.begin(),
                  kApprovedPpuCompilerOptions.end(),
                  name) == kApprovedPpuCompilerOptions.end()) {
      artifact_error("kernel has unapproved PPU compiler option " + name);
    }
    try {
      result.ppu_compiler_options.emplace(name, value.as_string());
    } catch (const std::exception&) {
      artifact_error("kernel PPU compiler option value must be a string");
    }
  }

  const auto& launch = object.at("launch").as_object();
  require_fields(launch, {"grid", "block", "shared_memory"}, {},
                 "kernel launch");
  result.grid = unsigned_triplet(launch.at("grid"), "kernel grid", true);
  result.block = unsigned_triplet(launch.at("block"), "kernel block", true);
  std::size_t block_threads = 1;
  for (const unsigned int value_u32 : result.block) {
    block_threads = checked_multiply(block_threads, value_u32, "kernel block");
  }
  require_artifact(block_threads <= 1024, "kernel block exceeds 1024 threads");
  result.shared_memory = to_unsigned(launch.at("shared_memory").as_int(),
                                     "kernel shared memory",
                                     kMaximumSharedMemory);
  return result;
}

bool same_argument_abi(const KernelArgument& left,
                       const KernelArgument& right) {
  return left.kind == right.kind && left.uid == right.uid &&
         left.name == right.name && left.scalar_i32 == right.scalar_i32 &&
         left.scalar_f32 == right.scalar_f32 &&
         left.workspace_offset == right.workspace_offset &&
         left.storage_size == right.storage_size &&
         left.alignment == right.alignment;
}

ExecutionStage parse_stage(
    const json::Value& value,
    const GraphDocument& graph,
    const std::vector<TensorArtifact>& artifact_tensors,
    const std::map<std::int64_t, std::size_t>& tensor_indices,
    const std::filesystem::path& artifact_root) {
  const auto& object = value.as_object();
  require_fields(object,
                 {"stage_id", "source_node_ids", "dependencies", "operation",
                  "kernel", "variants", "tuning"},
                 {},
                 "execution stage");
  ExecutionStage result;
  result.stage_id = to_size(object.at("stage_id").as_int(), "stage ID");
  result.source_node_ids = integer_array(
      object.at("source_node_ids"), "stage source_node_ids", graph.nodes.size());
  require_artifact(!result.source_node_ids.empty(),
                   "stage source_node_ids is empty");
  std::set<std::int64_t> source_ids;
  for (const std::int64_t node_id : result.source_node_ids) {
    require_artifact(node_id >= 0 &&
                         node_id < static_cast<std::int64_t>(graph.nodes.size()) &&
                         source_ids.insert(node_id).second,
                     "stage source node ID is invalid or duplicated");
  }
  const auto dependency_values = integer_array(
      object.at("dependencies"), "stage dependencies", kMaximumStages);
  std::set<std::size_t> dependencies;
  for (const std::int64_t dependency : dependency_values) {
    const std::size_t value_size = to_size(dependency, "stage dependency");
    require_artifact(dependencies.insert(value_size).second,
                     "stage dependency is duplicated");
    result.dependencies.push_back(value_size);
  }
  result.operation = object.at("operation").as_string();
  require_artifact(is_identifier(result.operation),
                   "stage operation is invalid");

  const auto& kernel = object.at("kernel").as_object();
  require_fields(kernel,
                 {"provider", "ownership", "function", "registry_sha256",
                  "materialized_source"},
                 {},
                 "stage kernel");
  result.provider = kernel.at("provider").as_string();
  result.ownership = kernel.at("ownership").as_string();
  result.function_name = kernel.at("function").as_string();
  result.registry_sha256 = kernel.at("registry_sha256").as_string();
  require_artifact(is_identifier(result.provider) &&
                       (result.ownership == "common" ||
                        result.ownership == "platform") &&
                       is_identifier(result.function_name) &&
                       is_sha256(result.registry_sha256),
                   "stage kernel identity is invalid");
  const auto& source = kernel.at("materialized_source").as_object();
  require_fields(source, {"path", "size", "sha256"}, {},
                 "materialized source");
  result.source_relative_path = source.at("path").as_string();
  result.source = safe_relative_path(artifact_root,
                                     result.source_relative_path.string(),
                                     "source path",
                                     true);
  const std::size_t declared_size =
      to_size(source.at("size").as_int(), "source size");
  std::error_code filesystem_error;
  require_artifact(
      std::filesystem::file_size(result.source, filesystem_error) == declared_size &&
          !filesystem_error,
      "source size does not match artifact metadata");
  result.source_sha256 = source.at("sha256").as_string();
  require_artifact(is_sha256(result.source_sha256), "source hash is invalid");
  require_artifact(flagdnn::native::sha256_file(result.source) ==
                       result.source_sha256,
                   "source hash does not match artifact bytes");

  const auto& variants = object.at("variants").as_array();
  require_artifact(!variants.empty() && variants.size() <= kMaximumVariants,
                   "stage variants is empty or too large");
  std::set<std::string> variant_ids;
  for (const auto& variant : variants) {
    KernelVariant parsed =
        parse_variant(variant, graph, artifact_tensors, tensor_indices);
    require_artifact(variant_ids.insert(parsed.variant_id).second,
                     "kernel variant ID is duplicated");
    if (!result.variants.empty()) {
      const KernelVariant& baseline = result.variants.front();
      require_artifact(parsed.runtime_signature == baseline.runtime_signature &&
                           parsed.arguments.size() == baseline.arguments.size(),
                       "kernel variants have incompatible argument ABI");
      for (std::size_t index = 0; index < parsed.arguments.size(); ++index) {
        require_artifact(
            same_argument_abi(parsed.arguments[index], baseline.arguments[index]),
            "kernel variants have incompatible argument ABI");
      }
    }
    result.variants.push_back(std::move(parsed));
  }

  const json::Value& tuning = object.at("tuning");
  if (tuning.is_null()) {
    require_artifact(result.variants.size() == 1,
                     "multiple variants require complete autotune identity");
  } else {
    const auto& tuning_object = tuning.as_object();
    require_fields(tuning_object,
                   {"warmup", "repetitions", "candidate_identity",
                    "selection_cache"},
                   {},
                   "stage tuning");
    require_artifact(result.variants.size() >= 2,
                     "autotune stage requires multiple variants");
    result.autotune = true;
    result.warmup =
        to_unsigned(tuning_object.at("warmup").as_int(), "tuning warmup", 100000);
    result.repetitions = to_unsigned(tuning_object.at("repetitions").as_int(),
                                     "tuning repetitions",
                                     100000);
    require_artifact(result.warmup != 0 && result.repetitions != 0,
                     "tuning iteration count must be positive");
    result.candidate_identity =
        tuning_object.at("candidate_identity").as_string();
    require_artifact(is_sha256(result.candidate_identity),
                     "tuning candidate_identity is invalid");
    const std::string& selection =
        tuning_object.at("selection_cache").as_string();
    result.selection_cache = safe_relative_path(
        artifact_root, selection, "selection cache path", false);
  }
  return result;
}

std::vector<std::int64_t> boundary_tensor_uids(
    const ExecutionStage& stage,
    const GraphDocument& graph,
    const std::map<std::int64_t, std::int64_t>& producers,
    const std::map<std::int64_t, std::vector<std::int64_t>>& consumers,
    const std::map<std::int64_t, std::size_t>& node_stages) {
  std::set<std::int64_t> stage_nodes(stage.source_node_ids.begin(),
                                     stage.source_node_ids.end());
  std::set<std::int64_t> seen;
  std::vector<std::int64_t> result;
  for (const std::int64_t node_id : stage.source_node_ids) {
    const GraphNode& node = graph.nodes.at(static_cast<std::size_t>(node_id));
    for (const std::int64_t uid : node.inputs) {
      const auto producer = producers.find(uid);
      if ((producer == producers.end() ||
           !stage_nodes.contains(producer->second)) &&
          seen.insert(uid).second) {
        result.push_back(uid);
      }
    }
  }
  for (const std::int64_t node_id : stage.source_node_ids) {
    const GraphNode& node = graph.nodes.at(static_cast<std::size_t>(node_id));
    for (const std::int64_t uid : node.outputs) {
      bool boundary =
          std::find(graph.external_uids.begin(), graph.external_uids.end(), uid) !=
          graph.external_uids.end();
      const auto consumer = consumers.find(uid);
      if (consumer != consumers.end()) {
        boundary = boundary || std::any_of(
                                   consumer->second.begin(),
                                   consumer->second.end(),
                                   [&](std::int64_t value) {
                                     return node_stages.at(value) != stage.stage_id;
                                   });
      }
      if (boundary && seen.insert(uid).second) {
        result.push_back(uid);
      }
    }
  }
  return result;
}

void validate_program_graph(ExecutionProgramArtifact& artifact,
                            const GraphDocument& graph) {
  std::map<std::int64_t, std::size_t> node_stages;
  std::map<std::size_t, std::size_t> stage_indices;
  for (std::size_t index = 0; index < artifact.stages.size(); ++index) {
    const ExecutionStage& stage = artifact.stages[index];
    require_artifact(stage.stage_id < artifact.stages.size() &&
                         stage_indices.emplace(stage.stage_id, index).second,
                     "stage ID is invalid or duplicated");
    for (const std::int64_t node_id : stage.source_node_ids) {
      require_artifact(node_stages.emplace(node_id, stage.stage_id).second,
                       "source node is assigned to multiple stages");
    }
  }
  require_artifact(node_stages.size() == graph.nodes.size(),
                   "execution stages do not cover every Graph node");
  for (const auto& [node_id, stage] : node_stages) {
    (void)stage;
    require_artifact(node_id >= 0 &&
                         node_id < static_cast<std::int64_t>(graph.nodes.size()),
                     "execution stage references an unknown Graph node");
  }

  std::vector<unsigned int> color(artifact.stages.size(), 0);
  const auto visit = [&](const auto& self, std::size_t stage_id) -> void {
    if (color[stage_id] == 1) {
      artifact_error("stage dependency graph contains a cycle");
    }
    if (color[stage_id] == 2) {
      return;
    }
    color[stage_id] = 1;
    const ExecutionStage& stage =
        artifact.stages.at(stage_indices.at(stage_id));
    for (const std::size_t dependency : stage.dependencies) {
      require_artifact(stage_indices.contains(dependency),
                       "stage dependency references a missing stage");
      self(self, dependency);
    }
    color[stage_id] = 2;
  };
  for (const auto& [stage_id, index] : stage_indices) {
    (void)index;
    visit(visit, stage_id);
  }

  std::map<std::int64_t, std::int64_t> producers;
  std::map<std::int64_t, std::vector<std::int64_t>> consumers;
  std::set<std::int64_t> materialized_virtual_uids;
  for (const GraphNode& node : graph.nodes) {
    for (const std::int64_t uid : node.outputs) {
      producers[uid] = node.id;
    }
    for (const std::int64_t uid : node.inputs) {
      consumers[uid].push_back(node.id);
    }
  }
  for (const ExecutionStage& stage : artifact.stages) {
    std::set<std::size_t> expected_dependencies;
    for (const std::int64_t node_id : stage.source_node_ids) {
      for (const std::int64_t uid :
           graph.nodes.at(static_cast<std::size_t>(node_id)).inputs) {
        const auto producer = producers.find(uid);
        if (producer != producers.end()) {
          const std::size_t producer_stage = node_stages.at(producer->second);
          if (producer_stage != stage.stage_id) {
            expected_dependencies.insert(producer_stage);
          }
        }
      }
    }
    const std::set<std::size_t> actual_dependencies(stage.dependencies.begin(),
                                                     stage.dependencies.end());
    require_artifact(actual_dependencies == expected_dependencies,
                     "stage is missing dependency required by Graph dataflow");

    const std::vector<std::int64_t> expected_uids = boundary_tensor_uids(
        stage, graph, producers, consumers, node_stages);
    for (const KernelVariant& variant : stage.variants) {
      std::vector<std::int64_t> actual_uids;
      for (const KernelArgument& argument : variant.arguments) {
        if (argument.kind == ArgumentKind::kTensor ||
            argument.kind == ArgumentKind::kWorkspaceTensor) {
          actual_uids.push_back(argument.uid);
          if (argument.kind == ArgumentKind::kWorkspaceTensor) {
            materialized_virtual_uids.insert(argument.uid);
          }
        }
      }
      // A kernel ABI may order tensor pointers differently from the frontend
      // node ports.  UIDs are the binding identity, so require an exact,
      // duplicate-free boundary set while preserving the declared ABI order
      // for launch.
      std::sort(actual_uids.begin(), actual_uids.end());
      std::vector<std::int64_t> sorted_expected = expected_uids;
      std::sort(sorted_expected.begin(), sorted_expected.end());
      require_artifact(actual_uids.size() == sorted_expected.size() &&
                           std::adjacent_find(actual_uids.begin(),
                                              actual_uids.end()) ==
                               actual_uids.end() &&
                           actual_uids == sorted_expected,
                       "kernel tensor argument UID set does not match stage");
    }
  }
  for (const TensorArtifact& tensor : artifact.tensors) {
    if (tensor.is_virtual) {
      require_artifact(
          tensor.workspace_offset.has_value() ==
              materialized_virtual_uids.contains(tensor.uid),
          "virtual tensor workspace materialization does not match stage "
          "boundaries");
    }
  }
}

ExecutionProgramArtifact load_impl(const EngineBuildContext& context,
                                   const flagdnnBackendBuildInputV2& input) {
  require_artifact(input.struct_size >= sizeof(flagdnnBackendBuildInputV2),
                   "backend build input is too small");
  require_artifact(input.graph_ir != nullptr && input.graph_ir_size != 0 &&
                       input.graph_ir_size <= kMaximumDocumentBytes,
                   "Graph IR pointer or size is invalid");
  require_artifact(input.artifact_directory != nullptr &&
                       input.artifact_directory[0] != '\0',
                   "artifact directory is missing");
  require_artifact(input.request_sha256 != nullptr &&
                       is_sha256(input.request_sha256),
                   "request SHA-256 is invalid");
  const std::string_view graph_ir{
      static_cast<const char*>(input.graph_ir), input.graph_ir_size};
  require_artifact(flagdnn::native::sha256(graph_ir) == input.request_sha256,
                   "request SHA-256 does not match Graph IR bytes");
  GraphDocument graph = parse_graph(graph_ir);
  require_artifact(graph.target == context.target_fingerprint,
                   "Graph IR target does not match PPU context");

  std::error_code filesystem_error;
  const std::filesystem::path artifact_root = std::filesystem::canonical(
      std::filesystem::path(input.artifact_directory), filesystem_error);
  require_artifact(!filesystem_error &&
                       std::filesystem::is_directory(artifact_root),
                   "artifact directory is invalid");
  const std::filesystem::path manifest_path =
      safe_relative_path(artifact_root, "manifest.json", "manifest path", true);
  const std::string manifest_bytes =
      read_file(manifest_path, kMaximumDocumentBytes, "manifest");
  const json::Value manifest_value = json::parse(manifest_bytes);
  const auto& manifest = manifest_value.as_object();
  require_fields(
      manifest,
      {"schema_version", "artifact_kind", "flagdnn_version",
       "graph_ir_schema_version", "backend_abi_version", "backend", "target",
       "warp_size", "engine", "request_sha256", "compiler", "external_uids",
       "tensor_count", "tensors", "workspace", "program"},
      {},
      "manifest");
  require_artifact(manifest.at("schema_version").as_int() ==
                       kArtifactSchemaVersion,
                   "manifest schema_version mismatch");
  require_artifact(manifest.at("artifact_kind").as_string() ==
                       "flagdnn_execution_program",
                   "manifest artifact_kind mismatch");
  require_artifact(manifest.at("flagdnn_version").as_string() == graph.version &&
                       graph.version == FLAGDNN_VERSION_STRING,
                   "manifest FlagDNN version mismatch");
  require_artifact(manifest.at("graph_ir_schema_version").as_int() == 3,
                   "manifest Graph IR schema mismatch");
  require_artifact(manifest.at("backend_abi_version").as_int() ==
                       FLAGDNN_BACKEND_ABI_VERSION_V2,
                   "manifest backend ABI mismatch");

  ExecutionProgramArtifact result;
  result.backend = manifest.at("backend").as_string();
  result.target = manifest.at("target").as_string();
  result.engine = manifest.at("engine").as_string();
  result.request_sha256 = manifest.at("request_sha256").as_string();
  require_artifact(result.backend == "thead", "manifest backend mismatch");
  require_artifact(result.target == context.target_fingerprint &&
                       result.target == graph.target,
                   "manifest target mismatch");
  require_artifact(result.engine == "libtriton_jit",
                   "manifest engine mismatch");
  require_artifact(manifest.at("warp_size").as_int() == 32,
                   "manifest PPU warp_size mismatch");
  require_artifact(result.request_sha256 == input.request_sha256,
                   "manifest request SHA-256 mismatch");

  const auto& compiler = manifest.at("compiler").as_object();
  require_fields(compiler,
                 {"provider", "provider_version", "identity_sha256"},
                 {},
                 "manifest compiler");
  result.compiler_provider = compiler.at("provider").as_string();
  result.compiler_provider_version =
      compiler.at("provider_version").as_string();
  result.compiler_identity = compiler.at("identity_sha256").as_string();
  require_artifact(result.compiler_provider == "thead_triton" &&
                       result.compiler_provider_version == "1" &&
                       is_sha256(result.compiler_identity) &&
                       result.compiler_identity == graph.compiler_identity,
                   "manifest compiler identity mismatch");

  const auto& workspace = manifest.at("workspace").as_object();
  require_fields(workspace, {"size", "alignment"}, {}, "manifest workspace");
  result.workspace_size =
      to_size(workspace.at("size").as_int(), "workspace size");
  result.workspace_alignment =
      to_size(workspace.at("alignment").as_int(), "workspace alignment");
  require_artifact(result.workspace_alignment == 256,
                   "workspace alignment must be exactly 256");

  const auto& manifest_tensors = manifest.at("tensors").as_array();
  require_artifact(to_size(manifest.at("tensor_count").as_int(),
                           "manifest tensor_count") == manifest_tensors.size() &&
                       manifest_tensors.size() == graph.tensors.size(),
                   "manifest tensor_count mismatch");
  std::map<std::int64_t, std::size_t> tensor_indices;
  std::vector<std::pair<std::size_t, std::size_t>> workspace_ranges;
  for (std::size_t index = 0; index < manifest_tensors.size(); ++index) {
    const auto& tensor = manifest_tensors[index].as_object();
    const bool is_virtual = tensor.at("virtual").as_bool();
    require_fields(tensor,
                   {"uid", "data_type", "dimensions", "strides", "alignment",
                    "virtual", "storage_size"},
                   is_virtual
                       ? std::initializer_list<std::string_view>{
                             "workspace_offset"}
                       : std::initializer_list<std::string_view>{},
                   "manifest tensor");
    TensorArtifact parsed;
    parsed.uid = tensor.at("uid").as_int();
    require_artifact(tensor_indices.emplace(parsed.uid, index).second,
                     "manifest tensor UID is duplicated");
    require_artifact(parsed.uid == graph.tensors[index].uid,
                     "manifest tensor UID/order mismatch");
    parsed.data_type = tensor.at("data_type").as_string();
    parsed.dimensions =
        integer_array(tensor.at("dimensions"), "manifest dimensions", 8);
    parsed.strides =
        integer_array(tensor.at("strides"), "manifest strides", 8);
    parsed.alignment =
        to_size(tensor.at("alignment").as_int(), "manifest alignment");
    parsed.is_virtual = is_virtual;
    parsed.storage_size =
        to_size(tensor.at("storage_size").as_int(), "manifest storage_size");
    const GraphTensor& expected = graph.tensors[index];
    require_artifact(parsed.data_type == expected.data_type &&
                         parsed.dimensions == expected.dimensions &&
                         parsed.strides == expected.strides &&
                         parsed.alignment == expected.alignment &&
                         parsed.is_virtual == expected.is_virtual &&
                         parsed.storage_size == expected.storage_size,
                     "manifest tensor metadata does not match Graph IR");
    if (parsed.is_virtual && tensor.contains("workspace_offset")) {
      const std::size_t workspace_offset = to_size(
          tensor.at("workspace_offset").as_int(), "workspace offset");
      parsed.workspace_offset = workspace_offset;
      require_artifact(workspace_offset % parsed.alignment == 0,
                       "virtual tensor workspace alignment mismatch");
      const std::size_t end = checked_add(workspace_offset,
                                          parsed.storage_size,
                                          "workspace range");
      require_artifact(end <= result.workspace_size,
                       "virtual tensor workspace range exceeds workspace");
      workspace_ranges.emplace_back(workspace_offset, end);
    }
    result.tensors.push_back(std::move(parsed));
  }
  std::sort(workspace_ranges.begin(), workspace_ranges.end());
  for (std::size_t index = 1; index < workspace_ranges.size(); ++index) {
    require_artifact(workspace_ranges[index - 1].second <=
                         workspace_ranges[index].first,
                     "virtual tensor workspace overlap");
  }

  const auto external_values = integer_array(
      manifest.at("external_uids"), "manifest external_uids", kMaximumTensors);
  result.external_uids = external_values;
  require_artifact(result.external_uids == graph.external_uids,
                   "manifest external_uids mismatch Graph IR");

  const auto& program = manifest.at("program").as_object();
  require_fields(program, {"schema_version", "stage_count", "stages"}, {},
                 "execution program");
  require_artifact(program.at("schema_version").as_int() ==
                       kExecutionProgramVersion,
                   "execution program schema_version mismatch");
  const auto& stages = program.at("stages").as_array();
  require_artifact(to_size(program.at("stage_count").as_int(),
                           "program stage_count") == stages.size() &&
                       !stages.empty() && stages.size() <= kMaximumStages,
                   "execution program stage_count mismatch or overflow");
  result.stages.reserve(stages.size());
  for (const auto& stage : stages) {
    result.stages.push_back(
        parse_stage(stage,
                    graph,
                    result.tensors,
                    tensor_indices,
                    artifact_root));
  }
  validate_program_graph(result, graph);
  return result;
}

}  // namespace

ExecutionProgramArtifact load_and_validate_artifact(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input) {
  try {
    return load_impl(context, input);
  } catch (const TheadError&) {
    throw;
  } catch (const std::exception& error) {
    artifact_error(error.what());
  }
}

}  // namespace flagdnn::thead
