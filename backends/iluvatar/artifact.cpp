/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/artifact.hpp"

#include "backends/iluvatar/error.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace flagdnn::iluvatar {
namespace {

using JsonValue = flagdnn::native::json::Value;

constexpr std::size_t kMaximumMetadataSize = 16U << 20;
constexpr std::size_t kMaximumSourceSize = 16U << 20;
constexpr std::size_t kMaximumTensorRank = 64;
constexpr std::size_t kMaximumTensors = 1U << 20;
constexpr std::size_t kMaximumVariants = 1024;
constexpr std::size_t kMaximumAlignment = 1U << 20;

[[noreturn]] void artifact_error(std::string message) {
  throw IluvatarError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      std::move(message));
}

bool is_lower_sha256(std::string_view value) {
  return value.size() == 64 &&
         std::all_of(value.begin(), value.end(), [](unsigned char character) {
           return std::isdigit(character) != 0 ||
                  (character >= 'a' && character <= 'f');
         });
}

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::size_t checked_size(std::int64_t value, std::string_view field) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    artifact_error("artifact field is outside size_t: " + std::string(field));
  }
  return static_cast<std::size_t>(value);
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        std::string_view field) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    artifact_error("artifact size addition overflow: " + std::string(field));
  }
  return left + right;
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view field) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    artifact_error("artifact size multiplication overflow: " +
                   std::string(field));
  }
  return left * right;
}

unsigned int checked_positive_unsigned(std::int64_t value,
                                       std::string_view field) {
  if (value <= 0 || static_cast<std::uint64_t>(value) >
                        std::numeric_limits<unsigned int>::max()) {
    artifact_error("artifact field is not a positive unsigned integer: " +
                   std::string(field));
  }
  return static_cast<unsigned int>(value);
}

unsigned int checked_nonnegative_unsigned(std::int64_t value,
                                          std::string_view field) {
  if (value < 0 || static_cast<std::uint64_t>(value) >
                       std::numeric_limits<unsigned int>::max()) {
    artifact_error("artifact field is not an unsigned integer: " +
                   std::string(field));
  }
  return static_cast<unsigned int>(value);
}

std::string read_text_file(const std::filesystem::path &path,
                           std::size_t maximum_size,
                           std::string_view description) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > maximum_size) {
    artifact_error("cannot stat " + std::string(description) +
                   " or it exceeds its size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    artifact_error("cannot open " + std::string(description));
  }
  std::string bytes(static_cast<std::size_t>(file_size), '\0');
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!input && !bytes.empty()) {
    artifact_error("cannot read " + std::string(description));
  }
  return bytes;
}

std::size_t element_size(std::string_view data_type) {
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
  artifact_error("artifact tensor data type is invalid");
}

std::vector<std::int64_t> positive_integer_array(const JsonValue &value,
                                                 std::string_view field) {
  const auto &array = value.as_array();
  // Rank-zero tensors use matching empty dimension/stride arrays and still
  // occupy one element.  The rank equality check rejects partial emptiness.
  if (array.size() > kMaximumTensorRank) {
    artifact_error("artifact tensor rank is invalid: " + std::string(field));
  }
  std::vector<std::int64_t> result;
  result.reserve(array.size());
  for (const auto &entry : array) {
    const std::int64_t integer = entry.as_int();
    if (integer <= 0) {
      artifact_error("artifact tensor dimensions/strides must be positive: " +
                     std::string(field));
    }
    result.push_back(integer);
  }
  return result;
}

TensorArtifact parse_graph_tensor(const JsonValue &value) {
  TensorArtifact tensor;
  tensor.uid = value.at("uid").as_int();
  if (tensor.uid <= 0) {
    artifact_error("graph tensor UID must be positive");
  }
  tensor.data_type = value.at("data_type").as_string();
  const std::size_t bytes_per_element = element_size(tensor.data_type);
  tensor.dimensions =
      positive_integer_array(value.at("dimensions"), "dimensions");
  tensor.strides = positive_integer_array(value.at("strides"), "strides");
  if (tensor.dimensions.size() != tensor.strides.size()) {
    artifact_error("graph tensor dimensions and strides differ in rank");
  }
  tensor.alignment =
      checked_size(value.at("alignment").as_int(), "tensor.alignment");
  if (!is_power_of_two(tensor.alignment) ||
      tensor.alignment > kMaximumAlignment) {
    artifact_error("graph tensor alignment is invalid");
  }
  tensor.is_virtual = value.at("virtual").as_bool();

  std::size_t maximum_element_offset = 0;
  for (std::size_t index = 0; index < tensor.dimensions.size(); ++index) {
    const std::size_t extent =
        checked_size(tensor.dimensions[index] - 1, "tensor.dimension");
    const std::size_t stride =
        checked_size(tensor.strides[index], "tensor.stride");
    maximum_element_offset =
        checked_add(maximum_element_offset,
                    checked_multiply(extent, stride, "tensor storage extent"),
                    "tensor storage extent");
  }
  tensor.storage_size = checked_multiply(
      checked_add(maximum_element_offset, 1, "tensor element count"),
      bytes_per_element, "tensor storage bytes");
  return tensor;
}

struct GraphNode {
  std::int64_t id = 0;
  std::string operation;
  std::vector<std::int64_t> inputs;
  std::vector<std::int64_t> outputs;
};

std::vector<std::int64_t>
parse_ports(const JsonValue &value,
            const std::map<std::int64_t, TensorArtifact> &tensors) {
  std::vector<std::int64_t> result;
  for (const auto &port : value.as_array()) {
    if (port.at("name").as_string().empty()) {
      artifact_error("graph port name is empty");
    }
    const std::int64_t uid = port.at("uid").as_int();
    if (tensors.find(uid) == tensors.end()) {
      artifact_error("graph port references an unknown tensor UID");
    }
    result.push_back(uid);
  }
  return result;
}

std::array<unsigned int, 3> parse_triplet(const JsonValue &value,
                                          std::string_view field) {
  const auto &array = value.as_array();
  if (array.size() != 3) {
    artifact_error("artifact launch triplet has the wrong length: " +
                   std::string(field));
  }
  return {checked_positive_unsigned(array[0].as_int(), field),
          checked_positive_unsigned(array[1].as_int(), field),
          checked_positive_unsigned(array[2].as_int(), field)};
}

bool printable_ascii(std::string_view value, std::size_t maximum_size,
                     bool identifier) {
  if (value.empty() || value.size() > maximum_size) {
    return false;
  }
  if (identifier &&
      (std::isalpha(static_cast<unsigned char>(value.front())) == 0 &&
       value.front() != '_')) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [&](unsigned char character) {
    if (identifier) {
      return std::isalnum(character) != 0 || character == '_';
    }
    return character >= 0x21U && character <= 0x7eU;
  });
}

bool path_is_below(const std::filesystem::path &root,
                   const std::filesystem::path &child) {
  auto root_iterator = root.begin();
  auto child_iterator = child.begin();
  for (; root_iterator != root.end(); ++root_iterator, ++child_iterator) {
    if (child_iterator == child.end() || *root_iterator != *child_iterator) {
      return false;
    }
  }
  return true;
}

std::filesystem::path
validate_source(const std::filesystem::path &artifact_directory,
                const JsonValue &descriptor,
                std::filesystem::path &relative_output,
                std::string &hash_output) {
  const std::string path_text = descriptor.at("path").as_string();
  const std::filesystem::path relative(path_text);
  if (path_text.empty() || relative.is_absolute() || relative.has_root_path() ||
      relative.lexically_normal() != relative) {
    artifact_error("artifact materialized source path is unsafe");
  }
  for (const auto &component : relative) {
    if (component == "." || component == ".." || component.empty()) {
      artifact_error("artifact materialized source path is unsafe");
    }
  }

  std::error_code error;
  const std::filesystem::path canonical_root =
      std::filesystem::canonical(artifact_directory, error);
  if (error || !std::filesystem::is_directory(canonical_root)) {
    artifact_error("artifact directory is not a canonical directory");
  }
  const std::filesystem::path canonical_source =
      std::filesystem::canonical(canonical_root / relative, error);
  if (error || !path_is_below(canonical_root, canonical_source) ||
      !std::filesystem::is_regular_file(canonical_source)) {
    artifact_error("artifact materialized source escapes its directory");
  }

  const std::size_t expected_size =
      checked_size(descriptor.at("size").as_int(), "source.size");
  const std::uintmax_t actual_size =
      std::filesystem::file_size(canonical_source, error);
  if (error || expected_size == 0 || expected_size > kMaximumSourceSize ||
      actual_size != expected_size) {
    artifact_error("artifact materialized source size does not match");
  }
  hash_output = descriptor.at("sha256").as_string();
  if (!is_lower_sha256(hash_output) ||
      flagdnn::native::sha256_file(canonical_source) != hash_output) {
    artifact_error("artifact materialized source SHA-256 does not match");
  }
  relative_output = relative;
  return canonical_source;
}

KernelArgument
parse_argument(const JsonValue &value,
               const std::map<std::int64_t, TensorArtifact> &tensors,
               std::size_t workspace_size,
               std::map<std::int64_t, KernelArgument> &provider_workspaces) {
  KernelArgument argument;
  const std::string kind = value.at("kind").as_string();
  if (kind == "tensor" || kind == "workspace_tensor") {
    argument.uid = value.at("uid").as_int();
    if (argument.uid <= 0) {
      artifact_error("kernel tensor argument UID is invalid");
    }
    const bool workspace = kind == "workspace_tensor";
    argument.kind =
        workspace ? ArgumentKind::kWorkspaceTensor : ArgumentKind::kTensor;
    argument.storage_size =
        checked_size(value.at("size").as_int(), "argument.size");
    argument.alignment =
        checked_size(value.at("alignment").as_int(), "argument.alignment");
    if (workspace) {
      argument.workspace_offset = checked_size(
          value.at("workspace_offset").as_int(), "argument.workspace_offset");
    }

    const auto tensor = tensors.find(argument.uid);
    if (tensor != tensors.end()) {
      if (workspace != tensor->second.is_virtual) {
        artifact_error("kernel tensor argument kind does not match virtuality");
      }
      if (argument.storage_size != tensor->second.storage_size ||
          argument.alignment != tensor->second.alignment) {
        artifact_error("kernel tensor argument metadata differs from tensor");
      }
      if (workspace &&
          argument.workspace_offset != tensor->second.workspace_offset) {
        artifact_error("kernel workspace argument offset differs from tensor");
      }
    } else if (!workspace) {
      artifact_error("kernel argument references an unknown tensor UID");
    } else {
      const std::int64_t maximum_graph_uid = tensors.rbegin()->first;
      if (argument.uid <= maximum_graph_uid || argument.storage_size == 0 ||
          !is_power_of_two(argument.alignment) ||
          argument.alignment > kMaximumAlignment ||
          argument.workspace_offset % argument.alignment != 0 ||
          argument.workspace_offset > workspace_size ||
          argument.storage_size > workspace_size - argument.workspace_offset) {
        artifact_error("provider workspace tensor metadata is invalid");
      }
      const auto [registered, inserted] =
          provider_workspaces.emplace(argument.uid, argument);
      if (!inserted &&
          (registered->second.workspace_offset != argument.workspace_offset ||
           registered->second.storage_size != argument.storage_size ||
           registered->second.alignment != argument.alignment)) {
        artifact_error("provider workspace tensor metadata is inconsistent");
      }
      if (inserted) {
        const std::size_t argument_end =
            checked_add(argument.workspace_offset, argument.storage_size,
                        "provider workspace tensor range");
        for (const auto &[uid, graph_tensor] : tensors) {
          (void)uid;
          if (!graph_tensor.is_virtual) {
            continue;
          }
          const std::size_t graph_end = checked_add(
              graph_tensor.workspace_offset, graph_tensor.storage_size,
              "Graph workspace tensor range");
          if (argument.workspace_offset < graph_end &&
              graph_tensor.workspace_offset < argument_end) {
            artifact_error("provider workspace overlaps Graph workspace");
          }
        }
        for (const auto &[uid, other] : provider_workspaces) {
          if (uid == argument.uid) {
            continue;
          }
          const std::size_t other_end =
              checked_add(other.workspace_offset, other.storage_size,
                          "provider workspace tensor range");
          if (argument.workspace_offset < other_end &&
              other.workspace_offset < argument_end) {
            artifact_error("provider workspace tensor ranges overlap");
          }
        }
      }
    }
  } else if (kind == "scalar_i32") {
    argument.kind = ArgumentKind::kScalarI32;
    const std::int64_t scalar = value.at("value").as_int();
    if (scalar < std::numeric_limits<std::int32_t>::min() ||
        scalar > std::numeric_limits<std::int32_t>::max()) {
      artifact_error("kernel scalar_i32 value is out of range");
    }
    argument.scalar_i32 = static_cast<std::int32_t>(scalar);
  } else if (kind == "scalar_f32") {
    argument.kind = ArgumentKind::kScalarF32;
    const double scalar = value.at("value").as_double();
    if (!std::isfinite(scalar) ||
        std::abs(scalar) > std::numeric_limits<float>::max()) {
      artifact_error("kernel scalar_f32 value is out of range");
    }
    argument.scalar_f32 = static_cast<float>(scalar);
  } else {
    artifact_error("kernel argument kind is unsupported");
  }
  return argument;
}

bool arguments_equal(const std::vector<KernelArgument> &left,
                     const std::vector<KernelArgument> &right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.size(); ++index) {
    const KernelArgument &a = left[index];
    const KernelArgument &b = right[index];
    if (a.kind != b.kind || a.uid != b.uid || a.scalar_i32 != b.scalar_i32 ||
        a.scalar_f32 != b.scalar_f32 ||
        a.workspace_offset != b.workspace_offset ||
        a.storage_size != b.storage_size || a.alignment != b.alignment) {
      return false;
    }
  }
  return true;
}

KernelVariant
parse_variant(const JsonValue &value,
              const std::map<std::int64_t, TensorArtifact> &tensors,
              std::size_t workspace_size,
              std::map<std::int64_t, KernelArgument> &provider_workspaces) {
  KernelVariant result;
  result.variant_id = value.at("variant_id").as_string();
  if (!printable_ascii(result.variant_id, 128, true)) {
    artifact_error("kernel variant ID is invalid");
  }
  result.full_signature = value.at("full_signature").as_string();
  if (!printable_ascii(result.full_signature, 64U << 10, false)) {
    artifact_error("kernel full signature is invalid");
  }

  const std::size_t argument_count =
      checked_size(value.at("argument_count").as_int(), "argument_count");
  const auto &arguments = value.at("arguments").as_array();
  if (argument_count == 0 ||
      argument_count > FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS ||
      arguments.size() != argument_count) {
    artifact_error("kernel argument count is invalid");
  }
  result.arguments.reserve(argument_count);
  for (const auto &argument : arguments) {
    result.arguments.push_back(
        parse_argument(argument, tensors, workspace_size, provider_workspaces));
  }

  const auto &options = value.at("compile_options");
  result.num_warps = checked_positive_unsigned(options.at("num_warps").as_int(),
                                               "compile_options.num_warps");
  result.num_stages = checked_positive_unsigned(
      options.at("num_stages").as_int(), "compile_options.num_stages");
  if (result.num_warps > 16 || result.num_stages > 32) {
    artifact_error("IX compile options exceed safety limits");
  }

  const auto &launch = value.at("launch");
  result.grid = parse_triplet(launch.at("grid"), "launch.grid");
  result.block = parse_triplet(launch.at("block"), "launch.block");
  result.shared_memory = checked_nonnegative_unsigned(
      launch.at("shared_memory").as_int(), "launch.shared_memory");
  std::size_t grid_product = 1;
  for (const unsigned int dimension : result.grid) {
    grid_product = checked_multiply(grid_product, dimension, "grid product");
  }
  if (grid_product == 0 || result.block[0] != result.num_warps * 64U ||
      result.block[1] != 1U || result.block[2] != 1U ||
      result.shared_memory != 0U) {
    artifact_error("IX launch metadata is inconsistent with warp64/JIT-derived "
                   "shared memory");
  }
  return result;
}

std::vector<std::int64_t> unique_node_ids(const JsonValue &value,
                                          std::string_view field) {
  const auto &array = value.as_array();
  if (array.empty()) {
    artifact_error("execution stage has no source graph nodes");
  }
  std::vector<std::int64_t> result;
  result.reserve(array.size());
  for (const auto &entry : array) {
    const std::int64_t id = entry.as_int();
    if (id < 0 || std::find(result.begin(), result.end(), id) != result.end()) {
      artifact_error("execution stage has invalid " + std::string(field));
    }
    result.push_back(id);
  }
  return result;
}

std::vector<std::size_t> dependencies(const JsonValue &value,
                                      std::size_t stage_id) {
  std::vector<std::size_t> result;
  for (const auto &entry : value.as_array()) {
    const std::size_t dependency =
        checked_size(entry.as_int(), "stage dependency");
    if (dependency >= stage_id ||
        (!result.empty() && dependency <= result.back())) {
      artifact_error("stage dependencies are cyclic, duplicated, or unsorted");
    }
    result.push_back(dependency);
  }
  return result;
}

bool tensors_equal(const TensorArtifact &left, const TensorArtifact &right) {
  return left.uid == right.uid && left.data_type == right.data_type &&
         left.dimensions == right.dimensions && left.strides == right.strides &&
         left.alignment == right.alignment &&
         left.is_virtual == right.is_virtual &&
         left.storage_size == right.storage_size;
}

} // namespace

ExecutionProgramArtifact
load_and_validate_artifact(const EngineBuildContext &context,
                           const flagdnnBackendBuildInputV2 &input) {
  try {
    if (input.struct_size < sizeof(flagdnnBackendBuildInputV2) ||
        input.graph_ir == nullptr || input.graph_ir_size == 0 ||
        input.graph_ir_size > kMaximumMetadataSize ||
        input.artifact_directory == nullptr ||
        input.request_sha256 == nullptr) {
      artifact_error("Iluvatar build input is incomplete");
    }
    const std::string_view graph_ir(static_cast<const char *>(input.graph_ir),
                                    input.graph_ir_size);
    const std::string request_hash(input.request_sha256);
    if (!is_lower_sha256(request_hash) ||
        flagdnn::native::sha256(graph_ir) != request_hash) {
      artifact_error("Graph IR request SHA-256 does not match build input");
    }

    const JsonValue request = flagdnn::native::json::parse(graph_ir);
    const std::string compiler_identity =
        request.at("compiler_identity").as_string();
    const bool request_autotune =
        request.at("build_options").at("autotune").as_bool();
    if (request.at("schema_version").as_int() != 3 ||
        request.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
        request.at("backend").as_string() != "iluvatar" ||
        request.at("target").as_string() != context.target_fingerprint ||
        context.target_fingerprint != "corex_71" ||
        !is_lower_sha256(compiler_identity)) {
      artifact_error("Iluvatar Graph IR identity is invalid");
    }

    const auto &graph = request.at("graph");
    const auto &graph_tensor_values = graph.at("tensors").as_array();
    const std::size_t graph_tensor_count =
        checked_size(graph.at("tensor_count").as_int(), "tensor_count");
    if (graph_tensor_count == 0 || graph_tensor_count > kMaximumTensors ||
        graph_tensor_values.size() != graph_tensor_count) {
      artifact_error("Graph IR tensor count is invalid");
    }
    std::map<std::int64_t, TensorArtifact> graph_tensors;
    for (const auto &value : graph_tensor_values) {
      TensorArtifact tensor = parse_graph_tensor(value);
      if (!graph_tensors.emplace(tensor.uid, std::move(tensor)).second) {
        artifact_error("Graph IR tensor UID is duplicated");
      }
    }

    const auto &graph_node_values = graph.at("nodes").as_array();
    const std::size_t graph_node_count =
        checked_size(graph.at("node_count").as_int(), "node_count");
    if (graph_node_count == 0 ||
        graph_node_count > FLAGDNN_BACKEND_MAX_EXECUTION_STAGES ||
        graph_node_values.size() != graph_node_count) {
      artifact_error("Graph IR node count is invalid");
    }
    std::vector<GraphNode> graph_nodes;
    std::map<std::int64_t, std::size_t> node_positions;
    std::map<std::int64_t, std::int64_t> tensor_producers;
    graph_nodes.reserve(graph_node_count);
    for (std::size_t position = 0; position < graph_node_count; ++position) {
      const auto &value = graph_node_values[position];
      GraphNode node;
      node.id = value.at("id").as_int();
      node.operation = value.at("type").as_string();
      if (node.id < 0 || node.operation.empty() ||
          !node_positions.emplace(node.id, position).second) {
        artifact_error("Graph IR node identity is invalid or duplicated");
      }
      node.inputs = parse_ports(value.at("inputs"), graph_tensors);
      node.outputs = parse_ports(value.at("outputs"), graph_tensors);
      if (node.outputs.empty()) {
        artifact_error("Graph IR node has no output");
      }
      for (const std::int64_t uid : node.outputs) {
        if (!tensor_producers.emplace(uid, node.id).second) {
          artifact_error("Graph IR tensor has multiple producers");
        }
      }
      graph_nodes.push_back(std::move(node));
    }
    std::map<std::int64_t, std::size_t> last_consumers;
    for (std::size_t position = 0; position < graph_nodes.size(); ++position) {
      for (const std::int64_t uid : graph_nodes[position].inputs) {
        last_consumers[uid] = position;
        const auto producer = tensor_producers.find(uid);
        if (graph_tensors.at(uid).is_virtual &&
            (producer == tensor_producers.end() ||
             node_positions.at(producer->second) >= position)) {
          artifact_error("Graph IR virtual tensor dependency is not ordered");
        }
      }
    }
    for (const auto &[uid, tensor] : graph_tensors) {
      if (!tensor.is_virtual) {
        continue;
      }
      const auto producer = tensor_producers.find(uid);
      if (producer == tensor_producers.end()) {
        artifact_error("Graph IR virtual tensor has no producer");
      }
      if (last_consumers.find(uid) == last_consumers.end()) {
        // Frontend operations may expose a virtual output which is disabled by
        // an operation attribute (for example SDPA generate_stats=false). It
        // is still a legal kernel ABI output and needs workspace at its
        // producer stage, but it has no Graph consumer.
        last_consumers.emplace(uid, node_positions.at(producer->second));
      }
    }

    const std::filesystem::path artifact_directory(input.artifact_directory);
    const JsonValue manifest = flagdnn::native::json::parse(
        read_text_file(artifact_directory / "manifest.json",
                       kMaximumMetadataSize, "Iluvatar artifact manifest"));
    if (manifest.at("schema_version").as_int() != kArtifactSchemaVersion ||
        manifest.at("artifact_kind").as_string() !=
            "flagdnn_execution_program" ||
        manifest.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
        manifest.at("graph_ir_schema_version").as_int() != 3 ||
        manifest.at("backend_abi_version").as_int() !=
            FLAGDNN_BACKEND_ABI_VERSION ||
        manifest.at("backend").as_string() != "iluvatar" ||
        manifest.at("target").as_string() != context.target_fingerprint ||
        manifest.at("warp_size").as_int() != 64 ||
        manifest.at("engine").as_string() != "libtriton_jit" ||
        manifest.at("request_sha256").as_string() != request_hash) {
      artifact_error("Iluvatar artifact identity does not match Graph IR");
    }

    ExecutionProgramArtifact result;
    result.backend = "iluvatar";
    result.target = context.target_fingerprint;
    result.engine = "libtriton_jit";
    result.request_sha256 = request_hash;
    const auto &compiler = manifest.at("compiler");
    result.compiler_provider = compiler.at("provider").as_string();
    result.compiler_provider_version =
        compiler.at("provider_version").as_string();
    result.compiler_identity = compiler.at("identity_sha256").as_string();
    if (result.compiler_provider.empty() ||
        result.compiler_provider_version.empty() ||
        result.compiler_identity != compiler_identity) {
      artifact_error("artifact compiler identity does not match request");
    }

    std::vector<std::int64_t> expected_external_uids;
    for (const auto &[uid, tensor] : graph_tensors) {
      if (!tensor.is_virtual) {
        expected_external_uids.push_back(uid);
      }
    }
    for (const auto &value : manifest.at("external_uids").as_array()) {
      const std::int64_t uid = value.as_int();
      if (uid <= 0 || (!result.external_uids.empty() &&
                       uid <= result.external_uids.back())) {
        artifact_error("artifact external UIDs are invalid or unsorted");
      }
      result.external_uids.push_back(uid);
    }
    if (result.external_uids != expected_external_uids) {
      artifact_error("artifact external binding set differs from Graph IR");
    }

    const auto &workspace = manifest.at("workspace");
    result.workspace_size =
        checked_size(workspace.at("size").as_int(), "workspace.size");
    result.workspace_alignment =
        checked_size(workspace.at("alignment").as_int(), "workspace.alignment");
    if (!is_power_of_two(result.workspace_alignment) ||
        result.workspace_alignment > kMaximumAlignment ||
        (result.workspace_size != 0 &&
         result.workspace_size % result.workspace_alignment != 0)) {
      artifact_error("artifact workspace size/alignment is invalid");
    }

    const auto &manifest_tensors = manifest.at("tensors").as_array();
    if (checked_size(manifest.at("tensor_count").as_int(), "tensor_count") !=
            graph_tensor_count ||
        manifest_tensors.size() != graph_tensor_count) {
      artifact_error("artifact tensor table count differs from Graph IR");
    }
    std::map<std::int64_t, TensorArtifact> parsed_tensors;
    for (const auto &value : manifest_tensors) {
      TensorArtifact tensor = parse_graph_tensor(value);
      tensor.storage_size = checked_size(value.at("storage_size").as_int(),
                                         "tensor.storage_size");
      const auto graph_tensor = graph_tensors.find(tensor.uid);
      if (graph_tensor == graph_tensors.end() ||
          !tensors_equal(tensor, graph_tensor->second)) {
        artifact_error("artifact tensor differs from Graph IR tensor");
      }
      if (tensor.is_virtual) {
        tensor.workspace_offset = checked_size(
            value.at("workspace_offset").as_int(), "tensor.workspace_offset");
        if (tensor.workspace_offset % tensor.alignment != 0 ||
            tensor.workspace_offset > result.workspace_size ||
            tensor.storage_size >
                result.workspace_size - tensor.workspace_offset) {
          artifact_error("virtual tensor workspace range is invalid");
        }
      } else if (value.as_object().find("workspace_offset") !=
                 value.as_object().end()) {
        artifact_error("external tensor must not have a workspace offset");
      }
      if (!parsed_tensors.emplace(tensor.uid, tensor).second) {
        artifact_error("artifact tensor UID is duplicated");
      }
      result.tensors.push_back(std::move(tensor));
    }

    std::vector<const TensorArtifact *> virtual_tensors;
    for (const auto &[uid, tensor] : parsed_tensors) {
      if (tensor.is_virtual) {
        virtual_tensors.push_back(&tensor);
      }
    }
    for (std::size_t left = 0; left < virtual_tensors.size(); ++left) {
      const TensorArtifact &a = *virtual_tensors[left];
      const std::size_t a_start = node_positions.at(tensor_producers.at(a.uid));
      const std::size_t a_end = last_consumers.at(a.uid);
      const std::size_t a_memory_end = checked_add(
          a.workspace_offset, a.storage_size, "virtual tensor range");
      for (std::size_t right = left + 1; right < virtual_tensors.size();
           ++right) {
        const TensorArtifact &b = *virtual_tensors[right];
        const std::size_t b_start =
            node_positions.at(tensor_producers.at(b.uid));
        const std::size_t b_end = last_consumers.at(b.uid);
        const std::size_t b_memory_end = checked_add(
            b.workspace_offset, b.storage_size, "virtual tensor range");
        const bool live_overlap = a_start <= b_end && b_start <= a_end;
        const bool memory_overlap = a.workspace_offset < b_memory_end &&
                                    b.workspace_offset < a_memory_end;
        if (live_overlap && memory_overlap) {
          artifact_error("live virtual tensor workspace ranges overlap");
        }
      }
    }

    const auto &program = manifest.at("program");
    if (program.at("schema_version").as_int() != kExecutionProgramVersion) {
      artifact_error("Iluvatar execution program schema is unsupported");
    }
    const std::size_t stage_count =
        checked_size(program.at("stage_count").as_int(), "stage_count");
    const auto &stages = program.at("stages").as_array();
    if (stage_count == 0 ||
        stage_count > FLAGDNN_BACKEND_MAX_EXECUTION_STAGES ||
        stages.size() != stage_count) {
      artifact_error("Iluvatar execution stage count is invalid");
    }

    std::map<std::int64_t, std::vector<std::size_t>> node_stages;
    std::set<std::int64_t> bound_external_uids;
    std::map<std::int64_t, KernelArgument> provider_workspaces;
    result.stages.reserve(stage_count);
    for (std::size_t index = 0; index < stages.size(); ++index) {
      const auto &value = stages[index];
      ExecutionStage stage;
      stage.stage_id = checked_size(value.at("stage_id").as_int(), "stage_id");
      if (stage.stage_id != index) {
        artifact_error("execution stage IDs must match stable array order");
      }
      stage.source_node_ids =
          unique_node_ids(value.at("source_node_ids"), "source node IDs");
      for (const std::int64_t node_id : stage.source_node_ids) {
        if (node_positions.find(node_id) == node_positions.end()) {
          artifact_error("execution stage references an unknown Graph node");
        }
        node_stages[node_id].push_back(index);
      }
      stage.dependencies =
          dependencies(value.at("dependencies"), stage.stage_id);
      stage.operation = value.at("operation").as_string();
      const auto &kernel = value.at("kernel");
      stage.provider = kernel.at("provider").as_string();
      stage.ownership = kernel.at("ownership").as_string();
      stage.function_name = kernel.at("function").as_string();
      if (stage.operation.empty() || stage.provider.empty() ||
          (stage.ownership != "common" && stage.ownership != "platform") ||
          !printable_ascii(stage.function_name, 256, true)) {
        artifact_error("execution stage kernel identity is invalid");
      }
      stage.source =
          validate_source(artifact_directory, kernel.at("materialized_source"),
                          stage.source_relative_path, stage.source_sha256);

      const auto &variants = value.at("variants").as_array();
      if (variants.empty() || variants.size() > kMaximumVariants) {
        artifact_error("execution stage variant count is invalid");
      }
      std::set<std::string> variant_ids;
      stage.variants.reserve(variants.size());
      for (const auto &variant_value : variants) {
        KernelVariant variant =
            parse_variant(variant_value, parsed_tensors, result.workspace_size,
                          provider_workspaces);

        if (!variant_ids.insert(variant.variant_id).second) {
          artifact_error("execution stage variant ID is duplicated");
        }
        if (!stage.variants.empty() &&
            !arguments_equal(variant.arguments,
                             stage.variants.front().arguments)) {
          artifact_error("execution stage variants have incompatible ABIs");
        }
        for (const KernelArgument &argument : variant.arguments) {
          if (argument.kind == ArgumentKind::kTensor) {
            bound_external_uids.insert(argument.uid);
          }
        }
        stage.variants.push_back(std::move(variant));
      }
      if (stage.variants.size() > 1) {
        if (!request_autotune) {
          artifact_error(
              "artifact autotune variants do not match Graph IR options");
        }
        const auto &tuning = value.at("tuning");
        if (tuning.at("schema_version").as_int() != 1) {
          artifact_error("autotune metadata schema is unsupported");
        }
        stage.warmup = checked_nonnegative_unsigned(
            tuning.at("warmup").as_int(), "tuning.warmup");
        stage.repetitions = checked_positive_unsigned(
            tuning.at("repetitions").as_int(), "tuning.repetitions");
        stage.candidate_identity = tuning.at("candidate_identity").as_string();
        if (stage.warmup > 100 || stage.repetitions > 100 ||
            !is_lower_sha256(stage.candidate_identity) ||
            !is_lower_sha256(
                tuning.at("base_candidate_identity").as_string()) ||
            !is_lower_sha256(tuning.at("source_sha256").as_string()) ||
            tuning.at("key").as_string().empty() ||
            tuning.at("strategy").as_string().empty()) {
          artifact_error("autotune metadata is invalid");
        }
        stage.autotune = true;
        stage.selection_cache =
            artifact_directory /
            (".flagdnn-autotune-v1-stage-" + std::to_string(index) + "-" +
             stage.candidate_identity + ".json");
      }
      result.stages.push_back(std::move(stage));
    }

    for (const auto &[node_id, position] : node_positions) {
      (void)position;
      if (node_stages.find(node_id) == node_stages.end()) {
        artifact_error("execution program does not cover every Graph node");
      }
    }
    if (bound_external_uids !=
        std::set<std::int64_t>(result.external_uids.begin(),
                               result.external_uids.end())) {
      artifact_error("kernel ABIs do not cover the external binding set");
    }

    std::map<std::int64_t, std::size_t> previous_node_stage;
    for (const ExecutionStage &stage : result.stages) {
      std::set<std::size_t> expected_dependencies;
      for (const std::int64_t node_id : stage.source_node_ids) {
        const auto previous = previous_node_stage.find(node_id);
        if (previous != previous_node_stage.end()) {
          expected_dependencies.insert(previous->second);
          previous->second = stage.stage_id;
          continue;
        }
        const GraphNode &node = graph_nodes.at(node_positions.at(node_id));
        for (const std::int64_t uid : node.inputs) {
          const auto producer = tensor_producers.find(uid);
          if (producer == tensor_producers.end() ||
              std::find(stage.source_node_ids.begin(),
                        stage.source_node_ids.end(),
                        producer->second) != stage.source_node_ids.end()) {
            continue;
          }
          const auto producer_stages = node_stages.find(producer->second);
          if (producer_stages == node_stages.end() ||
              producer_stages->second.back() >= stage.stage_id) {
            artifact_error("execution stage order violates the Graph DAG");
          }
          expected_dependencies.insert(producer_stages->second.back());
        }
        previous_node_stage.emplace(node_id, stage.stage_id);
      }
      if (stage.dependencies !=
          std::vector<std::size_t>(expected_dependencies.begin(),
                                   expected_dependencies.end())) {
        artifact_error("execution stage dependencies differ from Graph DAG");
      }
    }
    return result;
  } catch (const IluvatarError &) {
    throw;
  } catch (const std::exception &error) {
    artifact_error("invalid Iluvatar artifact: " + std::string(error.what()));
  }
}

} // namespace flagdnn::iluvatar
