/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/ascend/extended_artifact.hpp"
#include "ascend_kernel_sources.hpp"
#include "backends/ascend/error.hpp"
#include "runtime/sha256.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string_view>

namespace flagdnn::ascend {
namespace {
using native::json::Value;
void check(bool ok, const char *message) {
  if (!ok)
    throw AscendError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, message);
}
std::size_t size(const Value &value) {
  const auto n = value.as_int();
  check(n >= 0, "negative Ascend extended size");
  return static_cast<std::size_t>(n);
}
std::size_t multiply(std::size_t a, std::size_t b) {
  check(!a || b <= static_cast<std::size_t>(INT64_MAX) / a,
        "Ascend extended size overflow");
  return a * b;
}
std::size_t width(std::string_view type) {
  if (type == "float32" || type == "int32")
    return 4;
  if (type == "float16" || type == "bfloat16")
    return 2;
  if (type == "boolean" || type == "fp8_e4m3" || type == "fp8_e5m2" ||
      type == "fp8_e8m0")
    return 1;
  check(false, "unsupported Ascend extended tensor type");
  return 0;
}
struct Tensor {
  std::size_t bytes, alignment, offset;
  bool temporary;
  std::size_t element_bytes;
};
std::string read_source(const std::filesystem::path &path, std::size_t bytes) {
  check(bytes > 0 && bytes <= (1U << 20),
        "invalid Ascend extended source size");
  check(!std::filesystem::is_symlink(path) &&
            std::filesystem::is_regular_file(path),
        "invalid Ascend extended source file");
  check(std::filesystem::file_size(path) == bytes,
        "Ascend extended source size mismatch");
  std::ifstream stream(path, std::ios::binary);
  std::string text(bytes, '\0');
  stream.read(text.data(), bytes);
  check(static_cast<std::size_t>(stream.gcount()) == bytes,
        "cannot read Ascend extended source");
  return text;
}
} // namespace

AscendArtifact parse_extended_artifact(const Value &request,
                                       const Value &manifest,
                                       const std::filesystem::path &directory) {
  const auto &graph = request.at("graph");
  const auto &nodes = graph.at("nodes").as_array();
  check(!nodes.empty() && nodes.size() <= 1024 &&
            size(graph.at("node_count")) == nodes.size() &&
            size(manifest.at("graph_node_count")) == nodes.size(),
        "invalid Ascend extended graph size");
  std::map<std::int64_t, Tensor> tensors;
  AscendArtifact result;
  const auto &table = graph.at("tensors").as_array();
  check(!table.empty() && size(graph.at("tensor_count")) == table.size(),
        "invalid Ascend extended tensor count");
  for (const auto &value : table) {
    const auto uid = value.at("uid").as_int();
    const auto &dims = value.at("dimensions").as_array();
    const auto &strides = value.at("strides").as_array();
    check(uid > 0 && !tensors.contains(uid) && dims.size() == strides.size() &&
              dims.size() <= 8,
          "invalid Ascend extended tensor metadata");
    std::size_t span = 1;
    for (std::size_t i = 0; i < dims.size(); ++i) {
      const auto d = size(dims[i]), stride = size(strides[i]);
      check(d > 0, "empty Ascend extended tensor");
      const auto tail = multiply(d - 1, stride);
      check(tail <= static_cast<std::size_t>(INT64_MAX) - span,
            "Ascend tensor span overflow");
      span += tail;
    }
    const auto alignment = size(value.at("alignment"));
    check(alignment && alignment <= 256 && !(alignment & (alignment - 1)),
          "invalid Ascend tensor alignment");
    Tensor tensor{multiply(span, width(value.at("data_type").as_string())),
                  alignment, 0, value.at("virtual").as_bool(),
                  width(value.at("data_type").as_string())};
    tensors.emplace(uid, tensor);
  }
  std::map<std::int64_t, std::int64_t> private_delta;
  std::int64_t next_uid = tensors.rbegin()->first;
  for (const auto &node : nodes)
    if (node.at("type").as_string() == "sdpa_backward") {
      check(next_uid < INT64_MAX, "Ascend private tensor UID overflow");
      ++next_uid;
      const auto q_uid =
          node.at("inputs").as_array().front().at("uid").as_int();
      const auto q =
          std::find_if(table.begin(), table.end(), [&](const auto &t) {
            return t.at("uid").as_int() == q_uid;
          });
      check(q != table.end() && q->at("dimensions").as_array().size() == 4,
            "invalid Ascend SDPA query shape");
      std::size_t bytes = 4;
      for (std::size_t i = 0; i < 3; ++i)
        bytes = multiply(bytes, size(q->at("dimensions").as_array()[i]));
      tensors.emplace(next_uid, Tensor{bytes, 256, 0, true, 4});
      private_delta.emplace(node.at("id").as_int(), next_uid);
    }
  // Same deterministic UID ordering and alignment as dispatch.extended.
  for (auto &[uid, tensor] : tensors) {
    if (tensor.temporary) {
      check(result.workspace_size <=
                static_cast<std::size_t>(INT64_MAX) - 255 - tensor.bytes,
            "Ascend workspace overflow");
      result.workspace_size = (result.workspace_size + 255) / 256 * 256;
      tensor.offset = result.workspace_size;
      result.workspace_size += tensor.bytes;
    } else
      result.binding_uids.push_back(uid);
  }
  check(size(manifest.at("workspace_size")) == result.workspace_size,
        "Ascend extended workspace mismatch");
  const auto &program = manifest.at("program");
  const auto &stages = program.at("stages").as_array();
  check(program.at("schema_version").as_int() == 4 &&
            size(program.at("stage_count")) == stages.size() &&
            !stages.empty() &&
            stages.size() <= FLAGDNN_BACKEND_MAX_EXECUTION_STAGES,
        "invalid Ascend extended program");
  std::map<std::int64_t, const Value *> node_table;
  for (const auto &node : nodes) {
    const auto id = node.at("id").as_int();
    check(id >= 0 && static_cast<std::size_t>(id) < nodes.size() &&
              node_table.emplace(id, &node).second,
          "invalid Ascend extended node ID");
  }
  std::set<std::int64_t> covered;
  std::string source_hashes = "[";
  for (std::size_t i = 0; i < stages.size(); ++i) {
    const auto &raw = stages[i];
    check(size(raw.at("stage_id")) == i &&
              raw.at("kernel_family").as_string() == "extended",
          "invalid Ascend extended stage");
    const auto &ids = raw.at("source_node_ids").as_array();
    check(ids.size() == 1 && node_table.contains(ids[0].as_int()),
          "invalid Ascend extended source node");
    const auto node_id = ids[0].as_int();
    const auto &node = *node_table.at(node_id);
    check(raw.at("operation").as_string() == node.at("type").as_string(),
          "Ascend extended operation mismatch");
    check(covered.insert(node_id).second ||
              (node.at("type").as_string() == "concatenate" ||
               node.at("type").as_string() == "sdpa_backward"),
          "duplicate Ascend extended producer");
    AscendStageArtifact stage;
    stage.stage_id = i;
    stage.source_node_ids = {static_cast<std::size_t>(node_id)};
    stage.operation = raw.at("operation").as_string();
    for (const auto &d : raw.at("dependencies").as_array()) {
      check(size(d) < i, "non-topological Ascend extended dependency");
      stage.dependencies.push_back(size(d));
    }
    std::set<std::int64_t> ports;
    for (const char *direction : {"inputs", "outputs"})
      for (const auto &port : node.at(direction).as_array())
        ports.insert(port.at("uid").as_int());
    if (private_delta.contains(node_id))
      ports.insert(private_delta.at(node_id));
    const auto &kernel = raw.at("kernel");
    const auto source_name = kernel.at("source").as_string();
    const auto source_hash = kernel.at("source_sha256").as_string();
    const auto pin =
        std::find_if(kAscendKernelSources.begin(),
                     kAscendKernelSources.end(), [&](const auto &p) {
                       return p.first == source_name && p.second == source_hash;
                     });
    check(pin != kAscendKernelSources.end(),
          "Ascend extended kernel source is not pinned to this build");
    const auto &asset = kernel.at("materialized_source");
    const std::string filename = asset.at("file").as_string();
    check(filename == "source-" + request.at("compiler_identity").as_string() +
                          "-" + source_hash + ".py" &&
              asset.at("sha256").as_string() == source_hash,
          "invalid Ascend extended materialized source identity");
    stage.source = directory / filename;
    stage.source_sha256 = source_hash;
    stage.entry_point = kernel.at("entry_point").as_string();
    const auto text = read_source(stage.source, size(asset.at("size")));
    check(native::sha256(text) == source_hash &&
              text.find("def " + stage.entry_point + "(") != std::string::npos,
          "Ascend extended source content mismatch");
    const auto &args = raw.at("argument_sources").as_array();
    check(!args.empty() && args.size() <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
          "invalid Ascend extended argument count");
    for (std::size_t j = 0; j < args.size(); ++j) {
      const auto &a = args[j];
      if (a.at("source").as_string() == "scalar") {
        ArgumentSource argument;
        argument.index = j;
        argument.name = a.at("name").as_string();
        argument.source = ArgumentSourceKind::kScalar;
        check(size(a.at("index")) == j, "invalid Ascend scalar index");
        const auto type = a.at("type").as_string();
        if (type == "i32") {
          const auto value = a.at("value").as_int();
          check(value >= INT32_MIN && value <= INT32_MAX,
                "Ascend scalar exceeds int32");
          argument.type = RawArgumentType::kI32;
          argument.scalar = static_cast<std::int32_t>(value);
        } else {
          check(type == "f32", "invalid Ascend scalar type");
          const auto value = a.at("value").as_double();
          check(std::isfinite(value) &&
                    std::abs(value) <= std::numeric_limits<float>::max(),
                "Ascend scalar exceeds float32");
          argument.type = RawArgumentType::kF32;
          argument.scalar = static_cast<float>(value);
        }
        stage.arguments.push_back(argument);
        continue;
      }
      const auto uid = a.at("uid").as_int();
      check(ports.contains(uid) && tensors.contains(uid) &&
                size(a.at("index")) == j &&
                a.at("type").as_string() == "pointer",
            "invalid Ascend extended tensor binding");
      const auto &tensor = tensors.at(uid);
      ArgumentSource argument;
      argument.index = j;
      argument.name = a.at("name").as_string();
      argument.uid = uid;
      argument.size = tensor.bytes;
      argument.alignment = tensor.temporary ? 256 : tensor.alignment;
      argument.source = tensor.temporary ? ArgumentSourceKind::kGraphWorkspace
                                         : ArgumentSourceKind::kBinding;
      argument.workspace_offset = tensor.offset;
      check(size(a.at("size")) == argument.size &&
                size(a.at("alignment")) == argument.alignment &&
                a.at("source").as_string() ==
                    (tensor.temporary ? "graph_workspace" : "binding"),
            "Ascend extended argument metadata mismatch");
      if (tensor.temporary)
        check(size(a.at("offset")) == tensor.offset,
              "Ascend extended workspace offset mismatch");
      stage.arguments.push_back(argument);
    }
    const auto &candidates = raw.at("candidates").as_array();
    check(candidates.size() == 1 &&
              candidates[0].at("candidate_id").as_string() == "default" &&
              candidates[0].at("launch_abi").as_string() == "ltj_npu_raw_v1",
          "invalid Ascend extended candidate");
    const auto &payload = candidates[0].at("payload");
    check(payload.at("schema_version").as_int() == 1 &&
              payload.at("source_path").as_string() == filename &&
              payload.at("source_sha256").as_string() == source_hash &&
              payload.at("entry_point").as_string() == stage.entry_point,
          "Ascend extended launch identity mismatch");
    LtjNpuRawCandidate candidate;
    candidate.candidate_id = "default";
    candidate.standalone_compilation = true;
    candidate.source = stage.source;
    candidate.source_sha256 = source_hash;
    candidate.entry_point = stage.entry_point;
    candidate.full_signature = payload.at("full_signature").as_string();
    check(candidate.full_signature.size() < (1U << 20),
          "Ascend extended signature too large");
    std::istringstream signature(candidate.full_signature);
    std::string token;
    std::size_t pointers = 0;
    while (std::getline(signature, token, ',')) {
      check(!token.empty(), "empty Ascend extended signature token");
      if (token[0] == '*') {
        check(token == "*fp32" || token == "*fp16" || token == "*bf16" ||
                  token == "*i32" || token == "*i8" || token == "*u8" ||
                  token == "*u16" || token == "*i16",
              "invalid Ascend extended pointer token");
        const auto argument_index = candidate.argument_types.size();
        check(argument_index < stage.arguments.size() &&
                  stage.arguments[argument_index].type ==
                      RawArgumentType::kPointer,
              "Ascend extended pointer position mismatch");
        const auto uid = stage.arguments[argument_index].uid;
        const std::size_t token_bytes =
            token == "*fp32" || token == "*i32" ? 4
            : token == "*fp16" || token == "*bf16" || token == "*u16" ||
                    token == "*i16"
                ? 2
                : 1;
        check(tensors.at(uid).element_bytes == token_bytes,
              "Ascend extended pointer storage width mismatch");
        ++pointers;
        candidate.argument_types.push_back(RawArgumentType::kPointer);
      } else if (token == "i32" || token == "fp32") {
        candidate.argument_types.push_back(
            token == "i32" ? RawArgumentType::kI32 : RawArgumentType::kF32);
      } else {
        char *end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        check(end == token.c_str() + token.size() && std::isfinite(value),
              "invalid Ascend extended constexpr");
      }
    }
    check(pointers > 0 && candidate.argument_types.size() == args.size(),
          "Ascend extended signature argument count mismatch");
    for (std::size_t j = 0; j < args.size(); ++j)
      check(candidate.argument_types[j] == stage.arguments[j].type,
            "Ascend extended signature argument type mismatch");
    const auto &grid = payload.at("grid").as_array();
    check(grid.size() == 3, "invalid Ascend extended grid");
    std::size_t blocks = 1;
    for (std::size_t j = 0; j < 3; ++j) {
      const auto n = size(grid[j]);
      check(n > 0 && n <= INT32_MAX, "invalid Ascend extended grid dimension");
      blocks = multiply(blocks, n);
      candidate.grid[j] = static_cast<unsigned>(n);
    }
    check(blocks <= UINT32_MAX, "Ascend extended grid overflow");
    const auto &options = payload.at("compile_options");
    check(options.at("num_warps").as_int() == 4 &&
              options.at("num_stages").as_int() == 1,
          "invalid Ascend extended compile options");
    candidate.num_warps = 4;
    candidate.num_stages = 1;
    check(!raw.at("autotune").at("enabled").as_bool(),
          "Ascend extended candidate must be fixed");
    stage.selected_candidate = "default";
    stage.candidates.push_back(std::move(candidate));
    result.stages.push_back(std::move(stage));
    if (i)
      source_hashes += ',';
    source_hashes += '"' + source_hash + '"';
  }
  source_hashes += ']';
  check(covered.size() == nodes.size() &&
            native::sha256(source_hashes) ==
                manifest.at("source_sha256").as_string(),
        "Ascend extended graph/source coverage mismatch");
  return result;
}
} // namespace flagdnn::ascend
