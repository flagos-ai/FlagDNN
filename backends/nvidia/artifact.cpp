/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/artifact.hpp"

#include "backends/nvidia/error.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <flagdnn/version.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace flagdnn::cuda {
namespace {

constexpr std::int64_t kArtifactSchemaVersion = 4;
constexpr std::int64_t kExecutionProgramVersion = 2;
constexpr std::size_t kMaximumAutotuneCandidates = 1024;

std::string read_text_file(const std::filesystem::path& path,
                           std::size_t maximum_size) {
  std::error_code error;
  const std::uintmax_t file_size = std::filesystem::file_size(path, error);
  if (error || file_size > maximum_size) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "cannot stat artifact metadata or it exceeds size limit");
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "cannot open artifact metadata");
  }
  std::string result(static_cast<std::size_t>(file_size), '\0');
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input && !result.empty()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "cannot read artifact metadata");
  }
  return result;
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  return std::all_of(
      value.begin(), value.end(), [](const unsigned char character) {
        return std::isxdigit(character) != 0;
      });
}

unsigned int checked_positive_unsigned(std::int64_t value,
                                       const char* field) {
  if (value <= 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<unsigned int>::max()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("invalid artifact field: ") + field);
  }
  return static_cast<unsigned int>(value);
}

unsigned int checked_nonnegative_unsigned(std::int64_t value,
                                          const char* field) {
  if (value < 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<unsigned int>::max()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("invalid artifact field: ") + field);
  }
  return static_cast<unsigned int>(value);
}

std::size_t checked_size(std::int64_t value, const char* field) {
  if (value < 0 ||
      static_cast<std::uint64_t>(value) >
          std::numeric_limits<std::size_t>::max()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("invalid artifact field: ") + field);
  }
  return static_cast<std::size_t>(value);
}

std::int32_t checked_i32(std::int64_t value, const char* field) {
  if (value < std::numeric_limits<std::int32_t>::min() ||
      value > std::numeric_limits<std::int32_t>::max()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("invalid artifact field: ") + field);
  }
  return static_cast<std::int32_t>(value);
}

float checked_f32(double value, const char* field) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max()) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("invalid artifact field: ") + field);
  }
  return static_cast<float>(value);
}

std::array<unsigned int, 3> parse_triplet(
    const flagdnn::native::json::Value& value,
    const char* field) {
  const auto& array = value.as_array();
  if (array.size() != 3) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
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

std::filesystem::path validate_file(
    const std::filesystem::path& directory,
    const flagdnn::native::json::Value& descriptor,
    std::size_t maximum_size,
    const char* label) {
  const std::string name = descriptor.at("file").as_string();
  if (!is_safe_basename(name)) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    std::string("artifact ") + label + " path is unsafe");
  }
  const std::filesystem::path path = directory / name;
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    throw CudaError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        std::string("artifact ") + label +
            " is missing or not a regular file");
  }
  const std::size_t expected_size =
      checked_size(descriptor.at("size").as_int(), label);
  const std::uintmax_t actual_size = std::filesystem::file_size(path, error);
  if (error || expected_size == 0 || expected_size > maximum_size ||
      actual_size != expected_size) {
    throw CudaError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        std::string("artifact ") + label + " size does not match manifest");
  }
  const std::string expected_hash = descriptor.at("sha256").as_string();
  if (!is_sha256(expected_hash) ||
      flagdnn::native::sha256_file(path) != expected_hash) {
    throw CudaError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        std::string("artifact ") + label + " SHA-256 does not match manifest");
  }
  return path;
}

bool is_identifier(std::string_view value) {
  if (value.empty() || value.size() > 256 ||
      (std::isalpha(static_cast<unsigned char>(value.front())) == 0 &&
       value.front() != '_')) {
    return false;
  }
  return std::all_of(
      value.begin() + 1, value.end(), [](const unsigned char character) {
        return std::isalnum(character) != 0 || character == '_';
      });
}

std::vector<ArgumentSpec> parse_argument_abi(
    const flagdnn::native::json::Value& value,
    std::size_t workspace_size,
    std::vector<std::int64_t>& binding_uids) {
  const auto& abi = value.as_array();
  if (abi.size() < 3 ||
      abi.size() > FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "artifact argument ABI count is invalid");
  }
  if (abi[abi.size() - 2].at("kind").as_string() !=
          "global_scratch_pointer" ||
      abi[abi.size() - 1].at("kind").as_string() !=
          "profile_scratch_pointer") {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "artifact hidden scratch ABI is incompatible");
  }

  std::vector<ArgumentSpec> result;
  result.reserve(abi.size() - 2);
  for (std::size_t index = 0; index + 2 < abi.size(); ++index) {
    const std::string& kind = abi[index].at("kind").as_string();
    if (kind == "tensor") {
      const auto& argument_object = abi[index].as_object();
      const std::int64_t uid = abi[index].at("uid").as_int();
      const std::size_t size = checked_size(
          abi[index].at("size").as_int(), "argument_abi.size");
      const auto alignment_entry = argument_object.find("alignment");
      const std::size_t alignment =
          alignment_entry == argument_object.end()
              ? 1
              : checked_size(alignment_entry->second.as_int(),
                             "argument_abi.alignment");
      if (uid <= 0 || size == 0 || alignment == 0 ||
          (alignment & (alignment - 1)) != 0) {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "artifact tensor argument metadata is invalid");
      }
      result.push_back(
          {ArgumentKind::kTensor, uid, 0, 0.0F, 0, size, alignment});
      if (std::find(binding_uids.begin(), binding_uids.end(), uid) ==
          binding_uids.end()) {
        binding_uids.push_back(uid);
      }
    } else if (kind == "workspace_tensor") {
      const std::int64_t uid = abi[index].at("uid").as_int();
      const std::size_t offset = checked_size(
          abi[index].at("offset").as_int(), "argument_abi.offset");
      const std::size_t size = checked_size(
          abi[index].at("size").as_int(), "argument_abi.size");
      if (uid <= 0 || size == 0 || offset % 256 != 0 ||
          offset > workspace_size || size > workspace_size - offset) {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "artifact workspace tensor range is invalid");
      }
      result.push_back(
          {ArgumentKind::kWorkspaceTensor, uid, 0, 0.0F, offset, size});
    } else if (kind == "scalar_i32") {
      result.push_back(
          {ArgumentKind::kScalarI32,
           0,
           checked_i32(abi[index].at("value").as_int(),
                       "argument_abi.value"),
           0.0F,
           0,
           0});
    } else if (kind == "scalar_f32") {
      result.push_back(
          {ArgumentKind::kScalarF32,
           0,
           0,
           checked_f32(abi[index].at("value").as_double(),
                       "argument_abi.value"),
           0,
           0});
    } else {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact contains an unsupported argument kind");
    }
  }
  return result;
}

CudaKernelArtifact parse_kernel(
    EngineKind engine,
    const flagdnn::native::json::Value& entry,
    const std::filesystem::path& artifact_directory,
    std::size_t workspace_size) {
  const std::string kernel_source_hash =
      entry.at("source_sha256").as_string();
  if (!is_sha256(kernel_source_hash)) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "artifact kernel source identity is invalid");
  }

  CudaKernelArtifact result;
  const auto& entry_object = entry.as_object();
  const auto variant_entry = entry_object.find("variant_id");
  if (variant_entry != entry_object.end()) {
    result.variant_id = variant_entry->second.as_string();
  }
  if (result.variant_id.empty() || result.variant_id.size() > 128 ||
      std::any_of(
          result.variant_id.begin(),
          result.variant_id.end(),
          [](const unsigned char character) {
            return std::isalnum(character) == 0 && character != '_' &&
                   character != '-';
          })) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "artifact variant ID is invalid");
  }

  if (engine == EngineKind::kExternalArtifact) {
    result.binary = validate_file(artifact_directory,
                                  entry.at("binary"),
                                  1U << 30,
                                  "binary");
    result.entry_symbol = entry.at("entry_symbol").as_string();
    if (result.entry_symbol.empty() || result.entry_symbol.size() > 1024) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact entry symbol is invalid");
    }
  } else {
    result.full_signature = entry.at("full_signature").as_string();
    if (result.full_signature.empty() ||
        result.full_signature.size() > (64U << 10) ||
        std::any_of(
            result.full_signature.begin(),
            result.full_signature.end(),
            [](const unsigned char character) {
              return character < 0x21U || character > 0x7eU;
            })) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "libtriton_jit full signature is invalid");
    }
    const auto& options = entry.at("compile_options");
    result.num_warps = checked_positive_unsigned(
        options.at("num_warps").as_int(), "compile_options.num_warps");
    result.num_stages = checked_positive_unsigned(
        options.at("num_stages").as_int(), "compile_options.num_stages");
    if (result.num_warps > 32 || result.num_stages > 32) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "libtriton_jit compile options exceed safety limits");
    }
  }

  result.arguments = parse_argument_abi(
      entry.at("argument_abi"), workspace_size, result.binding_uids);
  const auto& launch = entry.at("launch");
  result.grid = parse_triplet(launch.at("grid"), "launch.grid");
  result.block = parse_triplet(launch.at("block"), "launch.block");
  const auto cluster = parse_triplet(launch.at("cluster"), "launch.cluster");
  const std::uint64_t block_threads =
      static_cast<std::uint64_t>(result.block[0]) * result.block[1] *
      result.block[2];
  if (block_threads > 1024) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "CUDA block contains more than 1024 threads");
  }
  if (cluster != std::array<unsigned int, 3>{1, 1, 1} ||
      launch.at("num_ctas").as_int() != 1) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                    "NVIDIA backend v2 does not support cluster launch");
  }
  result.shared_memory = checked_nonnegative_unsigned(
      launch.at("shared_memory").as_int(), "launch.shared_memory");
  result.global_scratch_size = checked_size(
      launch.at("global_scratch_size").as_int(),
      "launch.global_scratch_size");
  result.profile_scratch_size = checked_size(
      launch.at("profile_scratch_size").as_int(),
      "launch.profile_scratch_size");
  if (engine == EngineKind::kExternalArtifact &&
      (result.global_scratch_size != 0 ||
       result.profile_scratch_size != 0)) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                    "external NVIDIA artifacts do not support scratch buffers yet");
  }
  if (engine == EngineKind::kLibTritonJit &&
      (result.global_scratch_size == 0 ||
       result.global_scratch_size > workspace_size ||
       result.global_scratch_size % 256 != 0 ||
       result.profile_scratch_size != 0)) {
    throw CudaError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "libtriton_jit scratch metadata is incompatible with workspace");
  }
  const bool valid_jit_block =
      result.block[1] == 1U && result.block[2] == 1U &&
      result.block[0] == result.num_warps * 32U;
  if (engine == EngineKind::kLibTritonJit &&
      (!valid_jit_block || result.shared_memory != 0)) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "libtriton_jit launch plan is inconsistent");
  }
  return result;
}

bool same_argument_abi(const CudaKernelArtifact& left,
                       const CudaKernelArtifact& right) {
  if (left.binding_uids != right.binding_uids ||
      left.arguments.size() != right.arguments.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.arguments.size(); ++index) {
    const ArgumentSpec& a = left.arguments[index];
    const ArgumentSpec& b = right.arguments[index];
    if (a.kind != b.kind || a.uid != b.uid ||
        a.scalar_i32 != b.scalar_i32 ||
        a.scalar_f32 != b.scalar_f32 ||
        a.workspace_offset != b.workspace_offset ||
        a.storage_size != b.storage_size ||
        a.alignment != b.alignment) {
      return false;
    }
  }
  return true;
}

EngineKind parse_engine(std::string_view value) {
  if (value == "external_artifact") {
    return EngineKind::kExternalArtifact;
  }
  if (value == "libtriton_jit") {
    return EngineKind::kLibTritonJit;
  }
  throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                  "artifact execution engine is invalid");
}

}  // namespace

CudaArtifact parse_cuda_artifact(const EngineBuildContext& context,
                                 const flagdnnBackendBuildInputV2& input) {
  try {
    require(input.graph_ir != nullptr && input.graph_ir_size != 0,
            "graph IR is empty");
    require(input.artifact_directory != nullptr,
            "artifact directory is null");
    require(input.request_sha256 != nullptr, "request SHA-256 is null");
    const std::string_view graph_ir(
        static_cast<const char*>(input.graph_ir), input.graph_ir_size);
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
        request_root.at("backend").as_string() != "nvidia" ||
        request_root.at("target").as_string() != context.target_fingerprint ||
        !is_sha256(compiler_identity)) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "CUDA build request identity is invalid");
    }

    const std::filesystem::path artifact_directory(input.artifact_directory);
    const auto root = flagdnn::native::json::parse(
        read_text_file(artifact_directory / "manifest.json", 16U << 20));
    if (root.at("schema_version").as_int() != kArtifactSchemaVersion ||
        root.at("artifact_kind").as_string() !=
            "flagdnn_execution_program" ||
        root.at("flagdnn_version").as_string() != FLAGDNN_VERSION_STRING ||
        root.at("backend").as_string() != "nvidia" ||
        root.at("target").as_string() != context.target_fingerprint ||
        root.at("request_sha256").as_string() != request_hash ||
        root.at("compiler").at("identity_sha256").as_string() !=
            compiler_identity) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact target or version does not match build input");
    }
    const std::string source_hash = root.at("source_sha256").as_string();
    if (!is_sha256(source_hash) ||
        root.at("compiler").at("provider").as_string().empty() ||
        root.at("compiler").at("triton_version").as_string().empty()) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact compiler/source identity is invalid");
    }

    CudaArtifact result;
    const auto& root_object = root.as_object();
    const auto workspace_entry = root_object.find("workspace_size");
    result.workspace_size =
        workspace_entry == root_object.end()
            ? 0
            : checked_size(workspace_entry->second.as_int(),
                           "workspace_size");
    const std::size_t graph_node_count = checked_size(
        root.at("graph_node_count").as_int(), "graph_node_count");
    const auto& program = root.at("program");
    if (program.at("schema_version").as_int() !=
        kExecutionProgramVersion) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "CUDA Execution Program version is unsupported");
    }
    const auto& stages = program.at("stages").as_array();
    const std::size_t stage_count = checked_size(
        program.at("stage_count").as_int(), "stage_count");
    if (stage_count == 0 ||
        stage_count > FLAGDNN_BACKEND_MAX_EXECUTION_STAGES ||
        stages.size() != stage_count) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact Execution Program stage count is invalid");
    }

    result.stages.reserve(stage_count);
    bool engine_initialized = false;
    for (std::size_t index = 0; index < stages.size(); ++index) {
      const auto& stage = stages[index];
      if (checked_size(stage.at("stage_id").as_int(), "stage_id") != index ||
          stage.at("kind").as_string() != "kernel") {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "artifact Execution Program stage is invalid");
      }
      const auto& source_nodes = stage.at("source_node_ids").as_array();
      if (source_nodes.empty()) {
        throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                        "execution stage has no source graph nodes");
      }
      std::vector<std::size_t> seen_source_nodes;
      for (const auto& source_node : source_nodes) {
        const std::size_t node_id =
            checked_size(source_node.as_int(), "source_node_id");
        if (node_id >= graph_node_count ||
            std::find(seen_source_nodes.begin(),
                      seen_source_nodes.end(),
                      node_id) != seen_source_nodes.end()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "execution stage source node list is invalid");
        }
        seen_source_nodes.push_back(node_id);
      }
      const auto& dependencies = stage.at("dependencies").as_array();
      std::vector<std::size_t> seen_dependencies;
      for (const auto& dependency : dependencies) {
        const std::size_t dependency_id =
            checked_size(dependency.as_int(), "stage dependency");
        if (dependency_id >= index ||
            std::find(seen_dependencies.begin(),
                      seen_dependencies.end(),
                      dependency_id) != seen_dependencies.end()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
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
        throw CudaError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        "mixed execution engines are not supported yet");
      }

      CudaStageArtifact parsed_stage;
      parsed_stage.selection_cache =
          artifact_directory /
          (".flagdnn-autotune-v1-stage-" + std::to_string(index) + "-" +
           context.device_identity + ".json");
      if (stage_engine == EngineKind::kLibTritonJit) {
        const auto& kernel = stage.at("kernel");
        parsed_stage.source = validate_file(
            artifact_directory,
            kernel.at("materialized_source"),
            1U << 20,
            "JIT source");
        parsed_stage.function_name = kernel.at("function").as_string();
        if (!is_identifier(parsed_stage.function_name)) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "libtriton_jit function name is invalid");
        }
      }

      const auto& stage_object = stage.as_object();
      const auto variants_entry = stage_object.find("variants");
      if (variants_entry == stage_object.end()) {
        parsed_stage.variants.push_back(parse_kernel(
            stage_engine, stage, artifact_directory, result.workspace_size));
      } else {
        parsed_stage.autotune = true;
        const auto& variants = variants_entry->second.as_array();
        if (variants.size() < 2 ||
            variants.size() > kMaximumAutotuneCandidates) {
          throw CudaError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "autotune stage candidate count must be in [2, 1024]");
        }
        const auto& tuning = stage.at("tuning");
        if (tuning.at("schema_version").as_int() != 1) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "autotune metadata schema is unsupported");
        }
        parsed_stage.warmup = checked_nonnegative_unsigned(
            tuning.at("warmup").as_int(), "tuning.warmup");
        parsed_stage.repetitions = checked_positive_unsigned(
            tuning.at("repetitions").as_int(), "tuning.repetitions");
        parsed_stage.candidate_identity =
            tuning.at("candidate_identity").as_string();
        const std::string tuning_source_hash =
            tuning.at("source_sha256").as_string();
        if (parsed_stage.warmup > 100 ||
            parsed_stage.repetitions > 100 ||
            !is_sha256(parsed_stage.candidate_identity) ||
            !is_sha256(tuning_source_hash) ||
            tuning.at("key").as_string().empty() ||
            tuning.at("strategy").as_string().empty()) {
          throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                          "autotune metadata is invalid");
        }
        parsed_stage.variants.reserve(variants.size());
        for (const auto& variant : variants) {
          CudaKernelArtifact candidate = parse_kernel(
              stage_engine,
              variant,
              artifact_directory,
              result.workspace_size);
          if (std::any_of(
                  parsed_stage.variants.begin(),
                  parsed_stage.variants.end(),
                  [&](const CudaKernelArtifact& existing) {
                    return existing.variant_id == candidate.variant_id;
                  })) {
            throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                            "autotune variant IDs must be unique");
          }
          if (!parsed_stage.variants.empty() &&
              !same_argument_abi(candidate, parsed_stage.variants.front())) {
            throw CudaError(
                FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                "autotune variants have incompatible argument ABIs");
          }
          parsed_stage.variants.push_back(std::move(candidate));
        }
      }

      for (const std::int64_t uid :
           parsed_stage.variants.front().binding_uids) {
        if (std::find(result.binding_uids.begin(),
                      result.binding_uids.end(),
                      uid) == result.binding_uids.end()) {
          result.binding_uids.push_back(uid);
        }
      }
      result.stages.push_back(std::move(parsed_stage));
    }
    if (result.binding_uids.empty()) {
      throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                      "artifact has no external tensor bindings");
    }
    return result;
  } catch (const CudaError&) {
    throw;
  } catch (const std::exception& error) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                    "invalid CUDA artifact manifest: " +
                        std::string(error.what()));
  }
}

}  // namespace flagdnn::cuda
