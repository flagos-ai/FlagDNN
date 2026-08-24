/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ILUVATAR_ARTIFACT_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_ARTIFACT_HPP_

#include "backends/backend_api.h"
#include "backends/iluvatar/context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace flagdnn::iluvatar {

inline constexpr std::int64_t kArtifactSchemaVersion = 1;
inline constexpr std::int64_t kExecutionProgramVersion = 1;

enum class ArgumentKind {
  kTensor,
  kWorkspaceTensor,
  kScalarI32,
  kScalarF32,
};

struct KernelArgument {
  ArgumentKind kind = ArgumentKind::kTensor;
  std::int64_t uid = 0;
  std::int32_t scalar_i32 = 0;
  float scalar_f32 = 0.0F;
  std::size_t workspace_offset = 0;
  std::size_t storage_size = 0;
  std::size_t alignment = 1;
};

struct KernelVariant {
  std::string variant_id;
  std::string full_signature;
  std::vector<KernelArgument> arguments;
  unsigned int num_warps = 0;
  unsigned int num_stages = 0;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

struct ExecutionStage {
  std::size_t stage_id = 0;
  std::vector<std::int64_t> source_node_ids;
  std::vector<std::size_t> dependencies;
  std::string operation;
  std::string provider;
  std::string ownership;
  std::string function_name;
  std::filesystem::path source;
  std::filesystem::path source_relative_path;
  std::string source_sha256;
  std::vector<KernelVariant> variants;
  bool autotune = false;
  unsigned int warmup = 0;
  unsigned int repetitions = 1;
  std::string candidate_identity;
  std::filesystem::path selection_cache;
};

struct TensorArtifact {
  std::int64_t uid = 0;
  std::string data_type;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t alignment = 1;
  bool is_virtual = false;
  std::size_t storage_size = 0;
  std::size_t workspace_offset = 0;
};

struct ExecutionProgramArtifact {
  std::string backend;
  std::string target;
  std::string engine;
  std::string request_sha256;
  std::string compiler_provider;
  std::string compiler_provider_version;
  std::string compiler_identity;
  std::vector<std::int64_t> external_uids;
  std::vector<TensorArtifact> tensors;
  std::size_t workspace_size = 0;
  std::size_t workspace_alignment = 1;
  std::vector<ExecutionStage> stages;
};

[[nodiscard]] ExecutionProgramArtifact
load_and_validate_artifact(const EngineBuildContext &context,
                           const flagdnnBackendBuildInputV2 &input);

} // namespace flagdnn::iluvatar

#endif // FLAGDNN_BACKENDS_ILUVATAR_ARTIFACT_HPP_
