/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_ARTIFACT_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_ARTIFACT_HPP_

#include "backends/backend_api.h"
#include "backends/mthreads/context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace flagdnn::mthreads {

inline constexpr std::int64_t kMthreadsArtifactSchemaVersion = 1;
inline constexpr std::int64_t kMthreadsExecutionProgramVersion = 1;

enum class ArgumentKind {
  kTensor,
  kWorkspace,
  kScalarI32,
  kScalarF32,
};

struct ArgumentSpec {
  ArgumentKind kind = ArgumentKind::kTensor;
  std::string semantic_name;
  std::string data_type;
  std::int64_t uid = 0;
  std::uint32_t scalar_bits = 0;
  std::size_t storage_size = 0;
  std::size_t alignment = 1;
  bool tensor_descriptor = false;
  std::array<std::int32_t, 2> descriptor_shape = {0, 0};
  std::array<std::int64_t, 2> descriptor_strides = {0, 0};
  std::array<std::uint32_t, 2> descriptor_block_shape = {0, 0};
};

struct KernelVariantArtifact {
  std::string variant_id;
  std::filesystem::path source;
  std::string function_name;
  std::string full_signature;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  unsigned int num_warps = 0;
  unsigned int num_stages = 0;
  std::vector<ArgumentSpec> arguments;
};

struct StageArtifact {
  std::size_t id = 0;
  std::int64_t node_id = 0;
  std::vector<std::size_t> dependencies;
  std::filesystem::path source;
  std::string function_name;
  std::vector<KernelVariantArtifact> variants;
  bool autotune = false;
  unsigned int warmup = 0;
  unsigned int repetitions = 0;
  std::filesystem::path selection_cache;
};

struct MthreadsArtifact {
  std::vector<StageArtifact> stages;
  std::vector<std::int64_t> binding_uids;
  std::size_t workspace_size = 0;
  std::size_t workspace_alignment = 1;
};

[[nodiscard]] MthreadsArtifact parse_mthreads_artifact(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input);

}  // namespace flagdnn::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_ARTIFACT_HPP_
