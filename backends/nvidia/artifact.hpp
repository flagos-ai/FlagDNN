/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_ARTIFACT_HPP_
#define FLAGDNN_BACKENDS_CUDA_ARTIFACT_HPP_

#include "backends/backend_api.h"
#include "backends/nvidia/context.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace flagdnn::cuda {

enum class EngineKind {
  kExternalArtifact,
  kLibTritonJit,
};

enum class ArgumentKind {
  kTensor,
  kWorkspaceTensor,
  kScalarI32,
  kScalarF32,
};

struct ArgumentSpec {
  ArgumentKind kind = ArgumentKind::kTensor;
  std::int64_t uid = 0;
  std::int32_t scalar_i32 = 0;
  float scalar_f32 = 0.0F;
  std::size_t workspace_offset = 0;
  std::size_t storage_size = 0;
  std::size_t alignment = 1;
};

struct CudaKernelArtifact {
  std::string variant_id = "default";
  std::filesystem::path binary;
  std::string entry_symbol;
  std::string full_signature;
  unsigned int num_warps = 0;
  unsigned int num_stages = 0;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
  std::size_t global_scratch_size = 0;
  std::size_t profile_scratch_size = 0;
  std::vector<ArgumentSpec> arguments;
  std::vector<std::int64_t> binding_uids;
};

struct CudaStageArtifact {
  std::filesystem::path source;
  std::string function_name;
  std::vector<CudaKernelArtifact> variants;
  bool autotune = false;
  unsigned int warmup = 0;
  unsigned int repetitions = 1;
  std::string candidate_identity;
  std::filesystem::path selection_cache;
};

struct CudaArtifact {
  EngineKind engine = EngineKind::kExternalArtifact;
  std::vector<CudaStageArtifact> stages;
  std::vector<std::int64_t> binding_uids;
  std::size_t workspace_size = 0;
};

[[nodiscard]] CudaArtifact parse_cuda_artifact(
    const EngineBuildContext& context,
    const flagdnnBackendBuildInputV2& input);

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_ARTIFACT_HPP_
