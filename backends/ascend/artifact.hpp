/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_ARTIFACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_ARTIFACT_HPP_

#include "backends/backend_api.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace flagdnn::ascend {

enum class RawArgumentType {
  kPointer,
  kI32,
  kI64,
  kF32,
  kF64,
};

enum class StorageDataType {
  kFloat32,
  kFloat16,
  kBFloat16,
  kBoolean,
};

enum class ArgumentSourceKind {
  kBinding,
  kGraphWorkspace,
  kScalar,
};

enum class KernelFamily {
  kBinary,
  kUnary,
  kTernary,
  kLayout,
  kReduction,
  kMatMul,
  kConvolutionFprop,
  kBatchNorm,
  kBatchNormInference,
  kRmsNorm,
  kLayerNorm,
};

struct ArgumentSource {
  std::size_t index = 0;
  std::string name;
  ArgumentSourceKind source = ArgumentSourceKind::kBinding;
  RawArgumentType type = RawArgumentType::kPointer;
  std::int64_t uid = 0;
  std::size_t size = 0;
  std::size_t alignment = 1;
  std::size_t workspace_offset = 0;
  std::variant<std::int32_t, std::int64_t, float, double> scalar =
      std::int32_t{0};
};

struct LtjNpuRawCandidate {
  std::string candidate_id;
  std::filesystem::path source;
  std::string source_sha256;
  std::string entry_point;
  std::string full_signature;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  unsigned int block_size = 1;
  unsigned int num_warps = 1;
  unsigned int num_stages = 1;
  std::vector<RawArgumentType> argument_types;
  bool standalone_compilation = false;
};

struct AscendStageArtifact {
  std::size_t stage_id = 0;
  std::vector<std::size_t> source_node_ids;
  std::vector<std::size_t> dependencies;
  std::string operation;
  KernelFamily kernel_family = KernelFamily::kBinary;
  std::filesystem::path source;
  std::string source_sha256;
  std::string entry_point;
  std::vector<ArgumentSource> arguments;
  std::vector<LtjNpuRawCandidate> candidates;
  bool autotune = false;
  unsigned int warmup = 0;
  unsigned int repetitions = 0;
  std::string candidate_identity;
  std::string selected_candidate;
  std::vector<StorageDataType> tensor_storage_data_types;
  std::size_t input_count = 0;
  std::int32_t n_elements = 0;
  double alpha = 1.0;
  double negative_slope = 0.0;
  double lower_clip = 0.0;
  double upper_clip = 0.0;
  bool has_upper_clip = false;
  double swish_beta = 1.0;
  double elu_alpha = 1.0;
  double softplus_beta = 1.0;
  std::array<std::int64_t, 8> dimensions{};
  std::array<std::int64_t, 8> left_strides{};
  std::array<std::int64_t, 8> right_strides{};
  std::array<std::int64_t, 8> mask_strides{};
  std::array<std::int64_t, 8> output_strides{};
  std::int64_t matmul_batch = 0;
  std::int64_t matmul_m = 0;
  std::int64_t matmul_n = 0;
  std::int64_t matmul_k = 0;
  std::array<std::int64_t, 6> matmul_batch_dimensions{};
  std::array<std::int64_t, 6> matmul_a_batch_strides{};
  std::array<std::int64_t, 6> matmul_b_batch_strides{};
  std::array<std::int64_t, 6> matmul_output_batch_strides{};
  std::int64_t matmul_a_stride_m = 0;
  std::int64_t matmul_a_stride_k = 0;
  std::int64_t matmul_b_stride_k = 0;
  std::int64_t matmul_b_stride_n = 0;
  std::int64_t matmul_output_stride_m = 0;
  std::int64_t matmul_output_stride_n = 0;
  std::int64_t convolution_spatial_rank = 0;
  std::int64_t convolution_groups = 0;
  std::int64_t convolution_input_channels = 0;
  std::int64_t convolution_output_channels = 0;
  std::int64_t convolution_channels_per_group = 0;
  std::array<std::int64_t, 5> convolution_input_dimensions{};
  std::array<std::int64_t, 5> convolution_input_strides{};
  std::array<std::int64_t, 5> convolution_filter_dimensions{};
  std::array<std::int64_t, 5> convolution_filter_strides{};
  std::array<std::int64_t, 5> convolution_output_dimensions{};
  std::array<std::int64_t, 5> convolution_output_strides{};
  std::array<std::int64_t, 3> convolution_pre_padding{};
  std::array<std::int64_t, 3> convolution_post_padding{};
  std::array<std::int64_t, 3> convolution_stride{};
  std::array<std::int64_t, 3> convolution_dilation{};
  std::int64_t batchnorm_rank = 0;
  std::int64_t batchnorm_batch = 0;
  std::int64_t batchnorm_channels = 0;
  std::int64_t batchnorm_spatial = 0;
  std::int64_t batchnorm_reduction_elements = 0;
  double batchnorm_epsilon = 0.0;
  double batchnorm_momentum = 0.0;
  std::array<std::int64_t, 8> batchnorm_dimensions{};
  std::array<std::int64_t, 8> batchnorm_x_strides{};
  std::array<std::int64_t, 8> batchnorm_y_strides{};
  std::int64_t rmsnorm_rows = 0;
  std::int64_t rmsnorm_normalized_elements = 0;
  double rmsnorm_epsilon = 0.0;
  std::int64_t layernorm_rows = 0;
  std::int64_t layernorm_normalized_elements = 0;
  double layernorm_epsilon = 0.0;
  std::int64_t layout_input_base = 0;
  std::array<std::int64_t, 8> layout_input_dimensions{};
  std::array<std::int64_t, 8> layout_input_strides{};
  std::array<std::int64_t, 8> layout_output_dimensions{};
  std::int64_t reduction_axis = 0;
  std::int64_t reduction_rank = 0;
  std::int64_t reduction_output_rank = 0;
  std::int64_t reduction_keep_dimensions = 0;
  std::int64_t reduction_outer = 0;
  std::int64_t reduction_size = 0;
  std::int64_t reduction_inner = 0;
  std::array<std::int64_t, 8> reduction_input_dimensions{};
  std::array<std::int64_t, 8> reduction_input_strides{};
  std::array<std::int64_t, 8> reduction_output_dimensions{};
  std::size_t workspace_offset = 0;
  std::size_t workspace_size = 0;
  std::size_t workspace_alignment = 1;

  [[nodiscard]] StorageDataType input_type(std::size_t index) const {
    return tensor_storage_data_types.at(index);
  }

  [[nodiscard]] StorageDataType output_type() const {
    return tensor_storage_data_types.at(input_count);
  }
};

struct AscendArtifact {
  std::vector<AscendStageArtifact> stages;
  std::vector<std::int64_t> binding_uids;
  std::size_t workspace_size = 0;
};

[[nodiscard]] AscendArtifact parse_ascend_artifact(
    const std::string& target_fingerprint,
    std::uint32_t ai_core_count,
    const flagdnnBackendBuildInputV2& input);

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_ARTIFACT_HPP_
