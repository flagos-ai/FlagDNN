/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_COMMON_CASE_HPP_
#define FLAGDNN_BENCHMARK_COMMON_CASE_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::benchmarking {

enum class Operation {
  kRelu,
  kPointwise,
  kAdd,
  kReduction,
  kConvolutionFprop,
  kConvolutionDgrad,
  kConvolutionWgrad,
  kMatmul,
  kReshape,
  kTranspose,
  kSlice,
  kLayernorm,
  kRmsnorm,
  kBatchnorm,
  kBatchnormInference,
  kGraph,
};

enum class ConvolutionMode {
  kCrossCorrelation,
  kConvolution,
};

enum class InputDomain {
  kReal,
  kPositive,
  kScaled,
  kTan,
  kDivisor,
  kPower,
  kModulo,
  kModuloSigned,
  kComparison,
  kLogical,
};

struct TensorSpec {
  std::int64_t uid = 0;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t binding_byte_offset = 0;
};

struct ConvolutionAttributes {
  std::int32_t spatial_rank = 0;
  std::vector<std::int64_t> pre_padding;
  std::vector<std::int64_t> post_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
  ConvolutionMode mode = ConvolutionMode::kCrossCorrelation;
};

struct ReshapeAttributes {
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  bool logical = true;
};

struct TransposeAttributes {
  std::vector<std::int64_t> permutation;
};

struct SliceAttributes {
  std::vector<std::pair<std::int64_t, std::int64_t>> slices;
  std::vector<std::int64_t> strides;
};

struct NormalizationAttributes {
  double epsilon = 1.0e-5;
  double momentum = 0.1;
};

struct BenchmarkConfig {
  int warmup_iterations = 10;
  int sample_count = 20;
  int iterations_per_sample = 50;
};

struct GraphNodeSpec {
  std::string name;
  Operation operation = Operation::kPointwise;
  std::vector<std::int64_t> input_uids;
  std::int64_t output_uid = 0;
  flagdnnPointwiseMode_t pointwise_mode = FLAGDNN_POINTWISE_NOT_SET;
  flagdnnPointwiseAttributes_t pointwise_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  double alpha = 1.0;
  ConvolutionAttributes convolution;
};

struct GraphDescription {
  std::vector<TensorSpec> intermediates;
  std::vector<GraphNodeSpec> nodes;
};

inline flagdnnPointwiseAttributes_t default_pointwise_attributes() {
  flagdnnPointwiseAttributes_t result =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  return result;
}

struct BenchmarkCase {
  std::string name;
  Operation operation = Operation::kRelu;
  std::vector<TensorSpec> tensors;
  std::size_t output_count = 1;
  flagdnnPointwiseMode_t pointwise_mode = FLAGDNN_POINTWISE_NOT_SET;
  flagdnnPointwiseAttributes_t pointwise_attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  InputDomain input_domain = InputDomain::kReal;
  std::vector<InputDomain> input_domains;
  double add_alpha = 1.0;
  flagdnnReductionMode_t reduction_mode = FLAGDNN_REDUCTION_ADD;
  std::int32_t reduction_axis = -1;
  bool keep_dimensions = false;
  ConvolutionAttributes convolution;
  ReshapeAttributes reshape;
  TransposeAttributes transpose;
  SliceAttributes slice;
  NormalizationAttributes normalization;
  GraphDescription graph;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  BenchmarkConfig benchmark;
};

inline std::vector<std::int64_t> contiguous_strides(
    const std::vector<std::int64_t>& dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  std::int64_t stride = 1;
  for (std::size_t index = dimensions.size(); index != 0; --index) {
    result[index - 1] = stride;
    stride *= dimensions[index - 1];
  }
  return result;
}

inline TensorSpec tensor(std::int64_t uid,
                         std::initializer_list<std::int64_t> dimensions,
                         flagdnnDataType_t data_type =
                             FLAGDNN_DATA_FLOAT32) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions.assign(dimensions);
  result.strides = contiguous_strides(result.dimensions);
  return result;
}

inline TensorSpec strided_tensor(
    std::int64_t uid,
    std::initializer_list<std::int64_t> dimensions,
    std::initializer_list<std::int64_t> strides,
    flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions.assign(dimensions);
  result.strides.assign(strides);
  return result;
}

inline std::size_t input_tensor_count(const BenchmarkCase& specification) {
  if (specification.output_count == 0 ||
      specification.output_count > specification.tensors.size()) {
    throw std::invalid_argument("case output count is invalid");
  }
  return specification.tensors.size() - specification.output_count;
}

inline const TensorSpec& output_tensor(
    const BenchmarkCase& specification, std::size_t output_index = 0) {
  if (output_index >= specification.output_count) {
    throw std::out_of_range("case output index is invalid");
  }
  return specification.tensors.at(
      input_tensor_count(specification) + output_index);
}

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_COMMON_CASE_HPP_
