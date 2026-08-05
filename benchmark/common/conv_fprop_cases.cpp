/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

using Shape = std::vector<std::int64_t>;

struct ConvCaseDefinition {
  Shape input;
  Shape filter;
  Shape stride;
  Shape pre_padding;
  Shape post_padding;
  Shape dilation;
  std::int64_t groups = 1;
  bool channels_last = false;
  std::string label;
  bool output_channels_last_only = false;
};

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bfloat16";
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
  }
  return "invalid";
}

std::string shape_name(const Shape& shape) {
  std::string result;
  for (const std::int64_t dimension : shape) {
    if (!result.empty()) {
      result += 'x';
    }
    result += std::to_string(dimension);
  }
  return result;
}

std::vector<std::int64_t> tensor_strides(const Shape& dimensions,
                                         bool channels_last) {
  if (!channels_last) {
    return contiguous_strides(dimensions);
  }
  const std::int64_t channels = dimensions[1];
  if (dimensions.size() == 3) {
    const std::int64_t length = dimensions[2];
    return {channels * length, 1, channels};
  }
  if (dimensions.size() == 4) {
    const std::int64_t height = dimensions[2];
    const std::int64_t width = dimensions[3];
    return {channels * height * width, 1, width * channels, channels};
  }
  if (dimensions.size() == 5) {
    const std::int64_t depth = dimensions[2];
    const std::int64_t height = dimensions[3];
    const std::int64_t width = dimensions[4];
    return {channels * depth * height * width,
            1,
            height * width * channels,
            width * channels,
            channels};
  }
  throw std::invalid_argument(
      "channels-last convolution tensor rank must be 3, 4, or 5");
}

TensorSpec make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type,
                       bool channels_last) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = tensor_strides(dimensions, channels_last);
  return result;
}

std::int64_t output_dimension(std::int64_t input,
                              std::int64_t filter,
                              std::int64_t pre_padding,
                              std::int64_t post_padding,
                              std::int64_t stride,
                              std::int64_t dilation) {
  return (input + pre_padding + post_padding -
          dilation * (filter - 1) - 1) /
             stride +
         1;
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type) {
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2;
  specification.relative_tolerance = 5.0e-2;
}

BenchmarkCase make_case(const ConvCaseDefinition& definition,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark) {
  if (definition.input.size() < 3 || definition.input.size() > 5 ||
      definition.filter.size() != definition.input.size()) {
    throw std::invalid_argument(
        "convolution case tensors must have matching rank 3, 4, or 5");
  }
  const std::size_t spatial_rank = definition.input.size() - 2;
  if (definition.stride.size() != spatial_rank ||
      definition.pre_padding.size() != spatial_rank ||
      definition.post_padding.size() != spatial_rank ||
      definition.dilation.size() != spatial_rank) {
    throw std::invalid_argument(
        "convolution case spatial attributes have the wrong rank");
  }
  Shape output = {definition.input[0], definition.filter[0]};
  output.reserve(spatial_rank + 2);
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    output.push_back(output_dimension(definition.input[axis + 2],
                                      definition.filter[axis + 2],
                                      definition.pre_padding[axis],
                                      definition.post_padding[axis],
                                      definition.stride[axis],
                                      definition.dilation[axis]));
  }
  BenchmarkCase result;
  result.name = "conv" + std::to_string(spatial_rank) + "d_fprop_" +
                std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" + definition.label + "_" +
                shape_name(definition.input) + "_by_" +
                shape_name(definition.filter);
  result.operation = Operation::kConvolutionFprop;
  result.tensors = {
      make_tensor(uid,
                  definition.input,
                  data_type,
                  definition.channels_last),
      make_tensor(uid + 1,
                  definition.filter,
                  data_type,
                  definition.channels_last),
      make_tensor(uid + 2,
                  output,
                  data_type,
                  definition.channels_last ||
                      definition.output_channels_last_only),
  };
  result.convolution.spatial_rank =
      static_cast<std::int32_t>(spatial_rank);
  result.convolution.pre_padding = definition.pre_padding;
  result.convolution.post_padding = definition.post_padding;
  result.convolution.stride = definition.stride;
  result.convolution.dilation = definition.dilation;
  result.convolution.groups = definition.groups;
  set_tolerance(result, data_type);
  if (benchmark) {
    result.benchmark.iterations_per_sample = 10;
    if (data_type == FLAGDNN_DATA_FLOAT32) {
      double reduction_extent = static_cast<double>(definition.filter[1]);
      for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
        reduction_extent *=
            static_cast<double>(definition.filter[axis + 2]);
      }
      result.absolute_tolerance =
          1.0e-1 * std::sqrt(std::max(1.0, reduction_extent / 1152.0));
      result.relative_tolerance = 1.0e-1;
    }
  }
  return result;
}

const std::vector<ConvCaseDefinition>& old_correctness_cases() {
  static const std::vector<ConvCaseDefinition> cases = {
      {{2, 8, 16, 16},
       {16, 8, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       true,
       "channels_last_symmetric"},
      {{1, 4, 15, 17},
       {6, 4, 3, 5},
       {2, 1},
       {1, 2},
       {1, 2},
       {1, 1},
       1,
       true,
       "channels_last_nonuniform_stride"},
      {{2, 3, 8, 8},
       {5, 3, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1},
       1,
       true,
       "channels_last_1x1"},
      {{2, 4, 12, 13},
       {7, 4, 3, 3},
       {1, 2},
       {1, 0},
       {2, 3},
       {1, 1},
       1,
       true,
       "channels_last_asymmetric_padding"},
      {{1, 5, 19, 21},
       {9, 5, 3, 3},
       {1, 1},
       {2, 1},
       {0, 3},
       {2, 1},
       1,
       true,
       "channels_last_asymmetric_dilation"},
  };
  return cases;
}

const std::vector<ConvCaseDefinition>& old_benchmark_cases() {
  static const std::vector<ConvCaseDefinition> cases = {
      {{8, 32, 32, 32},
       {64, 32, 3, 3},
       {1, 1},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "standard_3x3"},
      {{8, 64, 28, 28},
       {128, 64, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1},
       1,
       false,
       "standard_1x1"},
      {{8, 64, 56, 56},
       {128, 64, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "stride2"},
      {{4, 64, 32, 32},
       {64, 64, 3, 3},
       {1, 1},
       {2, 2},
       {2, 2},
       {2, 2},
       1,
       false,
       "dilation2"},
      {{4, 32, 35, 37},
       {48, 32, 3, 5},
       {1, 2},
       {1, 0},
       {1, 2},
       {1, 1},
       1,
       false,
       "asymmetric"},
      {{1, 3, 640, 640},
       {16, 3, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_n_stem"},
      {{1, 3, 640, 640},
       {32, 3, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_s_stem"},
      {{1, 3, 640, 640},
       {64, 3, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_ml_stem"},
      {{1, 3, 640, 640},
       {96, 3, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_x_stem"},
      {{1, 128, 40, 40},
       {256, 128, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_n_p5"},
      {{1, 256, 40, 40},
       {512, 256, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_s_p5"},
      {{1, 512, 40, 40},
       {512, 512, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_ml_p5"},
      {{1, 768, 40, 40},
       {768, 768, 3, 3},
       {2, 2},
       {1, 1},
       {1, 1},
       {1, 1},
       1,
       false,
       "yolo_x_p5"},
  };
  return cases;
}

const std::vector<ConvCaseDefinition>& old_non2d_correctness_cases() {
  static const std::vector<ConvCaseDefinition> cases = {
      {{2, 4, 16},
       {6, 4, 3},
       {1},
       {1},
       {1},
       {1},
       1,
       false,
       "contiguous_input_nwc_output_symmetric",
       true},
      {{2, 4, 19},
       {7, 4, 3},
       {1},
       {2},
       {0},
       {2},
       1,
       false,
       "contiguous_input_nwc_output_asymmetric_dilation",
       true},
      {{1, 2, 5, 6, 7},
       {4, 2, 3, 3, 3},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       1,
       true,
       "channels_last_3d_symmetric"},
      {{1, 2, 6, 7, 8},
       {3, 2, 2, 3, 3},
       {1, 1, 1},
       {1, 0, 1},
       {0, 1, 2},
       {1, 1, 1},
       1,
       true,
       "channels_last_3d_asymmetric"},
  };
  return cases;
}

const std::vector<ConvCaseDefinition>& old_non2d_benchmark_cases() {
  static const std::vector<ConvCaseDefinition> cases = {
      {{16, 32, 256},
       {64, 32, 3},
       {1},
       {1},
       {1},
       {1},
       1,
       false,
       "contiguous_input_nwc_output_symmetric",
       true},
      {{8, 64, 255},
       {96, 64, 5},
       {2},
       {2},
       {1},
       {1},
       1,
       false,
       "contiguous_input_nwc_output_asymmetric",
       true},
      {{2, 8, 8, 16, 16},
       {16, 8, 3, 3, 3},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       1,
       false,
       "contiguous_symmetric"},
      {{1, 8, 10, 12, 14},
       {12, 8, 2, 3, 3},
       {1, 1, 1},
       {1, 0, 1},
       {0, 1, 2},
       {1, 1, 1},
       1,
       false,
       "contiguous_asymmetric"},
  };
  return cases;
}

}  // namespace

std::vector<BenchmarkCase> conv_fprop_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(34);
  std::int64_t uid = 800;

  ConvCaseDefinition smoke{
      {1, 2, 5, 5},
      {2, 2, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      false,
      "nchw_smoke"};
  BenchmarkCase smoke_case =
      make_case(smoke, FLAGDNN_DATA_FLOAT32, uid, false);
  smoke_case.absolute_tolerance = 2.0e-5;
  smoke_case.relative_tolerance = 1.0e-5;
  result.push_back(std::move(smoke_case));
  uid += 3;

  for (const ConvCaseDefinition& definition : old_correctness_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition, data_type, uid, false));
      uid += 3;
    }
  }

  const ConvCaseDefinition dispatch{
      {1, 128, 40, 40},
      {256, 128, 3, 3},
      {2, 2},
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      false,
      "nchw_im2col_dispatch"};
  const ConvCaseDefinition grouped{
      {1, 4, 7, 7},
      {6, 2, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      2,
      false,
      "nchw_groups2"};
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_case(dispatch, data_type, uid, false));
    uid += 3;
    result.push_back(make_case(grouped, data_type, uid, false));
    uid += 3;
  }
  for (const ConvCaseDefinition& definition :
       old_non2d_correctness_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition, data_type, uid, false));
      uid += 3;
    }
  }
  return result;
}

std::vector<BenchmarkCase> conv_fprop_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(51);
  std::int64_t uid = 1000;
  for (const ConvCaseDefinition& definition : old_benchmark_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition, data_type, uid, true));
      uid += 3;
    }
  }
  for (const ConvCaseDefinition& definition : old_non2d_benchmark_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition, data_type, uid, true));
      uid += 3;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
