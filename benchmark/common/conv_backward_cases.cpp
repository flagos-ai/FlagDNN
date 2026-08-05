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
                   bool benchmark,
                   Operation operation,
                   ConvolutionMode mode =
                       ConvolutionMode::kCrossCorrelation) {
  if (operation != Operation::kConvolutionDgrad &&
      operation != Operation::kConvolutionWgrad) {
    throw std::invalid_argument(
        "backward convolution case has an invalid operation");
  }
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
  Shape loss = {definition.input[0], definition.filter[0]};
  loss.reserve(spatial_rank + 2);
  for (std::size_t axis = 0; axis < spatial_rank; ++axis) {
    loss.push_back(output_dimension(definition.input[axis + 2],
                                    definition.filter[axis + 2],
                                    definition.pre_padding[axis],
                                    definition.post_padding[axis],
                                    definition.stride[axis],
                                    definition.dilation[axis]));
  }

  const bool data_gradient =
      operation == Operation::kConvolutionDgrad;
  BenchmarkCase result;
  result.name = "conv" + std::to_string(spatial_rank) + "d_" +
                std::string(data_gradient ? "dgrad_" : "wgrad_") +
                std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" + definition.label + "_" +
                shape_name(definition.input) + "_by_" +
                shape_name(definition.filter) +
                (mode == ConvolutionMode::kConvolution
                     ? "_convolution_mode"
                     : "");
  result.operation = operation;
  const TensorSpec loss_tensor =
      make_tensor(uid, loss, data_type, false);
  const TensorSpec image_tensor =
      make_tensor(uid + 1, definition.input, data_type, false);
  const TensorSpec filter_tensor =
      make_tensor(uid + 2, definition.filter, data_type, false);
  if (data_gradient) {
    result.tensors = {loss_tensor, filter_tensor, image_tensor};
  } else {
    result.tensors = {loss_tensor, image_tensor, filter_tensor};
  }
  result.convolution.spatial_rank =
      static_cast<std::int32_t>(spatial_rank);
  result.convolution.pre_padding = definition.pre_padding;
  result.convolution.post_padding = definition.post_padding;
  result.convolution.stride = definition.stride;
  result.convolution.dilation = definition.dilation;
  result.convolution.groups = definition.groups;
  result.convolution.mode = mode;
  set_tolerance(result, data_type);
  if (data_gradient) {
    result.absolute_tolerance =
        data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 7.0e-2;
    result.relative_tolerance = 7.0e-2;
  } else {
    result.absolute_tolerance =
        data_type == FLAGDNN_DATA_FLOAT16 ? 8.0e-2 : 2.0e-1;
    result.relative_tolerance = 1.0e-1;
  }
  if (benchmark) {
    result.benchmark.iterations_per_sample = 10;
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
       "contiguous_symmetric"},
      {{1, 4, 15, 17},
       {6, 4, 3, 5},
       {2, 1},
       {1, 2},
       {1, 2},
       {1, 1},
       1,
       true,
       "contiguous_nonuniform_stride"},
      {{2, 3, 8, 8},
       {5, 3, 1, 1},
       {1, 1},
       {0, 0},
       {0, 0},
       {1, 1},
       1,
       true,
       "contiguous_1x1"},
      {{2, 4, 12, 13},
       {7, 4, 3, 3},
       {1, 2},
       {1, 0},
       {2, 3},
       {1, 1},
       1,
       true,
       "contiguous_asymmetric_padding"},
      {{1, 5, 19, 21},
       {9, 5, 3, 3},
       {1, 1},
       {2, 1},
       {0, 3},
       {2, 1},
       1,
       true,
       "contiguous_asymmetric_dilation"},
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
       "contiguous_symmetric",
       true},
      {{2, 4, 19},
       {7, 4, 3},
       {1},
       {2},
       {0},
       {2},
       1,
       false,
       "contiguous_asymmetric_dilation",
       true},
      {{1, 2, 5, 6, 7},
       {4, 2, 3, 3, 3},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       {1, 1, 1},
       1,
       true,
       "contiguous_symmetric"},
      {{1, 2, 6, 7, 8},
       {3, 2, 2, 3, 3},
       {1, 1, 1},
       {1, 0, 1},
       {0, 1, 2},
       {1, 1, 1},
       1,
       true,
       "contiguous_asymmetric"},
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
       "contiguous_symmetric",
       true},
      {{8, 64, 255},
       {96, 64, 5},
       {2},
       {2},
       {1},
       {1},
       1,
       false,
       "contiguous_asymmetric",
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

std::vector<BenchmarkCase> conv_dgrad_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(31);
  std::int64_t uid = 6000;
  for (const ConvCaseDefinition& definition : old_correctness_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition,
                                 data_type,
                                 uid,
                                 false,
                                 Operation::kConvolutionDgrad));
      uid += 3;
    }
  }
  for (const ConvCaseDefinition& definition :
       old_non2d_correctness_cases()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition,
                                 data_type,
                                 uid,
                                 false,
                                 Operation::kConvolutionDgrad));
      uid += 3;
    }
  }
  const ConvCaseDefinition convolution_mode{
      {2, 4, 8, 9},
      {6, 4, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      false,
      "explicit_filter_flip"};
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_case(convolution_mode,
                               data_type,
                               uid,
                               false,
                               Operation::kConvolutionDgrad,
                               ConvolutionMode::kConvolution));
    uid += 3;
  }
  const ConvCaseDefinition grouped{
      {1, 4, 7, 7},
      {6, 2, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      2,
      false,
      "groups2"};
  result.push_back(make_case(grouped,
                             FLAGDNN_DATA_FLOAT32,
                             uid,
                             false,
                             Operation::kConvolutionDgrad));
  return result;
}

std::vector<BenchmarkCase> conv_wgrad_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(46);
  std::int64_t uid = 7000;
  const auto append_definition = [&](const ConvCaseDefinition& definition) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition,
                                 data_type,
                                 uid,
                                 false,
                                 Operation::kConvolutionWgrad));
      uid += 3;
    }
  };
  for (const ConvCaseDefinition& definition : old_correctness_cases()) {
    append_definition(definition);
  }
  for (const ConvCaseDefinition& definition :
       old_non2d_correctness_cases()) {
    append_definition(definition);
  }
  const auto& benchmark_2d = old_benchmark_cases();
  for (const std::size_t index : {std::size_t{9},
                                  std::size_t{10},
                                  std::size_t{11}}) {
    append_definition(benchmark_2d.at(index));
  }
  const auto& benchmark_nd = old_non2d_benchmark_cases();
  append_definition(benchmark_nd.at(2));
  append_definition(benchmark_nd.at(3));

  const ConvCaseDefinition convolution_mode{
      {2, 4, 8, 9},
      {6, 4, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      1,
      false,
      "explicit_filter_flip"};
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_case(convolution_mode,
                               data_type,
                               uid,
                               false,
                               Operation::kConvolutionWgrad,
                               ConvolutionMode::kConvolution));
    uid += 3;
  }
  const ConvCaseDefinition grouped{
      {1, 4, 7, 7},
      {6, 2, 3, 3},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1},
      2,
      false,
      "groups2"};
  result.push_back(make_case(grouped,
                             FLAGDNN_DATA_FLOAT32,
                             uid,
                             false,
                             Operation::kConvolutionWgrad));
  return result;
}

std::vector<BenchmarkCase> conv_dgrad_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(45);
  std::int64_t uid = 8000;
  const auto append_definition = [&](const ConvCaseDefinition& definition) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition,
                                 data_type,
                                 uid,
                                 true,
                                 Operation::kConvolutionDgrad));
      uid += 3;
    }
  };
  for (const ConvCaseDefinition& definition :
       old_non2d_benchmark_cases()) {
    append_definition(definition);
  }
  const auto& benchmark_2d = old_benchmark_cases();
  for (std::size_t index = 0; index < benchmark_2d.size(); ++index) {
    if (index != 3 && index != 4) {
      append_definition(benchmark_2d[index]);
    }
  }
  return result;
}

std::vector<BenchmarkCase> conv_wgrad_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(45);
  std::int64_t uid = 9000;
  const auto append_definition = [&](const ConvCaseDefinition& definition) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(definition,
                                 data_type,
                                 uid,
                                 true,
                                 Operation::kConvolutionWgrad));
      uid += 3;
    }
  };
  for (const ConvCaseDefinition& definition :
       old_non2d_benchmark_cases()) {
    append_definition(definition);
  }
  const auto& benchmark_2d = old_benchmark_cases();
  for (std::size_t index = 0; index < benchmark_2d.size(); ++index) {
    if (index != 3 && index != 4) {
      append_definition(benchmark_2d[index]);
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
