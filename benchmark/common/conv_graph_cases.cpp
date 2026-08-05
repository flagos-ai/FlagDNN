/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

using Shape = std::vector<std::int64_t>;

struct ConvBiasReluShape {
  Shape input;
  Shape filter;
  Shape stride;
  Shape padding;
  Shape dilation;
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
      result += "x";
    }
    result += std::to_string(dimension);
  }
  return result;
}

std::vector<std::int64_t> channels_last_strides(
    const Shape& dimensions) {
  const std::int64_t channels = dimensions[1];
  const std::int64_t height = dimensions[2];
  const std::int64_t width = dimensions[3];
  return {channels * height * width, 1, width * channels, channels};
}

TensorSpec channels_last_tensor(std::int64_t uid,
                                const Shape& dimensions,
                                flagdnnDataType_t data_type) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = channels_last_strides(dimensions);
  return result;
}

Shape output_shape(const ConvBiasReluShape& shape) {
  const std::int64_t output_height =
      1 + (shape.input[2] + 2 * shape.padding[0] -
           shape.dilation[0] * (shape.filter[2] - 1) - 1) /
              shape.stride[0];
  const std::int64_t output_width =
      1 + (shape.input[3] + 2 * shape.padding[1] -
           shape.dilation[1] * (shape.filter[3] - 1) - 1) /
              shape.stride[1];
  return {shape.input[0], shape.filter[0], output_height, output_width};
}

BenchmarkCase make_case(const ConvBiasReluShape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid) {
  const Shape output_dimensions = output_shape(shape);
  const Shape bias_dimensions = {1, shape.filter[0], 1, 1};

  BenchmarkCase result;
  result.name = "conv_bias_relu_perf_" + data_type_name(data_type) +
                "_x" + shape_name(shape.input) +
                "_w" + shape_name(shape.filter) +
                "_s" + shape_name(shape.stride) +
                "_p" + shape_name(shape.padding) +
                "_d" + shape_name(shape.dilation);
  result.operation = Operation::kGraph;
  result.input_domain = InputDomain::kReal;

  const TensorSpec input =
      channels_last_tensor(uid, shape.input, data_type);
  const TensorSpec filter =
      channels_last_tensor(uid + 1, shape.filter, data_type);
  const TensorSpec bias =
      channels_last_tensor(uid + 2, bias_dimensions, data_type);
  const TensorSpec convolution =
      channels_last_tensor(uid + 3, output_dimensions, data_type);
  const TensorSpec biased =
      channels_last_tensor(uid + 4, output_dimensions, data_type);
  const TensorSpec output =
      channels_last_tensor(uid + 5, output_dimensions, data_type);
  result.tensors = {input, filter, bias, output};
  result.graph.intermediates = {convolution, biased};

  GraphNodeSpec convolution_node;
  convolution_node.name = "convolution_fprop";
  convolution_node.operation = Operation::kConvolutionFprop;
  convolution_node.input_uids = {input.uid, filter.uid};
  convolution_node.output_uid = convolution.uid;
  convolution_node.convolution.spatial_rank = 2;
  convolution_node.convolution.pre_padding = shape.padding;
  convolution_node.convolution.post_padding = shape.padding;
  convolution_node.convolution.stride = shape.stride;
  convolution_node.convolution.dilation = shape.dilation;
  convolution_node.convolution.groups = 1;

  GraphNodeSpec bias_node;
  bias_node.name = "bias_add";
  bias_node.operation = Operation::kPointwise;
  bias_node.input_uids = {convolution.uid, bias.uid};
  bias_node.output_uid = biased.uid;
  bias_node.pointwise_mode = FLAGDNN_POINTWISE_ADD;

  GraphNodeSpec relu_node;
  relu_node.name = "relu";
  relu_node.operation = Operation::kPointwise;
  relu_node.input_uids = {biased.uid};
  relu_node.output_uid = output.uid;
  relu_node.pointwise_mode = FLAGDNN_POINTWISE_RELU_FWD;

  result.graph.nodes = {
      std::move(convolution_node),
      std::move(bias_node),
      std::move(relu_node),
  };
  result.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 : 5.0e-2;
  result.relative_tolerance = 5.0e-2;
  return result;
}

const std::vector<ConvBiasReluShape>& shapes() {
  static const std::vector<ConvBiasReluShape> result = {
      {{2, 8, 16, 16}, {16, 8, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
      {{1, 4, 15, 17}, {6, 4, 3, 5}, {2, 1}, {1, 2}, {1, 1}},
      {{2, 3, 8, 8}, {5, 3, 1, 1}, {1, 1}, {0, 0}, {1, 1}},
      {{1, 6, 20, 18}, {8, 6, 5, 3}, {1, 2}, {2, 1}, {1, 1}},
      {{2, 4, 12, 10}, {7, 4, 3, 3}, {2, 2}, {1, 1}, {1, 1}},
      {{1, 5, 19, 21}, {9, 5, 3, 3}, {1, 1}, {2, 2}, {2, 2}},
      {{1, 3, 32, 32}, {8, 3, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
      {{2, 8, 9, 11}, {4, 8, 1, 1}, {1, 1}, {0, 0}, {1, 1}},
      {{1, 4, 18, 18}, {4, 4, 3, 3}, {1, 1}, {0, 0}, {1, 1}},
      {{2, 12, 13, 15}, {10, 12, 3, 3}, {1, 1}, {1, 1}, {1, 1}},
  };
  return result;
}

}  // namespace

std::vector<BenchmarkCase> conv_bias_relu_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(shapes().size() * kDataTypes.size());
  std::int64_t uid = 13000;
  for (const ConvBiasReluShape& shape : shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid));
      uid += 6;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
