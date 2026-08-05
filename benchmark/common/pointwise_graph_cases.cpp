/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

using Shape = std::vector<std::int64_t>;
using ShapePair = std::pair<Shape, Shape>;

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

std::vector<std::int64_t> pointwise_strides(const Shape& dimensions) {
  if (dimensions.size() != 4) {
    return contiguous_strides(dimensions);
  }
  const std::int64_t channels = dimensions[1];
  const std::int64_t height = dimensions[2];
  const std::int64_t width = dimensions[3];
  return {channels * height * width, 1, width * channels, channels};
}

TensorSpec pointwise_tensor(std::int64_t uid,
                            const Shape& dimensions,
                            flagdnnDataType_t data_type) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = pointwise_strides(dimensions);
  return result;
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type) {
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  specification.relative_tolerance = 1.0e-2;
}

BenchmarkCase make_add_square_case(const ShapePair& shapes,
                              flagdnnDataType_t data_type,
                              std::int64_t uid,
                              bool benchmark) {
  if (shapes.first != shapes.second) {
    throw std::invalid_argument(
        "add_square graph catalog requires equal input shapes");
  }
  BenchmarkCase result;
  result.name = std::string("add_square") +
                (benchmark ? "_perf_" : "_") +
                data_type_name(data_type) + "_" +
                shape_name(shapes.first);
  result.operation = Operation::kGraph;
  result.input_domain = InputDomain::kReal;

  const TensorSpec left =
      pointwise_tensor(uid, shapes.first, data_type);
  const TensorSpec right =
      pointwise_tensor(uid + 1, shapes.second, data_type);
  const TensorSpec square =
      pointwise_tensor(uid + 2, shapes.second, data_type);
  const TensorSpec output =
      pointwise_tensor(uid + 3, shapes.first, data_type);
  result.tensors = {left, right, output};
  result.graph.intermediates = {square};

  GraphNodeSpec square_node;
  square_node.name = "square";
  square_node.operation = Operation::kPointwise;
  square_node.input_uids = {right.uid, right.uid};
  square_node.output_uid = square.uid;
  square_node.pointwise_mode = FLAGDNN_POINTWISE_MUL;

  GraphNodeSpec add_node;
  add_node.name = "add_square";
  add_node.operation = Operation::kPointwise;
  add_node.input_uids = {left.uid, square.uid};
  add_node.output_uid = output.uid;
  add_node.pointwise_mode = FLAGDNN_POINTWISE_ADD;

  result.graph.nodes = {std::move(square_node), std::move(add_node)};
  set_tolerance(result, data_type);
  return result;
}

const std::vector<ShapePair>& correctness_shapes() {
  static const std::vector<ShapePair> shapes = {
      {{1, 1, 16}, {1, 1, 16}},
      {{2, 4, 8}, {2, 4, 8}},
      {{1, 4, 8, 16}, {1, 4, 8, 16}},
      {{2, 4, 8, 16}, {2, 4, 8, 16}},
      {{1, 3, 17}, {1, 3, 17}},
      {{3, 5, 7}, {3, 5, 7}},
      {{1, 3, 5, 7}, {1, 3, 5, 7}},
      {{2, 3, 5, 7}, {2, 3, 5, 7}},
  };
  return shapes;
}

const std::vector<ShapePair>& benchmark_shapes() {
  static const std::vector<ShapePair> shapes = {
      {{1, 1, 1024}, {1, 1, 1024}},
      {{8, 16, 32}, {8, 16, 32}},
      {{4, 16, 64, 128}, {4, 16, 64, 128}},
      {{8, 16, 64, 128}, {8, 16, 64, 128}},
      {{1, 1, 1000}, {1, 1, 1000}},
      {{3, 257, 513}, {3, 257, 513}},
      {{3, 7, 65, 129}, {3, 7, 65, 129}},
      {{5, 7, 65, 129}, {5, 7, 65, 129}},
  };
  return shapes;
}

std::vector<BenchmarkCase> make_cases(const std::vector<ShapePair>& shapes,
                                 bool benchmark) {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = benchmark ? 12000 : 11000;
  for (const ShapePair& shape : shapes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_add_square_case(shape, data_type, uid, benchmark));
      uid += 4;
    }
  }
  return result;
}

}  // namespace

std::vector<BenchmarkCase> add_square_cases() {
  return make_cases(correctness_shapes(), false);
}

std::vector<BenchmarkCase> add_square_benchmark_cases() {
  return make_cases(benchmark_shapes(), true);
}

}  // namespace flagdnn::benchmarking
