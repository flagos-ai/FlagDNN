/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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
      result += 'x';
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

Shape broadcast_shape(const Shape& left, const Shape& right) {
  const std::size_t rank = std::max(left.size(), right.size());
  Shape result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t left_dimension =
        trailing < left.size() ? left[left.size() - 1 - trailing] : 1;
    const std::int64_t right_dimension =
        trailing < right.size() ? right[right.size() - 1 - trailing] : 1;
    result[rank - 1 - trailing] =
        std::max(left_dimension, right_dimension);
  }
  return result;
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type) {
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  specification.relative_tolerance = 1.0e-2;
}

BenchmarkCase make_case(const ShapePair& shapes,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   std::string prefix,
                   double alpha = 1.0) {
  const Shape output = broadcast_shape(shapes.first, shapes.second);
  BenchmarkCase result;
  result.name = std::move(prefix) + "_" + data_type_name(data_type) + "_" +
                shape_name(shapes.first) + "_by_" +
                shape_name(shapes.second);
  result.operation = Operation::kAdd;
  result.tensors = {
      pointwise_tensor(uid, shapes.first, data_type),
      pointwise_tensor(uid + 1, shapes.second, data_type),
      pointwise_tensor(uid + 2, output, data_type),
  };
  result.add_alpha = alpha;
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
      {{2, 3, 17}, {1, 3, 17}},
      {{2, 3, 17}, {2, 1, 17}},
      {{3, 5, 7}, {3, 1, 7}},
      {{3, 5, 7}, {1, 5, 7}},
  };
  return shapes;
}

const std::vector<ShapePair>& multi_axis_broadcast_shapes() {
  static const std::vector<ShapePair> shapes = {
      {{2, 3, 17}, {1, 1, 17}},
      {{2, 3, 5, 7}, {1, 3, 1, 7}},
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

}  // namespace

std::vector<BenchmarkCase> add_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(49);
  std::int64_t uid = 300;

  for (const ShapePair& shapes : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_case(shapes, data_type, uid, "add"));
      uid += 3;
    }
  }
  for (const ShapePair& shapes : multi_axis_broadcast_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_case(shapes, data_type, uid, "add_multi_axis_broadcast"));
      uid += 3;
    }
  }
  for (const flagdnnDataType_t data_type : kDataTypes) {
    for (const double alpha : {0.5, -2.0}) {
      const std::string alpha_name = alpha > 0.0 ? "alpha_0_5" : "alpha_neg_2";
      result.push_back(make_case({{2, 4, 8}, {2, 4, 8}},
                                 data_type,
                                 uid,
                                 "add_" + alpha_name,
                                 alpha));
      uid += 3;
    }
  }

  BenchmarkCase strided;
  strided.name = "add_strided_broadcast_alpha_neg_0_75_fp32_2x3x4";
  strided.operation = Operation::kAdd;
  strided.tensors = {
      strided_tensor(uid, {2, 3, 4}, {31, 9, 2}),
      strided_tensor(uid + 1, {1, 4}, {13, 3}),
      strided_tensor(uid + 2, {2, 3, 4}, {37, 11, 2})};
  strided.add_alpha = -0.75;
  strided.absolute_tolerance = 1.0e-6;
  strided.relative_tolerance = 1.0e-6;
  result.push_back(std::move(strided));

  return result;
}

std::vector<BenchmarkCase> add_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(benchmark_shapes().size() * kDataTypes.size());
  std::int64_t uid = 600;
  for (const ShapePair& shapes : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_case(shapes, data_type, uid, "add_perf"));
      uid += 3;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
