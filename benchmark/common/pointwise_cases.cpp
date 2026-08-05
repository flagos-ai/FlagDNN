/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

using Shape = std::vector<std::int64_t>;

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

bool expects_exact_results(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CEIL ||
         mode == FLAGDNN_POINTWISE_FLOOR ||
         mode == FLAGDNN_POINTWISE_IDENTITY ||
         mode == FLAGDNN_POINTWISE_LOGICAL_NOT;
}

std::vector<flagdnnDataType_t> input_data_types(
    flagdnnPointwiseMode_t mode) {
  if (mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
    return {FLAGDNN_DATA_BOOLEAN};
  }
  return std::vector<flagdnnDataType_t>(
      kDataTypes.begin(), kDataTypes.end());
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type,
                   flagdnnPointwiseMode_t mode) {
  if (expects_exact_results(mode)) {
    specification.absolute_tolerance = 0.0;
    specification.relative_tolerance = 0.0;
    return;
  }
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  specification.relative_tolerance = 1.0e-2;
}

BenchmarkCase make_case(const Shape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   flagdnnPointwiseMode_t mode,
                   std::string_view operation_name,
                   InputDomain input_domain,
                   bool benchmark,
                   const flagdnnPointwiseAttributes_t& attributes) {
  BenchmarkCase result;
  result.name = std::string(operation_name) +
                (benchmark ? "_perf_" : "_") +
                data_type_name(data_type) + "_" + shape_name(shape);
  result.operation = Operation::kPointwise;
  result.pointwise_mode = mode;
  result.pointwise_attributes = attributes;
  result.input_domain = input_domain;
  result.tensors = {
      pointwise_tensor(uid, shape, data_type),
      pointwise_tensor(uid + 1, shape, data_type),
  };
  set_tolerance(result, data_type, mode);
  return result;
}

const std::vector<Shape>& correctness_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 16},
      {2, 4, 8},
      {1, 4, 8, 16},
      {2, 4, 8, 16},
      {1, 3, 17},
      {3, 5, 7},
      {1, 3, 5, 7},
      {2, 3, 5, 7},
  };
  return shapes;
}

const std::vector<Shape>& identity_correctness_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 1},
      {2, 3, 4},
      {4, 5, 6},
      {1, 8, 16},
      {3, 1, 17},
      {2, 4, 8},
      {5, 7, 11},
      {1, 33, 65},
      {2, 16, 257},
      {4, 32, 128},
  };
  return shapes;
}

const std::vector<Shape>& benchmark_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 1024},
      {8, 16, 32},
      {4, 16, 64, 128},
      {8, 16, 64, 128},
      {1, 1, 1000},
      {3, 257, 513},
      {3, 7, 65, 129},
      {5, 7, 65, 129},
  };
  return shapes;
}

const std::vector<Shape>& identity_benchmark_shapes() {
  static const std::vector<Shape> shapes = {
      {1, 1, 1},
      {2, 3, 4},
      {8, 16, 32},
      {64, 64, 64},
      {16, 64, 128},
      {16, 256, 256},
      {32, 128, 256},
      {4, 1024, 1024},
      {16, 512, 1024},
      {2, 2048, 2048},
      {8, 1024, 2048},
  };
  return shapes;
}

std::vector<Shape> selected_benchmark_shapes(flagdnnPointwiseMode_t mode) {
  const std::vector<Shape>& all = benchmark_shapes();
  switch (mode) {
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return {all[0], all[5]};
    case FLAGDNN_POINTWISE_CEIL:
      return {all[2], all[5]};
    case FLAGDNN_POINTWISE_FLOOR:
      return {all[2]};
    case FLAGDNN_POINTWISE_ERF:
    case FLAGDNN_POINTWISE_SIN:
    case FLAGDNN_POINTWISE_COS:
    case FLAGDNN_POINTWISE_TAN:
      return {all[5]};
    case FLAGDNN_POINTWISE_IDENTITY:
      return identity_benchmark_shapes();
    default:
      return all;
  }
}

}  // namespace

std::vector<BenchmarkCase> unary_pointwise_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain,
    flagdnnPointwiseAttributes_t attributes) {
  const std::vector<Shape>& shapes =
      mode == FLAGDNN_POINTWISE_IDENTITY ? identity_correctness_shapes()
                                         : correctness_shapes();
  std::vector<BenchmarkCase> result;
  const std::vector<flagdnnDataType_t> data_types = input_data_types(mode);
  result.reserve(shapes.size() * data_types.size() + data_types.size());
  std::int64_t uid = 1000;
  for (const Shape& shape : shapes) {
    for (const flagdnnDataType_t data_type : data_types) {
      result.push_back(make_case(shape,
                                 data_type,
                                 uid,
                                 mode,
                                 operation_name,
                                 input_domain,
                                 false,
                                 attributes));
      uid += 2;
    }
  }

  for (const flagdnnDataType_t data_type : data_types) {
    BenchmarkCase strided;
    strided.name = std::string(operation_name) + "_strided_" +
                   data_type_name(data_type) + "_2x3x4";
    strided.operation = Operation::kPointwise;
    strided.pointwise_mode = mode;
    strided.pointwise_attributes = attributes;
    strided.input_domain = input_domain;
    strided.tensors = {
        strided_tensor(uid, {2, 3, 4}, {31, 9, 2}, data_type),
        strided_tensor(uid + 1, {2, 3, 4}, {37, 11, 2}, data_type),
    };
    set_tolerance(strided, data_type, mode);
    result.push_back(std::move(strided));
    uid += 2;
  }
  return result;
}

std::vector<BenchmarkCase> unary_pointwise_benchmark_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain,
    flagdnnPointwiseAttributes_t attributes) {
  const std::vector<Shape> shapes = selected_benchmark_shapes(mode);
  std::vector<BenchmarkCase> result;
  const std::vector<flagdnnDataType_t> data_types = input_data_types(mode);
  result.reserve(shapes.size() * data_types.size());
  std::int64_t uid = 2000;
  for (const Shape& shape : shapes) {
    for (const flagdnnDataType_t data_type : data_types) {
      result.push_back(make_case(shape,
                                 data_type,
                                 uid,
                                 mode,
                                 operation_name,
                                 input_domain,
                                 true,
                                 attributes));
      uid += 2;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
