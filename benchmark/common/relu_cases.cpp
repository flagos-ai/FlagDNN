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

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type) {
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  specification.relative_tolerance = 1.0e-2;
}

BenchmarkCase make_case(const Shape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   std::string prefix) {
  BenchmarkCase result;
  result.name = std::move(prefix) + "_" + data_type_name(data_type) + "_" +
                shape_name(shape);
  result.operation = Operation::kRelu;
  result.tensors = {
      pointwise_tensor(uid, shape, data_type),
      pointwise_tensor(uid + 1, shape, data_type),
  };
  set_tolerance(result, data_type);
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

}  // namespace

std::vector<BenchmarkCase> relu_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(27);
  std::int64_t uid = 100;
  for (const Shape& shape : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, "relu"));
      uid += 2;
    }
  }

  for (const flagdnnDataType_t data_type : kDataTypes) {
    BenchmarkCase strided;
    strided.name =
        "relu_strided_" + data_type_name(data_type) + "_2x3x4";
    strided.operation = Operation::kRelu;
    strided.tensors = {
        strided_tensor(uid, {2, 3, 4}, {31, 9, 2}, data_type),
        strided_tensor(uid + 1, {2, 3, 4}, {37, 11, 2}, data_type),
    };
    set_tolerance(strided, data_type);
    result.push_back(std::move(strided));
    uid += 2;
  }
  return result;
}

std::vector<BenchmarkCase> relu_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(24);
  std::int64_t uid = 200;
  for (const Shape& shape : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, "relu_perf"));
      uid += 2;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
