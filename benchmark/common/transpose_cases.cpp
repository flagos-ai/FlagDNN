/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

using Shape = std::vector<std::int64_t>;
using TransposeShape = std::pair<Shape, Shape>;

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

TensorSpec make_tensor(std::int64_t uid,
                       Shape dimensions,
                       Shape strides,
                       flagdnnDataType_t data_type) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = std::move(dimensions);
  result.strides = std::move(strides);
  return result;
}

BenchmarkCase make_case(const TransposeShape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark) {
  const Shape input_strides = contiguous_strides(shape.first);
  Shape output_dimensions(shape.first.size());
  Shape output_strides(shape.first.size());
  for (std::size_t axis = 0; axis < shape.second.size(); ++axis) {
    const std::size_t source =
        static_cast<std::size_t>(shape.second[axis]);
    output_dimensions[axis] = shape.first[source];
    output_strides[axis] = input_strides[source];
  }
  BenchmarkCase result;
  result.name = "transpose_" +
                std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" +
                shape_name(shape.first);
  result.operation = Operation::kTranspose;
  result.tensors = {
      make_tensor(uid, shape.first, input_strides, data_type),
      make_tensor(uid + 1,
                  std::move(output_dimensions),
                  std::move(output_strides),
                  data_type),
  };
  result.transpose.permutation = shape.second;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

const std::vector<TransposeShape>& correctness_shapes() {
  static const std::vector<TransposeShape> result = {
      {{2, 3, 4}, {2, 0, 1}},
      {{1, 8, 16}, {0, 2, 1}},
      {{2, 3, 4, 5}, {0, 2, 3, 1}},
  };
  return result;
}

const std::vector<TransposeShape>& benchmark_shapes() {
  static const std::vector<TransposeShape> result = {
      {{8, 16, 32}, {2, 0, 1}},
      {{16, 64, 128}, {0, 2, 1}},
      {{32, 128, 256}, {1, 0, 2}},
      {{4, 64, 128, 32}, {0, 2, 3, 1}},
      {{2, 128, 128, 64}, {0, 3, 1, 2}},
  };
  return result;
}

}  // namespace

std::vector<BenchmarkCase> transpose_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 55000;
  for (const TransposeShape& shape : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, false));
      uid += 2;
    }
  }
  return result;
}

std::vector<BenchmarkCase> transpose_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 56000;
  for (const TransposeShape& shape : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, true));
      uid += 2;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
