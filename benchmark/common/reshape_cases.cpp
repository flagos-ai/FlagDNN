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
  return result.empty() ? "scalar" : result;
}

TensorSpec make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = contiguous_strides(dimensions);
  return result;
}

BenchmarkCase make_case(const ShapePair& shapes,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark) {
  BenchmarkCase result;
  result.name = "reshape_" + std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" +
                shape_name(shapes.first) + "_to_" +
                shape_name(shapes.second);
  result.operation = Operation::kReshape;
  result.tensors = {
      make_tensor(uid, shapes.first, data_type),
      make_tensor(uid + 1, shapes.second, data_type),
  };
  result.reshape.dimensions = shapes.second;
  result.reshape.strides = result.tensors[1].strides;
  result.reshape.logical = true;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

const std::vector<ShapePair>& correctness_shapes() {
  static const std::vector<ShapePair> result = {
      {{2, 3, 4}, {6, 4}},
      {{1, 8, 16}, {4, 32}},
      {{4, 5, 6}, {2, 3, 20}},
  };
  return result;
}

const std::vector<ShapePair>& benchmark_shapes() {
  static const std::vector<ShapePair> result = {
      {{8, 16, 32}, {128, 32}},
      {{16, 64, 128}, {1024, 128}},
      {{16, 256, 256}, {4096, 256}},
      {{32, 128, 256}, {4096, 256}},
      {{4, 1024, 1024}, {4096, 1024}},
  };
  return result;
}

}  // namespace

std::vector<BenchmarkCase> reshape_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 53000;
  for (const ShapePair& shapes : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shapes, data_type, uid, false));
      uid += 2;
    }
  }
  return result;
}

std::vector<BenchmarkCase> reshape_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 54000;
  for (const ShapePair& shapes : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shapes, data_type, uid, true));
      uid += 2;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
