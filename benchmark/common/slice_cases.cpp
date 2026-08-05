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
using SliceRange = std::pair<std::int64_t, std::int64_t>;

struct SliceShape {
  Shape input;
  std::vector<SliceRange> slices;
  Shape strides;
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

BenchmarkCase make_case(const SliceShape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark,
                   std::size_t case_index) {
  const Shape input_strides = contiguous_strides(shape.input);
  Shape output_dimensions(shape.input.size());
  Shape output_strides(shape.input.size());
  for (std::size_t axis = 0; axis < shape.input.size(); ++axis) {
    const std::int64_t start = shape.slices[axis].first;
    const std::int64_t limit = shape.slices[axis].second;
    const std::int64_t step = shape.strides[axis];
    output_dimensions[axis] = (limit - start + step - 1) / step;
    output_strides[axis] = input_strides[axis] * step;
  }
  BenchmarkCase result;
  result.name = "slice_" + std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_case" +
                std::to_string(case_index) + "_" +
                shape_name(shape.input);
  result.operation = Operation::kSlice;
  result.tensors = {
      make_tensor(uid, shape.input, input_strides, data_type),
      make_tensor(uid + 1,
                  std::move(output_dimensions),
                  std::move(output_strides),
                  data_type),
  };
  result.slice.slices = shape.slices;
  result.slice.strides = shape.strides;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

const std::vector<SliceShape>& correctness_shapes() {
  static const std::vector<SliceShape> result = {
      {{2, 4, 5}, {{0, 2}, {1, 4}, {0, 5}}, {1, 2, 1}},
      {{4, 6, 8}, {{1, 4}, {0, 6}, {2, 8}}, {1, 2, 3}},
      {{3, 5, 7, 2},
       {{0, 3}, {1, 5}, {0, 7}, {0, 2}},
       {1, 2, 1, 1}},
  };
  return result;
}

const std::vector<SliceShape>& benchmark_shapes() {
  static const std::vector<SliceShape> result = {
      {{8, 16, 32}, {{0, 8}, {2, 14}, {0, 32}}, {1, 2, 1}},
      {{16, 64, 128},
       {{1, 15}, {0, 64}, {8, 120}},
       {1, 2, 4}},
      {{32, 128, 256},
       {{0, 32}, {4, 124}, {16, 240}},
       {1, 3, 2}},
      {{4, 64, 128, 32},
       {{0, 4}, {8, 56}, {0, 128}, {4, 28}},
       {1, 2, 1, 2}},
  };
  return result;
}

}  // namespace

std::vector<BenchmarkCase> slice_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 57000;
  std::size_t case_index = 0;
  for (const SliceShape& shape : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_case(shape, data_type, uid, false, case_index));
      uid += 2;
    }
    ++case_index;
  }
  return result;
}

std::vector<BenchmarkCase> slice_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 58000;
  std::size_t case_index = 0;
  for (const SliceShape& shape : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_case(shape, data_type, uid, true, case_index));
      uid += 2;
    }
    ++case_index;
  }
  return result;
}

}  // namespace flagdnn::benchmarking
