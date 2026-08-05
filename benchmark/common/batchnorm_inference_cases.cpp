/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstdint>
#include <iterator>
#include <utility>
#include <string>
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
      result += "x";
    }
    result += std::to_string(dimension);
  }
  return result.empty() ? "scalar" : result;
}

TensorSpec make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type,
                       const std::vector<std::int64_t>& strides = {}) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = strides.empty() ? contiguous_strides(dimensions) : strides;
  return result;
}

std::vector<std::int64_t> channels_last_strides(const Shape& shape) {
  const std::int64_t channels = shape[1];
  const std::int64_t height = shape[2];
  const std::int64_t width = shape[3];
  return {channels * height * width, 1, width * channels, channels};
}

BenchmarkCase make_case(const Shape& shape,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark,
                   bool channels_last = false) {
  const std::int64_t channels = shape[1];
  const Shape parameter_shape = {1, channels, 1, 1};
  const std::vector<std::int64_t> data_strides =
      channels_last ? channels_last_strides(shape) : contiguous_strides(shape);

  BenchmarkCase result;
  result.name = "batchnorm_inference_" +
                std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" + shape_name(shape) +
                (channels_last ? "_channels_last" : "");
  result.operation = Operation::kBatchnormInference;
  result.tensors = {
      make_tensor(uid, shape, data_type, data_strides),
      make_tensor(uid + 1, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 2, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 3, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 4, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 5, shape, data_type, data_strides),
  };
  result.input_domains = {
      InputDomain::kReal,
      InputDomain::kReal,
      InputDomain::kPositive,
      InputDomain::kReal,
      InputDomain::kReal,
  };
  result.absolute_tolerance = data_type == FLAGDNN_DATA_FLOAT32 ? 1.0e-5 :
                              data_type == FLAGDNN_DATA_FLOAT16 ? 2.0e-2 :
                                                                 5.0e-2;
  result.relative_tolerance = result.absolute_tolerance;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

const std::vector<Shape>& correctness_shapes() {
  static const std::vector<Shape> result = {
      {2, 8, 16, 16},
      {4, 16, 8, 8},
      {2, 32, 7, 9},
  };
  return result;
}

const std::vector<Shape>& benchmark_shapes() {
  static const std::vector<Shape> result = {
      {8, 32, 32, 32},
      {16, 64, 16, 16},
      {4, 128, 16, 16},
      {8, 64, 56, 56},
      {16, 128, 28, 28},
      {16, 256, 14, 14},
      {8, 512, 7, 7},
      {32, 1024, 1, 1},
  };
  return result;
}

}  // namespace

namespace {

BenchmarkCase make_training_case(const Shape& shape,
                            flagdnnDataType_t data_type,
                            std::int64_t uid,
                            bool benchmark,
                            bool channels_last = false) {
  const std::int64_t channels = shape[1];
  const Shape parameter_shape = {1, channels, 1, 1};
  const std::vector<std::int64_t> data_strides =
      channels_last ? channels_last_strides(shape) : contiguous_strides(shape);

  BenchmarkCase result;
  result.name = "batchnorm_" +
                std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" + shape_name(shape) +
                (channels_last ? "_channels_last" : "");
  result.operation = Operation::kBatchnorm;
  result.output_count = 5;
  result.tensors = {
      make_tensor(uid, shape, data_type, data_strides),
      make_tensor(uid + 1, parameter_shape, data_type),
      make_tensor(uid + 2, parameter_shape, data_type),
      make_tensor(uid + 3, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 4, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 5, shape, data_type, data_strides),
      make_tensor(uid + 6, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 7, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 8, parameter_shape, FLAGDNN_DATA_FLOAT32),
      make_tensor(uid + 9, parameter_shape, FLAGDNN_DATA_FLOAT32),
  };
  result.input_domains = {
      InputDomain::kReal,
      InputDomain::kReal,
      InputDomain::kReal,
      InputDomain::kReal,
      InputDomain::kPositive,
  };
  result.normalization.epsilon = 1.0e-3;
  result.normalization.momentum = 0.1;
  result.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT32
          ? (benchmark ? 2.0e-3 : 2.0e-4)
          : (data_type == FLAGDNN_DATA_FLOAT16 ? 3.0e-2 : 7.0e-2);
  result.relative_tolerance = result.absolute_tolerance;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

}  // namespace


std::vector<BenchmarkCase> batchnorm_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 63000;
  const Shape shape = {2, 8, 8, 8};
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_training_case(shape, data_type, uid, false));
    uid += 10;
    result.push_back(
        make_training_case(shape, data_type, uid, false, true));
    uid += 10;
  }
  return result;
}

std::vector<BenchmarkCase> batchnorm_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 64000;
  for (const Shape& shape : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_training_case(shape, data_type, uid, true));
      uid += 10;
    }
  }
  return result;
}

std::vector<BenchmarkCase> batchnorm_inference_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 61000;
  for (const Shape& shape : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, false));
      uid += 6;
    }
  }
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(
        make_case(correctness_shapes().front(), data_type, uid, false, true));
    uid += 6;
  }
  return result;
}

std::vector<BenchmarkCase> batchnorm_inference_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 62000;
  for (const Shape& shape : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shape, data_type, uid, true));
      uid += 6;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking

namespace flagdnn::benchmarking {
namespace {

const std::vector<Shape>& normalization_benchmark_shapes() {
  static const std::vector<Shape> result = {
      {1, 128, 768},
      {8, 128, 1024},
      {4, 256, 2048},
      {2, 512, 4096},
      {3, 257, 513},
  };
  return result;
}

BenchmarkCase make_normalization_case(
    Operation operation,
    const char* operation_name,
    const Shape& shape,
    flagdnnDataType_t data_type,
    std::int64_t uid,
    bool benchmark) {
  Shape parameter_shape(shape.size(), 1);
  parameter_shape.back() = shape.back();
  Shape statistic_shape = shape;
  statistic_shape.back() = 1;
  BenchmarkCase result;
  result.name = std::string(operation_name) +
                (benchmark ? "_perf_" : "_") +
                data_type_name(data_type) + "_" + shape_name(shape);
  result.operation = operation;
  result.normalization.epsilon = 1.0e-3;
  if (operation == Operation::kLayernorm) {
    result.output_count = 3;
    result.tensors = {
        make_tensor(uid, shape, data_type),
        make_tensor(uid + 1, parameter_shape, data_type),
        make_tensor(uid + 2, parameter_shape, data_type),
        make_tensor(uid + 3, shape, data_type),
        make_tensor(uid + 4, statistic_shape, FLAGDNN_DATA_FLOAT32),
        make_tensor(uid + 5, statistic_shape, FLAGDNN_DATA_FLOAT32),
    };
  } else {
    result.output_count = 2;
    result.tensors = {
        make_tensor(uid, shape, data_type),
        make_tensor(uid + 1, parameter_shape, data_type),
        make_tensor(uid + 2, parameter_shape, data_type),
        make_tensor(uid + 3, shape, data_type),
        make_tensor(uid + 4, statistic_shape, FLAGDNN_DATA_FLOAT32),
    };
  }
  result.input_domains = {
      InputDomain::kReal, InputDomain::kReal, InputDomain::kReal};
  result.absolute_tolerance =
      data_type == FLAGDNN_DATA_FLOAT32 ? 2.0e-4 : 2.0e-2;
  result.relative_tolerance = result.absolute_tolerance;
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 20;
  }
  return result;
}

void append_cases(std::vector<BenchmarkCase>& destination,
                  std::vector<BenchmarkCase> source) {
  destination.insert(
      destination.end(),
      std::make_move_iterator(source.begin()),
      std::make_move_iterator(source.end()));
}

}  // namespace

std::vector<BenchmarkCase> layernorm_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 65000;
  for (const Shape& shape :
       std::vector<Shape>{{2, 5, 17}, {2, 4, 4096}}) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_normalization_case(
          Operation::kLayernorm,
          "layernorm",
          shape,
          data_type,
          uid,
          false));
      uid += 6;
    }
  }
  return result;
}

std::vector<BenchmarkCase> layernorm_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 66000;
  for (const Shape& shape : normalization_benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_normalization_case(
          Operation::kLayernorm,
          "layernorm",
          shape,
          data_type,
          uid,
          true));
      uid += 6;
    }
  }
  return result;
}

std::vector<BenchmarkCase> rmsnorm_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 67000;
  const Shape shape = {2, 5, 17};
  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_normalization_case(
        Operation::kRmsnorm,
        "rmsnorm",
        shape,
        data_type,
        uid,
        false));
    uid += 5;
  }
  return result;
}

std::vector<BenchmarkCase> rmsnorm_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 68000;
  for (const Shape& shape : normalization_benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_normalization_case(
          Operation::kRmsnorm,
          "rmsnorm",
          shape,
          data_type,
          uid,
          true));
      uid += 5;
    }
  }
  return result;
}

std::vector<BenchmarkCase> normalization_forward_cases() {
  std::vector<BenchmarkCase> result;
  append_cases(result, layernorm_cases());
  append_cases(result, rmsnorm_cases());
  append_cases(result, batchnorm_cases());
  return result;
}

}  // namespace flagdnn::benchmarking
