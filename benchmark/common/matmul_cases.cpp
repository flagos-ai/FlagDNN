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

Shape broadcast_batch(const Shape& a, const Shape& b) {
  const std::size_t a_rank = a.size() - 2;
  const std::size_t b_rank = b.size() - 2;
  const std::size_t rank = std::max(a_rank, b_rank);
  Shape result(rank, 1);
  for (std::size_t trailing = 0; trailing < rank; ++trailing) {
    const std::int64_t a_dimension =
        trailing < a_rank ? a[a_rank - 1 - trailing] : 1;
    const std::int64_t b_dimension =
        trailing < b_rank ? b[b_rank - 1 - trailing] : 1;
    if (a_dimension != b_dimension && a_dimension != 1 &&
        b_dimension != 1) {
      throw std::invalid_argument(
          "MatMul case batch dimensions are incompatible");
    }
    result[rank - 1 - trailing] =
        std::max(a_dimension, b_dimension);
  }
  return result;
}

Shape output_shape(const Shape& a, const Shape& b) {
  if (a.size() < 2 || b.size() < 2 || a.back() != b[b.size() - 2]) {
    throw std::invalid_argument("MatMul case shapes are invalid");
  }
  Shape result = broadcast_batch(a, b);
  result.push_back(a[a.size() - 2]);
  result.push_back(b.back());
  return result;
}

TensorSpec make_tensor(std::int64_t uid,
                       const Shape& dimensions,
                       flagdnnDataType_t data_type,
                       Shape strides = {}) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = strides.empty() ? contiguous_strides(dimensions)
                                   : std::move(strides);
  return result;
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type,
                   std::int64_t k) {
  if (data_type == FLAGDNN_DATA_FLOAT16) {
    specification.absolute_tolerance = 5.0e-2;
    specification.relative_tolerance = 5.0e-2;
  } else if (data_type == FLAGDNN_DATA_BFLOAT16) {
    specification.absolute_tolerance = 1.0e-1;
    specification.relative_tolerance = 5.0e-2;
  } else {
    specification.absolute_tolerance =
        5.0e-3 * std::sqrt(std::max(1.0, static_cast<double>(k) / 512.0));
    specification.relative_tolerance = 5.0e-3;
  }
}

BenchmarkCase make_case(const ShapePair& shapes,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   bool benchmark) {
  const Shape output = output_shape(shapes.first, shapes.second);
  BenchmarkCase result;
  result.name = "matmul_" + std::string(benchmark ? "perf_" : "") +
                data_type_name(data_type) + "_" +
                shape_name(shapes.first) + "_by_" +
                shape_name(shapes.second);
  result.operation = Operation::kMatmul;
  result.tensors = {
      make_tensor(uid, shapes.first, data_type),
      make_tensor(uid + 1, shapes.second, data_type),
      make_tensor(uid + 2, output, data_type),
  };
  set_tolerance(result, data_type, shapes.first.back());
  if (benchmark && data_type == FLAGDNN_DATA_FLOAT32) {
    // cuDNN may select TF32 for FLOAT MatMul. The functional suite keeps the
    // IEEE host oracle; this performance gate only rejects differences beyond
    // the expected TF32 envelope before timing both GPU providers.
    result.absolute_tolerance =
        1.0e-1 * std::sqrt(static_cast<double>(shapes.first.back()) / 512.0);
    result.relative_tolerance = 5.0e-2;
  }
  if (benchmark) {
    result.benchmark.warmup_iterations = 5;
    result.benchmark.sample_count = 10;
    result.benchmark.iterations_per_sample = 5;
  }
  return result;
}

const std::vector<ShapePair>& correctness_shapes() {
  static const std::vector<ShapePair> result = {
      {{4, 16, 32}, {4, 32, 24}},
      {{8, 32, 64}, {8, 64, 32}},
      {{16, 32, 128}, {16, 128, 64}},
      {{4, 17, 30}, {4, 30, 23}},
      {{2, 65, 130}, {2, 130, 33}},
      {{1, 64, 64}, {1, 64, 64}},
      {{32, 64}, {64, 24}},
      {{2, 1, 17, 30}, {3, 30, 23}},
  };
  return result;
}

const std::vector<ShapePair>& benchmark_shapes() {
  static const std::vector<ShapePair> result = {
      {{4, 16, 32}, {4, 32, 24}},
      {{8, 32, 64}, {8, 64, 32}},
      {{32, 512, 512}, {32, 512, 512}},
      {{16, 1024, 1024}, {16, 1024, 1024}},
      {{8, 2048, 2048}, {8, 2048, 2048}},
      {{4, 4096, 4096}, {4, 4096, 4096}},
      {{16, 2048, 512}, {16, 512, 2048}},
      {{32, 1024, 4096}, {32, 4096, 1024}},
  };
  return result;
}

}  // namespace

std::vector<BenchmarkCase> matmul_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 51000;
  for (const ShapePair& shapes : correctness_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shapes, data_type, uid, false));
      uid += 3;
    }
  }
  for (const flagdnnDataType_t data_type : kDataTypes) {
    BenchmarkCase strided = make_case(
        {{2, 17, 30}, {2, 30, 23}}, data_type, uid, false);
    strided.name = "matmul_" + data_type_name(data_type) +
                   "_strided_2x17x30_by_2x30x23";
    strided.tensors[0].strides = {600, 31, 1};
    strided.tensors[1].strides = {800, 1, 32};
    strided.tensors[2].strides = {500, 25, 1};
    result.push_back(std::move(strided));
    uid += 3;
  }
  return result;
}

std::vector<BenchmarkCase> matmul_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 52000;
  for (const ShapePair& shapes : benchmark_shapes()) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_case(shapes, data_type, uid, true));
      uid += 3;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
