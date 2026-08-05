/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <algorithm>
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

TensorSpec strided_pointwise_tensor(
    std::int64_t uid,
    const Shape& dimensions,
    const Shape& strides,
    flagdnnDataType_t data_type) {
  TensorSpec result;
  result.uid = uid;
  result.data_type = data_type;
  result.dimensions = dimensions;
  result.strides = strides;
  return result;
}

Shape broadcast_dimensions(const Shape& left, const Shape& right) {
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
  if (data_type == FLAGDNN_DATA_BOOLEAN) {
    specification.absolute_tolerance = 0.0;
    specification.relative_tolerance = 0.0;
    return;
  }
  specification.absolute_tolerance =
      data_type == FLAGDNN_DATA_BFLOAT16 ? 5.0e-2 : 2.0e-2;
  specification.relative_tolerance = 1.0e-2;
}

bool is_comparison_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool is_logical_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

std::vector<flagdnnDataType_t> input_data_types(
    flagdnnPointwiseMode_t mode) {
  if (is_logical_mode(mode)) {
    return {FLAGDNN_DATA_BOOLEAN};
  }
  return std::vector<flagdnnDataType_t>(
      kDataTypes.begin(), kDataTypes.end());
}

flagdnnDataType_t output_data_type(flagdnnPointwiseMode_t mode,
                                   flagdnnDataType_t input_data_type) {
  if (is_comparison_mode(mode) || is_logical_mode(mode)) {
    return FLAGDNN_DATA_BOOLEAN;
  }
  return input_data_type;
}

BenchmarkCase make_case(const ShapePair& shapes,
                   flagdnnDataType_t data_type,
                   std::int64_t uid,
                   flagdnnPointwiseMode_t mode,
                   std::string_view operation_name,
                   InputDomain input_domain,
                   bool benchmark,
                   double alpha = 1.0) {
  BenchmarkCase result;
  result.name = std::string(operation_name) +
                (benchmark ? "_perf_" : "_") +
                data_type_name(data_type) + "_" +
                shape_name(shapes.first) + "_by_" +
                shape_name(shapes.second);
  result.operation = Operation::kPointwise;
  result.pointwise_mode = mode;
  result.input_domain =
      is_comparison_mode(mode) ? InputDomain::kComparison : input_domain;
  result.add_alpha = alpha;
  const Shape output_dimensions =
      broadcast_dimensions(shapes.first, shapes.second);
  result.tensors = {
      pointwise_tensor(uid, shapes.first, data_type),
      pointwise_tensor(uid + 1, shapes.second, data_type),
      pointwise_tensor(
          uid + 2, output_dimensions, output_data_type(mode, data_type)),
  };
  set_tolerance(result, output_data_type(mode, data_type));
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

const std::vector<ShapePair>& broadcast_shapes() {
  static const std::vector<ShapePair> shapes = {
      {{2, 3, 17}, {1, 3, 17}},
      {{2, 3, 17}, {2, 1, 17}},
      {{3, 5, 7}, {3, 1, 7}},
      {{3, 5, 7}, {1, 5, 7}},
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

std::vector<BenchmarkCase> binary_pointwise_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain) {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 3000;
  const std::vector<flagdnnDataType_t> data_types = input_data_types(mode);
  const auto append_shapes = [&](const std::vector<ShapePair>& shapes) {
    for (const ShapePair& shape : shapes) {
      for (const flagdnnDataType_t data_type : data_types) {
        result.push_back(make_case(shape,
                                   data_type,
                                   uid,
                                   mode,
                                   operation_name,
                                   input_domain,
                                   false));
        uid += 3;
      }
    }
  };

  append_shapes(correctness_shapes());
  if (mode != FLAGDNN_POINTWISE_POW &&
      mode != FLAGDNN_POINTWISE_SIGMOID_BWD &&
      !is_logical_mode(mode)) {
    append_shapes(broadcast_shapes());
  }

  if (mode == FLAGDNN_POINTWISE_SUB) {
    for (const double alpha : {0.5, -2.0}) {
      for (const flagdnnDataType_t data_type : data_types) {
        BenchmarkCase specification = make_case({{2, 4, 8}, {2, 4, 8}},
                                           data_type,
                                           uid,
                                           mode,
                                           operation_name,
                                           input_domain,
                                           false,
                                           alpha);
        specification.name += alpha > 0.0 ? "_alpha_0p5" : "_alpha_neg2";
        result.push_back(std::move(specification));
        uid += 3;
      }
    }
  }

  if (mode == FLAGDNN_POINTWISE_MOD) {
    for (const flagdnnDataType_t data_type : data_types) {
      result.push_back(make_case({{1, 1, 6}, {1, 1, 6}},
                                 data_type,
                                 uid,
                                 mode,
                                 operation_name,
                                 InputDomain::kModuloSigned,
                                 false));
      result.back().name += "_signed";
      uid += 3;
    }
  }

  for (const flagdnnDataType_t data_type : data_types) {
    const bool requires_equal_shapes =
        mode == FLAGDNN_POINTWISE_SIGMOID_BWD;
    BenchmarkCase strided;
    strided.name = std::string(operation_name) + "_strided_" +
                   data_type_name(data_type) + "_2x3x4";
    strided.operation = Operation::kPointwise;
    strided.pointwise_mode = mode;
    strided.input_domain =
        is_comparison_mode(mode) ? InputDomain::kComparison : input_domain;
    strided.tensors = {
        strided_tensor(uid, {2, 3, 4}, {31, 9, 2}, data_type),
        strided_pointwise_tensor(
            uid + 1,
            requires_equal_shapes ? Shape{2, 3, 4} : Shape{1, 3, 4},
            requires_equal_shapes ? Shape{31, 9, 2} : Shape{29, 8, 2},
            data_type),
        strided_tensor(uid + 2,
                       {2, 3, 4},
                       {37, 11, 2},
                       output_data_type(mode, data_type)),
    };
    set_tolerance(strided, output_data_type(mode, data_type));
    result.push_back(std::move(strided));
    uid += 3;
  }
  return result;
}

std::vector<BenchmarkCase> binary_pointwise_benchmark_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain) {
  const std::vector<ShapePair>& shapes = benchmark_shapes();
  std::vector<BenchmarkCase> result;
  std::int64_t uid = 4000;
  const std::vector<flagdnnDataType_t> data_types = input_data_types(mode);
  for (std::size_t index = 0; index < shapes.size(); ++index) {
    if (mode == FLAGDNN_POINTWISE_MOD && index != 0 && index != 4) {
      continue;
    }
    for (const flagdnnDataType_t data_type : data_types) {
      result.push_back(make_case(shapes[index],
                                 data_type,
                                 uid,
                                 mode,
                                 operation_name,
                                 input_domain,
                                 true));
      uid += 3;
    }
  }
  return result;
}

namespace {

using ShapeTriple = std::array<Shape, 3>;

BenchmarkCase make_binary_select_case(const ShapeTriple& shapes,
                                 flagdnnDataType_t data_type,
                                 std::int64_t uid,
                                 bool benchmark) {
  const Shape value_dimensions =
      broadcast_dimensions(shapes[0], shapes[1]);
  const Shape output_dimensions =
      broadcast_dimensions(value_dimensions, shapes[2]);
  BenchmarkCase result;
  result.name = std::string("binary_select") +
                (benchmark ? "_perf_" : "_") +
                data_type_name(data_type) + "_" +
                shape_name(shapes[0]) + "_by_" +
                shape_name(shapes[1]) + "_mask_" +
                shape_name(shapes[2]);
  result.operation = Operation::kPointwise;
  result.pointwise_mode = FLAGDNN_POINTWISE_BINARY_SELECT;
  result.input_domain = InputDomain::kReal;
  result.input_domains = {
      InputDomain::kReal, InputDomain::kReal, InputDomain::kLogical};
  result.tensors = {
      pointwise_tensor(uid, shapes[0], data_type),
      pointwise_tensor(uid + 1, shapes[1], data_type),
      pointwise_tensor(uid + 2, shapes[2], FLAGDNN_DATA_BOOLEAN),
      pointwise_tensor(uid + 3, output_dimensions, data_type),
  };
  set_tolerance(result, data_type);
  return result;
}

const std::vector<ShapeTriple>& binary_select_correctness_shapes() {
  static const std::vector<ShapeTriple> shapes = {
      {{{1, 1, 16}, {1, 1, 16}, {1, 1, 16}}},
      {{{2, 4, 8}, {2, 4, 8}, {2, 4, 8}}},
      {{{1, 4, 8, 16}, {1, 4, 8, 16}, {1, 4, 8, 16}}},
      {{{2, 4, 8, 16}, {2, 4, 8, 16}, {2, 4, 8, 16}}},
      {{{1, 3, 17}, {1, 3, 17}, {1, 3, 17}}},
      {{{3, 5, 7}, {3, 5, 7}, {3, 5, 7}}},
      {{{2, 3, 17}, {1, 3, 17}, {2, 3, 17}}},
      {{{3, 5, 7}, {3, 5, 7}, {3, 1, 7}}},
  };
  return shapes;
}

const std::vector<ShapeTriple>& binary_select_performance_shapes() {
  static const std::vector<ShapeTriple> shapes = {
      {{{1, 1, 1024}, {1, 1, 1024}, {1, 1, 1024}}},
      {{{8, 16, 32}, {8, 16, 32}, {8, 16, 32}}},
      {{{4, 16, 64, 128},
        {4, 16, 64, 128},
        {4, 16, 64, 128}}},
      {{{8, 16, 64, 128},
        {8, 16, 64, 128},
        {8, 16, 64, 128}}},
      {{{1, 1, 1000}, {1, 1, 1000}, {1, 1, 1000}}},
      {{{3, 257, 513}, {3, 257, 513}, {3, 257, 513}}},
      {{{3, 7, 65, 129},
        {3, 7, 65, 129},
        {3, 7, 65, 129}}},
      {{{5, 7, 65, 129},
        {5, 7, 65, 129},
        {5, 7, 65, 129}}},
  };
  return shapes;
}

std::vector<BenchmarkCase> make_binary_select_cases(
    const std::vector<ShapeTriple>& shapes, bool benchmark) {
  std::vector<BenchmarkCase> result;
  std::int64_t uid = benchmark ? 21000 : 20000;
  for (const ShapeTriple& shape : shapes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(
          make_binary_select_case(shape, data_type, uid, benchmark));
      uid += 4;
    }
  }
  return result;
}

}  // namespace

std::vector<BenchmarkCase> binary_select_cases() {
  std::vector<BenchmarkCase> result = make_binary_select_cases(
      binary_select_correctness_shapes(), false);
  std::int64_t uid = 22000;
  for (const flagdnnDataType_t data_type : kDataTypes) {
    BenchmarkCase strided = make_binary_select_case(
        {{{2, 3, 4}, {2, 3, 4}, {2, 3, 4}}},
        data_type,
        uid,
        false);
    strided.name = "binary_select_strided_" + data_type_name(data_type);
    strided.tensors = {
        strided_tensor(uid, {2, 3, 4}, {31, 9, 2}, data_type),
        strided_tensor(uid + 1, {2, 3, 4}, {37, 11, 2}, data_type),
        strided_tensor(
            uid + 2, {2, 3, 4}, {41, 13, 3}, FLAGDNN_DATA_BOOLEAN),
        strided_tensor(uid + 3, {2, 3, 4}, {43, 14, 3}, data_type),
    };
    result.push_back(std::move(strided));
    uid += 4;
  }
  return result;
}

std::vector<BenchmarkCase> binary_select_benchmark_cases() {
  return make_binary_select_cases(
      binary_select_performance_shapes(), true);
}

}  // namespace flagdnn::benchmarking
