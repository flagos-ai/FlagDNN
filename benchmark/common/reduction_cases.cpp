/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "cases.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

constexpr std::array<flagdnnDataType_t, 3> kDataTypes = {
    FLAGDNN_DATA_FLOAT32,
    FLAGDNN_DATA_FLOAT16,
    FLAGDNN_DATA_BFLOAT16,
};

constexpr std::array<flagdnnReductionMode_t, 3> kModes = {
    FLAGDNN_REDUCTION_ADD,
    FLAGDNN_REDUCTION_AVG,
    FLAGDNN_REDUCTION_MUL,
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

std::string mode_name(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "sum";
    case FLAGDNN_REDUCTION_AVG:
      return "avg";
    case FLAGDNN_REDUCTION_MUL:
      return "mul";
  }
  return "invalid";
}

void set_tolerance(BenchmarkCase& specification,
                   flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      specification.absolute_tolerance = 2.0e-5;
      specification.relative_tolerance = 1.0e-5;
      return;
    case FLAGDNN_DATA_FLOAT16:
      specification.absolute_tolerance = 5.0e-2;
      specification.relative_tolerance = 1.0e-2;
      return;
    case FLAGDNN_DATA_BFLOAT16:
      specification.absolute_tolerance = 8.0e-2;
      specification.relative_tolerance = 1.0e-2;
      return;
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      break;
    case FLAGDNN_DATA_BOOLEAN:
      specification.absolute_tolerance = 0.0;
      specification.relative_tolerance = 0.0;
      return;
  }
}

BenchmarkCase make_regular_case(flagdnnReductionMode_t mode,
                           flagdnnDataType_t data_type,
                           std::int64_t uid) {
  BenchmarkCase result;
  result.name = "reduction_" + mode_name(mode) + "_" +
                data_type_name(data_type) +
                "_axis1_keepdim_2x4x8x8";
  result.operation = Operation::kReduction;
  result.tensors = {
      tensor(uid, {2, 4, 8, 8}, data_type),
      tensor(uid + 1, {2, 1, 8, 8}, data_type),
  };
  result.reduction_mode = mode;
  result.reduction_axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result, data_type);
  return result;
}

BenchmarkCase make_channels_last_case(flagdnnReductionMode_t mode,
                                 flagdnnDataType_t data_type,
                                 std::int64_t uid) {
  BenchmarkCase result;
  result.name = "reduction_" + mode_name(mode) + "_" +
                data_type_name(data_type) +
                "_channels_last_axis1_keepdim_2x3x5x5";
  result.operation = Operation::kReduction;
  result.tensors = {
      strided_tensor(uid, {2, 3, 5, 5}, {75, 1, 15, 3}, data_type),
      tensor(uid + 1, {2, 1, 5, 5}, data_type),
  };
  result.reduction_mode = mode;
  result.reduction_axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result, data_type);
  return result;
}

BenchmarkCase make_unaligned_entrance_case(flagdnnDataType_t data_type,
                                     std::int64_t uid) {
  BenchmarkCase result;
  result.name = "reduction_sum_" + data_type_name(data_type) +
                "_unaligned_input_entrance_axis1_2x4x8x8";
  result.operation = Operation::kReduction;
  TensorSpec input = strided_tensor(
      uid, {2, 4, 8, 8}, {288, 72, 9, 1}, data_type);
  input.binding_byte_offset =
      data_type == FLAGDNN_DATA_FLOAT32 ? 4U : 2U;
  result.tensors = {
      std::move(input),
      tensor(uid + 1, {2, 1, 8, 8}, data_type),
  };
  result.reduction_mode = FLAGDNN_REDUCTION_ADD;
  result.reduction_axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result, data_type);
  return result;
}

BenchmarkCase make_benchmark_case(flagdnnReductionMode_t mode,
                             flagdnnDataType_t data_type,
                             std::int64_t uid) {
  const bool multiply = mode == FLAGDNN_REDUCTION_MUL;
  const std::int64_t channels = multiply ? 4 : 8;
  const std::int64_t spatial = multiply ? 16 : 32;
  const std::string shape = multiply ? "8x4x16x16" : "8x8x32x32";
  BenchmarkCase result;
  result.name = "reduction_" + mode_name(mode) + "_perf_" +
                data_type_name(data_type) + "_axis1_keepdim_" + shape;
  result.operation = Operation::kReduction;
  result.tensors = {
      tensor(uid, {8, channels, spatial, spatial}, data_type),
      tensor(uid + 1, {8, 1, spatial, spatial}, data_type),
  };
  result.reduction_mode = mode;
  result.reduction_axis = 1;
  result.keep_dimensions = true;
  set_tolerance(result, data_type);
  return result;
}

}  // namespace

std::vector<BenchmarkCase> reduction_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(23);

  BenchmarkCase last_axis;
  last_axis.name = "reduction_sum_fp32_7x256";
  last_axis.operation = Operation::kReduction;
  last_axis.tensors = {tensor(6, {7, 256}), tensor(7, {7})};
  last_axis.reduction_mode = FLAGDNN_REDUCTION_ADD;
  last_axis.reduction_axis = -1;
  last_axis.keep_dimensions = false;
  set_tolerance(last_axis, FLAGDNN_DATA_FLOAT32);
  result.push_back(std::move(last_axis));

  std::int64_t uid = 100;
  for (const flagdnnReductionMode_t mode : kModes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_regular_case(mode, data_type, uid));
      uid += 2;
      result.push_back(make_channels_last_case(mode, data_type, uid));
      uid += 2;
    }
  }

  for (const flagdnnDataType_t data_type : kDataTypes) {
    result.push_back(make_unaligned_entrance_case(data_type, uid));
    uid += 2;
  }

  BenchmarkCase scalar;
  scalar.name = "reduction_sum_fp32_scalar_8";
  scalar.operation = Operation::kReduction;
  scalar.tensors = {tensor(uid, {8}), tensor(uid + 1, {})};
  scalar.reduction_mode = FLAGDNN_REDUCTION_ADD;
  scalar.reduction_axis = 0;
  scalar.keep_dimensions = false;
  set_tolerance(scalar, FLAGDNN_DATA_FLOAT32);
  result.push_back(std::move(scalar));

  return result;
}

std::vector<BenchmarkCase> reduction_benchmark_cases() {
  std::vector<BenchmarkCase> result;
  result.reserve(9);
  std::int64_t uid = 200;
  for (const flagdnnReductionMode_t mode : kModes) {
    for (const flagdnnDataType_t data_type : kDataTypes) {
      result.push_back(make_benchmark_case(mode, data_type, uid));
      uid += 2;
    }
  }
  return result;
}

}  // namespace flagdnn::benchmarking
