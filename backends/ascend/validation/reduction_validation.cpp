/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/reduction_validation.hpp"

#include "validation/tensor_io.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace tensor_io = flagdnn::validation::ascend::tensor_io;

std::string mode_name(flagdnnReductionMode_t mode) {
  switch (mode) {
    case FLAGDNN_REDUCTION_ADD:
      return "sum";
    case FLAGDNN_REDUCTION_AVG:
      return "avg";
    case FLAGDNN_REDUCTION_MUL:
      return "mul";
  }
  throw std::invalid_argument("Ascend reduction mode is invalid");
}

ReductionTestCase make_special_case(flagdnnReductionMode_t mode) {
  ReductionTestCase result;
  result.name = "reduction_" + mode_name(mode) +
                "_ascend_gapped_offsets_padding_fp32";
  result.input = {79000,
                  FLAGDNN_DATA_FLOAT32,
                  {2, 3, 4},
                  {43, 11, 2},
                  32};
  result.output = {79001,
                   FLAGDNN_DATA_FLOAT32,
                   {2, 4},
                   {13, 2},
                   64};
  result.mode = mode;
  result.axis = 1;
  result.keep_dimensions = false;
  result.absolute_tolerance = 2.0e-5;
  result.relative_tolerance = 1.0e-5;
  validate_reduction_case(result);
  (void)tensor_io::encoded_byte_count(result.input);
  (void)tensor_io::encoded_byte_count(result.output);
  return result;
}

std::vector<std::int64_t> logical_coordinates(
    std::size_t logical_index,
    std::span<const std::int64_t> dimensions) {
  std::vector<std::int64_t> result(dimensions.size());
  for (std::size_t axis = dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(dimensions[current]);
    result[current] = static_cast<std::int64_t>(logical_index % dimension);
    logical_index /= dimension;
  }
  return result;
}

std::size_t row_major_index(
    std::span<const std::int64_t> coordinates,
    std::span<const std::int64_t> dimensions) {
  if (coordinates.size() != dimensions.size()) {
    throw std::invalid_argument(
        "reduction host-oracle coordinate rank differs");
  }
  std::size_t result = 0;
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (coordinates[axis] < 0 || coordinates[axis] >= dimensions[axis]) {
      throw std::out_of_range(
          "reduction host-oracle coordinate is out of range");
    }
    result = result * static_cast<std::size_t>(dimensions[axis]) +
             static_cast<std::size_t>(coordinates[axis]);
  }
  return result;
}

}  // namespace

std::vector<ReductionTestCase> make_ascend_reduction_cases(
    std::span<const ReductionTestCase> common_cases,
    flagdnnReductionMode_t mode) {
  if (common_cases.empty()) {
    throw std::invalid_argument(
        "Ascend reduction common catalog must contain cases");
  }
  std::vector<ReductionTestCase> result;
  result.reserve(common_cases.size() + 1);
  for (const ReductionTestCase& test_case : common_cases) {
    validate_reduction_case(test_case);
    if (test_case.mode == mode) {
      result.push_back(test_case);
    }
  }
  if (result.empty()) {
    throw std::invalid_argument(
        "Ascend reduction common catalog does not contain the requested mode");
  }
  result.push_back(make_special_case(mode));
  return result;
}

std::vector<float> make_reduction_input(
    const ReductionTestCase& test_case) {
  validate_reduction_case(test_case);
  std::vector<float> result(tensor_io::element_count(test_case.input));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered = static_cast<int>((index * 17U) % 41U) - 20;
    const float value = static_cast<float>(centered) / 13.0F;
    result[index] = test_case.mode == FLAGDNN_REDUCTION_MUL
                        ? 1.0F + value * 0.125F
                        : value;
  }
  return tensor_io::quantize(result, test_case.input.data_type);
}

std::vector<float> reduction_host_oracle(
    const ReductionTestCase& test_case,
    std::span<const float> logical_input) {
  validate_reduction_case(test_case);
  if (logical_input.size() != tensor_io::element_count(test_case.input)) {
    throw std::invalid_argument(
        "reduction host-oracle input size is invalid");
  }
  std::int64_t axis = test_case.axis;
  if (axis < 0) {
    axis += static_cast<std::int64_t>(test_case.input.dimensions.size());
  }
  const std::size_t reduced_axis = static_cast<std::size_t>(axis);
  const std::int64_t reduction_extent =
      test_case.input.dimensions[reduced_axis];
  std::vector<float> result(tensor_io::element_count(test_case.output));
  for (std::size_t output_index = 0; output_index < result.size();
       ++output_index) {
    const std::vector<std::int64_t> output_coordinates =
        logical_coordinates(output_index, test_case.output.dimensions);
    std::vector<std::int64_t> input_coordinates(
        test_case.input.dimensions.size());
    if (test_case.keep_dimensions) {
      input_coordinates = output_coordinates;
    } else {
      std::size_t output_axis = 0;
      for (std::size_t input_axis = 0;
           input_axis < input_coordinates.size();
           ++input_axis) {
        if (input_axis == reduced_axis) {
          continue;
        }
        input_coordinates[input_axis] = output_coordinates[output_axis++];
      }
    }

    float accumulator =
        test_case.mode == FLAGDNN_REDUCTION_MUL ? 1.0F : 0.0F;
    for (std::int64_t reduction_index = 0;
         reduction_index < reduction_extent;
         ++reduction_index) {
      input_coordinates[reduced_axis] = reduction_index;
      const float value = logical_input[row_major_index(
          input_coordinates, test_case.input.dimensions)];
      if (test_case.mode == FLAGDNN_REDUCTION_MUL) {
        accumulator *= value;
      } else {
        accumulator += value;
      }
    }
    if (test_case.mode == FLAGDNN_REDUCTION_AVG) {
      accumulator /= static_cast<float>(reduction_extent);
    }
    result[output_index] = accumulator;
  }
  return tensor_io::quantize(result, test_case.output.data_type);
}

}  // namespace flagdnn::testing
