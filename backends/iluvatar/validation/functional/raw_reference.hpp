// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "common/pointwise.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/pointwise.hpp"
#include <cstring>
namespace flagdnn::iluvatar::validation::functional {
using Bytes = std::vector<std::uint8_t>;
inline Bytes raw_pattern(const testing::TestTensor &tensor,
                         std::size_t salt = 0) {
  Bytes result(element_count(tensor) * data_type_size(tensor.data_type));
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<std::uint8_t>(i * 37 + salt * 83);
  return result;
}
inline std::size_t broadcast_index(std::size_t logical,
                                   const testing::TestTensor &input,
                                   const testing::TestTensor &output) {
  std::size_t result = 0, stride = 1;
  const auto leading = output.dimensions.size() - input.dimensions.size();
  for (std::size_t a = output.dimensions.size(); a > 0; --a) {
    const auto coordinate = logical % output.dimensions[a - 1];
    logical /= output.dimensions[a - 1];
    if (a - 1 < leading)
      continue;
    const auto extent = input.dimensions[a - 1 - leading];
    if (extent != 1)
      result += coordinate * stride;
    stride *= extent;
  }
  return result;
}
inline void run_integer_case(FunctionalSuite &suite, const CasePlan &plan,
                             flagdnnPointwiseMode_t mode, std::int32_t alpha,
                             const BuildExecutable &build,
                             bool square = false) {
  std::vector<Bytes> inputs;
  for (std::size_t i = 0; i < plan.inputs.size(); ++i) {
    const auto &tensor = plan.inputs[i].tensor;
    Bytes values(element_count(tensor) * data_type_size(tensor.data_type));
    for (std::size_t j = 0; j < element_count(tensor); ++j) {
      if (tensor.data_type == FLAGDNN_DATA_BOOLEAN)
        values[j] = j % 2;
      else {
        const auto value = testing::pointwise_integer_input(j, i, mode);
        std::memcpy(values.data() + j * 4, &value, 4);
      }
    }
    inputs.push_back(std::move(values));
  }
  const auto &output = plan.outputs.front().tensor;
  const auto width = data_type_size(output.data_type);
  Bytes expected(element_count(output) * width);
  for (std::size_t j = 0; j < element_count(output); ++j) {
    std::int32_t operands[2] = {};
    for (std::size_t i = 0; i < 2; ++i) {
      const auto offset = broadcast_index(j, plan.inputs[i].tensor, output);
      std::memcpy(&operands[i], inputs[i].data() + offset * 4, 4);
    }
    const bool predicate =
        inputs.size() == 3 &&
        inputs[2][broadcast_index(j, plan.inputs[2].tensor, output)] != 0;
    // The public composite is left + right * right, with INT32 wrap at
    // the virtual tensor between the multiplication and the addition.
    if (square)
      operands[1] = reference::cpu::pointwise_integer_reference(
          FLAGDNN_POINTWISE_MUL, operands[1], operands[1], false, 1);
    const auto value = reference::cpu::pointwise_integer_reference(
        mode, operands[0], operands[1], predicate, alpha);
    if (width == 1)
      expected[j] = value != 0;
    else
      std::memcpy(expected.data() + j * 4, &value, 4);
  }
  suite.run_raw(plan, build, inputs, {expected});
}
} // namespace flagdnn::iluvatar::validation::functional
