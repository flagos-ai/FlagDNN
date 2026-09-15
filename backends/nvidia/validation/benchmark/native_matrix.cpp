/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <numeric>

#include "common/convolution.hpp"
#include "common/matmul.hpp"
#include "common/reduction.hpp"
#include "validation/functional/cudnn_precision.hpp"
#include "validation/functional/cudnn_reduction.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
namespace {
template <class Specification>
struct Case {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  Specification specification;
};
std::vector<std::vector<float>> matrix_inputs(
    const std::vector<TestTensor>& tensors) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < tensors.size(); ++input) {
    std::vector<float> values(cuda::element_count(tensors[input]));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] =
          static_cast<float>(static_cast<int>((i * 17 + input * 7) % 61) - 30) /
          19.0F;
    result.push_back(std::move(values));
  }
  return result;
}
}  // namespace
int run_native_precision_benchmark(int argc, char** argv,
                                   std::string_view operation) {
  if (operation == "matmul") {
    std::vector<Case<MatmulTestCase>> cases;
    for (const auto& value : make_matmul_cases())
      if (value.input_precision &&
          value.input_precision == cuda::selected_input_precision() &&
          value.output.dimensions.size() <= 3)
        cases.push_back(
            {value.name, {value.a, value.b}, {value.output}, value});
    return cuda::run_paired_cases(
        argc, argv, std::span<const Case<MatmulTestCase>>(cases),
        "FLAGDNN_NATIVE_CASE",
        [](flagdnn::Handle& handle, const auto& value) {
          return build_flagdnn_matmul(handle, value.specification);
        },
        [](const auto& value) { return matrix_inputs(value.inputs); },
        [](const auto& value) {
          return build_matmul_reference(value.specification);
        },
        [](const auto& value, std::size_t) {
          return cuda::PairedTolerance{value.specification.absolute_tolerance,
                                       value.specification.relative_tolerance};
        },
        true);
  }
  const auto direction =
      operation == "conv_fprop"   ? ConvolutionDirection::kFprop
      : operation == "conv_dgrad" ? ConvolutionDirection::kDgrad
                                  : ConvolutionDirection::kWgrad;
  std::vector<Case<ConvolutionTestCase>> cases;
  for (const auto& value : make_convolution_cases(direction)) {
    if (!value.input_precision ||
        value.input_precision != cuda::selected_input_precision())
      continue;
    const auto inputs = direction == ConvolutionDirection::kFprop
                            ? std::vector{value.x, value.w}
                        : direction == ConvolutionDirection::kDgrad
                            ? std::vector{value.y, value.w}
                            : std::vector{value.y, value.x};
    cases.push_back(
        {value.name, inputs, {convolution_output_tensor(value)}, value});
  }
  return cuda::run_paired_cases(
      argc, argv, std::span<const Case<ConvolutionTestCase>>(cases),
      "FLAGDNN_NATIVE_CASE",
      [](flagdnn::Handle& handle, const auto& value) {
        return build_flagdnn_convolution(handle, value.specification);
      },
      [](const auto& value) { return matrix_inputs(value.inputs); },
      [](const auto& value) {
        return build_convolution_reference(value.specification);
      },
      [](const auto& value, std::size_t) {
        return cuda::PairedTolerance{value.specification.absolute_tolerance,
                                     value.specification.relative_tolerance};
      },
      true);
}
int run_native_reduction_benchmark(int argc, char** argv) {
  std::vector<Case<ReductionTestCase>> cases;
  for (const auto& value : make_reduction_cases())
    if (cuda::cudnn_reduction_case(value) &&
        value.name.find("_to_fp32_") != std::string::npos)
      cases.push_back({value.name, {value.input}, {value.output}, value});
  return cuda::run_paired_cases(
      argc, argv, std::span<const Case<ReductionTestCase>>(cases),
      "FLAGDNN_NATIVE_CASE",
      [](flagdnn::Handle& handle, const auto& value) {
        return build_flagdnn_reduction(handle, value.specification);
      },
      [](const auto& value) {
        return std::vector<std::vector<float>>{
            reduction_host_input(value.specification)};
      },
      [](const auto& value) {
        return build_reduction_reference(value.specification);
      },
      [](const auto& value, std::size_t) {
        return cuda::PairedTolerance{value.specification.absolute_tolerance,
                                     value.specification.relative_tolerance};
      },
      true);
}
}  // namespace flagdnn::testing
