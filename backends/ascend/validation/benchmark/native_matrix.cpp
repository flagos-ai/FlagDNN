/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <numeric>

#include "common/convolution.hpp"
#include "common/matmul.hpp"
#include "common/reduction.hpp"
#include "validation/functional/aclnn_extended.hpp"

#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
namespace {
template <class Specification> struct Case {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  Specification specification;
};
std::vector<std::vector<float>>
matrix_inputs(const std::vector<TestTensor> &tensors) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < tensors.size(); ++input) {
    std::vector<float> values(ascend::io::element_count(tensors[input]));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] =
          static_cast<float>(static_cast<int>((i * 17 + input * 7) % 61) - 30) /
          19.0F;
    result.push_back(std::move(values));
  }
  return result;
}
} // namespace
int run_native_precision_benchmark(int argc, char **argv,
                                   std::string_view operation) {
  if (std::getenv("FLAGDNN_INPUT_PRECISION") &&
      std::atoi(std::getenv("FLAGDNN_INPUT_PRECISION")) == 2) {
    std::cout << "CAPABILITY_UNAVAILABLE: Ascend 910B ACLNN exposes HF32, not "
                 "the CUDA TF32 input-precision contract\n";
    return 77;
  }
  if (operation == "matmul") {
    std::vector<Case<MatmulTestCase>> cases;
    for (const auto &value : make_matmul_cases())
      if (value.input_precision &&
          value.input_precision ==
              (std::getenv("FLAGDNN_INPUT_PRECISION")
                   ? std::atoi(std::getenv("FLAGDNN_INPUT_PRECISION"))
                   : 1) &&
          value.output.dimensions.size() <= 3)
        cases.push_back(
            {value.name, {value.a, value.b}, {value.output}, value});
    return ascend::run_paired_cases(
        argc, argv, std::span<const Case<MatmulTestCase>>(cases),
        "FLAGDNN_NATIVE_CASE",
        [](flagdnn::Handle &handle, const auto &value) {
          return build_flagdnn_matmul(handle, value.specification);
        },
        [](const auto &value) { return matrix_inputs(value.inputs); },
        [](const auto &value) {
          auto c = value.specification;
          c.output = value.outputs[0];
          return build_matmul_reference(c);
        },
        [](const auto &value, std::size_t) {
          return ascend::PairedTolerance{
              value.specification.absolute_tolerance,
              value.specification.relative_tolerance};
        },
        true);
  }
  const auto direction =
      operation == "conv_fprop"   ? ConvolutionDirection::kFprop
      : operation == "conv_dgrad" ? ConvolutionDirection::kDgrad
                                  : ConvolutionDirection::kWgrad;
  std::vector<Case<ConvolutionTestCase>> cases;
  for (const auto &value : make_convolution_cases(direction)) {
    if (!value.input_precision ||
        value.input_precision !=
            (std::getenv("FLAGDNN_INPUT_PRECISION")
                 ? std::atoi(std::getenv("FLAGDNN_INPUT_PRECISION"))
                 : 1))
      continue;
    const auto inputs = direction == ConvolutionDirection::kFprop
                            ? std::vector{value.x, value.w}
                        : direction == ConvolutionDirection::kDgrad
                            ? std::vector{value.y, value.w}
                            : std::vector{value.y, value.x};
    cases.push_back(
        {value.name, inputs, {convolution_output_tensor(value)}, value});
  }
  return ascend::run_paired_cases(
      argc, argv, std::span<const Case<ConvolutionTestCase>>(cases),
      "FLAGDNN_NATIVE_CASE",
      [](flagdnn::Handle &handle, const auto &value) {
        return build_flagdnn_convolution(handle, value.specification);
      },
      [](const auto &value) { return matrix_inputs(value.inputs); },
      [](const auto &value) {
        auto c = value.specification;
        if (c.direction == ConvolutionDirection::kFprop) {
          c.y = value.outputs[0];
          return build_convolution_reference(c);
        }
        if (c.direction == ConvolutionDirection::kDgrad)
          c.x = value.outputs[0];
        else
          c.w = value.outputs[0];
        return build_aclnn_convolution_backward(c);
      },
      [](const auto &value, std::size_t) {
        return ascend::PairedTolerance{value.specification.absolute_tolerance,
                                       value.specification.relative_tolerance};
      },
      true);
}
} // namespace flagdnn::testing
