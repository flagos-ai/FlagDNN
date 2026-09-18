// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/convolution.hpp"
#include "common/dtype_runner.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"
#include "functional/paired.hpp"
namespace flagdnn::testing {
namespace {
namespace f = flagdnn::validation::thead::functional;
template <class Specification>
struct Case {
  std::string name;
  std::vector<TestTensor> inputs, outputs, reference_outputs;
  Specification specification;
};
template <class T>
Case<T> wrapped(const T& t, std::vector<TestTensor> inputs,
                std::vector<TestTensor> outputs) {
  return {t.name, std::move(inputs), outputs, outputs, t};
}
template <class T>
std::vector<std::vector<float>> values(const Case<T>& t) {
  std::vector<std::vector<float>> result;
  for (std::size_t input = 0; input < t.inputs.size(); ++input) {
    std::vector<float> data(f::element_count(t.inputs[input]));
    for (std::size_t i = 0; i < data.size(); ++i)
      data[i] =
          static_cast<float>(static_cast<int>((i * 17 + input * 7) % 61) - 30) /
          19.0F;
    result.push_back(std::move(data));
  }
  return result;
}
template <class T, class Build, class Reference, class Input>
int run(int argc, char** argv, const std::vector<Case<T>>& cases,
        std::string_view operation, std::string_view category, Build build,
        Reference reference, Input input) {
  return f::run_paired_cases(
      argc, argv, std::span<const Case<T>>(cases), std::string(operation),
      [build](flagdnn::Handle& h, const Case<T>& t) {
        return build(h, t.specification);
      },
      input, true,
      [reference](const Case<T>& t) { return reference(t.specification); },
      std::string(category));
}
int precision() {
  const char* value = std::getenv("FLAGDNN_INPUT_PRECISION");
  if (!value ||
      (std::string_view(value) != "1" && std::string_view(value) != "2"))
    throw std::invalid_argument(
        "dtype benchmark requires explicit IEEE or TF32 precision");
  return std::atoi(value);
}
}  // namespace
int run_native_dtype_benchmark(int argc, char** argv,
                               std::string_view operation,
                               std::string_view category) {
  if (category == "precision") {
    const int selected = precision();
    const auto label = selected == 1 ? "ieee" : "tf32";
    if (operation == "matmul") {
      std::vector<Case<MatmulTestCase>> cases;
      for (const auto& t : make_matmul_cases())
        if (t.input_precision == selected && t.output.dimensions.size() <= 3)
          cases.push_back(wrapped(t, {t.a, t.b}, {t.output}));
      return run(argc, argv, cases, operation, label, build_flagdnn_matmul,
                 build_matmul_reference, values<MatmulTestCase>);
    }
    const auto direction =
        operation == "conv_fprop"   ? ConvolutionDirection::kFprop
        : operation == "conv_dgrad" ? ConvolutionDirection::kDgrad
                                    : ConvolutionDirection::kWgrad;
    std::vector<Case<ConvolutionTestCase>> cases;
    for (const auto& t : make_convolution_cases(direction))
      if (t.input_precision == selected) {
        const auto inputs =
            direction == ConvolutionDirection::kFprop   ? std::vector{t.x, t.w}
            : direction == ConvolutionDirection::kDgrad ? std::vector{t.y, t.w}
                                                        : std::vector{t.y, t.x};
        cases.push_back(wrapped(t, inputs, {convolution_output_tensor(t)}));
      }
    return run(argc, argv, cases, operation, label, build_flagdnn_convolution,
               build_convolution_reference, values<ConvolutionTestCase>);
  }
  if (category == "fp32_output") {
    std::vector<Case<ReductionTestCase>> cases;
    for (const auto& t : make_reduction_cases())
      if (t.name.find("_to_fp32_") != std::string::npos)
        cases.push_back(wrapped(t, {t.input}, {t.output}));
    return run(argc, argv, cases, operation, category, build_flagdnn_reduction,
               build_reduction_reference, [](const Case<ReductionTestCase>& t) {
                 return std::vector<std::vector<float>>{
                     reduction_host_input(t.specification)};
               });
  }
  if (category == "copy" && operation != "identity") {
    const auto mode = operation == "reshape"     ? LayoutOperation::kReshape
                      : operation == "transpose" ? LayoutOperation::kTranspose
                                                 : LayoutOperation::kSlice;
    std::vector<Case<LayoutTestCase>> cases;
    for (const auto& t : make_layout_cases(mode)) {
      auto value = wrapped(t, {t.input}, {t.output});
      cases.push_back(std::move(value));
    }
    return run(argc, argv, cases, operation, category, build_flagdnn_layout,
               flagdnn::validation::thead::make_acdnn_raw_layout_reference,
               values<LayoutTestCase>);
  }
  if (category == "boolean" ||
      (category == "copy" && operation == "identity")) {
    PointwiseCaseDefinition definition;
    definition.operation_name = operation;
    definition.mode =
        operation == "identity"      ? FLAGDNN_POINTWISE_IDENTITY
        : operation == "logical_not" ? FLAGDNN_POINTWISE_LOGICAL_NOT
        : operation == "logical_and" ? FLAGDNN_POINTWISE_LOGICAL_AND
                                     : FLAGDNN_POINTWISE_LOGICAL_OR;
    definition.input_domain = category == "boolean"
                                  ? PointwiseInputDomain::kLogical
                                  : PointwiseInputDomain::kReal;
    const auto shared = operation == "identity" || operation == "logical_not"
                            ? make_unary_pointwise_cases(definition)
                            : make_binary_pointwise_cases(definition);
    std::vector<Case<PointwiseTestCase>> cases;
    for (const auto& t : shared)
      cases.push_back(wrapped(t, t.inputs, {t.output}));
    return run(
        argc, argv, cases, operation, category, build_flagdnn_pointwise,
        [](const PointwiseTestCase& test) -> std::unique_ptr<TestExecutable> {
          if (test.mode == FLAGDNN_POINTWISE_IDENTITY)
            return std::make_unique<flagdnn::validation::thead::AcdnnRawCopy>(
                test.inputs.at(0), test.output);
          return build_pointwise_reference(test);
        },
        values<PointwiseTestCase>);
  }
  throw std::invalid_argument("unknown THead dtype benchmark category");
}
}  // namespace flagdnn::testing
