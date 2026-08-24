// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/convolution.hpp"
#include "functional/runner_support.hpp"

#include <stdexcept>
#include <string_view>

namespace flagdnn::testing {
namespace {

std::string_view convolution_operation(ConvolutionDirection direction) {
  switch (direction) {
  case ConvolutionDirection::kFprop:
    return "conv_fprop";
  case ConvolutionDirection::kDgrad:
    return "conv_dgrad";
  case ConvolutionDirection::kWgrad:
    return "conv_wgrad";
  }
  throw std::invalid_argument("convolution direction is invalid");
}

std::string_view convolution_marker(ConvolutionDirection direction) {
  switch (direction) {
  case ConvolutionDirection::kFprop:
    return "FLAGDNN_CONV_FPROP_FUNCTIONAL";
  case ConvolutionDirection::kDgrad:
    return "FLAGDNN_CONV_DGRAD_FUNCTIONAL";
  case ConvolutionDirection::kWgrad:
    return "FLAGDNN_CONV_WGRAD_FUNCTIONAL";
  }
  throw std::invalid_argument("convolution direction is invalid");
}

} // namespace

int run_convolution_functional_test(int argc, char **argv,
                                    std::span<const ConvolutionTestCase> cases,
                                    ConvolutionDirection expected_direction) {
  namespace functional = iluvatar::validation::functional;
  const std::string operation(convolution_operation(expected_direction));
  functional::FunctionalSuite suite(
      argc, argv, operation,
      std::string(convolution_marker(expected_direction)));
  for (const ConvolutionTestCase &test_case : cases) {
    validate_convolution_case(test_case);
    if (test_case.direction != expected_direction) {
      throw std::invalid_argument("convolution case direction mismatch");
    }
    functional::CasePlan plan{
        .operation = operation,
        .case_name = test_case.name,
        .inputs = {},
        .outputs = {{convolution_output_tensor(test_case),
                     test_case.absolute_tolerance, test_case.relative_tolerance,
                     "output"}},
    };
    if (expected_direction == ConvolutionDirection::kFprop) {
      plan.inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                     {test_case.w, functional::InputDomain::kReal, {}}};
    } else if (expected_direction == ConvolutionDirection::kDgrad) {
      plan.inputs = {{test_case.y, functional::InputDomain::kReal, {}},
                     {test_case.w, functional::InputDomain::kReal, {}}};
    } else {
      plan.inputs = {{test_case.y, functional::InputDomain::kReal, {}},
                     {test_case.x, functional::InputDomain::kReal, {}}};
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_convolution(suite.handle(), test_case);
        },
        [&test_case] { return build_convolution_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
