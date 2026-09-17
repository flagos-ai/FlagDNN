// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/composite.hpp"
#include "functional/raw_reference.hpp"
#include "functional/runner_support.hpp"

#include <algorithm>
#include <vector>

namespace flagdnn::testing {

int run_add_square_functional_test(int argc, char **argv,
                                   std::span<const AddSquareTestCase> cases) {
  namespace functional = iluvatar::validation::functional;
  functional::FunctionalSuite suite(argc, argv, "add_square",
                                    "FLAGDNN_ADD_SQUARE_FUNCTIONAL");
  for (const AddSquareTestCase &test_case : cases) {
    validate_composite_case(test_case);
    functional::CasePlan plan{
        .operation = "add_square",
        .case_name = test_case.name,
        .inputs = {{test_case.left, functional::InputDomain::kReal, {}},
                   {test_case.right, functional::InputDomain::kReal, {}}},
        .outputs = {{test_case.output, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "output"}}};
    if (test_case.left.data_type == FLAGDNN_DATA_INT32) {
      functional::run_integer_case(
          suite, plan, FLAGDNN_POINTWISE_ADD, 1,
          [&] { return build_flagdnn_add_square(suite.handle(), test_case); },
          true);
      continue;
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_add_square(suite.handle(), test_case);
        },
        [&test_case] { return build_add_square_reference(test_case); });
  }
  return suite.finish();
}

int run_conv_bias_relu_functional_test(
    int argc, char **argv, std::span<const ConvBiasReluTestCase> cases) {
  namespace functional = iluvatar::validation::functional;
  functional::FunctionalSuite suite(argc, argv, "conv_bias_relu",
                                    "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL");
  for (const ConvBiasReluTestCase &test_case : cases) {
    validate_composite_case(test_case);
    suite.run(
        {.operation = "conv_bias_relu",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.w, functional::InputDomain::kReal, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {{test_case.output, test_case.absolute_tolerance,
                      test_case.relative_tolerance, "output"}}},
        [&suite, &test_case] {
          return build_flagdnn_conv_bias_relu(suite.handle(), test_case);
        },
        [&test_case] { return build_conv_bias_relu_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
