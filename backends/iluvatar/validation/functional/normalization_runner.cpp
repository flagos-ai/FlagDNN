// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/normalization.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {
namespace {

namespace functional = iluvatar::validation::functional;

functional::PlannedOutput output(const TestTensor &tensor,
                                 double absolute_tolerance,
                                 double relative_tolerance, const char *label) {
  return {tensor, absolute_tolerance, relative_tolerance, label};
}

} // namespace

int run_layernorm_functional_test(int argc, char **argv,
                                  std::span<const LayernormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "layernorm",
                                    "FLAGDNN_LAYERNORM_FUNCTIONAL");
  for (const LayernormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "layernorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y"),
                     output(test_case.mean, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "mean"),
                     output(test_case.inv_variance,
                            test_case.absolute_tolerance,
                            test_case.relative_tolerance, "inv_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_layernorm(suite.handle(), test_case);
        },
        [&test_case] { return build_layernorm_reference(test_case); });
  }
  return suite.finish();
}

int run_rmsnorm_functional_test(int argc, char **argv,
                                std::span<const RmsnormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "rmsnorm",
                                    "FLAGDNN_RMSNORM_FUNCTIONAL");
  for (const RmsnormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "rmsnorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y"),
                     output(test_case.inv_variance,
                            test_case.absolute_tolerance,
                            test_case.relative_tolerance, "inv_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_rmsnorm(suite.handle(), test_case);
        },
        [&test_case] { return build_rmsnorm_reference(test_case); });
  }
  return suite.finish();
}

int run_batchnorm_functional_test(int argc, char **argv,
                                  std::span<const BatchnormTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "batchnorm",
                                    "FLAGDNN_BATCHNORM_FUNCTIONAL");
  for (const BatchnormTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "batchnorm",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}},
                    {test_case.previous_running_mean,
                     functional::InputDomain::kReal,
                     {}},
                    {test_case.previous_running_variance,
                     functional::InputDomain::kVariance,
                     {}}},
         .outputs =
             {output(test_case.y, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "y"),
              output(test_case.mean, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "mean"),
              output(test_case.inv_variance, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "inv_variance"),
              output(test_case.next_running_mean, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "next_running_mean"),
              output(test_case.next_running_variance,
                     test_case.absolute_tolerance, test_case.relative_tolerance,
                     "next_running_variance")}},
        [&suite, &test_case] {
          return build_flagdnn_batchnorm(suite.handle(), test_case);
        },
        [&test_case] { return build_batchnorm_reference(test_case); });
  }
  return suite.finish();
}

int run_batchnorm_inference_functional_test(
    int argc, char **argv, std::span<const BatchnormInferenceTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "batchnorm_inference",
                                    "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL");
  for (const BatchnormInferenceTestCase &test_case : cases) {
    validate_normalization_case(test_case);
    suite.run(
        {.operation = "batchnorm_inference",
         .case_name = test_case.name,
         .inputs = {{test_case.x, functional::InputDomain::kReal, {}},
                    {test_case.mean, functional::InputDomain::kReal, {}},
                    {test_case.inv_variance,
                     functional::InputDomain::kVariance,
                     {}},
                    {test_case.scale, functional::InputDomain::kUnitScale, {}},
                    {test_case.bias, functional::InputDomain::kReal, {}}},
         .outputs = {output(test_case.y, test_case.absolute_tolerance,
                            test_case.relative_tolerance, "y")}},
        [&suite, &test_case] {
          return build_flagdnn_batchnorm_inference(suite.handle(), test_case);
        },
        [&test_case] {
          return build_batchnorm_inference_reference(test_case);
        });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
