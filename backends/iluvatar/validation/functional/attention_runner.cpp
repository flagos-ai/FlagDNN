// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/attention.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {
namespace {

namespace functional = iluvatar::validation::functional;

functional::PlannedInput
input(const TestTensor &tensor,
      functional::InputDomain domain = functional::InputDomain::kReal) {
  return {tensor, domain, {}};
}

functional::PlannedInput scalar(const Fp8Scalar &value) {
  return {value.tensor, functional::InputDomain::kReal, {value.value}};
}

functional::PlannedOutput output(const TestTensor &tensor,
                                 double absolute_tolerance,
                                 double relative_tolerance, const char *label) {
  return {tensor, absolute_tolerance, relative_tolerance, label};
}

} // namespace

int run_sdpa_functional_test(int argc, char **argv,
                             std::span<const SdpaTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "sdpa",
                                    "FLAGDNN_SDPA_FUNCTIONAL");
  for (const SdpaTestCase &test_case : cases) {
    validate_sdpa_case(test_case);
    functional::CasePlan plan{
        .operation = "sdpa",
        .case_name = test_case.name,
        .inputs = {input(test_case.q), input(test_case.k), input(test_case.v)},
        .outputs = {output(test_case.output,
                           test_case.output_absolute_tolerance,
                           test_case.output_relative_tolerance, "output")},
    };
    if (test_case.bias.has_value()) {
      plan.inputs.push_back(input(*test_case.bias));
    }
    if (test_case.stats.has_value()) {
      plan.outputs.push_back(
          output(*test_case.stats, test_case.stats_absolute_tolerance,
                 test_case.stats_relative_tolerance, "stats"));
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_sdpa(suite.handle(), test_case);
        },
        [&test_case] { return build_sdpa_reference(test_case); });
  }
  return suite.finish();
}

int run_sdpa_backward_functional_test(
    int argc, char **argv, std::span<const SdpaBackwardTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "sdpa_backward",
                                    "FLAGDNN_SDPA_BACKWARD_FUNCTIONAL");
  for (const SdpaBackwardTestCase &test_case : cases) {
    validate_sdpa_backward_case(test_case);
    functional::CasePlan plan{
        .operation = "sdpa_backward",
        .case_name = test_case.name,
        .inputs = {input(test_case.q), input(test_case.k), input(test_case.v),
                   input(test_case.output), input(test_case.doutput),
                   input(test_case.stats, functional::InputDomain::kStats)},
        .outputs = {output(test_case.dq, test_case.absolute_tolerance,
                           test_case.relative_tolerance, "dq"),
                    output(test_case.dk, test_case.absolute_tolerance,
                           test_case.relative_tolerance, "dk"),
                    output(test_case.dv, test_case.absolute_tolerance,
                           test_case.relative_tolerance, "dv")},
    };
    if (test_case.bias.has_value()) {
      plan.inputs.push_back(input(*test_case.bias));
    }
    if (test_case.dbias.has_value()) {
      plan.outputs.push_back(output(*test_case.dbias,
                                    test_case.absolute_tolerance,
                                    test_case.relative_tolerance, "dbias"));
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_sdpa_backward(suite.handle(), test_case);
        },
        [&test_case] { return build_sdpa_backward_reference(test_case); });
  }
  return suite.finish();
}

int run_sdpa_fp8_functional_test(int argc, char **argv,
                                 std::span<const SdpaFp8TestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "sdpa_fp8",
                                    "FLAGDNN_SDPA_FP8_FUNCTIONAL");
  for (const SdpaFp8TestCase &test_case : cases) {
    validate_sdpa_fp8_case(test_case);
    functional::CasePlan plan{
        .operation = "sdpa_fp8",
        .case_name = test_case.name,
        .inputs = {input(test_case.q), input(test_case.k), input(test_case.v),
                   scalar(test_case.descale_q), scalar(test_case.descale_k),
                   scalar(test_case.descale_v), scalar(test_case.descale_s),
                   scalar(test_case.scale_s), scalar(test_case.scale_o)},
        .outputs = {output(test_case.output,
                           test_case.output_absolute_tolerance,
                           test_case.output_relative_tolerance, "output"),
                    output(test_case.amax_s, test_case.amax_absolute_tolerance,
                           test_case.amax_relative_tolerance, "amax_s"),
                    output(test_case.amax_o, test_case.amax_absolute_tolerance,
                           test_case.amax_relative_tolerance, "amax_o")},
    };
    if (test_case.bias.has_value()) {
      plan.inputs.push_back(input(*test_case.bias));
    }
    if (test_case.stats.has_value()) {
      plan.outputs.push_back(
          output(*test_case.stats, test_case.stats_absolute_tolerance,
                 test_case.stats_relative_tolerance, "stats"));
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_sdpa_fp8(suite.handle(), test_case);
        },
        [&test_case] { return build_sdpa_fp8_reference(test_case); });
  }
  return suite.finish();
}

int run_sdpa_fp8_backward_functional_test(
    int argc, char **argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  functional::FunctionalSuite suite(argc, argv, "sdpa_fp8_backward",
                                    "FLAGDNN_SDPA_FP8_BACKWARD_FUNCTIONAL");
  for (const SdpaFp8BackwardTestCase &test_case : cases) {
    validate_sdpa_fp8_backward_case(test_case);
    const std::vector<functional::PlannedInput> inputs = {
        input(test_case.q),
        input(test_case.k),
        input(test_case.v),
        input(test_case.output),
        input(test_case.doutput),
        input(test_case.stats, functional::InputDomain::kStats),
        scalar(test_case.descale_q),
        scalar(test_case.descale_k),
        scalar(test_case.descale_v),
        scalar(test_case.descale_o),
        scalar(test_case.descale_doutput),
        scalar(test_case.descale_s),
        scalar(test_case.descale_dp),
        scalar(test_case.scale_s),
        scalar(test_case.scale_dq),
        scalar(test_case.scale_dk),
        scalar(test_case.scale_dv),
        scalar(test_case.scale_dp)};
    const std::vector<functional::PlannedOutput> outputs = {
        output(test_case.dq, test_case.gradient_absolute_tolerance,
               test_case.gradient_relative_tolerance, "dq"),
        output(test_case.dk, test_case.gradient_absolute_tolerance,
               test_case.gradient_relative_tolerance, "dk"),
        output(test_case.dv, test_case.gradient_absolute_tolerance,
               test_case.gradient_relative_tolerance, "dv"),
        output(test_case.amax_dq, test_case.amax_absolute_tolerance,
               test_case.amax_relative_tolerance, "amax_dq"),
        output(test_case.amax_dk, test_case.amax_absolute_tolerance,
               test_case.amax_relative_tolerance, "amax_dk"),
        output(test_case.amax_dv, test_case.amax_absolute_tolerance,
               test_case.amax_relative_tolerance, "amax_dv"),
        output(test_case.amax_dp, test_case.amax_absolute_tolerance,
               test_case.amax_relative_tolerance, "amax_dp")};
    suite.run(
        {.operation = "sdpa_fp8_backward",
         .case_name = test_case.name,
         .inputs = inputs,
         .outputs = outputs},
        [&suite, &test_case] {
          return build_flagdnn_sdpa_fp8_backward(suite.handle(), test_case);
        },
        [&test_case] { return build_sdpa_fp8_backward_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
