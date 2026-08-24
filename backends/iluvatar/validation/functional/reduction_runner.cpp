// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/reduction.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {

int run_reduction_functional_test(int argc, char **argv,
                                  std::span<const ReductionTestCase> cases) {
  namespace functional = iluvatar::validation::functional;
  functional::FunctionalSuite suite(argc, argv, "reduction",
                                    "FLAGDNN_REDUCTION_FUNCTIONAL");
  for (const ReductionTestCase &test_case : cases) {
    validate_reduction_case(test_case);
    const functional::InputDomain domain =
        test_case.mode == FLAGDNN_REDUCTION_MUL
            ? functional::InputDomain::kUnitScale
            : functional::InputDomain::kReal;
    suite.run(
        {.operation = "reduction",
         .case_name = test_case.name,
         .inputs = {{test_case.input, domain, {}}},
         .outputs = {{test_case.output, test_case.absolute_tolerance,
                      test_case.relative_tolerance, "output"}}},
        [&suite, &test_case] {
          return build_flagdnn_reduction(suite.handle(), test_case);
        },
        [&test_case] { return build_reduction_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
