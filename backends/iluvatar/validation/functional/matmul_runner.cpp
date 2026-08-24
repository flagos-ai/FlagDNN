// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/matmul.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {

int run_matmul_functional_test(int argc, char **argv,
                               std::span<const MatmulTestCase> cases) {
  namespace functional = iluvatar::validation::functional;
  functional::FunctionalSuite suite(argc, argv, "matmul",
                                    "FLAGDNN_MATMUL_FUNCTIONAL");
  for (const MatmulTestCase &test_case : cases) {
    validate_matmul_case(test_case);
    suite.run(
        {.operation = "matmul",
         .case_name = test_case.name,
         .inputs = {{test_case.a, functional::InputDomain::kReal, {}},
                    {test_case.b, functional::InputDomain::kReal, {}}},
         .outputs = {{test_case.output, test_case.absolute_tolerance,
                      test_case.relative_tolerance, "output"}}},
        [&suite, &test_case] {
          return build_flagdnn_matmul(suite.handle(), test_case);
        },
        [&test_case] { return build_matmul_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
