// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/add.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {

int run_add_functional_test(int argc, char **argv,
                            std::span<const AddTestCase> cases) {
  namespace functional = iluvatar::validation::functional;
  functional::FunctionalSuite suite(argc, argv, "add",
                                    "FLAGDNN_ADD_FUNCTIONAL");
  for (const AddTestCase &test_case : cases) {
    validate_add_case(test_case);
    suite.run(
        {.operation = "add",
         .case_name = test_case.name,
         .inputs = {{test_case.left, functional::InputDomain::kReal, {}},
                    {test_case.right, functional::InputDomain::kReal, {}}},
         .outputs = {{test_case.output, test_case.absolute_tolerance,
                      test_case.relative_tolerance, "output"}}},
        [&suite, &test_case] {
          return build_flagdnn_add(suite.handle(), test_case);
        },
        [&test_case] { return build_add_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
