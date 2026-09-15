// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/add.hpp"
#include "functional/runner_support.hpp"

#include <algorithm>
#include <vector>

namespace flagdnn::testing {

int run_add_functional_test(int argc, char **argv,
                            std::span<const AddTestCase> cases) {
  // This backend's reference adapter currently accepts floating storage.
  std::vector<AddTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const AddTestCase &test_case) {
    const auto type = test_case.left.data_type;
    return type != FLAGDNN_DATA_FLOAT32 && type != FLAGDNN_DATA_FLOAT16 &&
           type != FLAGDNN_DATA_BFLOAT16;
  });
  cases = supported_cases;

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
