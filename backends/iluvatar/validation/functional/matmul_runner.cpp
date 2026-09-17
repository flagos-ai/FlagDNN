// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/matmul.hpp"
#include "functional/capability_skips.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/matrix.hpp"

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
        [&test_case] { return build_matmul_reference(test_case); },
        [&test_case](const auto &inputs) {
          return std::vector<std::vector<float>>{
              reference::cpu::evaluate_matmul(
                  {test_case.a.dimensions, test_case.b.dimensions,
                   test_case.output.dimensions, test_case.a.data_type,
                   // Iluvatar defaults to IEEE; the shared CPU helper defaults
                   // to NVIDIA TF32 for some aligned FP32 shapes.
                   test_case.input_precision == 0 ? 1
                                                  : test_case.input_precision},
                  inputs[0], inputs[1])};
        });
  }
  iluvatar::validation::emit_plain_fp8_matmul_skips();
  return suite.finish();
}

} // namespace flagdnn::testing
