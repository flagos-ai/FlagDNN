/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"
#include "pointwise_runner_support.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <span>
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

  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  return support::run_suite(
      argc, argv, "add", "FLAGDNN_ADD_FUNCTIONAL",
      [&](const std::function<flagdnn::Handle &()> &get_handle,
          hv::Stream &stream, std::size_t &matched, std::size_t &executed,
          std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_ADD_CASE");
        for (const AddTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          validate_add_case(test_case);
          const std::array<TestTensor, 2> inputs = {test_case.left,
                                                    test_case.right};
          const std::array<PointwiseInputDomain, 2> domains = {
              PointwiseInputDomain::kReal, PointwiseInputDomain::kReal};
          const support::CaseResult result = support::run_case(
              "add", test_case.name, inputs, test_case.output, domains,
              support::fixed_operation(hv::HipdnnPointwiseKind::kAdd,
                                       test_case.alpha),
              test_case.absolute_tolerance, test_case.relative_tolerance,
              get_handle, stream,
              [&](flagdnn::Handle &value) {
                return build_flagdnn_add(value, test_case);
              },
              [&] { return build_add_reference(test_case); });
          result == support::CaseResult::kExecuted ? ++executed : ++skipped;
        }
      });
}

} // namespace flagdnn::testing
