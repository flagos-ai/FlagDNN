/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"
#include "pointwise_runner_support.hpp"

#include <cstdlib>
#include <functional>
#include <span>
#include <string>
#include <string_view>

namespace flagdnn::testing {

int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name) {
  namespace support = hygon_functional::pointwise;
  namespace hv = validation::hygon;
  const std::string family =
      cases.empty() ? std::string("pointwise")
                    : support::case_operation_name(cases.front().name);
  return support::run_suite(
      argc, argv, family, suite_name,
      [&](const std::function<flagdnn::Handle &()> &get_handle,
          hv::Stream &stream, std::size_t &matched, std::size_t &executed,
          std::size_t &skipped) {
        const char *filter = std::getenv("FLAGDNN_POINTWISE_CASE");
        for (const PointwiseTestCase &test_case : cases) {
          if (filter != nullptr &&
              test_case.name.find(filter) == std::string::npos) {
            continue;
          }
          ++matched;
          validate_pointwise_case(test_case);
          const std::string operation =
              support::case_operation_name(test_case.name);
          support::CaseResult result;
          if (support::uses_hygon_cpu_reference(test_case.mode)) {
            result = support::run_cpu_case(
                operation, test_case.name, test_case.inputs,
                test_case.output, test_case.input_domains, test_case.mode,
                test_case.absolute_tolerance, test_case.relative_tolerance,
                get_handle, stream, [&](flagdnn::Handle &value) {
                  return build_flagdnn_pointwise(value, test_case);
                });
          } else {
            result = support::run_case(
                operation, test_case.name, test_case.inputs,
                test_case.output, test_case.input_domains,
                hv::make_hipdnn_pointwise_operation(
                    test_case.mode, test_case.attributes, test_case.alpha),
                test_case.absolute_tolerance, test_case.relative_tolerance,
                get_handle, stream,
                [&](flagdnn::Handle &value) {
                  return build_flagdnn_pointwise(value, test_case);
                },
                [&] { return build_pointwise_reference(test_case); });
          }
          result == support::CaseResult::kExecuted ? ++executed : ++skipped;
        }
      });
}

} // namespace flagdnn::testing
