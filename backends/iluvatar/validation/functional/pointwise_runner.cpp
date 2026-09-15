// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/pointwise.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::testing {
namespace {

std::string operation_from_marker(std::string_view marker) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_FUNCTIONAL";
  if (!marker.starts_with(prefix) || !marker.ends_with(suffix) ||
      marker.size() <= prefix.size() + suffix.size()) {
    throw std::invalid_argument("pointwise suite marker is invalid");
  }
  std::string result(marker.substr(
      prefix.size(), marker.size() - prefix.size() - suffix.size()));
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return result;
}

} // namespace

int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name) {
  // This adapter currently validates floating storage (and logical BOOL).
  // Other shared dtype/output combinations are enabled with their backend
  // support.
  std::vector<PointwiseTestCase> supported_cases(cases.begin(), cases.end());
  std::erase_if(supported_cases, [](const PointwiseTestCase& test_case) {
    const bool logical = test_case.mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
                         test_case.mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
                         test_case.mode == FLAGDNN_POINTWISE_BINARY_SELECT;
    return std::any_of(test_case.inputs.begin(), test_case.inputs.end(),
                       [logical](const TestTensor& tensor) {
                         const auto type = tensor.data_type;
                         return type != FLAGDNN_DATA_FLOAT32 &&
                                type != FLAGDNN_DATA_FLOAT16 &&
                                type != FLAGDNN_DATA_BFLOAT16 &&
                                !(logical && type == FLAGDNN_DATA_BOOLEAN);
                       });
  });
  cases = supported_cases;

  namespace functional = iluvatar::validation::functional;
  const std::string operation = operation_from_marker(suite_name);
  functional::FunctionalSuite suite(argc, argv, operation,
                                    std::string(suite_name));
  for (const PointwiseTestCase &test_case : cases) {
    validate_pointwise_case(test_case);
    functional::CasePlan plan{
        .operation = operation,
        .case_name = test_case.name,
        .inputs = {},
        .outputs = {{test_case.output, test_case.absolute_tolerance,
                     test_case.relative_tolerance, "output"}},
    };
    for (std::size_t index = 0; index < test_case.inputs.size(); ++index) {
      plan.inputs.push_back(
          {test_case.inputs[index],
           functional::pointwise_input_domain(
               static_cast<int>(test_case.input_domains[index])),
           {}});
    }
    suite.run(
        plan,
        [&suite, &test_case] {
          return build_flagdnn_pointwise(suite.handle(), test_case);
        },
        [&test_case] { return build_pointwise_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
