// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/layout.hpp"
#include "functional/runner_support.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace flagdnn::testing {
namespace {

std::string layout_operation(std::string_view marker) {
  constexpr std::string_view prefix = "FLAGDNN_";
  constexpr std::string_view suffix = "_FUNCTIONAL";
  if (!marker.starts_with(prefix) || !marker.ends_with(suffix)) {
    throw std::invalid_argument("layout suite marker is invalid");
  }
  std::string result(marker.substr(
      prefix.size(), marker.size() - prefix.size() - suffix.size()));
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char value) {
                   return static_cast<char>(std::tolower(value));
                 });
  return result;
}

} // namespace

int run_layout_functional_test(int argc, char **argv,
                               std::span<const LayoutTestCase> cases,
                               std::string_view suite_name) {
  namespace functional = iluvatar::validation::functional;
  const std::string operation = layout_operation(suite_name);
  functional::FunctionalSuite suite(argc, argv, operation,
                                    std::string(suite_name));
  for (const LayoutTestCase &test_case : cases) {
    validate_layout_case(test_case);
    suite.run(
        {.operation = operation,
         .case_name = test_case.name,
         .inputs = {{test_case.input, functional::InputDomain::kReal, {}}},
         .outputs = {{test_case.output, 0.0, 0.0, "output"}}},
        [&suite, &test_case] {
          return build_flagdnn_layout(suite.handle(), test_case);
        },
        [&test_case] { return build_layout_reference(test_case); });
  }
  return suite.finish();
}

} // namespace flagdnn::testing
