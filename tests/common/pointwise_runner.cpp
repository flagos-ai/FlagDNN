/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/pointwise.hpp"
namespace flagdnn::testing {
int run_unary_pointwise_functional_test(
    int argc, char** argv, const PointwiseCaseDefinition& definition,
    std::string_view suite_name) {
  const auto cases = make_unary_pointwise_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

int run_binary_pointwise_functional_test(
    int argc, char** argv, const PointwiseCaseDefinition& definition,
    std::string_view suite_name) {
  const auto cases = make_binary_pointwise_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

int run_binary_select_functional_test(int argc, char** argv,
                                      const PointwiseCaseDefinition& definition,
                                      std::string_view suite_name) {
  const auto cases = make_binary_select_cases(definition);
  return run_pointwise_functional_test(argc, argv, cases, suite_name);
}

}  // namespace flagdnn::testing
