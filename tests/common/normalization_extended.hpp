/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_NORMALIZATION_EXTENDED_HPP_
#define FLAGDNN_TESTS_COMMON_NORMALIZATION_EXTENDED_HPP_
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/common.hpp"
namespace flagdnn::testing {
struct ExtendedNormalizationTestCase {
  std::string name, operation;
  std::vector<TestTensor> inputs, outputs;
  std::vector<std::size_t> axes;
  double epsilon = 1.0e-5;
};
std::vector<ExtendedNormalizationTestCase> make_extended_normalization_cases(
    const std::string& operation);
std::unique_ptr<TestExecutable> build_flagdnn_extended_normalization(
    flagdnn::Handle& handle, const ExtendedNormalizationTestCase& test_case);
int run_extended_normalization_functional_test(
    int argc, char** argv, std::span<const ExtendedNormalizationTestCase> cases,
    bool benchmark = false);
}  // namespace flagdnn::testing
#endif
