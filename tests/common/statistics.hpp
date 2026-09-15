/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_STATISTICS_HPP_
#define FLAGDNN_TESTS_COMMON_STATISTICS_HPP_
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/common.hpp"
namespace flagdnn::testing {
struct StatisticsTestCase {
  std::string name;
  std::string operation = "genstats";
  std::vector<TestTensor> inputs;
  std::vector<TestTensor> outputs;
  double epsilon = 1.0e-5, accum_count = 17.0, momentum = 0.1;
};
[[nodiscard]] std::vector<StatisticsTestCase> make_genstats_cases();
[[nodiscard]] std::vector<StatisticsTestCase> make_bn_finalize_cases();
float statistics_input_value(const StatisticsTestCase& test_case,
                             std::size_t input, std::size_t index);
[[nodiscard]] std::unique_ptr<TestExecutable> build_flagdnn_statistics(
    flagdnn::Handle& handle, const StatisticsTestCase& test_case);
int run_statistics_functional_test(int argc, char** argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark = false);
}  // namespace flagdnn::testing
#endif
