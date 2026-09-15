/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_RANDOM_HPP_
#define FLAGDNN_TESTS_COMMON_RANDOM_HPP_
#include <memory>
#include <span>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct RngTestCase {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  std::int64_t seed = 0, offset = 0;
  int distribution = 1;
  double probability = 0.5;
};
std::vector<RngTestCase> make_rng_cases();
std::unique_ptr<TestExecutable> build_flagdnn_rng(flagdnn::Handle& handle,
                                                  const RngTestCase& test_case);
int run_rng_functional_test(int argc, char** argv,
                            std::span<const RngTestCase> cases,
                            bool benchmark = false);
}  // namespace flagdnn::testing
#endif
