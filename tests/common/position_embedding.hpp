/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_POSITION_EMBEDDING_HPP_
#define FLAGDNN_TESTS_COMMON_POSITION_EMBEDDING_HPP_
#include <memory>
#include <span>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct RoPETestCase {
  std::string name, operation;
  std::vector<TestTensor> inputs, outputs;
  std::int64_t rope_dim = 0;
  float output_scale = 1.0F;
};
std::vector<RoPETestCase> make_rope_cases(bool backward);
std::unique_ptr<TestExecutable> build_flagdnn_rope(
    flagdnn::Handle& handle, const RoPETestCase& test_case);
std::vector<std::vector<float>> rope_inputs(const RoPETestCase& test_case);
int run_rope_functional_test(int argc, char** argv,
                             std::span<const RoPETestCase> cases,
                             bool benchmark = false);
}  // namespace flagdnn::testing
#endif
