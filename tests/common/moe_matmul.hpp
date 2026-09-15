/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_MOE_MATMUL_HPP_
#define FLAGDNN_TESTS_COMMON_MOE_MATMUL_HPP_
#include <memory>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct MoeMatmulTestCase {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  int mode = 0, top_k = 1;
  bool backward = false;
  std::vector<std::int32_t> offsets, token_index, token_ks;
};
std::vector<MoeMatmulTestCase> make_moe_matmul_cases(bool backward);
std::unique_ptr<TestExecutable> build_flagdnn_moe_matmul(
    flagdnn::Handle&, const MoeMatmulTestCase&);
std::vector<std::vector<float>> moe_matmul_inputs(const MoeMatmulTestCase&);
int run_moe_matmul_functional_test(int, char**,
                                   std::span<const MoeMatmulTestCase>,
                                   bool benchmark = false);
}  // namespace flagdnn::testing
#endif
