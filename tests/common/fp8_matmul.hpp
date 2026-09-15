/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_FP8_MATMUL_HPP_
#define FLAGDNN_TESTS_COMMON_FP8_MATMUL_HPP_
#include <memory>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct Fp8MatmulTestCase {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  int scale_mode = 0;
  int scale_variant = 0;
  bool plain_matmul = false;
};
std::vector<Fp8MatmulTestCase> make_fp8_matmul_cases();
std::unique_ptr<TestExecutable> build_flagdnn_fp8_matmul(
    flagdnn::Handle&, const Fp8MatmulTestCase&);
std::vector<std::vector<float>> fp8_matmul_inputs(const Fp8MatmulTestCase&);
int run_fp8_matmul_functional_test(int, char**,
                                   std::span<const Fp8MatmulTestCase>,
                                   bool benchmark = false);
}  // namespace flagdnn::testing
#endif
