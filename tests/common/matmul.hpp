/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_MATMUL_HPP_
#define FLAGDNN_TESTS_COMMON_MATMUL_HPP_

#include "common/common.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

struct MatmulTestCase {
  std::string name;
  TestTensor a;
  TestTensor b;
  TestTensor output;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using MatmulExecutable = TestExecutable;

[[nodiscard]] std::vector<MatmulTestCase> make_matmul_cases();
void validate_matmul_case(const MatmulTestCase& test_case);

[[nodiscard]] std::unique_ptr<MatmulExecutable> build_flagdnn_matmul(
    flagdnn::Handle& handle,
    const MatmulTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<MatmulExecutable> build_matmul_reference(
    const MatmulTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/matmul_runner.cpp. */
int run_matmul_functional_test(int argc,
                               char** argv,
                               std::span<const MatmulTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_MATMUL_HPP_
