/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_ADD_HPP_
#define FLAGDNN_TESTS_COMMON_ADD_HPP_

#include "common/common.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

struct AddTestCase {
  std::string name;
  TestTensor left;
  TestTensor right;
  TestTensor output;
  double alpha = 1.0;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using AddExecutable = TestExecutable;

[[nodiscard]] std::vector<AddTestCase> make_add_cases();
void validate_add_case(const AddTestCase& test_case);

[[nodiscard]] std::unique_ptr<AddExecutable> build_flagdnn_add(
    flagdnn::Handle& handle,
    const AddTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<AddExecutable> build_add_reference(
    const AddTestCase& test_case);

/* Implemented by the selected platform runner and called by test_add.cpp. */
int run_add_functional_test(int argc,
                            char** argv,
                            std::span<const AddTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_ADD_HPP_
