/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_REDUCTION_HPP_
#define FLAGDNN_TESTS_COMMON_REDUCTION_HPP_

#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/common.hpp"

namespace flagdnn::testing {

struct ReductionTestCase {
  std::string name;
  TestTensor input;
  TestTensor output;
  flagdnnReductionMode_t mode = FLAGDNN_REDUCTION_ADD;
  std::int32_t axis = 0;
  bool keep_dimensions = false;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using ReductionExecutable = TestExecutable;

[[nodiscard]] std::vector<ReductionTestCase> make_reduction_cases();
std::vector<float> reduction_host_input(const ReductionTestCase& test_case);

void validate_reduction_case(const ReductionTestCase& test_case);

[[nodiscard]] std::unique_ptr<ReductionExecutable> build_flagdnn_reduction(
    flagdnn::Handle& handle, const ReductionTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional
 * adapter. */
[[nodiscard]] TestTensor reduction_reference_input_tensor(
    const ReductionTestCase& test_case);
[[nodiscard]] std::unique_ptr<ReductionExecutable> build_reduction_reference(
    const ReductionTestCase& test_case);

/* Implemented by
 * backends/<platform>/validation/functional/reduction_runner.cpp. */
int run_reduction_functional_test(int argc, char** argv,
                                  std::span<const ReductionTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_REDUCTION_HPP_
