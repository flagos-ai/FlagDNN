/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_RESAMPLE_HPP_
#define FLAGDNN_TESTS_COMMON_RESAMPLE_HPP_
#include <memory>
#include <span>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct ResampleTestCase {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  int mode = 5, padding = 3;
  std::vector<std::int64_t> window, stride, pre, post;
  bool align_corners = false;
};
std::vector<ResampleTestCase> make_resample_cases();
std::unique_ptr<TestExecutable> build_flagdnn_resample(
    flagdnn::Handle& handle, const ResampleTestCase& test_case);
std::vector<std::vector<float>> resample_inputs(
    const ResampleTestCase& test_case);
int run_resample_functional_test(int argc, char** argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark = false);
}  // namespace flagdnn::testing
#endif
