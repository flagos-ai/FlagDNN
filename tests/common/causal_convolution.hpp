/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TESTS_COMMON_CAUSAL_CONVOLUTION_HPP_
#define FLAGDNN_TESTS_COMMON_CAUSAL_CONVOLUTION_HPP_
#include <memory>
#include <span>
#include <string>

#include "common/common.hpp"
namespace flagdnn::testing {
struct CausalConvolutionTestCase {
  std::string name;
  std::vector<TestTensor> inputs, outputs;
  int precision = 0;
  std::int64_t dilation = 1;
  bool silu = false;
};
std::vector<CausalConvolutionTestCase> make_causal_convolution_cases();
std::unique_ptr<TestExecutable> build_flagdnn_causal_convolution(
    flagdnn::Handle& handle, const CausalConvolutionTestCase& test_case);
std::vector<std::vector<float>> causal_convolution_inputs(
    const CausalConvolutionTestCase& test_case);
int run_causal_convolution_functional_test(
    int argc, char** argv, std::span<const CausalConvolutionTestCase> cases,
    bool benchmark = false);
}  // namespace flagdnn::testing
#endif
