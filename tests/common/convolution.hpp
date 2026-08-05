/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_CONVOLUTION_HPP_
#define FLAGDNN_TESTS_COMMON_CONVOLUTION_HPP_

#include "common/common.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

enum class ConvolutionDirection {
  kFprop,
  kDgrad,
  kWgrad,
};

enum class ConvolutionMode {
  kCrossCorrelation,
  kConvolution,
};

/*
 * Every convolution case names the same mathematical tensors:
 *   Y = convolution(X, W)
 * FProp consumes X/W and produces Y; Dgrad consumes Y/W and produces X;
 * Wgrad consumes Y/X and produces W.  Keeping these roles stable makes the
 * three public Graph APIs and their bindings directly comparable.
 */
struct ConvolutionTestCase {
  std::string name;
  ConvolutionDirection direction = ConvolutionDirection::kFprop;
  TestTensor x;
  TestTensor w;
  TestTensor y;
  std::vector<std::int64_t> pre_padding;
  std::vector<std::int64_t> post_padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  std::int64_t groups = 1;
  ConvolutionMode mode = ConvolutionMode::kCrossCorrelation;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using ConvolutionExecutable = TestExecutable;

[[nodiscard]] std::vector<ConvolutionTestCase> make_convolution_cases(
    ConvolutionDirection direction);
void validate_convolution_case(const ConvolutionTestCase& test_case);
[[nodiscard]] const TestTensor& convolution_output_tensor(
    const ConvolutionTestCase& test_case);

[[nodiscard]] std::unique_ptr<ConvolutionExecutable>
build_flagdnn_convolution(flagdnn::Handle& handle,
                          const ConvolutionTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<ConvolutionExecutable>
build_convolution_reference(const ConvolutionTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/convolution_runner.cpp. */
int run_convolution_functional_test(
    int argc,
    char** argv,
    std::span<const ConvolutionTestCase> cases,
    ConvolutionDirection expected_direction);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_CONVOLUTION_HPP_
