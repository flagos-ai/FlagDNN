/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_COMPOSITE_HPP_
#define FLAGDNN_TESTS_COMMON_COMPOSITE_HPP_

#include "common/common.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

struct AddSquareTestCase {
  std::string name;
  TestTensor left;
  TestTensor right;
  TestTensor output;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

struct ConvBiasReluTestCase {
  std::string name;
  TestTensor x;
  TestTensor w;
  TestTensor bias;
  TestTensor output;
  std::vector<std::int64_t> padding;
  std::vector<std::int64_t> stride;
  std::vector<std::int64_t> dilation;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using CompositeExecutable = TestExecutable;

[[nodiscard]] std::vector<AddSquareTestCase> make_add_square_cases();
[[nodiscard]] std::vector<ConvBiasReluTestCase>
make_conv_bias_relu_cases();
void validate_composite_case(const AddSquareTestCase& test_case);
void validate_composite_case(const ConvBiasReluTestCase& test_case);

[[nodiscard]] std::unique_ptr<CompositeExecutable> build_flagdnn_add_square(
    flagdnn::Handle& handle,
    const AddSquareTestCase& test_case);
[[nodiscard]] std::unique_ptr<CompositeExecutable>
build_flagdnn_conv_bias_relu(flagdnn::Handle& handle,
                             const ConvBiasReluTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<CompositeExecutable> build_add_square_reference(
    const AddSquareTestCase& test_case);
[[nodiscard]] std::unique_ptr<CompositeExecutable>
build_conv_bias_relu_reference(const ConvBiasReluTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/composite_runner.cpp. */
int run_add_square_functional_test(
    int argc,
    char** argv,
    std::span<const AddSquareTestCase> cases);
int run_conv_bias_relu_functional_test(
    int argc,
    char** argv,
    std::span<const ConvBiasReluTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_COMPOSITE_HPP_
