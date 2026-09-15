/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_POINTWISE_HPP_
#define FLAGDNN_TESTS_COMMON_POINTWISE_HPP_

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "common/common.hpp"

namespace flagdnn::testing {

enum class PointwiseInputDomain {
  kReal,
  kPositive,
  kScaled,
  kTan,
  kDivisor,
  kModulo,
  kPower,
  kModuloSigned,
  kComparison,
  kLogical,
};

struct PointwiseTestCase {
  std::string name;
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  std::vector<TestTensor> inputs;
  TestTensor output;
  std::vector<PointwiseInputDomain> input_domains;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  double alpha = 1.0;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

struct PointwiseCaseDefinition {
  std::string operation_name;
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  PointwiseInputDomain input_domain = PointwiseInputDomain::kReal;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  bool autotune = false;
};

using PointwiseExecutable = TestExecutable;

[[nodiscard]] std::vector<PointwiseTestCase> make_unary_pointwise_cases(
    const PointwiseCaseDefinition& definition);
[[nodiscard]] std::vector<PointwiseTestCase> make_binary_pointwise_cases(
    const PointwiseCaseDefinition& definition);
[[nodiscard]] std::vector<PointwiseTestCase> make_binary_select_cases(
    const PointwiseCaseDefinition& definition);

int run_unary_pointwise_functional_test(
    int argc, char** argv, const PointwiseCaseDefinition& definition,
    std::string_view suite_name);
int run_binary_pointwise_functional_test(
    int argc, char** argv, const PointwiseCaseDefinition& definition,
    std::string_view suite_name);
int run_binary_select_functional_test(int argc, char** argv,
                                      const PointwiseCaseDefinition& definition,
                                      std::string_view suite_name);

// Exact signed integer oracle; values never pass through floating point.
[[nodiscard]] std::int32_t pointwise_integer_input(std::size_t index,
                                                   std::size_t input,
                                                   flagdnnPointwiseMode_t mode);

void validate_pointwise_case(const PointwiseTestCase& test_case);

[[nodiscard]] std::unique_ptr<PointwiseExecutable> build_flagdnn_pointwise(
    flagdnn::Handle& handle, const PointwiseTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation/functional
 * adapter. */
[[nodiscard]] std::unique_ptr<PointwiseExecutable> build_pointwise_reference(
    const PointwiseTestCase& test_case);

/* Implemented by
 * backends/<platform>/validation/functional/pointwise_runner.cpp. */
int run_pointwise_functional_test(int argc, char** argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view suite_name);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_POINTWISE_HPP_
