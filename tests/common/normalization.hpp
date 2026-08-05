/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_NORMALIZATION_HPP_
#define FLAGDNN_TESTS_COMMON_NORMALIZATION_HPP_

#include "common/common.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

struct LayernormTestCase {
  std::string name;
  TestTensor x;
  TestTensor scale;
  TestTensor bias;
  TestTensor y;
  TestTensor mean;
  TestTensor inv_variance;
  double epsilon = 1.0e-3;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

struct RmsnormTestCase {
  std::string name;
  TestTensor x;
  TestTensor scale;
  TestTensor bias;
  TestTensor y;
  TestTensor inv_variance;
  double epsilon = 1.0e-3;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

struct BatchnormTestCase {
  std::string name;
  TestTensor x;
  TestTensor scale;
  TestTensor bias;
  TestTensor previous_running_mean;
  TestTensor previous_running_variance;
  TestTensor y;
  TestTensor mean;
  TestTensor inv_variance;
  TestTensor next_running_mean;
  TestTensor next_running_variance;
  double epsilon = 1.0e-3;
  double momentum = 0.1;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

struct BatchnormInferenceTestCase {
  std::string name;
  TestTensor x;
  TestTensor mean;
  TestTensor inv_variance;
  TestTensor scale;
  TestTensor bias;
  TestTensor y;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool autotune = false;
};

using NormalizationExecutable = TestExecutable;

[[nodiscard]] std::vector<LayernormTestCase> make_layernorm_cases();
[[nodiscard]] std::vector<RmsnormTestCase> make_rmsnorm_cases();
[[nodiscard]] std::vector<BatchnormTestCase> make_batchnorm_cases();
[[nodiscard]] std::vector<BatchnormInferenceTestCase>
make_batchnorm_inference_cases();

void validate_normalization_case(const LayernormTestCase& test_case);
void validate_normalization_case(const RmsnormTestCase& test_case);
void validate_normalization_case(const BatchnormTestCase& test_case);
void validate_normalization_case(
    const BatchnormInferenceTestCase& test_case);

[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_flagdnn_layernorm(flagdnn::Handle& handle,
                        const LayernormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_flagdnn_rmsnorm(flagdnn::Handle& handle,
                      const RmsnormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_flagdnn_batchnorm(flagdnn::Handle& handle,
                        const BatchnormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_flagdnn_batchnorm_inference(
    flagdnn::Handle& handle,
    const BatchnormInferenceTestCase& test_case);

/* Platform reference may require a different physical layout for X/Y. */
[[nodiscard]] TestTensor batchnorm_reference_data_tensor(
    const TestTensor& tensor);

/* Implemented by the selected backends/<platform>/validation/functional adapter. */
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_layernorm_reference(const LayernormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_rmsnorm_reference(const RmsnormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_batchnorm_reference(const BatchnormTestCase& test_case);
[[nodiscard]] std::unique_ptr<NormalizationExecutable>
build_batchnorm_inference_reference(
    const BatchnormInferenceTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/normalization_runner.cpp. */
int run_layernorm_functional_test(
    int argc,
    char** argv,
    std::span<const LayernormTestCase> cases);
int run_rmsnorm_functional_test(
    int argc,
    char** argv,
    std::span<const RmsnormTestCase> cases);
int run_batchnorm_functional_test(
    int argc,
    char** argv,
    std::span<const BatchnormTestCase> cases);
int run_batchnorm_inference_functional_test(
    int argc,
    char** argv,
    std::span<const BatchnormInferenceTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_NORMALIZATION_HPP_
