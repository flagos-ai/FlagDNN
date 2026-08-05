/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_TESTS_COMMON_ATTENTION_HPP_
#define FLAGDNN_TESTS_COMMON_ATTENTION_HPP_

#include "common/common.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace flagdnn::testing {

enum class AttentionDiagonalAlignment {
  kTopLeft,
  kBottomRight,
};

struct AttentionOptions {
  std::optional<float> attention_scale;
  std::optional<std::int64_t> diagonal_band_left_bound;
  std::optional<std::int64_t> diagonal_band_right_bound;
  AttentionDiagonalAlignment diagonal_alignment =
      AttentionDiagonalAlignment::kTopLeft;
};

struct SdpaTestCase {
  std::string name;
  TestTensor q;
  TestTensor k;
  TestTensor v;
  std::optional<TestTensor> bias;
  TestTensor output;
  std::optional<TestTensor> stats;
  AttentionOptions options;
  double output_absolute_tolerance = 0.0;
  double output_relative_tolerance = 0.0;
  double stats_absolute_tolerance = 0.0;
  double stats_relative_tolerance = 0.0;
  bool autotune = false;
};

struct SdpaBackwardTestCase {
  std::string name;
  TestTensor q;
  TestTensor k;
  TestTensor v;
  std::optional<TestTensor> bias;
  TestTensor output;
  TestTensor doutput;
  TestTensor stats;
  TestTensor dq;
  TestTensor dk;
  TestTensor dv;
  std::optional<TestTensor> dbias;
  AttentionOptions options;
  double absolute_tolerance = 0.0;
  double relative_tolerance = 0.0;
  bool deterministic = false;
  bool autotune = false;
};

struct Fp8Scalar {
  TestTensor tensor;
  float value = 1.0F;
};

struct SdpaFp8TestCase {
  std::string name;
  TestTensor q;
  TestTensor k;
  TestTensor v;
  Fp8Scalar descale_q;
  Fp8Scalar descale_k;
  Fp8Scalar descale_v;
  Fp8Scalar descale_s;
  Fp8Scalar scale_s;
  Fp8Scalar scale_o;
  std::optional<TestTensor> bias;
  TestTensor output;
  std::optional<TestTensor> stats;
  TestTensor amax_s;
  TestTensor amax_o;
  AttentionOptions options;
  double output_absolute_tolerance = 0.0;
  double output_relative_tolerance = 0.0;
  double stats_absolute_tolerance = 0.0;
  double stats_relative_tolerance = 0.0;
  double amax_absolute_tolerance = 0.0;
  double amax_relative_tolerance = 0.0;
  bool autotune = false;
};

struct SdpaFp8BackwardTestCase {
  std::string name;
  TestTensor q;
  TestTensor k;
  TestTensor v;
  TestTensor output;
  TestTensor doutput;
  TestTensor stats;
  Fp8Scalar descale_q;
  Fp8Scalar descale_k;
  Fp8Scalar descale_v;
  Fp8Scalar descale_o;
  Fp8Scalar descale_doutput;
  Fp8Scalar descale_s;
  Fp8Scalar descale_dp;
  Fp8Scalar scale_s;
  Fp8Scalar scale_dq;
  Fp8Scalar scale_dk;
  Fp8Scalar scale_dv;
  Fp8Scalar scale_dp;
  TestTensor dq;
  TestTensor dk;
  TestTensor dv;
  TestTensor amax_dq;
  TestTensor amax_dk;
  TestTensor amax_dv;
  TestTensor amax_dp;
  AttentionOptions options;
  double gradient_absolute_tolerance = 0.0;
  double gradient_relative_tolerance = 0.0;
  double amax_absolute_tolerance = 0.0;
  double amax_relative_tolerance = 0.0;
  bool autotune = false;
};

using AttentionExecutable = TestExecutable;

[[nodiscard]] std::vector<SdpaTestCase> make_sdpa_cases();
[[nodiscard]] std::vector<SdpaBackwardTestCase>
make_sdpa_backward_cases();
[[nodiscard]] std::vector<SdpaFp8TestCase> make_sdpa_fp8_cases();
[[nodiscard]] std::vector<SdpaFp8BackwardTestCase>
make_sdpa_fp8_backward_cases();

void validate_sdpa_case(const SdpaTestCase& test_case);
void validate_sdpa_backward_case(const SdpaBackwardTestCase& test_case);
void validate_sdpa_fp8_case(const SdpaFp8TestCase& test_case);
void validate_sdpa_fp8_backward_case(
    const SdpaFp8BackwardTestCase& test_case);

[[nodiscard]] std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa(
    flagdnn::Handle& handle,
    const SdpaTestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable>
build_flagdnn_sdpa_backward(flagdnn::Handle& handle,
                            const SdpaBackwardTestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable> build_flagdnn_sdpa_fp8(
    flagdnn::Handle& handle,
    const SdpaFp8TestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable>
build_flagdnn_sdpa_fp8_backward(
    flagdnn::Handle& handle,
    const SdpaFp8BackwardTestCase& test_case);

/* Implemented by the selected backends/<platform>/validation adapter. */
[[nodiscard]] std::unique_ptr<AttentionExecutable> build_sdpa_reference(
    const SdpaTestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable>
build_sdpa_backward_reference(const SdpaBackwardTestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable> build_sdpa_fp8_reference(
    const SdpaFp8TestCase& test_case);
[[nodiscard]] std::unique_ptr<AttentionExecutable>
build_sdpa_fp8_backward_reference(
    const SdpaFp8BackwardTestCase& test_case);

/* Implemented by backends/<platform>/validation/functional/attention_runner.cpp. */
int run_sdpa_functional_test(int argc,
                             char** argv,
                             std::span<const SdpaTestCase> cases);
int run_sdpa_backward_functional_test(
    int argc,
    char** argv,
    std::span<const SdpaBackwardTestCase> cases);
int run_sdpa_fp8_functional_test(
    int argc,
    char** argv,
    std::span<const SdpaFp8TestCase> cases);
int run_sdpa_fp8_backward_functional_test(
    int argc,
    char** argv,
    std::span<const SdpaFp8BackwardTestCase> cases);

}  // namespace flagdnn::testing

#endif  // FLAGDNN_TESTS_COMMON_ATTENTION_HPP_
