/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_CUDNN_EXTENDED_HPP_
#define FLAGDNN_NVIDIA_CUDNN_EXTENDED_HPP_
#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
std::unique_ptr<TestExecutable> build_cudnn_bilinear_resample(
    const ResampleTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_index(
    const IndexTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_concatenate_transform(
    const IndexTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_fp8_matmul(
    const Fp8MatmulTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_moe_matmul(
    const MoeMatmulTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_resample(
    const ResampleTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_extended_normalization(
    const ExtendedNormalizationTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_causal_convolution(
    const CausalConvolutionTestCase& test_case);
std::unique_ptr<TestExecutable> build_cudnn_statistics(
    const StatisticsTestCase& test_case);
}  // namespace flagdnn::testing
#endif
