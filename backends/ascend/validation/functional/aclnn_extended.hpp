/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/causal_convolution.hpp"
#include "common/convolution.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/pointwise.hpp"
#include "common/position_embedding.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_moe_matmul(const MoeMatmulTestCase &);
std::unique_ptr<TestExecutable>
build_aclnn_convolution_backward(const ConvolutionTestCase &);
std::unique_ptr<TestExecutable>
build_aclnn_causal_convolution(const CausalConvolutionTestCase &);
std::unique_ptr<TestExecutable>
build_aclnn_activation_backward(const PointwiseTestCase &);
std::unique_ptr<TestExecutable>
build_aclnn_extended_normalization(const ExtendedNormalizationTestCase &);
std::unique_ptr<TestExecutable> build_aclnn_index(const IndexTestCase &);
std::unique_ptr<TestExecutable> build_aclnn_resample(const ResampleTestCase &);
std::unique_ptr<TestExecutable> build_aclnn_rope(const RoPETestCase &);
std::unique_ptr<TestExecutable>
build_aclnn_statistics(const StatisticsTestCase &);
} // namespace flagdnn::testing
