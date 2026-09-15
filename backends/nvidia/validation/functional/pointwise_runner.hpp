/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_POINTWISE_RUNNER_HPP_
#define FLAGDNN_NVIDIA_POINTWISE_RUNNER_HPP_
#include "common/pointwise.hpp"
namespace flagdnn::testing {
int run_cudnn_pointwise_tests(int argc, char** argv,
                              std::span<const PointwiseTestCase> cases,
                              std::string_view suite_name, bool benchmark);
}  // namespace flagdnn::testing
#endif
