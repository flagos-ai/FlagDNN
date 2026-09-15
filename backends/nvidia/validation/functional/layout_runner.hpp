/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_LAYOUT_RUNNER_HPP_
#define FLAGDNN_NVIDIA_LAYOUT_RUNNER_HPP_
#include "common/layout.hpp"
namespace flagdnn::testing {
int run_cudnn_layout_tests(int argc, char** argv,
                           std::span<const LayoutTestCase> cases,
                           bool benchmark = false);
}  // namespace flagdnn::testing
#endif
