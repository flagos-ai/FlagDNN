/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_BENCHMARK_DTYPE_RUNNER_HPP_
#define FLAGDNN_BENCHMARK_DTYPE_RUNNER_HPP_
#include <string_view>
namespace flagdnn::testing {
int run_native_dtype_benchmark(int argc, char** argv,
                               std::string_view operation,
                               std::string_view category);
}
#endif
