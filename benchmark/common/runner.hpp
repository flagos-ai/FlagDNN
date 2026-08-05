/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_COMMON_RUNNER_HPP_
#define FLAGDNN_BENCHMARK_COMMON_RUNNER_HPP_

#include "case.hpp"

#include <span>
#include <string_view>

namespace flagdnn::benchmarking {

/*
 * Implemented by the selected backends/<platform>/validation/benchmark adapter.
 * Every platform reuses the same benchmark/test_<op>.cpp workloads.
 */
int run_benchmark_suite(int argc,
                        char** argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name);

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_COMMON_RUNNER_HPP_
