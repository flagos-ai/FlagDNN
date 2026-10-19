/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_

#include <span>
#include <vector>

#include "common/case.hpp"

namespace flagdnn::benchmarking {

// Unmeasurable native-reference workloads are removed in the shared
// factories. Every platform receives the same retained performance cases.
inline std::vector<BenchmarkCase> cudnn_benchmark_cases(
    std::span<const BenchmarkCase> cases) {
  return {cases.begin(), cases.end()};
}

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CASES_HPP_
