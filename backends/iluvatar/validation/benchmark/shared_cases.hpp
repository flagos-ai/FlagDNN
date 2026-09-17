// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_ILUVATAR_SHARED_BENCHMARK_CASES_HPP_
#define FLAGDNN_ILUVATAR_SHARED_BENCHMARK_CASES_HPP_
#include "common/case.hpp"
#include <string_view>
namespace flagdnn::iluvatar::validation::benchmark {
// NVIDIA's additional dtype/precision groups use the functional generators.
// Append their workloads to the ordinary suite so accounting stays per
// operator.
std::vector<flagdnn::benchmarking::BenchmarkCase>
shared_benchmark_cases(std::string_view marker);
} // namespace flagdnn::iluvatar::validation::benchmark
#endif
