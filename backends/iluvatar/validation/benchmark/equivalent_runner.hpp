// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "benchmark/equivalent_baseline.hpp"
#include "functional/runner_support.hpp"

namespace flagdnn::benchmarking {
using EquivalentOracle = std::function<std::vector<std::vector<float>>(
    const std::vector<std::vector<float>> &)>;
void run_equivalent_native(
    const BenchmarkCase &specification,
    const iluvatar::validation::functional::BuildExecutable &build,
    const EquivalentOracle &oracle, const NativeBaselineMetadata &metadata);
} // namespace flagdnn::benchmarking
