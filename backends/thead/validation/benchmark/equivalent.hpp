// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#ifndef FLAGDNN_THEAD_BENCHMARK_EQUIVALENT_HPP_
#define FLAGDNN_THEAD_BENCHMARK_EQUIVALENT_HPP_
#include "cpu_validation.hpp"
namespace flagdnn::validation::thead {
void run_equivalent_benchmark(
    const benchmarking::BenchmarkCase& specification,
    benchmarking::BenchmarkExecutable& executable, DeviceStream& stream,
    const benchmarking::thead::NativeBaselineMetadata* metadata = nullptr);
}
#endif
