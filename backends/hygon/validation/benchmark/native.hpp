/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/benchmark_provider.hpp"
#include <functional>
#include <vector>
namespace flagdnn {
class Handle;
}
namespace flagdnn::benchmarking {
struct NativeBaselineMetadata;
using NativeBuilder = std::function<std::unique_ptr<BenchmarkExecutable>(
    flagdnn::Handle &, const BenchmarkCase &)>;
using NativeCpuOracle = std::function<std::vector<std::vector<float>>(
    const BenchmarkCase &, const std::vector<std::vector<float>> &)>;
using NativeMetadata =
    std::function<NativeBaselineMetadata(const BenchmarkCase &)>;
int run_hygon_benchmark_suite(int argc, char **argv,
                              std::span<const BenchmarkCase> cases,
                              std::string_view suite_name,
                              const NativeBuilder &builder = {},
                              const NativeCpuOracle &oracle = {},
                              const NativeMetadata &metadata = {});
} // namespace flagdnn::benchmarking

namespace flagdnn::testing {
int run_native_extended_benchmark(int argc, char **argv,
                                  std::string_view operation);
int run_native_index_benchmark(int argc, char **argv,
                               std::string_view operation);
} // namespace flagdnn::testing
