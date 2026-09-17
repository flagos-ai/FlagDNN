/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/benchmark_provider.hpp"
#include <functional>
namespace flagdnn {
class Handle;
}
namespace flagdnn::benchmarking {
using NativeBuilder = std::function<std::unique_ptr<BenchmarkExecutable>(
    flagdnn::Handle &, const BenchmarkCase &)>;
int run_hygon_benchmark_suite(int argc, char **argv,
                              std::span<const BenchmarkCase> cases,
                              std::string_view suite_name,
                              const NativeBuilder &builder = {});
} // namespace flagdnn::benchmarking
