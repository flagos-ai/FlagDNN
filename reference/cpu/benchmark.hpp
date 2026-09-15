/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_BENCHMARK_HPP_
#define FLAGDNN_REFERENCE_CPU_BENCHMARK_HPP_
#include <flagdnn/flagdnn.h>

#include <functional>
#include <span>
#include <vector>
namespace flagdnn::benchmarking {
struct BenchmarkCase;
}
namespace flagdnn::reference::cpu {
// Round logical values to the requested storage dtype and back to float.
using OutputQuantizer = std::function<std::vector<float>(std::span<const float>,
                                                         flagdnnDataType_t)>;
std::vector<std::vector<float>> evaluate_benchmark(
    const benchmarking::BenchmarkCase& specification,
    const std::vector<std::vector<float>>& inputs, OutputQuantizer quantize);
}  // namespace flagdnn::reference::cpu
#endif
