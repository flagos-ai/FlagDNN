/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_TESTS_CPU_CUDNN_BENCHMARK_HPP_
#define FLAGDNN_REFERENCE_TESTS_CPU_CUDNN_BENCHMARK_HPP_
#include "cpu_cudnn.hpp"
#include "benchmark/common/case.hpp"
namespace flagdnn::reference::tests {
namespace b = flagdnn::benchmarking;
b::BenchmarkCase formula_case(b::Operation operation,
    const std::vector<t::TestTensor>& inputs, const std::vector<t::TestTensor>& outputs);
void check_formula(const b::BenchmarkCase&, const Result&, double atol, double rtol);
}
#endif
