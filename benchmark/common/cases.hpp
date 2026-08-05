/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BENCHMARK_CASES_CASES_HPP_
#define FLAGDNN_BENCHMARK_CASES_CASES_HPP_

#include "case.hpp"

#include <string_view>
#include <vector>

namespace flagdnn::benchmarking {

[[nodiscard]] std::vector<BenchmarkCase> unary_pointwise_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain = InputDomain::kReal,
    flagdnnPointwiseAttributes_t attributes =
        default_pointwise_attributes());
[[nodiscard]] std::vector<BenchmarkCase> unary_pointwise_benchmark_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain = InputDomain::kReal,
    flagdnnPointwiseAttributes_t attributes =
        default_pointwise_attributes());
[[nodiscard]] std::vector<BenchmarkCase> binary_pointwise_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain = InputDomain::kReal);
[[nodiscard]] std::vector<BenchmarkCase> binary_select_cases();
[[nodiscard]] std::vector<BenchmarkCase> binary_select_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> binary_pointwise_benchmark_cases(
    flagdnnPointwiseMode_t mode,
    std::string_view operation_name,
    InputDomain input_domain = InputDomain::kReal);
[[nodiscard]] std::vector<BenchmarkCase> relu_cases();
[[nodiscard]] std::vector<BenchmarkCase> relu_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> add_cases();
[[nodiscard]] std::vector<BenchmarkCase> add_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> reduction_cases();
[[nodiscard]] std::vector<BenchmarkCase> reduction_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> matmul_cases();
[[nodiscard]] std::vector<BenchmarkCase> matmul_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> reshape_cases();
[[nodiscard]] std::vector<BenchmarkCase> reshape_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> transpose_cases();
[[nodiscard]] std::vector<BenchmarkCase> transpose_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> slice_cases();
[[nodiscard]] std::vector<BenchmarkCase> slice_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_fprop_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_fprop_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_dgrad_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_dgrad_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_wgrad_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_wgrad_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> layernorm_cases();
[[nodiscard]] std::vector<BenchmarkCase> layernorm_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> rmsnorm_cases();
[[nodiscard]] std::vector<BenchmarkCase> rmsnorm_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> normalization_forward_cases();
[[nodiscard]] std::vector<BenchmarkCase> batchnorm_cases();
[[nodiscard]] std::vector<BenchmarkCase> batchnorm_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> batchnorm_inference_cases();
[[nodiscard]] std::vector<BenchmarkCase> batchnorm_inference_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> add_square_cases();
[[nodiscard]] std::vector<BenchmarkCase> add_square_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> conv_bias_relu_benchmark_cases();
[[nodiscard]] std::vector<BenchmarkCase> all_cases();

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BENCHMARK_CASES_CASES_HPP_
