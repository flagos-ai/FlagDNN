/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_OPS_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_OPS_HPP_

#include "common/benchmark_provider.hpp"

#include <memory>

namespace flagdnn::benchmarking::cudnn_detail {

[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_pointwise(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_relu(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_reduction(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_matmul(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_reshape(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_transpose(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_slice(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_convolution_fprop(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_convolution_dgrad(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_convolution_wgrad(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_layernorm(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_rmsnorm(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_batchnorm(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_batchnorm_inference(
    const BenchmarkCase& specification);
[[nodiscard]] std::unique_ptr<BenchmarkExecutable> build_graph(
    const BenchmarkCase& specification);

}  // namespace flagdnn::benchmarking::cudnn_detail

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_OPS_HPP_
