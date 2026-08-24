// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_OPS_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_OPS_HPP_

#include "common/benchmark_provider.hpp"
#include "common/common.hpp"

#include <memory>

namespace flagdnn::iluvatar::validation::benchmark {

[[nodiscard]] flagdnn::testing::TestTensor
to_test_tensor(const flagdnn::benchmarking::TensorSpec &tensor);

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_pointwise_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_reduction_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_layout_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_matmul_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_convolution_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_normalization_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);
[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
build_graph_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification);

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_OPS_HPP_
