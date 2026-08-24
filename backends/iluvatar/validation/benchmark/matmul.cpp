// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

namespace flagdnn::iluvatar::validation::benchmark {

std::unique_ptr<flagdnn::testing::TestExecutable>
build_matmul_reference(const flagdnn::benchmarking::BenchmarkCase &) {
  throw flagdnn::benchmarking::BenchmarkUnsupportedError(
      "NO_EXACT_CUDNN_PRIMITIVE");
}

} // namespace flagdnn::iluvatar::validation::benchmark
