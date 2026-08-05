/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::benchmarking::binary_pointwise_benchmark_cases(
      FLAGDNN_POINTWISE_SIGMOID_BWD,
      "sigmoid_backward",
      flagdnn::benchmarking::InputDomain::kReal);
  return flagdnn::benchmarking::run_benchmark_suite(
      argc,
      argv,
      std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_SIGMOID_BACKWARD_BENCHMARK");
}
