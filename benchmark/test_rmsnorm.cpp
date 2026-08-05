/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::benchmarking::rmsnorm_benchmark_cases();
  return flagdnn::benchmarking::run_benchmark_suite(
      argc,
      argv,
      std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_RMSNORM_BENCHMARK");
}
