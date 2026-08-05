/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  auto attributes = flagdnn::benchmarking::default_pointwise_attributes();
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  attributes.softplus_beta = 1.0;
  const auto cases =
      flagdnn::benchmarking::unary_pointwise_benchmark_cases(
          FLAGDNN_POINTWISE_SOFTPLUS_FWD,
          "softplus",
          flagdnn::benchmarking::InputDomain::kReal,
          attributes);
  return flagdnn::benchmarking::run_benchmark_suite(
      argc,
      argv,
      std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_SOFTPLUS_BENCHMARK");
}
