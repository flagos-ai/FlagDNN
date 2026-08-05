/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  auto attributes = flagdnn::benchmarking::default_pointwise_attributes();
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  attributes.swish_beta = 1.25;
  const auto cases =
      flagdnn::benchmarking::unary_pointwise_benchmark_cases(
          FLAGDNN_POINTWISE_SWISH_FWD,
          "swish",
          flagdnn::benchmarking::InputDomain::kReal,
          attributes);
  return flagdnn::benchmarking::run_benchmark_suite(
      argc,
      argv,
      std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_SWISH_BENCHMARK");
}
