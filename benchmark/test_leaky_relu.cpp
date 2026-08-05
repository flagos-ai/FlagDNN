/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  auto attributes = flagdnn::benchmarking::default_pointwise_attributes();
  attributes.flags =
      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  attributes.relu_lower_clip_slope = 0.2;
  const auto cases =
      flagdnn::benchmarking::unary_pointwise_benchmark_cases(
          FLAGDNN_POINTWISE_RELU_FWD,
          "leaky_relu",
          flagdnn::benchmarking::InputDomain::kReal,
          attributes);
  return flagdnn::benchmarking::run_benchmark_suite(
      argc,
      argv,
      std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_LEAKY_RELU_BENCHMARK");
}
