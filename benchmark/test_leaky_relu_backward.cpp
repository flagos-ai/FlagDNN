/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/cases.hpp"
#include "common/runner.hpp"

#include <span>

int main(int argc, char** argv) {
  auto cases = flagdnn::benchmarking::binary_pointwise_benchmark_cases(
      FLAGDNN_POINTWISE_RELU_BWD, "leaky_relu_backward",
      flagdnn::benchmarking::InputDomain::kReal);
  for (auto& test_case : cases) {
    test_case.pointwise_attributes.flags =
        FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
    test_case.pointwise_attributes.relu_lower_clip_slope = 0.2;
  }
  return flagdnn::benchmarking::run_benchmark_suite(
      argc, argv, std::span<const flagdnn::benchmarking::BenchmarkCase>(cases),
      "FLAGDNN_LEAKY_RELU_BACKWARD_BENCHMARK");
}
