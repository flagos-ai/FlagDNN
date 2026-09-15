/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/attention_runner.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_attention_benchmark_test(
      argc, argv, flagdnn::testing::AttentionBenchmarkOperation::kFp8Forward);
}
