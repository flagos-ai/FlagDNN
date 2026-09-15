/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/native_runner.hpp"
int main(int argc, char** argv) {
  return flagdnn::testing::run_native_benchmark_test(argc, argv, "resample");
}
