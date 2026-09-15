/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/dtype_runner.hpp"
int main(int argc, char** argv) {
  if (argc != 5) return 2;
  return flagdnn::testing::run_native_dtype_benchmark(argc - 2, argv, argv[3],
                                                      argv[4]);
}
