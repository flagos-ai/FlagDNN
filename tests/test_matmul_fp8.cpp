/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/fp8_matmul.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_fp8_matmul_cases();
  return flagdnn::testing::run_fp8_matmul_functional_test(argc, argv, cases);
}
