/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/moe_matmul.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_moe_matmul_cases(true);
  return flagdnn::testing::run_moe_matmul_functional_test(argc, argv, cases);
}
