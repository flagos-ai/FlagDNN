/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/matmul.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_matmul_cases();
  return flagdnn::testing::run_matmul_functional_test(
      argc,
      argv,
      std::span<const flagdnn::testing::MatmulTestCase>(cases));
}
