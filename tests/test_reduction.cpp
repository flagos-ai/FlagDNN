/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/reduction.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_reduction_cases();
  return flagdnn::testing::run_reduction_functional_test(
      argc,
      argv,
      std::span<const flagdnn::testing::ReductionTestCase>(cases));
}
