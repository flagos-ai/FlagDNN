/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/random.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_rng_cases();
  return flagdnn::testing::run_rng_functional_test(argc, argv, cases);
}
