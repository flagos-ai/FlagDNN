/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/position_embedding.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_rope_cases(true);
  return flagdnn::testing::run_rope_functional_test(argc, argv, cases);
}
