/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/normalization_extended.hpp"
int main(int argc, char** argv) {
  const auto cases =
      flagdnn::testing::make_extended_normalization_cases("rmsnorm_backward");
  return flagdnn::testing::run_extended_normalization_functional_test(
      argc, argv, cases);
}
