/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/resample.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_resample_cases();
  return flagdnn::testing::run_resample_functional_test(argc, argv, cases);
}
