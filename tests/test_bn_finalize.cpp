/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/statistics.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_bn_finalize_cases();
  return flagdnn::testing::run_statistics_functional_test(argc, argv, cases);
}
