/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_add_cases();
  return flagdnn::testing::run_add_functional_test(
      argc, argv, std::span<const flagdnn::testing::AddTestCase>(cases));
}
