/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_add_square_cases();
  return flagdnn::testing::run_add_square_functional_test(
      argc, argv, std::span(cases));
}
