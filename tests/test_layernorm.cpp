/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_layernorm_cases();
  return flagdnn::testing::run_layernorm_functional_test(
      argc, argv, std::span(cases));
}
