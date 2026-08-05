/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/attention.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_sdpa_fp8_cases();
  return flagdnn::testing::run_sdpa_fp8_functional_test(
      argc, argv, std::span(cases));
}
