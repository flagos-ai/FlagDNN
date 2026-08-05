/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include <span>

int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_batchnorm_cases();
  return flagdnn::testing::run_batchnorm_functional_test(
      argc, argv, std::span(cases));
}
