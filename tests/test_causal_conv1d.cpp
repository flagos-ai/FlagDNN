/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/causal_convolution.hpp"
int main(int argc, char** argv) {
  const auto cases = flagdnn::testing::make_causal_convolution_cases();
  return flagdnn::testing::run_causal_convolution_functional_test(argc, argv,
                                                                  cases);
}
