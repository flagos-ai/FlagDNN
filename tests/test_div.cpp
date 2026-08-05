/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "div",
       .mode = FLAGDNN_POINTWISE_DIV,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kDivisor},
      "FLAGDNN_DIV_FUNCTIONAL");
}
