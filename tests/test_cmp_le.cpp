/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "cmp_le",
       .mode = FLAGDNN_POINTWISE_CMP_LE,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kComparison},
      "FLAGDNN_CMP_LE_FUNCTIONAL");
}
