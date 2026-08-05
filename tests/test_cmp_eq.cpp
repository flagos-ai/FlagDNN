/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "cmp_eq",
       .mode = FLAGDNN_POINTWISE_CMP_EQ,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kComparison},
      "FLAGDNN_CMP_EQ_FUNCTIONAL");
}
