/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "pow",
       .mode = FLAGDNN_POINTWISE_POW,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kPower},
      "FLAGDNN_POW_FUNCTIONAL");
}
