/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "mod",
       .mode = FLAGDNN_POINTWISE_MOD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kModulo},
      "FLAGDNN_MOD_FUNCTIONAL");
}
