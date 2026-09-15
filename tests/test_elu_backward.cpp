/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc, argv,
      {.operation_name = "elu_backward",
       .mode = FLAGDNN_POINTWISE_ELU_BWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal},
      "FLAGDNN_ELU_BACKWARD_FUNCTIONAL");
}
