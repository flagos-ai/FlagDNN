/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_unary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "gelu",
       .mode = FLAGDNN_POINTWISE_GELU_FWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal},
      "FLAGDNN_GELU_FUNCTIONAL");
}
