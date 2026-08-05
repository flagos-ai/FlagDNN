/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_unary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "rsqrt",
       .mode = FLAGDNN_POINTWISE_RSQRT,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kPositive},
      "FLAGDNN_RSQRT_FUNCTIONAL");
}
