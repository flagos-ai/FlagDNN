/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  return flagdnn::testing::run_binary_select_functional_test(
      argc,
      argv,
      {.operation_name = "binary_select",
       .mode = FLAGDNN_POINTWISE_BINARY_SELECT,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal},
      "FLAGDNN_BINARY_SELECT_FUNCTIONAL");
}
