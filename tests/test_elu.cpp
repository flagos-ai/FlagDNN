/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
  attributes.elu_alpha = 1.0;
  return flagdnn::testing::run_unary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "elu",
       .mode = FLAGDNN_POINTWISE_ELU_FWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal,
       .attributes = attributes},
      "FLAGDNN_ELU_FUNCTIONAL");
}
