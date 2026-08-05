/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  attributes.softplus_beta = 1.0;
  return flagdnn::testing::run_unary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "softplus",
       .mode = FLAGDNN_POINTWISE_SOFTPLUS_FWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal,
       .attributes = attributes},
      "FLAGDNN_SOFTPLUS_FUNCTIONAL");
}
