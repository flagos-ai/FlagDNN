/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  attributes.swish_beta = 1.25;
  return flagdnn::testing::run_unary_pointwise_functional_test(
      argc,
      argv,
      {.operation_name = "swish",
       .mode = FLAGDNN_POINTWISE_SWISH_FWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal,
       .attributes = attributes},
      "FLAGDNN_SWISH_FUNCTIONAL");
}
