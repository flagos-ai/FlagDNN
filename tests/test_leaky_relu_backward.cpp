/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

int main(int argc, char** argv) {
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  attributes.relu_lower_clip_slope = 0.2;
  return flagdnn::testing::run_binary_pointwise_functional_test(
      argc, argv,
      {.operation_name = "leaky_relu_backward",
       .mode = FLAGDNN_POINTWISE_RELU_BWD,
       .input_domain = flagdnn::testing::PointwiseInputDomain::kReal,
       .attributes = attributes},
      "FLAGDNN_LEAKY_RELU_BACKWARD_FUNCTIONAL");
}
