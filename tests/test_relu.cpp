/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/pointwise.hpp"

#include <span>

int main(int argc, char** argv) {
  const flagdnn::testing::PointwiseCaseDefinition definition{
      .operation_name = "relu",
      .mode = FLAGDNN_POINTWISE_RELU_FWD,
      .input_domain = flagdnn::testing::PointwiseInputDomain::kReal,
  };
  const auto cases =
      flagdnn::testing::make_unary_pointwise_cases(definition);
  return flagdnn::testing::run_pointwise_functional_test(
      argc,
      argv,
      std::span<const flagdnn::testing::PointwiseTestCase>(cases),
      "FLAGDNN_RELU_FUNCTIONAL");
}
