/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"

#include <span>

int main(int argc, char** argv) {
  using flagdnn::testing::ConvolutionDirection;
  const auto cases = flagdnn::testing::make_convolution_cases(
      ConvolutionDirection::kFprop);
  return flagdnn::testing::run_convolution_functional_test(
      argc, argv, std::span(cases), ConvolutionDirection::kFprop);
}
