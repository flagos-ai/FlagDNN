/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"
#include "convolution_runner_support.hpp"
#include "precision.hpp"

#include <span>
#include <stdexcept>
#include <vector>

namespace flagdnn::testing {
int run_convolution_functional_test(int argc, char **argv,
                                    std::span<const ConvolutionTestCase> cases,
                                    ConvolutionDirection expected_direction) {
  namespace support = hygon_functional::convolution;
  std::vector<ConvolutionTestCase> selected;
  const int precision = validation::hygon::selected_input_precision();
  for (const auto& value : cases)
    if (value.input_precision == precision) selected.push_back(value);
  return support::run_suite(
      argc, argv, std::span<const ConvolutionTestCase>(selected),
      "FLAGDNN_CONVOLUTION_FUNCTIONAL", "FLAGDNN_CONVOLUTION_CASE",
      [&](const ConvolutionTestCase &test_case) {
        if (test_case.direction != expected_direction) {
          throw std::invalid_argument(
              "convolution suite contains the wrong direction");
        }
      });
}

} // namespace flagdnn::testing
