// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/composite.hpp"
#include "convolution_reference.hpp"
#include "pointwise_reference.hpp"

namespace flagdnn::testing {

std::unique_ptr<CompositeExecutable>
build_add_square_reference(const AddSquareTestCase &test_case) {
  return iluvatar::validation::make_classic_add_square_reference(
      test_case.left, test_case.right, test_case.output);
}

std::unique_ptr<CompositeExecutable>
build_conv_bias_relu_reference(const ConvBiasReluTestCase &test_case) {
  return iluvatar::validation::make_classic_conv_bias_relu_reference(test_case);
}

} // namespace flagdnn::testing
