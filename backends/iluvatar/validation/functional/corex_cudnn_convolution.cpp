// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/convolution.hpp"
#include "convolution_reference.hpp"

namespace flagdnn::testing {

std::unique_ptr<ConvolutionExecutable>
build_convolution_reference(const ConvolutionTestCase &test_case) {
  return iluvatar::validation::make_classic_convolution_reference(test_case);
}

} // namespace flagdnn::testing
