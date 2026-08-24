// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_CONVOLUTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_CONVOLUTION_REFERENCE_HPP_

#include "common/composite.hpp"
#include "common/convolution.hpp"

#include <memory>

namespace flagdnn::iluvatar::validation {

[[nodiscard]] std::unique_ptr<testing::ConvolutionExecutable>
make_classic_convolution_reference(
    const testing::ConvolutionTestCase &test_case);

[[nodiscard]] std::unique_ptr<testing::CompositeExecutable>
make_classic_conv_bias_relu_reference(
    const testing::ConvBiasReluTestCase &test_case);

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_CONVOLUTION_REFERENCE_HPP_
