// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_CONVOLUTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_CONVOLUTION_REFERENCE_HPP_

#include "capability.hpp"
#include "common/composite.hpp"
#include "common/convolution.hpp"

#include <memory>

namespace flagdnn::validation::thead {

[[nodiscard]] std::unique_ptr<flagdnn::testing::ConvolutionExecutable>
make_acdnn_convolution_reference(
    const flagdnn::testing::ConvolutionTestCase &test_case,
    const CapabilityRecord &capability);

[[nodiscard]] std::unique_ptr<flagdnn::testing::CompositeExecutable>
make_acdnn_conv_bias_relu_reference(
    const flagdnn::testing::ConvBiasReluTestCase &test_case,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_CONVOLUTION_REFERENCE_HPP_
