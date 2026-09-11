// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_NORMALIZATION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_NORMALIZATION_REFERENCE_HPP_

#include "capability.hpp"
#include "common/normalization.hpp"

#include <memory>

namespace flagdnn::validation::thead {

[[nodiscard]] std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_layernorm_reference(
    const flagdnn::testing::LayernormTestCase &test_case,
    const CapabilityRecord &capability);

[[nodiscard]] std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_rmsnorm_reference(
    const flagdnn::testing::RmsnormTestCase &test_case,
    const CapabilityRecord &capability);

[[nodiscard]] std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_batchnorm_reference(
    const flagdnn::testing::BatchnormTestCase &test_case,
    const CapabilityRecord &capability);

[[nodiscard]] std::unique_ptr<flagdnn::testing::NormalizationExecutable>
make_acdnn_batchnorm_inference_reference(
    const flagdnn::testing::BatchnormInferenceTestCase &test_case,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_NORMALIZATION_REFERENCE_HPP_
