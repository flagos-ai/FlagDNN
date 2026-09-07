// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REDUCTION_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REDUCTION_REFERENCE_HPP_

#include "capability.hpp"
#include "common/reduction.hpp"

#include <memory>

namespace flagdnn::validation::thead {

[[nodiscard]] std::unique_ptr<flagdnn::testing::ReductionExecutable>
make_acdnn_reduction_reference(
    const flagdnn::testing::ReductionTestCase &test_case,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REDUCTION_REFERENCE_HPP_
