// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_COMPOSITE_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_COMPOSITE_REFERENCE_HPP_

#include "capability.hpp"
#include "common/common.hpp"

#include <memory>

namespace flagdnn::validation::thead {

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_add_square_reference(
    const flagdnn::testing::TestTensor &left,
    const flagdnn::testing::TestTensor &right,
    const flagdnn::testing::TestTensor &output,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_COMPOSITE_REFERENCE_HPP_
