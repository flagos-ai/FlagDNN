// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_TENSOR_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_TENSOR_REFERENCE_HPP_

#include "common/layout.hpp"
#include "common/reduction.hpp"

#include <memory>

namespace flagdnn::iluvatar::validation {

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_reduction_reference(
    const flagdnn::testing::ReductionTestCase &test_case);

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_layout_reference(
    const flagdnn::testing::LayoutTestCase &test_case);

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_TENSOR_REFERENCE_HPP_
