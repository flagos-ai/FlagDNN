// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_MATMUL_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_MATMUL_REFERENCE_HPP_

#include "capability.hpp"
#include "common/matmul.hpp"

#include <memory>

namespace flagdnn::validation::thead {

[[nodiscard]] std::unique_ptr<flagdnn::testing::MatmulExecutable>
make_acdnn_matmul_reference(
    const flagdnn::testing::MatmulTestCase &test_case,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_MATMUL_REFERENCE_HPP_
