// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_LAYOUT_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_LAYOUT_REFERENCE_HPP_

#include "capability.hpp"
#include "common/layout.hpp"

#include <cstdint>
#include <memory>
#include <span>

namespace flagdnn::validation::thead {

[[nodiscard]] bool legacy_acdnn_transpose_descriptor_compatible(
    std::span<const std::int64_t> input_dimensions,
    std::span<const std::int64_t> permutation);

[[nodiscard]] std::unique_ptr<flagdnn::testing::LayoutExecutable>
make_acdnn_layout_reference(
    const flagdnn::testing::LayoutTestCase &test_case,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_LAYOUT_REFERENCE_HPP_
