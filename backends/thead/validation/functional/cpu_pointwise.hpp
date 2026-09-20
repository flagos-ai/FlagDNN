// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_CPU_POINTWISE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_CPU_POINTWISE_HPP_

#include "common/pointwise.hpp"
#include "tensor_io.hpp"

#include <string_view>

namespace flagdnn::validation::thead::functional {

// Execute the production graph and compare with the independent CPU oracle.
// add_square represents left + right * right; the virtual square is quantized
// to its graph-declared output dtype before the addition.
void run_cpu_pointwise_case(
    const flagdnn::testing::PointwiseTestCase& test_case,
    flagdnn::testing::TestExecutable& production, DeviceStream& stream,
    std::string_view fallback_reason, bool add_square = false);

}  // namespace flagdnn::validation::thead::functional

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_FUNCTIONAL_CPU_POINTWISE_HPP_
