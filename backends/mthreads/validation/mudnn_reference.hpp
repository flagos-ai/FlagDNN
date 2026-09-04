/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REFERENCE_HPP_

#include "common/add.hpp"

#include <memory>

namespace flagdnn::validation::mthreads {

[[nodiscard]] std::unique_ptr<flagdnn::testing::AddExecutable>
build_mudnn_add_reference(
    const flagdnn::testing::AddTestCase& test_case);

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUDNN_REFERENCE_HPP_
