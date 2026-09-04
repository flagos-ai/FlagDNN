/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/add.hpp"

#include "backends/mthreads/validation/mudnn_reference.hpp"

#include <memory>

namespace flagdnn::testing {

std::unique_ptr<AddExecutable> build_add_reference(
    const AddTestCase& test_case) {
  return validation::mthreads::build_mudnn_add_reference(test_case);
}

}  // namespace flagdnn::testing
