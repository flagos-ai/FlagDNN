// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/add.hpp"
#include "pointwise_reference.hpp"

namespace flagdnn::testing {

std::unique_ptr<AddExecutable>
build_add_reference(const AddTestCase &test_case) {
  return iluvatar::validation::make_classic_pointwise_reference(
      {.mode = FLAGDNN_POINTWISE_ADD,
       .inputs = {test_case.left, test_case.right},
       .output = test_case.output,
       .alpha = test_case.alpha});
}

} // namespace flagdnn::testing
