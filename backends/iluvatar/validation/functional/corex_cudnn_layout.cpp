// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/layout.hpp"
#include "tensor_reference.hpp"

namespace flagdnn::testing {

std::unique_ptr<LayoutExecutable>
build_layout_reference(const LayoutTestCase &test_case) {
  return iluvatar::validation::make_classic_layout_reference(test_case);
}

} // namespace flagdnn::testing
