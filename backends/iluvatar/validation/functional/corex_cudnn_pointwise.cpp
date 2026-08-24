// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/pointwise.hpp"
#include "pointwise_reference.hpp"

namespace flagdnn::testing {

std::unique_ptr<PointwiseExecutable>
build_pointwise_reference(const PointwiseTestCase &test_case) {
  return iluvatar::validation::make_classic_pointwise_reference(
      {.mode = test_case.mode,
       .inputs = test_case.inputs,
       .output = test_case.output,
       .attributes = test_case.attributes,
       .alpha = test_case.alpha});
}

} // namespace flagdnn::testing
