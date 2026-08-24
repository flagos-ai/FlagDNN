// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/reduction.hpp"
#include "tensor_reference.hpp"

namespace flagdnn::testing {

TestTensor
reduction_reference_input_tensor(const ReductionTestCase &test_case) {
  return test_case.input;
}

std::unique_ptr<ReductionExecutable>
build_reduction_reference(const ReductionTestCase &test_case) {
  return iluvatar::validation::make_classic_reduction_reference(test_case);
}

} // namespace flagdnn::testing
