// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "tensor_reference.hpp"

#include <stdexcept>

namespace flagdnn::iluvatar::validation::benchmark {

std::unique_ptr<flagdnn::testing::TestExecutable> build_reduction_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation != flagdnn::benchmarking::Operation::kReduction ||
      specification.tensors.size() != 2 || specification.output_count != 1) {
    throw std::invalid_argument(
        "CoreX cuDNN reduction benchmark case is invalid");
  }
  flagdnn::testing::ReductionTestCase test_case;
  test_case.name = specification.name;
  test_case.input = to_test_tensor(specification.tensors[0]);
  test_case.output = to_test_tensor(specification.tensors[1]);
  test_case.mode = specification.reduction_mode;
  test_case.axis = specification.reduction_axis;
  test_case.keep_dimensions = specification.keep_dimensions;
  test_case.absolute_tolerance = specification.absolute_tolerance;
  test_case.relative_tolerance = specification.relative_tolerance;
  return validation::make_classic_reduction_reference(test_case);
}

} // namespace flagdnn::iluvatar::validation::benchmark
