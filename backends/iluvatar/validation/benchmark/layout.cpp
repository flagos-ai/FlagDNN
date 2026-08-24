// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "tensor_reference.hpp"

#include <stdexcept>

namespace flagdnn::iluvatar::validation::benchmark {

std::unique_ptr<flagdnn::testing::TestExecutable> build_layout_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  using flagdnn::benchmarking::Operation;
  if (specification.tensors.size() != 2 || specification.output_count != 1) {
    throw std::invalid_argument(
        "CoreX cuDNN layout benchmark tensor count is invalid");
  }
  flagdnn::testing::LayoutTestCase test_case;
  test_case.name = specification.name;
  test_case.input = to_test_tensor(specification.tensors[0]);
  test_case.output = to_test_tensor(specification.tensors[1]);
  switch (specification.operation) {
  case Operation::kReshape:
    test_case.operation = flagdnn::testing::LayoutOperation::kReshape;
    break;
  case Operation::kTranspose:
    test_case.operation = flagdnn::testing::LayoutOperation::kTranspose;
    test_case.permutation = specification.transpose.permutation;
    break;
  case Operation::kSlice:
    test_case.operation = flagdnn::testing::LayoutOperation::kSlice;
    test_case.slices = specification.slice.slices;
    test_case.slice_strides = specification.slice.strides;
    break;
  default:
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        "NO_EXACT_CUDNN_PRIMITIVE");
  }
  return validation::make_classic_layout_reference(test_case);
}

} // namespace flagdnn::iluvatar::validation::benchmark
