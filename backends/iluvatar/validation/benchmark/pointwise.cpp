// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "pointwise_reference.hpp"

#include <stdexcept>
#include <utility>

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

void require_tensor_count(
    const flagdnn::benchmarking::BenchmarkCase &specification,
    std::size_t count) {
  if (specification.tensors.size() != count ||
      specification.output_count != 1) {
    throw std::invalid_argument(
        "CoreX cuDNN pointwise benchmark tensor count is invalid");
  }
}

} // namespace

flagdnn::testing::TestTensor
to_test_tensor(const flagdnn::benchmarking::TensorSpec &tensor) {
  return {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
          tensor.binding_byte_offset};
}

std::unique_ptr<flagdnn::testing::TestExecutable> build_pointwise_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  using flagdnn::benchmarking::input_tensor_count;
  using flagdnn::benchmarking::Operation;

  validation::ClassicPointwiseReferenceSpec reference;
  switch (specification.operation) {
  case Operation::kRelu:
    require_tensor_count(specification, 2);
    reference.mode = FLAGDNN_POINTWISE_RELU_FWD;
    break;
  case Operation::kPointwise:
    if (specification.output_count != 1) {
      throw std::invalid_argument(
          "CoreX cuDNN pointwise output count is invalid");
    }
    reference.mode = specification.pointwise_mode;
    reference.attributes = specification.pointwise_attributes;
    reference.alpha = specification.add_alpha;
    break;
  case Operation::kAdd:
    require_tensor_count(specification, 3);
    reference.mode = FLAGDNN_POINTWISE_ADD;
    reference.alpha = specification.add_alpha;
    break;
  default:
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        "NO_EXACT_CUDNN_PRIMITIVE");
  }

  const std::size_t inputs = input_tensor_count(specification);
  if (inputs == 0 || inputs > 2) {
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        "NO_EXACT_CUDNN_PRIMITIVE");
  }
  reference.inputs.reserve(inputs);
  for (std::size_t index = 0; index < inputs; ++index) {
    reference.inputs.push_back(to_test_tensor(specification.tensors[index]));
  }
  reference.output =
      to_test_tensor(flagdnn::benchmarking::output_tensor(specification));
  return validation::make_classic_pointwise_reference(std::move(reference));
}

} // namespace flagdnn::iluvatar::validation::benchmark
