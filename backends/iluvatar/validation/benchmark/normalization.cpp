// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "common/normalization.hpp"

#include <stdexcept>

namespace flagdnn::iluvatar::validation::benchmark {

std::unique_ptr<flagdnn::testing::TestExecutable> build_normalization_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  using flagdnn::benchmarking::Operation;
  const auto tensor_at = [&](std::size_t index) {
    return to_test_tensor(specification.tensors.at(index));
  };
  switch (specification.operation) {
  case Operation::kLayernorm: {
    if (specification.tensors.size() != 6 || specification.output_count != 3) {
      throw std::invalid_argument("LayerNorm benchmark case is invalid");
    }
    flagdnn::testing::LayernormTestCase test_case;
    test_case.name = specification.name;
    test_case.x = tensor_at(0);
    test_case.scale = tensor_at(1);
    test_case.bias = tensor_at(2);
    test_case.y = tensor_at(3);
    test_case.mean = tensor_at(4);
    test_case.inv_variance = tensor_at(5);
    test_case.epsilon = specification.normalization.epsilon;
    return flagdnn::testing::build_layernorm_reference(test_case);
  }
  case Operation::kRmsnorm: {
    if (specification.tensors.size() != 5 || specification.output_count != 2) {
      throw std::invalid_argument("RMSNorm benchmark case is invalid");
    }
    flagdnn::testing::RmsnormTestCase test_case;
    test_case.name = specification.name;
    test_case.x = tensor_at(0);
    test_case.scale = tensor_at(1);
    test_case.bias = tensor_at(2);
    test_case.y = tensor_at(3);
    test_case.inv_variance = tensor_at(4);
    test_case.epsilon = specification.normalization.epsilon;
    return flagdnn::testing::build_rmsnorm_reference(test_case);
  }
  case Operation::kBatchnorm: {
    if (specification.tensors.size() != 10 || specification.output_count != 5) {
      throw std::invalid_argument("BatchNorm benchmark case is invalid");
    }
    flagdnn::testing::BatchnormTestCase test_case;
    test_case.name = specification.name;
    test_case.x = tensor_at(0);
    test_case.scale = tensor_at(1);
    test_case.bias = tensor_at(2);
    test_case.previous_running_mean = tensor_at(3);
    test_case.previous_running_variance = tensor_at(4);
    test_case.y = tensor_at(5);
    test_case.mean = tensor_at(6);
    test_case.inv_variance = tensor_at(7);
    test_case.next_running_mean = tensor_at(8);
    test_case.next_running_variance = tensor_at(9);
    test_case.epsilon = specification.normalization.epsilon;
    test_case.momentum = specification.normalization.momentum;
    return flagdnn::testing::build_batchnorm_reference(test_case);
  }
  case Operation::kBatchnormInference: {
    if (specification.tensors.size() != 6 || specification.output_count != 1) {
      throw std::invalid_argument(
          "BatchNorm inference benchmark case is invalid");
    }
    flagdnn::testing::BatchnormInferenceTestCase test_case;
    test_case.name = specification.name;
    test_case.x = tensor_at(0);
    test_case.mean = tensor_at(1);
    test_case.inv_variance = tensor_at(2);
    test_case.scale = tensor_at(3);
    test_case.bias = tensor_at(4);
    test_case.y = tensor_at(5);
    return flagdnn::testing::build_batchnorm_inference_reference(test_case);
  }
  default:
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        "NO_EXACT_CUDNN_PRIMITIVE");
  }
}

} // namespace flagdnn::iluvatar::validation::benchmark
