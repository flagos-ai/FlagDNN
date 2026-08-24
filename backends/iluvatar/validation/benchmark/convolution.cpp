// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "convolution_reference.hpp"

#include <stdexcept>

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

flagdnn::testing::ConvolutionMode
convolution_mode(flagdnn::benchmarking::ConvolutionMode mode) {
  switch (mode) {
  case flagdnn::benchmarking::ConvolutionMode::kCrossCorrelation:
    return flagdnn::testing::ConvolutionMode::kCrossCorrelation;
  case flagdnn::benchmarking::ConvolutionMode::kConvolution:
    return flagdnn::testing::ConvolutionMode::kConvolution;
  }
  throw std::invalid_argument("unknown benchmark convolution mode");
}

} // namespace

std::unique_ptr<flagdnn::testing::TestExecutable> build_convolution_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  using flagdnn::benchmarking::Operation;
  if (specification.tensors.size() != 3 || specification.output_count != 1) {
    throw std::invalid_argument(
        "CoreX cuDNN convolution benchmark tensor count is invalid");
  }
  flagdnn::testing::ConvolutionTestCase test_case;
  test_case.name = specification.name;
  switch (specification.operation) {
  case Operation::kConvolutionFprop:
    test_case.direction = flagdnn::testing::ConvolutionDirection::kFprop;
    test_case.x = to_test_tensor(specification.tensors[0]);
    test_case.w = to_test_tensor(specification.tensors[1]);
    test_case.y = to_test_tensor(specification.tensors[2]);
    break;
  case Operation::kConvolutionDgrad:
    test_case.direction = flagdnn::testing::ConvolutionDirection::kDgrad;
    test_case.y = to_test_tensor(specification.tensors[0]);
    test_case.w = to_test_tensor(specification.tensors[1]);
    test_case.x = to_test_tensor(specification.tensors[2]);
    break;
  case Operation::kConvolutionWgrad:
    test_case.direction = flagdnn::testing::ConvolutionDirection::kWgrad;
    test_case.y = to_test_tensor(specification.tensors[0]);
    test_case.x = to_test_tensor(specification.tensors[1]);
    test_case.w = to_test_tensor(specification.tensors[2]);
    break;
  default:
    throw flagdnn::benchmarking::BenchmarkUnsupportedError(
        "NO_EXACT_CUDNN_PRIMITIVE");
  }
  test_case.pre_padding = specification.convolution.pre_padding;
  test_case.post_padding = specification.convolution.post_padding;
  test_case.stride = specification.convolution.stride;
  test_case.dilation = specification.convolution.dilation;
  test_case.groups = specification.convolution.groups;
  test_case.mode = convolution_mode(specification.convolution.mode);
  test_case.absolute_tolerance = specification.absolute_tolerance;
  test_case.relative_tolerance = specification.relative_tolerance;
  return validation::make_classic_convolution_reference(test_case);
}

} // namespace flagdnn::iluvatar::validation::benchmark
