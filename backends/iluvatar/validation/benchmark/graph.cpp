// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/ops.hpp"

#include "convolution_reference.hpp"
#include "pointwise_reference.hpp"

#include <stdexcept>
#include <string_view>

namespace flagdnn::iluvatar::validation::benchmark {

std::unique_ptr<flagdnn::testing::TestExecutable> build_graph_reference(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  if (specification.operation != flagdnn::benchmarking::Operation::kGraph ||
      specification.output_count != 1) {
    throw std::invalid_argument("CoreX cuDNN Graph benchmark case is invalid");
  }
  if (specification.name.starts_with("add_square_perf_")) {
    if (specification.tensors.size() != 3) {
      throw std::invalid_argument("AddSquare benchmark case is invalid");
    }
    return validation::make_classic_add_square_reference(
        to_test_tensor(specification.tensors[0]),
        to_test_tensor(specification.tensors[1]),
        to_test_tensor(specification.tensors[2]));
  }
  if (specification.name.starts_with("conv_bias_relu_perf_")) {
    if (specification.tensors.size() != 4 ||
        specification.graph.nodes.empty()) {
      throw std::invalid_argument("ConvBiasRelu benchmark case is invalid");
    }
    const auto &convolution = specification.graph.nodes.front().convolution;
    if (convolution.pre_padding != convolution.post_padding) {
      throw flagdnn::benchmarking::BenchmarkUnsupportedError(
          "ATTRIBUTE_UNSUPPORTED");
    }
    flagdnn::testing::ConvBiasReluTestCase test_case;
    test_case.name = specification.name;
    test_case.x = to_test_tensor(specification.tensors[0]);
    test_case.w = to_test_tensor(specification.tensors[1]);
    test_case.bias = to_test_tensor(specification.tensors[2]);
    test_case.output = to_test_tensor(specification.tensors[3]);
    test_case.padding = convolution.pre_padding;
    test_case.stride = convolution.stride;
    test_case.dilation = convolution.dilation;
    test_case.absolute_tolerance = specification.absolute_tolerance;
    test_case.relative_tolerance = specification.relative_tolerance;
    return validation::make_classic_conv_bias_relu_reference(test_case);
  }
  throw flagdnn::benchmarking::BenchmarkUnsupportedError(
      "NO_EXACT_CUDNN_PRIMITIVE");
}

} // namespace flagdnn::iluvatar::validation::benchmark
