/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/convolution.hpp"
#include "common/pointwise.hpp"
#include "common/runner.hpp"
#include "validation/functional/paired.hpp"
#include <cstdlib>
namespace flagdnn::benchmarking {
int run_ascend_conv_bias_relu_benchmark(int, char **,
                                        std::span<const BenchmarkCase>);
int run_benchmark_suite(int argc, char **argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite) {
  if (!cases.empty()) {
    const auto &config = cases[0].benchmark;
    testing::ascend::paired_benchmark_config = {config.warmup_iterations,
                                                config.sample_count,
                                                config.iterations_per_sample};
  }
  if (!cases.empty() && cases[0].operation == Operation::kGraph)
    return run_ascend_conv_bias_relu_benchmark(argc, argv, cases);
  if (!cases.empty() && (cases[0].operation == Operation::kConvolutionDgrad ||
                         cases[0].operation == Operation::kConvolutionWgrad)) {
    std::vector<testing::ConvolutionTestCase> converted;
    const bool dx = cases[0].operation == Operation::kConvolutionDgrad;
    const auto direction = dx ? testing::ConvolutionDirection::kDgrad
                              : testing::ConvolutionDirection::kWgrad;
    for (const auto &c : cases) {
      testing::ConvolutionTestCase p;
      p.name = c.name;
      p.direction = direction;
      auto tensor = [](const TensorSpec &t) {
        return testing::TestTensor{t.uid, t.data_type, t.dimensions, t.strides,
                                   t.binding_byte_offset};
      };
      p.y = tensor(c.tensors[0]);
      p.x = tensor(c.tensors[dx ? 2 : 1]);
      p.w = tensor(c.tensors[dx ? 1 : 2]);
      p.pre_padding = c.convolution.pre_padding;
      p.post_padding = c.convolution.post_padding;
      p.stride = c.convolution.stride;
      p.dilation = c.convolution.dilation;
      p.groups = c.convolution.groups;
      p.mode = c.convolution.mode == ConvolutionMode::kConvolution
                   ? testing::ConvolutionMode::kConvolution
                   : testing::ConvolutionMode::kCrossCorrelation;
      p.absolute_tolerance = c.absolute_tolerance;
      p.relative_tolerance = c.relative_tolerance;
      converted.push_back(std::move(p));
    }
    if (setenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK", "1", 1) != 0)
      return 1;
    return testing::run_convolution_functional_test(argc, argv, converted,
                                                    direction);
  }
  std::vector<testing::PointwiseTestCase> converted;
  for (const auto &c : cases) {
    testing::PointwiseTestCase p;
    p.name = c.name;
    p.mode = c.pointwise_mode;
    p.attributes = c.pointwise_attributes;
    p.alpha = c.add_alpha;
    p.absolute_tolerance = c.absolute_tolerance;
    p.relative_tolerance = c.relative_tolerance;
    for (std::size_t i = 0; i < c.tensors.size(); ++i) {
      const auto &t = c.tensors[i];
      testing::TestTensor tensor{t.uid, t.data_type, t.dimensions, t.strides,
                                 t.binding_byte_offset};
      if (i + 1 == c.tensors.size())
        p.output = tensor;
      else {
        p.inputs.push_back(tensor);
        p.input_domains.push_back(testing::PointwiseInputDomain::kReal);
      }
    }
    converted.push_back(std::move(p));
  }
  if (setenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK", "1", 1) != 0)
    return 1;
  return testing::run_pointwise_functional_test(argc, argv, converted, suite);
}
} // namespace flagdnn::benchmarking
