/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/case.hpp"
#include "common/composite.hpp"
#include "validation/functional/paired.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_relu.h>
namespace flagdnn::benchmarking {
int run_ascend_conv_bias_relu_benchmark(int argc, char **argv,
                                        std::span<const BenchmarkCase> cases) {
  using namespace testing;
  using namespace testing::ascend;
  struct Case : ConvBiasReluTestCase {
    std::vector<TestTensor> inputs, outputs;
  };
  std::vector<Case> converted;
  for (const auto &c : cases) {
    Case p;
    p.name = c.name;
    auto tensor = [](const TensorSpec &t) {
      return TestTensor{t.uid, t.data_type, t.dimensions, t.strides,
                        t.binding_byte_offset};
    };
    p.x = tensor(c.tensors[0]);
    p.w = tensor(c.tensors[1]);
    p.bias = tensor(c.tensors[2]);
    p.output = tensor(c.tensors[3]);
    p.inputs = {p.x, p.w, p.bias};
    p.outputs = {p.output};
    const auto &conv = c.graph.nodes[0].convolution;
    p.padding = conv.pre_padding;
    p.stride = conv.stride;
    p.dilation = conv.dilation;
    p.absolute_tolerance = c.absolute_tolerance;
    p.relative_tolerance = c.relative_tolerance;
    converted.push_back(std::move(p));
  }
  return run_paired_cases<Case>(
      argc, argv, converted, "FLAGDNN_CONV_BIAS_RELU_CASE",
      build_flagdnn_conv_bias_relu,
      [](const Case &c) {
        std::vector<std::vector<float>> values;
        for (std::size_t k = 0; k < c.inputs.size(); ++k) {
          std::vector<float> v(io::element_count(c.inputs[k]));
          for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 17 + k * 7) % 31) - 15) / 31.0F;
          values.push_back(std::move(v));
        }
        return values;
      },
      [](const Case &c) -> std::unique_ptr<TestExecutable> {
        auto specs = c.inputs;
        specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
        return std::make_unique<Plan>(specs, [c](Plan &p) {
          auto *x = p.formatted(p.ports[0], ACL_FORMAT_NCHW);
          auto *weight = p.formatted(p.ports[1], ACL_FORMAT_NCHW);
          auto *conv = p.formatted(p.temporary(c.outputs[0]), ACL_FORMAT_NCHW);
          auto *biased = p.temporary(c.outputs[0]);
          auto *stride = p.array(c.stride);
          auto *padding = p.array(c.padding);
          auto *dilation = p.array(c.dilation);
          auto *zero = p.array({0, 0});
          auto *one = p.scalar(1);
          p.add(aclnnConvolution, [&](auto *w, auto **e) {
            return aclnnConvolutionGetWorkspaceSize(x, weight, nullptr, stride,
                                                    padding, dilation, false,
                                                    zero, 1, conv, 0, w, e);
          });
          p.add(aclnnAdd, [&](auto *w, auto **e) {
            return aclnnAddGetWorkspaceSize(conv, p.ports[2], one, biased, w,
                                            e);
          });
          p.add(aclnnRelu, [&](auto *w, auto **e) {
            return aclnnReluGetWorkspaceSize(biased, p.ports[3], w, e);
          });
        });
      },
      [](const Case &c, std::size_t) {
        return PairedTolerance{c.absolute_tolerance, c.relative_tolerance};
      },
      true);
}
} // namespace flagdnn::benchmarking
