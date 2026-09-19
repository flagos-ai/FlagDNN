/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_convolution_functional_test(int argc, char **argv,
                                    std::span<const ConvolutionTestCase> cases,
                                    ConvolutionDirection direction) {
  struct Case : ConvolutionTestCase {
    std::vector<TestTensor> inputs, outputs;
  };
  std::vector<Case> paired;
  for (const auto &c : cases) {
    if (c.direction != direction)
      throw std::invalid_argument("convolution direction mismatch");
    const bool dx = direction == ConvolutionDirection::kDgrad;
    paired.push_back({c, {c.y, dx ? c.w : c.x}, {dx ? c.x : c.w}});
  }
  return ascend::run_paired_cases<Case>(
      argc, argv, paired, "FLAGDNN_CONVOLUTION_CASE", build_flagdnn_convolution,
      [](const Case &c) {
        std::vector<std::vector<float>> inputs;
        for (std::size_t k = 0; k < c.inputs.size(); ++k) {
          std::vector<float> v(ascend::io::element_count(c.inputs[k]));
          for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 17 + k * 13) % 29) - 14) / float(31 + k);
          inputs.push_back(std::move(v));
        }
        return inputs;
      },
      [](const Case &c) {
        auto ref = static_cast<const ConvolutionTestCase &>(c);
        if (c.direction == ConvolutionDirection::kDgrad)
          ref.x = c.outputs[0];
        else
          ref.w = c.outputs[0];
        return build_aclnn_convolution_backward(ref);
      },
      [](const Case &c, std::size_t) {
        return ascend::PairedTolerance{c.absolute_tolerance,
                                       c.relative_tolerance};
      },
      std::getenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK") != nullptr);
}
} // namespace flagdnn::testing
