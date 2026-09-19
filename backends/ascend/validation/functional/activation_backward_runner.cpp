/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_pointwise_functional_test(int argc, char **argv,
                                  std::span<const PointwiseTestCase> cases,
                                  std::string_view) {
  struct Case : PointwiseTestCase {
    std::vector<TestTensor> outputs;
  };
  std::vector<Case> paired;
  for (const auto &c : cases)
    paired.push_back({c, {c.output}});
  return ascend::run_paired_cases<Case>(
      argc, argv, paired, "FLAGDNN_POINTWISE_CASE", build_flagdnn_pointwise,
      [](const Case &c) {
        std::vector<std::vector<float>> inputs;
        for (std::size_t k = 0; k < c.inputs.size(); ++k) {
          std::vector<float> v(ascend::io::element_count(c.inputs[k]));
          for (std::size_t i = 0; i < v.size(); ++i)
            v[i] = float(int((i * 17 + k * 11) % 41) - 20) / float(13 + k);
          inputs.push_back(std::move(v));
        }
        return inputs;
      },
      [](const Case &c) {
        auto spec = static_cast<const PointwiseTestCase &>(c);
        spec.output = c.outputs[0];
        return build_aclnn_activation_backward(spec);
      },
      [](const Case &c, std::size_t) {
        return ascend::PairedTolerance{c.absolute_tolerance,
                                       c.relative_tolerance};
      },
      std::getenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK") != nullptr);
}
} // namespace flagdnn::testing
