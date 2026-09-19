/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_math.hpp"
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_extended_normalization(const ExtendedNormalizationTestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    Math m{p};
    const bool backward = c.operation.ends_with("_backward"),
               rms = c.operation == "rmsnorm_backward";
    std::vector<std::int64_t> axes(c.axes.begin(), c.axes.end());
    const auto x = m.input(backward ? 1 : 0), scale = m.input(backward ? 2 : 1);
    if (!backward) {
      auto mean = m.reduce(x, axes, true);
      auto centered = m.add(x, mean, -1);
      auto variance = m.reduce(m.mul(centered, centered), axes, true);
      auto inv = m.rsqrt(m.shift(variance, c.epsilon));
      auto y = m.add(m.mul(m.mul(centered, inv), scale), m.input(2));
      m.output(y, 3);
      m.output(mean, 4);
      m.output(inv, 5);
    } else {
      auto dy = m.input(0);
      auto inv = m.input(rms ? 3 : 4);
      auto centered = rms ? x : m.add(x, m.input(3), -1);
      auto z = m.mul(centered, inv);
      auto gradient = m.mul(dy, scale);
      auto dx = m.add(gradient,
                      m.mul(z, m.reduce(m.mul(gradient, z), axes, true)), -1);
      if (!rms)
        dx = m.add(dx, m.reduce(gradient, axes, true), -1);
      dx = m.mul(dx, inv);
      std::vector<std::int64_t> parameter_axes;
      auto param_shape = scale.shape;
      param_shape.insert(param_shape.begin(),
                         x.shape.size() - param_shape.size(), 1);
      for (std::size_t i = 0; i < x.shape.size(); ++i)
        if (param_shape[i] == 1 && x.shape[i] != 1)
          parameter_axes.push_back(i);
      const auto output = c.inputs.size();
      m.output(dx, output);
      m.output(m.reduce(m.mul(dy, z), parameter_axes, false), output + 1);
      m.output(m.reduce(dy, parameter_axes, false), output + 2);
    }
  });
}
} // namespace flagdnn::testing
