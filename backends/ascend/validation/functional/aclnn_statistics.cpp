/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_math.hpp"
#include <aclnnop/aclnn_clamp.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_statistics(const StatisticsTestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    Math m{p};
    if (c.operation == "genstats") {
      auto x = m.input(0);
      std::vector<std::int64_t> axes;
      for (std::size_t i = 0; i < x.shape.size(); ++i)
        if (i != 1)
          axes.push_back(i);
      m.output(m.reduce(x, axes, false), 1);
      m.output(m.reduce(m.mul(x, x), axes, false), 2);
    } else {
      auto mean = m.scale(m.input(0), 1.0 / c.accum_count);
      auto variance = m.add(m.scale(m.input(1), 1.0 / c.accum_count),
                            m.mul(mean, mean), -1);
      auto clamped = m.temp(variance.shape);
      auto *zero = p.scalar(0);
      p.add(aclnnClampMin, [&](auto *w, auto **e) {
        return aclnnClampMinGetWorkspaceSize(variance.tensor, zero,
                                             clamped.tensor, w, e);
      });
      variance = clamped;
      auto inv = m.rsqrt(m.shift(variance, c.epsilon));
      auto eq_scale = m.mul(m.input(2), inv);
      auto eq_bias = m.add(m.input(3), m.mul(mean, eq_scale), -1);
      const auto out = c.inputs.size();
      m.output(eq_scale, out);
      m.output(eq_bias, out + 1);
      m.output(mean, out + 2);
      m.output(inv, out + 3);
      if (c.inputs.size() == 6) {
        m.output(m.add(m.scale(m.input(4), 1 - c.momentum),
                       m.scale(mean, c.momentum)),
                 out + 4);
        const double correction =
            c.accum_count > 1 ? c.accum_count / (c.accum_count - 1) : 0;
        m.output(m.add(m.scale(m.input(5), 1 - c.momentum),
                       m.scale(variance, c.momentum * correction)),
                 out + 5);
      }
    }
  });
}
} // namespace flagdnn::testing
