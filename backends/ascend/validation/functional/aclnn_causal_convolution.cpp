/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_convolution.h>
#include <aclnnop/aclnn_silu.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_causal_convolution(const CausalConvolutionTestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    const auto channels = c.inputs[0].dimensions[1],
               width = c.inputs[1].dimensions[1];
    auto *weight = p.formatted(p.view(1, {channels, 1, width}), ACL_FORMAT_NCL);
    auto *stride = p.array({1});
    auto *padding = p.array({(width - 1) * c.dilation, 0});
    auto *dilation = p.array({c.dilation});
    auto *zero = p.array({0});
    auto *output = c.silu ? p.temporary(c.outputs[0]) : p.ports.back();
    auto *input = p.formatted(p.ports[0], ACL_FORMAT_NCL);
    output = p.formatted(output, ACL_FORMAT_NCL);
    p.add(aclnnConvolution, [&](auto *w, auto **e) {
      return aclnnConvolutionGetWorkspaceSize(input, weight, p.ports[2], stride,
                                              padding, dilation, false, zero,
                                              channels, output, 0, w, e);
    });
    if (c.silu)
      p.add(aclnnSilu, [&](auto *w, auto **e) {
        return aclnnSiluGetWorkspaceSize(output, p.ports.back(), w, e);
      });
  });
}
} // namespace flagdnn::testing
