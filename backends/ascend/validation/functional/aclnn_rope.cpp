/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_math.hpp"
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_slice.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable> build_aclnn_rope(const RoPETestCase &c) {
  using namespace ascend;
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    Math m{p};
    auto x = m.input(0);
    const auto d = x.shape.back(), width = c.rope_dim ? c.rope_dim : d,
               half = width / 2;
    auto slice = [&](Math::Value v, std::int64_t start, std::int64_t end) {
      auto shape = v.shape;
      shape.back() = end - start;
      auto out = m.temp(shape);
      p.add(aclnnSlice, [&](auto *w, auto **e) {
        return aclnnSliceGetWorkspaceSize(v.tensor, 3, start, end, 1,
                                          out.tensor, w, e);
      });
      return out;
    };
    const auto prefix = d - width;
    auto a = slice(x, prefix, prefix + half), b = slice(x, prefix + half, d);
    auto source = m.input(1);
    Math::Value freqs{p.permuted(source.tensor, {1, 2, 0, 3}),
                      {1, 1, c.inputs[1].dimensions[0], width}};
    auto cosine = m.temp(freqs.shape), sine = m.temp(freqs.shape);
    p.add(aclnnCos, [&](auto *w, auto **e) {
      return aclnnCosGetWorkspaceSize(freqs.tensor, cosine.tensor, w, e);
    });
    p.add(aclnnSin, [&](auto *w, auto **e) {
      return aclnnSinGetWorkspaceSize(freqs.tensor, sine.tensor, w, e);
    });
    auto ca = slice(cosine, 0, half), cb = slice(cosine, half, width);
    auto sa = slice(sine, 0, half), sb = slice(sine, half, width);
    const bool backward = c.operation == "rope_backward";
    auto left =
        m.add(m.mul(a, ca), m.mul(b, backward ? sb : sa), backward ? 1 : -1);
    auto right =
        m.add(m.mul(b, cb), m.mul(a, backward ? sa : sb), backward ? -1 : 1);
    std::vector<aclTensor *> parts;
    if (prefix)
      parts.push_back(slice(x, 0, prefix).tensor);
    parts.push_back(left.tensor);
    parts.push_back(right.tensor);
    auto combined = m.temp(x.shape);
    auto *list = p.list(parts);
    p.add(aclnnCat, [&](auto *w, auto **e) {
      return aclnnCatGetWorkspaceSize(list, 3, combined.tensor, w, e);
    });
    m.output(m.scale(combined, c.output_scale), 2);
  });
}
} // namespace flagdnn::testing
