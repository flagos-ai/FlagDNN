/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "validation/functional/aclnn_extended.hpp"
#include "validation/functional/aclnn_plan.hpp"
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_fill_scalar.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_scatter_nd_update.h>
namespace flagdnn::testing {
std::unique_ptr<TestExecutable>
build_aclnn_moe_matmul(const MoeMatmulTestCase &c) {
  using namespace ascend;
  for (const auto &t : c.inputs)
    (void)dtype(t.data_type);
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(std::move(specs), [c](Plan &p) {
    const auto e = static_cast<std::int64_t>(c.offsets.size());
    const auto routed =
        c.backward ? c.inputs[0].dimensions[1] : c.outputs[0].dimensions[1];
    const auto k =
        c.backward ? c.inputs[1].dimensions[2] : c.inputs[0].dimensions[2];
    const auto n = c.outputs[0].dimensions[2];
    auto *tokens = p.ports[c.backward ? 1 : 0];
    auto *matrices = p.ports[c.backward ? 0 : 1];
    auto *out = p.ports.back();
    auto type = c.outputs[0].data_type;
    auto temporary = [&](std::vector<std::int64_t> shape, flagdnnDataType_t t) {
      return p.temporary(TestTensor{0, t, shape, dense_strides(shape)});
    };
    if (c.mode == 1) {
      auto *source = p.view(0, {c.inputs[0].dimensions[1], k});
      auto *index = p.view(3, {routed});
      tokens = temporary({routed, k}, type);
      p.add(aclnnIndexSelect, [&](auto *w, auto **exec) {
        return aclnnIndexSelectGetWorkspaceSize(source, 0, index, tokens, w,
                                                exec);
      });
    }
    if (c.mode == 2)
      out = temporary({1, routed, n}, type);
    if (c.backward) {
      auto *zero = p.scalar(0);
      p.add(aclnnInplaceFillScalar, [&](auto *w, auto **exec) {
        return aclnnInplaceFillScalarGetWorkspaceSize(out, zero, w, exec);
      });
    }
    // The reference fixture supplies expert boundaries; routing gathers and
    // scatters, and every contraction, execute through independent ACLNN ops.
    for (std::int64_t expert = 0; expert < e; ++expert) {
      const std::int64_t begin = c.offsets[expert],
                         end = expert + 1 < e ? c.offsets[expert + 1] : routed;
      const auto count = end - begin;
      if (!count)
        continue;
      auto *a = c.backward ? p.alias(tokens, {k, count}, {1, k}, begin * k)
                           : p.alias(tokens, {count, k}, {k, 1}, begin * k);
      auto *b = c.backward ? p.alias(matrices, {count, n}, {n, 1}, begin * n)
                           : p.alias(matrices, {k, n}, {n, 1}, expert * k * n);
      auto *y = c.backward ? p.alias(out, {k, n}, {n, 1}, expert * k * n)
                           : p.alias(out, {count, n}, {n, 1}, begin * n);
      p.add(aclnnMatmul, [&](auto *w, auto **exec) {
        return aclnnMatmulGetWorkspaceSize(a, b, y, 0, w, exec);
      });
    }
    if (c.mode == 2) {
      auto *token_index = p.view(3, {routed});
      auto *token_ks = p.view(4, {routed});
      auto *scaled = temporary({routed}, FLAGDNN_DATA_INT32);
      auto *index = temporary({routed}, FLAGDNN_DATA_INT32);
      auto *top = p.integer_scalar(c.top_k);
      auto *one = p.integer_scalar(1);
      p.add(aclnnMuls, [&](auto *w, auto **exec) {
        return aclnnMulsGetWorkspaceSize(token_index, top, scaled, w, exec);
      });
      p.add(aclnnAdd, [&](auto *w, auto **exec) {
        return aclnnAddGetWorkspaceSize(scaled, token_ks, one, index, w, exec);
      });
      auto *destination = p.view(c.inputs.size(), {routed, n});
      auto *source = p.alias(out, {routed, n}, {n, 1});
      if (routed == 1)
        p.copy(source, destination);
      else {
        auto *indices = p.alias(index, {routed, 1}, {1, 1});
        p.add(aclnnScatterNdUpdate, [&](auto *w, auto **exec) {
          return aclnnScatterNdUpdateGetWorkspaceSize(destination, indices,
                                                      source, w, exec);
        });
      }
    }
  });
}
} // namespace flagdnn::testing
