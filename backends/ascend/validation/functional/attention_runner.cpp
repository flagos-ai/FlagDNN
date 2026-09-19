/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/attention.hpp"
#include "validation/functional/aclnn_math.hpp"
#include "validation/functional/paired.hpp"
#include <aclnnop/aclnn_masked_fill_scalar.h>
#include <limits>
namespace flagdnn::testing {
namespace {
using namespace ascend;
struct ForwardCase : SdpaTestCase {
  std::vector<TestTensor> inputs, outputs;
};
struct BackwardCase : SdpaBackwardTestCase {
  std::vector<TestTensor> inputs, outputs;
};
struct AttentionMath {
  Math m;
  Math::Value repeat_heads(Math::Value a, std::int64_t heads) {
    if (a.shape[1] == heads)
      return a;
    const auto b = a.shape[0], h = a.shape[1], s = a.shape[2], d = a.shape[3];
    return m.reshape(
        m.expand(m.reshape(a, {b, h, 1, s, d}), {b, h, heads / h, s, d}),
        {b, heads, s, d});
  }
  Math::Value collapse_heads(Math::Value a, std::int64_t heads) {
    if (a.shape[1] == heads)
      return a;
    const auto b = a.shape[0], h = a.shape[1], s = a.shape[2], d = a.shape[3];
    return m.reshape(
        m.reduce(m.reshape(a, {b, heads, h / heads, s, d}), {2}, false),
        {b, heads, s, d});
  }
  Math::Value scores(Math::Value q, Math::Value k,
                     const AttentionOptions &options,
                     std::optional<std::size_t> bias) {
    auto a = m.scale(
        m.mm(q, m.transpose(k)),
        options.attention_scale.value_or(1.0F / std::sqrt(float(q.shape[3]))));
    if (bias) {
      a = m.add(a, m.input(*bias));
    }
    const auto sq = q.shape[2], sk = k.shape[2];
    if (options.diagonal_band_left_bound || options.diagonal_band_right_bound) {
      TestTensor spec{
          0, FLAGDNN_DATA_BOOLEAN, {1, 1, sq, sk}, {sq * sk, sq * sk, sk, 1}};
      auto *mask = m.p.temporary(spec);
      std::vector<std::uint8_t> bits(sq * sk);
      const auto shift =
          options.diagonal_alignment == AttentionDiagonalAlignment::kBottomRight
              ? sk - sq
              : 0;
      for (std::int64_t i = 0; i < sq; ++i)
        for (std::int64_t j = 0; j < sk; ++j)
          bits[i * sk + j] =
              (options.diagonal_band_left_bound &&
               j < i + shift - *options.diagonal_band_left_bound) ||
              (options.diagonal_band_right_bound &&
               j > i + shift + *options.diagonal_band_right_bound);
      m.p.initialize(mask, bits);
      auto *negative = m.p.scalar(-std::numeric_limits<double>::infinity());
      m.p.add(aclnnInplaceMaskedFillScalar, [&](auto *w, auto **e) {
        return aclnnInplaceMaskedFillScalarGetWorkspaceSize(a.tensor, mask,
                                                            negative, w, e);
      });
    }
    return a;
  }
};
std::vector<std::vector<float>> inputs(const std::vector<TestTensor> &tensors) {
  std::vector<std::vector<float>> result;
  for (std::size_t k = 0; k < tensors.size(); ++k) {
    std::vector<float> v(io::element_count(tensors[k]));
    for (std::size_t i = 0; i < v.size(); ++i)
      v[i] = float(int((i * 17 + k * 7) % 61) - 30) / 57.0F;
    result.push_back(std::move(v));
  }
  return result;
}
std::unique_ptr<TestExecutable> forward_reference(const ForwardCase &c) {
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(specs, [c](Plan &p) {
    AttentionMath a{{p}};
    auto q = a.m.input(0), k = a.repeat_heads(a.m.input(1), q.shape[1]),
         v = a.repeat_heads(a.m.input(2), q.shape[1]);
    auto scores = a.scores(
        q, k, c.options, c.bias ? std::optional<std::size_t>(3) : std::nullopt);
    auto lse = a.m.lse(scores);
    auto probability = a.m.exp(a.m.add(scores, lse, -1));
    a.m.output(a.m.mm(probability, v), c.inputs.size());
    if (c.stats)
      a.m.output(lse, c.inputs.size() + 1);
  });
}
std::unique_ptr<TestExecutable> backward_reference(const BackwardCase &c) {
  auto specs = c.inputs;
  specs.insert(specs.end(), c.outputs.begin(), c.outputs.end());
  return std::make_unique<Plan>(specs, [c](Plan &p) {
    AttentionMath a{{p}};
    const auto saved = c.bias ? 4U : 3U;
    auto q = a.m.input(0), k = a.repeat_heads(a.m.input(1), q.shape[1]),
         v = a.repeat_heads(a.m.input(2), q.shape[1]);
    auto scores = a.scores(
        q, k, c.options, c.bias ? std::optional<std::size_t>(3) : std::nullopt);
    auto lse = a.m.lse(scores);
    auto probability = a.m.exp(a.m.add(scores, lse, -1));
    a.m.output(a.m.mm(probability, v), saved);
    a.m.output(lse, saved + 2);
    // Saved forward activations are initialized once before measuring backward.
    p.run_initialization();
    q = a.m.input(0);
    k = a.repeat_heads(a.m.input(1), q.shape[1]);
    v = a.repeat_heads(a.m.input(2), q.shape[1]);
    scores = a.scores(q, k, c.options,
                      c.bias ? std::optional<std::size_t>(3) : std::nullopt);
    probability = a.m.exp(a.m.add(scores, a.m.input(saved + 2), -1));
    auto dy = a.m.input(saved + 1), output = a.m.input(saved);
    auto delta = a.m.reduce(a.m.mul(dy, output), {3}, false);
    auto ds =
        a.m.mul(probability, a.m.add(a.m.mm(dy, a.m.transpose(v)), delta, -1));
    const auto scale =
        c.options.attention_scale.value_or(1.0F / std::sqrt(float(q.shape[3])));
    const auto out = c.inputs.size();
    a.m.output(a.m.scale(a.m.mm(ds, k), scale), out);
    a.m.output(a.collapse_heads(a.m.scale(a.m.mm(a.m.transpose(ds), q), scale),
                                c.k.dimensions[1]),
               out + 1);
    a.m.output(a.collapse_heads(a.m.mm(a.m.transpose(probability), dy),
                                c.v.dimensions[1]),
               out + 2);
    if (c.dbias) {
      std::vector<std::int64_t> axes;
      const auto &shape = c.dbias->dimensions;
      for (std::size_t i = 0; i < 4; ++i)
        if (shape[i] == 1 && ds.shape[i] != 1)
          axes.push_back(i);
      a.m.output(a.m.reduce(ds, axes, false), out + 3);
    }
  });
}
} // namespace
int run_sdpa_functional_test(int argc, char **argv,
                             std::span<const SdpaTestCase> cases) {
  std::vector<ForwardCase> paired;
  for (const auto &c : cases) {
    ForwardCase p{c, {c.q, c.k, c.v}, {c.output}};
    if (c.bias)
      p.inputs.push_back(*c.bias);
    if (c.stats)
      p.outputs.push_back(*c.stats);
    paired.push_back(std::move(p));
  }
  return ascend::run_paired_cases<ForwardCase>(
      argc, argv, paired, "FLAGDNN_SDPA_CASE", build_flagdnn_sdpa,
      [](const ForwardCase &c) { return inputs(c.inputs); }, forward_reference,
      [](const ForwardCase &c, std::size_t i) {
        return PairedTolerance{
            i ? c.stats_absolute_tolerance : c.output_absolute_tolerance,
            i ? c.stats_relative_tolerance : c.output_relative_tolerance};
      },
      std::getenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK") != nullptr);
}
int run_sdpa_backward_functional_test(
    int argc, char **argv, std::span<const SdpaBackwardTestCase> cases) {
  std::vector<BackwardCase> paired;
  for (const auto &c : cases) {
    BackwardCase p{c, {c.q, c.k, c.v}, {c.dq, c.dk, c.dv}};
    if (c.bias)
      p.inputs.push_back(*c.bias);
    p.inputs.insert(p.inputs.end(), {c.output, c.doutput, c.stats});
    if (c.dbias)
      p.outputs.push_back(*c.dbias);
    paired.push_back(std::move(p));
  }
  return ascend::run_paired_cases<BackwardCase>(
      argc, argv, paired, "FLAGDNN_SDPA_BACKWARD_CASE",
      build_flagdnn_sdpa_backward,
      [](const BackwardCase &c) { return inputs(c.inputs); },
      backward_reference,
      [](const BackwardCase &c, std::size_t) {
        return PairedTolerance{c.absolute_tolerance, c.relative_tolerance};
      },
      std::getenv("FLAGDNN_ASCEND_PAIRED_BENCHMARK") != nullptr);
}
} // namespace flagdnn::testing
