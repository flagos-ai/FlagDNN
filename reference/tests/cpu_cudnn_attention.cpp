/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "cpu_cudnn.hpp"
#include "common/attention.hpp"
#include "reference/cpu/attention.hpp"
#include <tuple>

namespace flagdnn::reference::tests {
namespace {
template<class Case> cpu::AttentionParameters parameters(const Case& c) {
  cpu::AttentionParameters p;
  p.q = {c.q.dimensions}; p.k = {c.k.dimensions}; p.v = {c.v.dimensions};
  p.output = {c.output.dimensions};
  if (c.bias) p.bias = cpu::AttentionTensor{c.bias->dimensions};
  p.options = {c.options.attention_scale, c.options.diagonal_band_left_bound,
      c.options.diagonal_band_right_bound,
      c.options.diagonal_alignment == t::AttentionDiagonalAlignment::kTopLeft ?
      cpu::AttentionDiagonalAlignment::kTopLeft : cpu::AttentionDiagonalAlignment::kBottomRight};
  return p;
}
Arrays attention_inputs(const std::vector<t::TestTensor>& tensors, std::size_t seed) {
  Arrays values;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    Values input(io::element_count(tensors[i]));
    for (std::size_t j = 0; j < input.size(); ++j)
      input[j] = 0.5F * float(int((j * 37 + (seed + i) * 19) % 101) - 50) /
                 float(53 + seed + i);
    values.push_back(std::move(input));
  }
  return values;
}
template<class Case> auto key(const Case& c) {
  return std::tuple{int(c.q.data_type), bool(c.bias),
      c.q.dimensions[1] != c.k.dimensions[1],
      c.options.diagonal_band_left_bound.has_value(),
      c.options.diagonal_band_right_bound.has_value(),
      int(c.options.diagonal_alignment)};
}
}
void attention(Suite& suite) {
  std::set<decltype(key(t::SdpaTestCase{}))> covered;
  for (const auto& c : t::make_sdpa_cases()) {
    // Native cuDNN SDPA accepts half/bfloat16 on this qualified stack.
    // CPU attention itself consumes float values and has no dtype branch.
    if (c.q.data_type == FLAGDNN_DATA_FLOAT32) continue;
    if (io::element_count(c.q) > 20000 || !c.stats || !covered.insert(key(c)).second) continue;
    suite.run(c.name, [&] {
      auto p = parameters(c);
      std::vector inputs{c.q, c.k, c.v};
      if (c.bias) inputs.push_back(*c.bias);
      auto r = suite.execute(inputs, {c.output, *c.stats}, t::build_sdpa_reference(c), attention_inputs(inputs, 0));
      const auto expected = cpu::evaluate_attention_forward(
          p, r.inputs[0], r.inputs[1], r.inputs[2], c.bias ? r.inputs[3] : Values{});
      compare(r.outputs[0], quantize(expected.output, c.output.data_type),
              c.output_absolute_tolerance, c.output_relative_tolerance);
      compare(r.outputs[1], expected.stats, c.stats_absolute_tolerance, c.stats_relative_tolerance);
    });
  }
  covered.clear();
  for (const auto& c : t::make_sdpa_backward_cases()) {
    // Native cuDNN SDPA accepts half/bfloat16 on this qualified stack.
    // CPU attention itself consumes float values and has no dtype branch.
    if (c.q.data_type == FLAGDNN_DATA_FLOAT32) continue;
    if (io::element_count(c.q) > 20000 || !covered.insert(key(c)).second) continue;
    suite.run(c.name, [&] {
      auto p = parameters(c);
      p.doutput = {c.doutput.dimensions};
      if (c.dbias) p.dbias = cpu::AttentionTensor{c.dbias->dimensions};
      t::SdpaTestCase f;
      f.name = c.name + "_primal"; f.q = c.q; f.k = c.k; f.v = c.v;
      f.bias = c.bias; f.output = c.output; f.stats = c.stats; f.options = c.options;
      std::vector primal_inputs{c.q, c.k, c.v};
      if (c.bias) primal_inputs.push_back(*c.bias);
      auto primal = suite.execute(primal_inputs, {c.output, c.stats}, t::build_sdpa_reference(f), attention_inputs(primal_inputs, 10));
      const auto forward = cpu::evaluate_attention_forward(p, primal.inputs[0],
          primal.inputs[1], primal.inputs[2], c.bias ? primal.inputs[3] : Values{});
      compare(primal.outputs[0], quantize(forward.output, c.output.data_type),
              c.absolute_tolerance, c.relative_tolerance);
      compare(primal.outputs[1], forward.stats, 2e-4, 2e-4);
      auto inputs = primal_inputs;
      inputs.insert(inputs.end(), {c.output, c.doutput, c.stats});
      auto values = primal.inputs;
      values.insert(values.end(), {primal.outputs[0], attention_inputs({c.doutput}, 13)[0], primal.outputs[1]});
      std::vector outputs{c.dq, c.dk, c.dv};
      if (c.dbias) outputs.push_back(*c.dbias);
      suite.check(inputs, outputs, t::build_sdpa_backward_reference(c),
          [&](const Arrays& x) {
            const auto base = primal_inputs.size();
            // Use cuDNN's saved O/stats, never a CPU-produced primal.
            const auto out = cpu::evaluate_attention_backward(p, x[0], x[1], x[2],
                x[base + 1], c.bias ? x[3] : Values{}, {x[base], x[base + 2]});
            Arrays result{out.dq, out.dk, out.dv};
            if (c.dbias) result.push_back(out.dbias);
            return result;
          }, c.absolute_tolerance, c.relative_tolerance, values);
    });
  }
}
}  // namespace flagdnn::reference::tests
