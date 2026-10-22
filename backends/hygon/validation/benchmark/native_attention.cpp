/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <iomanip>
#include <map>
#include <sstream>
#include <type_traits>

#include "common/attention.hpp"
#include "common/attention_runner.hpp"
#include "equivalent_baseline.hpp"
#include "functional/fp8_attention_reference.hpp"
#include "native.hpp"
#include "reference/cpu/attention.hpp"
#include "tensor_io.hpp"

namespace flagdnn::testing {
namespace {
namespace bm = benchmarking;
namespace cpu = reference::cpu;
namespace io = validation::hygon::tensor_io;
namespace host = hygon_functional::host;
using Values = std::vector<std::vector<float>>;
using Limits = std::vector<std::pair<double, double>>;
using Json = native::json::Value;
void require(bool condition, const char* text) {
  if (!condition) throw std::runtime_error(text);
}
std::vector<float> values(const TestTensor& t, std::size_t seed, float scale) {
  std::vector<float> out(io::element_count(t));
  for (std::size_t i = 0; i < out.size(); ++i)
    out[i] =
        scale * float(int((i * 37 + seed * 19) % 101) - 50) / float(53 + seed);
  return out;
}
Values quantize(Values x, const std::vector<TestTensor>& tensors) {
  require(x.size() == tensors.size(), "attention input/output arity differs");
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = io::decode(io::encode(x[i], tensors[i].data_type),
                      tensors[i].data_type, x[i].size());
  return x;
}
template <class C>
cpu::AttentionParameters parameters(const C& c) {
  cpu::AttentionParameters p;
  p.q = {c.q.dimensions};
  p.k = {c.k.dimensions};
  p.v = {c.v.dimensions};
  p.output = {c.output.dimensions};
  if constexpr (requires { c.bias; })
    if (c.bias) p.bias = cpu::AttentionTensor{c.bias->dimensions};
  if constexpr (requires { c.doutput; }) p.doutput = {c.doutput.dimensions};
  if constexpr (requires { c.dbias; })
    if (c.dbias) p.dbias = cpu::AttentionTensor{c.dbias->dimensions};
  p.options = {
      c.options.attention_scale, c.options.diagonal_band_left_bound,
      c.options.diagonal_band_right_bound,
      c.options.diagonal_alignment == AttentionDiagonalAlignment::kTopLeft
          ? cpu::AttentionDiagonalAlignment::kTopLeft
          : cpu::AttentionDiagonalAlignment::kBottomRight};
  return p;
}
template <class C>
constexpr bool backward = std::is_same_v<C, SdpaBackwardTestCase> ||
                          std::is_same_v<C, SdpaFp8BackwardTestCase>;
template <class C>
constexpr bool fp8 = std::is_same_v<C, SdpaFp8TestCase> ||
                     std::is_same_v<C, SdpaFp8BackwardTestCase>;
template <class C>
std::vector<Fp8Scalar> scalars(const C& c) {
  if constexpr (std::is_same_v<C, SdpaFp8TestCase>)
    return {c.descale_q, c.descale_k, c.descale_v,
            c.descale_s, c.scale_s,   c.scale_o};
  else if constexpr (std::is_same_v<C, SdpaFp8BackwardTestCase>)
    return {c.descale_q,       c.descale_k, c.descale_v,  c.descale_o,
            c.descale_doutput, c.descale_s, c.descale_dp, c.scale_s,
            c.scale_dq,        c.scale_dk,  c.scale_dv,   c.scale_dp};
  else
    return {};
}
template <class C>
void attributes(const C& c, const Json& a) {
  const auto optional_int = [&](const char* key, const auto& value) {
    require(value ? !a.at(key).is_null() && a.at(key).as_int() == *value
                  : a.at(key).is_null(),
            "attention mask differs");
  };
  optional_int("diagonal_band_left_bound", c.options.diagonal_band_left_bound);
  optional_int("diagonal_band_right_bound",
               c.options.diagonal_band_right_bound);
  require(
      a.at("diagonal_alignment").as_string() ==
          (c.options.diagonal_alignment == AttentionDiagonalAlignment::kTopLeft
               ? "top_left"
               : "bottom_right"),
      "attention alignment differs");
  if (c.options.attention_scale) {
    // The historical collector serializes this float with six significant
    // digits. Compare that exact representation, not an arbitrary tolerance.
    std::ostringstream text;
    text << std::setprecision(6) << *c.options.attention_scale;
    require(a.at("attention_scale").as_double() == std::stod(text.str()),
            "attention scale differs");
  } else
    require(a.at("attention_scale").is_null(),
            "attention default scale differs");
  std::size_t expected = 4;
  if constexpr (backward<C>) {
    ++expected;
    require(a.at("backward_inputs_from_cudnn_forward").as_bool(),
            "attention primal provenance missing");
  }
  if constexpr (std::is_same_v<C, SdpaBackwardTestCase>) {
    ++expected;
    require(a.at("deterministic").as_bool() == c.deterministic,
            "attention determinism differs");
  }
  if constexpr (fp8<C>) {
    ++expected;
    const auto v = scalars(c);
    const auto& r = a.at("fp8_scalar_values_input_order").as_array();
    require(r.size() == v.size(), "attention scalar count differs");
    for (std::size_t i = 0; i < v.size(); ++i)
      require(r[i].as_double() == v[i].value, "attention FP8 scale differs");
  }
  require(a.as_object().size() == expected,
          "attention attribute count differs");
}
class Executable final : public bm::BenchmarkExecutable {
  std::unique_ptr<TestExecutable> value_;

 public:
  explicit Executable(std::unique_ptr<TestExecutable> value)
      : value_(std::move(value)) {}
  void prepare(std::span<const flagdnnBinding_t> b,
               flagdnnStream_t s) override {
    value_->prepare(b, s);
  }
  std::size_t workspace_size() const noexcept override {
    return value_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> b, void* w, std::size_t n,
               flagdnnStream_t s) override {
    value_->execute(b, w, n, s);
  }
};
struct Prepared {
  std::vector<TestTensor> inputs, outputs;
  Limits limits;
  std::function<Values()> generate;
  std::function<Values(const Values&)> oracle;
  std::function<std::unique_ptr<TestExecutable>(flagdnn::Handle&)> build;
  std::function<void(const Json&)> validate;
};
template <class C>
Prepared prepare(C c) {
  c.autotune = true;
  Prepared p;
  p.inputs = {c.q, c.k, c.v};
  if constexpr (backward<C>) {
    // Preserve the measured NVIDIA catalog's dO, O, stats binding order.
    p.inputs.insert(p.inputs.end(), {c.doutput, c.output, c.stats});
    p.outputs = {c.dq, c.dk, c.dv};
    if constexpr (fp8<C>) {
      p.outputs.insert(p.outputs.end(),
                       {c.amax_dq, c.amax_dk, c.amax_dv, c.amax_dp});
      p.limits = Limits(
          3, {c.gradient_absolute_tolerance, c.gradient_relative_tolerance});
      p.limits.insert(p.limits.end(), 4,
                      {c.amax_absolute_tolerance, c.amax_relative_tolerance});
    } else {
      if (c.dbias) p.outputs.push_back(*c.dbias);
      p.limits = Limits(p.outputs.size(),
                        {c.absolute_tolerance, c.relative_tolerance});
    }
  } else {
    p.outputs = {c.output};
    p.limits = {{c.output_absolute_tolerance, c.output_relative_tolerance}};
    if constexpr (fp8<C>) {
      p.outputs.insert(p.outputs.end(), {c.amax_s, c.amax_o});
      p.limits.insert(p.limits.end(), 2,
                      {c.amax_absolute_tolerance, c.amax_relative_tolerance});
    }
    if (c.stats) {
      p.outputs.push_back(*c.stats);
      p.limits.emplace_back(c.stats_absolute_tolerance,
                            c.stats_relative_tolerance);
    }
  }
  for (const auto& scalar : scalars(c)) p.inputs.push_back(scalar.tensor);
  if constexpr (requires { c.bias; })
    if (c.bias) p.inputs.push_back(*c.bias);
  p.generate = [c, inputs = p.inputs] {
    constexpr auto seed =
        fp8<C> ? (backward<C> ? 30 : 20) : (backward<C> ? 10 : 0);
    constexpr float scale = fp8<C> ? 1.0F : 0.5F;
    Values v{values(c.q, seed, scale), values(c.k, seed + 1, scale),
             values(c.v, seed + 2, scale)};
    if constexpr (backward<C>)
      v.insert(v.end(), {values(c.doutput, seed + 3, scale / 2),
                         std::vector<float>(io::element_count(c.output)),
                         std::vector<float>(io::element_count(c.stats))});
    for (const auto& s : scalars(c)) v.push_back({s.value});
    if constexpr (requires { c.bias; })
      if (c.bias)
        v.push_back(values(*c.bias, seed + (backward<C> ? 4 : 3),
                           fp8<C> ? 0.125F : 0.25F));
    v = quantize(std::move(v), inputs);
    if constexpr (backward<C>) {
      if constexpr (fp8<C>) {
        SdpaFp8TestCase f;
        f.q = c.q;
        f.k = c.k;
        f.v = c.v;
        f.output = c.output;
        f.stats = c.stats;
        f.options = c.options;
        f.descale_q = c.descale_q;
        f.descale_k = c.descale_k;
        f.descale_v = c.descale_v;
        f.descale_s = c.descale_s;
        f.scale_s = c.scale_s;
        f.scale_o.value = 1.0F / c.descale_o.value;
        const auto primal = host::fp8_forward_reference(f, v);
        v[4] = primal[0];
        v[5] = primal[1];
      } else {
        const auto primal = cpu::evaluate_attention_forward(
            parameters(c), v[0], v[1], v[2],
            c.bias ? v[6] : std::vector<float>{});
        v[4] = primal.output;
        v[5] = primal.stats;
      }
      v = quantize(std::move(v), inputs);
    }
    return v;
  };
  p.oracle = [c, outputs = p.outputs](const Values& v) {
    Values out;
    if constexpr (fp8<C>) {
      if constexpr (backward<C>) {
        auto reordered = v;
        std::swap(reordered[3], reordered[4]);
        out = host::fp8_backward_reference(c, reordered);
      } else {
        auto r = host::fp8_forward_reference(c, v);
        out = {r[0], r[c.stats ? 2 : 1], r[c.stats ? 3 : 2]};
        if (c.stats) out.push_back(r[1]);
      }
    } else if constexpr (backward<C>) {
      const auto r = cpu::evaluate_attention_backward(
          parameters(c), v[0], v[1], v[2], v[3],
          c.bias ? v[6] : std::vector<float>{}, {v[4], v[5]});
      out = {r.dq, r.dk, r.dv};
      if (c.dbias) out.push_back(r.dbias);
    } else {
      const auto r =
          cpu::evaluate_attention_forward(parameters(c), v[0], v[1], v[2],
                                          c.bias ? v[3] : std::vector<float>{});
      out = {r.output};
      if (c.stats) out.push_back(r.stats);
    }
    return quantize(std::move(out), outputs);
  };
  p.build = [c](flagdnn::Handle& h) {
    if constexpr (std::is_same_v<C, SdpaTestCase>)
      return build_flagdnn_sdpa(h, c);
    else if constexpr (std::is_same_v<C, SdpaBackwardTestCase>)
      return build_flagdnn_sdpa_backward(h, c);
    else if constexpr (std::is_same_v<C, SdpaFp8TestCase>)
      return build_flagdnn_sdpa_fp8(h, c);
    else
      return build_flagdnn_sdpa_fp8_backward(h, c);
  };
  p.validate = [c](const Json& value) { attributes(c, value); };
  return p;
}
template <class C>
int run(int argc, char** argv, std::string op, std::vector<C> catalog) {
  std::vector<bm::BenchmarkCase> cases;
  std::map<std::string, Prepared> prepared;
  for (const auto& c : catalog) {
    if (c.q.data_type == FLAGDNN_DATA_FLOAT32)
      continue;  // Same native NVIDIA performance catalog.
    auto p = prepare(c);
    bm::BenchmarkCase spec;
    spec.name = c.name;
    spec.operation = bm::Operation::kGraph;
    spec.benchmark = {5, 10, 20};
    spec.output_count = p.outputs.size();
    spec.absolute_tolerance = p.limits[0].first;
    spec.relative_tolerance = p.limits[0].second;
    for (const auto& t : p.inputs)
      spec.tensors.push_back(
          {t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
    for (const auto& t : p.outputs)
      spec.tensors.push_back(
          {t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
    cases.push_back(std::move(spec));
    require(prepared.emplace(c.name, std::move(p)).second,
            "duplicate attention case");
  }
  std::string marker = "FLAGDNN_";
  for (char c : op) marker += char(std::toupper(static_cast<unsigned char>(c)));
  marker += "_BENCHMARK";
  return bm::run_hygon_benchmark_suite(
      argc, argv, cases, marker,
      [&](flagdnn::Handle& h, const bm::BenchmarkCase& s) {
        return std::make_unique<Executable>(prepared.at(s.name).build(h));
      },
      [&](const bm::BenchmarkCase& s, const Values& v) {
        return prepared.at(s.name).oracle(v);
      },
      [&](const bm::BenchmarkCase& s) {
        const auto& p = prepared.at(s.name);
        bm::NativeBaselineMetadata m;
        m.operation = op;
        m.input_count = p.inputs.size();
        m.output_count = p.outputs.size();
        m.generate_inputs = p.generate;
        m.validate_attributes = p.validate;
        m.output_tolerances = p.limits;
        return m;
      });
}
}  // namespace
int run_attention_benchmark_test(int argc, char** argv,
                                 AttentionBenchmarkOperation op) {
  switch (op) {
    case AttentionBenchmarkOperation::kForward:
      return run(argc, argv, "sdpa", make_sdpa_benchmark_cases());
    case AttentionBenchmarkOperation::kBackward:
      return run(argc, argv, "sdpa_backward",
                 make_sdpa_backward_benchmark_cases());
    case AttentionBenchmarkOperation::kFp8Forward:
      return run(argc, argv, "sdpa_fp8", make_sdpa_fp8_benchmark_cases());
    case AttentionBenchmarkOperation::kFp8Backward:
      return run(argc, argv, "sdpa_fp8_backward",
                 make_sdpa_fp8_backward_benchmark_cases());
  }
  throw std::invalid_argument("unknown attention benchmark operation");
}
}  // namespace flagdnn::testing
