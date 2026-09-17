/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/attention_runner.hpp"
#include "attention_reference.hpp"
#include "common/attention.hpp"
#include "fp8_attention_reference.hpp"
#include "hipdnn_reference.hpp"
#include "host_runner.hpp"
#include "reference/cpu/attention.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace hv = validation::hygon;

constexpr int kSkipReturnCode = 77;

void append_tensor(std::vector<hv::ReferenceTensor> &result,
                   const TestTensor &tensor) {
  result.push_back(hv::as_reference_tensor(tensor));
}

void append_scalar(std::vector<hv::ReferenceTensor> &result,
                   const Fp8Scalar &scalar) {
  append_tensor(result, scalar.tensor);
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(7);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  if (test_case.stats.has_value()) {
    append_tensor(result, *test_case.stats);
  }
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaBackwardTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(12);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  append_tensor(result, test_case.doutput);
  append_tensor(result, test_case.stats);
  append_tensor(result, test_case.dq);
  append_tensor(result, test_case.dk);
  append_tensor(result, test_case.dv);
  if (test_case.dbias.has_value()) {
    append_tensor(result, *test_case.dbias);
  }
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaFp8TestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(16);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  append_scalar(result, test_case.descale_q);
  append_scalar(result, test_case.descale_k);
  append_scalar(result, test_case.descale_v);
  append_scalar(result, test_case.descale_s);
  append_scalar(result, test_case.scale_s);
  append_scalar(result, test_case.scale_o);
  if (test_case.bias.has_value()) {
    append_tensor(result, *test_case.bias);
  }
  append_tensor(result, test_case.output);
  if (test_case.stats.has_value()) {
    append_tensor(result, *test_case.stats);
  }
  append_tensor(result, test_case.amax_s);
  append_tensor(result, test_case.amax_o);
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const SdpaFp8BackwardTestCase &test_case) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(28);
  append_tensor(result, test_case.q);
  append_tensor(result, test_case.k);
  append_tensor(result, test_case.v);
  append_tensor(result, test_case.output);
  append_tensor(result, test_case.doutput);
  append_tensor(result, test_case.stats);
  append_scalar(result, test_case.descale_q);
  append_scalar(result, test_case.descale_k);
  append_scalar(result, test_case.descale_v);
  append_scalar(result, test_case.descale_o);
  append_scalar(result, test_case.descale_doutput);
  append_scalar(result, test_case.descale_s);
  append_scalar(result, test_case.descale_dp);
  append_scalar(result, test_case.scale_s);
  append_scalar(result, test_case.scale_dq);
  append_scalar(result, test_case.scale_dk);
  append_scalar(result, test_case.scale_dv);
  append_scalar(result, test_case.scale_dp);
  append_tensor(result, test_case.dq);
  append_tensor(result, test_case.dk);
  append_tensor(result, test_case.dv);
  append_tensor(result, test_case.amax_dq);
  append_tensor(result, test_case.amax_dk);
  append_tensor(result, test_case.amax_dv);
  append_tensor(result, test_case.amax_dp);
  return result;
}

void emit_skip(const hv::HipdnnAttentionOperation &operation,
               std::string_view case_name, std::string_view reason,
               std::span<const hv::ReferenceTensor> tensors) {
  std::cout << "[SKIP][hipdnn] op="
            << hv::hipdnn_attention_kind_name(operation.kind)
            << " case=" << case_name << " reason=" << reason << ' '
            << hv::hipdnn_environment() << ' '
            << hv::describe_reference_tensors(tensors) << std::endl;
}

template <typename Case, typename Validate, typename MakeTensors>
int run_unavailable_suite(int argc, char **argv, std::span<const Case> cases,
                          std::string_view suite_name,
                          std::string_view filter_environment,
                          hv::HipdnnAttentionKind kind, Validate &&validate,
                          MakeTensors &&make_tensors) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    const char *filter = std::getenv(std::string(filter_environment).c_str());
    const hv::HipdnnAttentionOperation operation{kind};
    std::size_t matched = 0;
    std::size_t skipped = 0;
    for (const Case &test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++matched;
      validate(test_case);
      const std::vector<hv::ReferenceTensor> tensors = make_tensors(test_case);
      const hv::HipdnnCapability capability =
          hv::hipdnn_attention_capability(operation, tensors);
      hv::require_valid_hipdnn_adapter_contract(
          capability, hv::hipdnn_attention_kind_name(operation.kind));
      if (capability.supported) {
        throw std::logic_error(
            "hipDNN attention capability unexpectedly became supported");
      }
      emit_skip(operation, test_case.name, capability.reason, tensors);
      ++skipped;
    }
    if (matched == 0) {
      throw std::runtime_error(std::string(suite_name) +
                               " filter matched no test cases");
    }
    std::cout << suite_name << ": SKIP cases=" << matched
              << " executed=0 skipped=" << skipped << std::endl;
    return kSkipReturnCode;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

} // namespace

namespace host = hygon_functional::host;
namespace cpu = reference::cpu;
// Match the NVIDIA attention runner's deterministic seeds and amplitudes.
std::vector<float> attention_values(const TestTensor &tensor,
                                    std::size_t tensor_index, float scale) {
  std::vector<float> result(host::io::element_count(tensor));
  for (std::size_t i = 0; i < result.size(); ++i) {
    const int centered = int((i * 37 + tensor_index * 19) % 101) - 50;
    result[i] = scale * float(centered) / float(53 + tensor_index);
  }
  return result;
}
template <class Case>
cpu::AttentionParameters cpu_attention_parameters(const Case &c) {
  cpu::AttentionParameters p;
  p.q.dimensions = c.q.dimensions;
  p.k.dimensions = c.k.dimensions;
  p.v.dimensions = c.v.dimensions;
  p.output.dimensions = c.output.dimensions;
  if (c.bias)
    p.bias = cpu::AttentionTensor{c.bias->dimensions};
  if constexpr (requires { c.dbias; }) {
    if (c.dbias)
      p.dbias = cpu::AttentionTensor{c.dbias->dimensions};
    p.doutput.dimensions = c.doutput.dimensions;
  }
  p.options.attention_scale = c.options.attention_scale;
  p.options.diagonal_band_left_bound = c.options.diagonal_band_left_bound;
  p.options.diagonal_band_right_bound = c.options.diagonal_band_right_bound;
  p.options.diagonal_alignment =
      c.options.diagonal_alignment == AttentionDiagonalAlignment::kTopLeft
          ? cpu::AttentionDiagonalAlignment::kTopLeft
          : cpu::AttentionDiagonalAlignment::kBottomRight;
  return p;
}
int run_sdpa_functional_test(int argc, char **argv,
                             std::span<const SdpaTestCase> cases) {
  return host::run_suite(
      argc, argv, cases, "sdpa", false,
      [](const SdpaTestCase &c, const auto &handle, auto &stream) {
        validate_sdpa_case(c);
        std::vector<TestTensor> inputs{c.q, c.k, c.v}, outputs{c.output};
        if (c.bias)
          inputs.push_back(*c.bias);
        if (c.stats)
          outputs.push_back(*c.stats);
        host::Values values{attention_values(c.q, 0, 0.5F),
                            attention_values(c.k, 1, 0.5F),
                            attention_values(c.v, 2, 0.5F)};
        if (c.bias)
          values.push_back(attention_values(*c.bias, 3, 0.25F));
        const auto p = cpu_attention_parameters(c);
        std::vector<std::pair<double, double>> limits{
            {c.output_absolute_tolerance, c.output_relative_tolerance}};
        if (c.stats)
          limits.emplace_back(c.stats_absolute_tolerance,
                              c.stats_relative_tolerance);
        host::run_case(
            c.name, inputs, outputs, values, handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_sdpa(h, c); },
            [&](const host::Values &v) {
              const auto ref = cpu::evaluate_attention_forward(
                  p, v[0], v[1], v[2],
                  c.bias ? std::span<const float>(v[3])
                         : std::span<const float>{});
              host::Values out{ref.output};
              if (c.stats)
                out.push_back(ref.stats);
              return out;
            },
            -1, -1, limits);
      });
}
int run_sdpa_backward_functional_test(
    int argc, char **argv, std::span<const SdpaBackwardTestCase> cases) {
  return host::run_suite(
      argc, argv, cases, "sdpa_backward", false,
      [](const SdpaBackwardTestCase &c, const auto &handle, auto &stream) {
        validate_sdpa_backward_case(c);
        std::vector<TestTensor> inputs{c.q,      c.k,       c.v,
                                       c.output, c.doutput, c.stats},
            outputs{c.dq, c.dk, c.dv};
        if (c.bias)
          inputs.push_back(*c.bias);
        if (c.dbias)
          outputs.push_back(*c.dbias);
        host::Values values{
            attention_values(c.q, 10, 0.5F),
            attention_values(c.k, 11, 0.5F),
            attention_values(c.v, 12, 0.5F),
            std::vector<float>(host::io::element_count(c.output)),
            attention_values(c.doutput, 13, 0.25F),
            std::vector<float>(host::io::element_count(c.stats))};
        if (c.bias)
          values.push_back(attention_values(*c.bias, 14, 0.25F));
        for (std::size_t i = 0; i < inputs.size(); ++i)
          values[i] =
              host::io::decode(host::io::encode(values[i], inputs[i].data_type),
                               inputs[i].data_type, values[i].size());
        const auto p = cpu_attention_parameters(c);
        const auto primal = cpu::evaluate_attention_forward(
            p, values[0], values[1], values[2],
            c.bias ? std::span<const float>(values[6])
                   : std::span<const float>{});
        values[3] = primal.output;
        values[5] = primal.stats;
        host::run_case(
            c.name, inputs, outputs, values, handle, stream,
            [&](flagdnn::Handle &h) {
              return build_flagdnn_sdpa_backward(h, c);
            },
            [&](const host::Values &v) {
              const auto ref = cpu::evaluate_attention_backward(
                  p, v[0], v[1], v[2], v[4],
                  c.bias ? std::span<const float>(v[6])
                         : std::span<const float>{},
                  primal);
              host::Values out{ref.dq, ref.dk, ref.dv};
              if (c.dbias)
                out.push_back(ref.dbias);
              return out;
            },
            c.absolute_tolerance, c.relative_tolerance);
      });
}

int run_sdpa_fp8_functional_test(int argc, char **argv,
                                 std::span<const SdpaFp8TestCase> cases) {
  return host::run_suite(
      argc, argv, cases, "sdpa_fp8", false,
      [](const SdpaFp8TestCase &c, const auto &handle, auto &stream) {
        validate_sdpa_fp8_case(c);
        std::vector<TestTensor> inputs{c.q, c.k, c.v}, outputs{c.output};
        host::Values values{attention_values(c.q, 20, 1.0F),
                            attention_values(c.k, 21, 1.0F),
                            attention_values(c.v, 22, 1.0F)};
        for (const auto *scalar : {&c.descale_q, &c.descale_k, &c.descale_v,
                                   &c.descale_s, &c.scale_s, &c.scale_o}) {
          inputs.push_back(scalar->tensor);
          values.push_back({scalar->value});
        }
        if (c.bias) {
          inputs.push_back(*c.bias);
          values.push_back(attention_values(*c.bias, 23, 0.125F));
        }
        std::vector<std::pair<double, double>> limits{
            {c.output_absolute_tolerance, c.output_relative_tolerance}};
        if (c.stats) {
          outputs.push_back(*c.stats);
          limits.emplace_back(c.stats_absolute_tolerance,
                              c.stats_relative_tolerance);
        }
        outputs.push_back(c.amax_s);
        outputs.push_back(c.amax_o);
        limits.insert(limits.end(), 2,
                      {c.amax_absolute_tolerance, c.amax_relative_tolerance});
        host::run_case(
            c.name, inputs, outputs, values, handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_sdpa_fp8(h, c); },
            [&](const host::Values &v) {
              return host::fp8_forward_reference(c, v);
            },
            -1, -1, limits);
      });
}
int run_sdpa_fp8_backward_functional_test(
    int argc, char **argv, std::span<const SdpaFp8BackwardTestCase> cases) {
  return host::run_suite(
      argc, argv, cases, "sdpa_fp8_backward", false,
      [](const SdpaFp8BackwardTestCase &c, const auto &handle, auto &stream) {
        validate_sdpa_fp8_backward_case(c);
        std::vector<TestTensor> inputs{c.q,      c.k,       c.v,
                                       c.output, c.doutput, c.stats},
            outputs{c.dq,      c.dk,      c.dv,     c.amax_dq,
                    c.amax_dk, c.amax_dv, c.amax_dp};
        host::Values values{
            attention_values(c.q, 30, 1.0F),
            attention_values(c.k, 31, 1.0F),
            attention_values(c.v, 32, 1.0F),
            std::vector<float>(host::io::element_count(c.output)),
            attention_values(c.doutput, 33, 0.5F),
            std::vector<float>(host::io::element_count(c.stats))};
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] =
              host::io::decode(host::io::encode(values[i], inputs[i].data_type),
                               inputs[i].data_type, values[i].size());
        SdpaFp8TestCase primal;
        primal.q = c.q;
        primal.k = c.k;
        primal.v = c.v;
        primal.output = c.output;
        primal.stats = c.stats;
        primal.options = c.options;
        primal.descale_q = c.descale_q;
        primal.descale_k = c.descale_k;
        primal.descale_v = c.descale_v;
        primal.descale_s = c.descale_s;
        primal.scale_s = c.scale_s;
        primal.scale_o.value = 1.0F / c.descale_o.value;
        const auto forward = host::fp8_forward_reference(primal, values);
        values[3] = forward[0];
        values[5] = forward[1];
        for (const auto *scalar :
             {&c.descale_q, &c.descale_k, &c.descale_v, &c.descale_o,
              &c.descale_doutput, &c.descale_s, &c.descale_dp, &c.scale_s,
              &c.scale_dq, &c.scale_dk, &c.scale_dv, &c.scale_dp}) {
          inputs.push_back(scalar->tensor);
          values.push_back({scalar->value});
        }
        std::vector<std::pair<double, double>> limits(
            3, {c.gradient_absolute_tolerance, c.gradient_relative_tolerance});
        limits.insert(limits.end(), 4,
                      {c.amax_absolute_tolerance, c.amax_relative_tolerance});
        host::run_case(
            c.name, inputs, outputs, values, handle, stream,
            [&](flagdnn::Handle &h) {
              return build_flagdnn_sdpa_fp8_backward(h, c);
            },
            [&](const host::Values &v) {
              return host::fp8_backward_reference(c, v);
            },
            -1, -1, limits);
      });
}

template <class Case, class Validate>
int run_attention_benchmark_catalog(int argc, char **argv,
                                    std::vector<Case> cases,
                                    std::string_view marker,
                                    hv::HipdnnAttentionKind kind,
                                    Validate validate) {
  std::erase_if(cases, [](const auto &value) {
    return value.q.data_type == FLAGDNN_DATA_FLOAT32;
  });
  return run_unavailable_suite(
      argc, argv, std::span<const Case>(cases), marker,
      "FLAGDNN_BENCHMARK_CASE", kind, validate,
      [](const Case &value) { return reference_tensors(value); });
}
int run_attention_benchmark_test(int argc, char **argv,
                                 AttentionBenchmarkOperation operation) {
  switch (operation) {
  case AttentionBenchmarkOperation::kForward:
    return run_attention_benchmark_catalog(
        argc, argv, make_sdpa_benchmark_cases(), "FLAGDNN_SDPA_BENCHMARK",
        hv::HipdnnAttentionKind::kSdpa, validate_sdpa_case);
  case AttentionBenchmarkOperation::kBackward:
    return run_attention_benchmark_catalog(
        argc, argv, make_sdpa_backward_benchmark_cases(),
        "FLAGDNN_SDPA_BACKWARD_BENCHMARK",
        hv::HipdnnAttentionKind::kSdpaBackward, validate_sdpa_backward_case);
  case AttentionBenchmarkOperation::kFp8Forward:
    return run_attention_benchmark_catalog(
        argc, argv, make_sdpa_fp8_benchmark_cases(),
        "FLAGDNN_SDPA_FP8_BENCHMARK", hv::HipdnnAttentionKind::kSdpaFp8,
        validate_sdpa_fp8_case);
  case AttentionBenchmarkOperation::kFp8Backward:
    return run_attention_benchmark_catalog(
        argc, argv, make_sdpa_fp8_backward_benchmark_cases(),
        "FLAGDNN_SDPA_FP8_BACKWARD_BENCHMARK",
        hv::HipdnnAttentionKind::kSdpaFp8Backward,
        validate_sdpa_fp8_backward_case);
  }
  throw std::invalid_argument("unknown attention benchmark operation");
}

} // namespace flagdnn::testing
