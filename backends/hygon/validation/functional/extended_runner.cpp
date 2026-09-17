/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/index.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/position_embedding.hpp"
#include "common/random.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include "host_runner.hpp"
#include "reference/cpu/causal_convolution.hpp"
#include "reference/cpu/fp8_matmul.hpp"
#include "reference/cpu/index.hpp"
#include "reference/cpu/moe_matmul.hpp"
#include "reference/cpu/normalization.hpp"
#include "reference/cpu/position_embedding.hpp"
#include "reference/cpu/random.hpp"
#include "reference/cpu/resample.hpp"
#include "reference/cpu/statistics.hpp"
namespace flagdnn::testing {
namespace host = hygon_functional::host;
namespace cpu = reference::cpu;
constexpr auto kPairedComparison =
    hygon_functional::ComparisonRule::kCombinedTolerance;
int run_statistics_functional_test(int argc, char **argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark) {
  return host::run_suite(
      argc, argv, cases, cases.empty() ? "extended" : cases.front().operation,
      benchmark,
      [](const StatisticsTestCase &c, const auto &handle, auto &stream) {
        auto values = host::default_inputs(c.inputs);
        for (std::size_t i = 0; i < values.size(); ++i)
          for (std::size_t j = 0; j < values[i].size(); ++j)
            values[i][j] = statistics_input_value(c, i, j);
        host::run_case(
            c.name, c.inputs, c.outputs, values, handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_statistics(h, c); },
            [&](const host::Values &v) {
              return cpu::evaluate_statistics(
                  c.operation, c.inputs[0].dimensions, c.outputs.size(),
                  c.epsilon, c.accum_count, c.momentum, v);
            },
            2e-5, 2e-4, {}, kPairedComparison);
      });
}
int run_index_functional_test(int argc, char **argv,
                              std::span<const IndexTestCase> cases,
                              bool benchmark) {
  return host::run_suite(
      argc, argv, cases, cases.empty() ? "extended" : cases.front().operation,
      benchmark, [](const IndexTestCase &c, const auto &handle, auto &stream) {
        if (c.operation == "concatenate") {
          const cpu::IndexParameters parameters{
              c.operation, host::shapes(c.inputs), c.output.dimensions, c.axis};
          host::run_copy(
              c.name, c.inputs, c.output, handle, stream,
              [&](std::size_t i) { return cpu::index_source(parameters, i); },
              [&](flagdnn::Handle &h) { return build_flagdnn_index(h, c); });
          return;
        }

        host::run_case(
            c.name, c.inputs, std::vector<TestTensor>{c.output},
            host::default_inputs(c.inputs), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_index(h, c); },
            [&](const host::Values &v) {
              return [&] {
                const cpu::IndexParameters parameters{
                    c.operation, host::shapes(c.inputs), c.output.dimensions,
                    c.axis};
                std::vector<float> output(host::io::element_count(c.output));
                for (std::size_t i = 0; i < output.size(); ++i) {
                  auto [input, offset] = cpu::index_source(parameters, i);
                  output[i] = c.operation == "gen_index" ? float(offset)
                                                         : v[input][offset];
                }
                return host::Values{output};
              }();
            },
            0.0, 0.0, {}, kPairedComparison);
      });
}
int run_moe_matmul_functional_test(int argc, char **argv,
                                   std::span<const MoeMatmulTestCase> cases,
                                   bool benchmark) {
  return host::run_suite(
      argc, argv, cases,
      !cases.empty() && cases.front().backward ? "moe_grouped_matmul_bwd"
                                               : "moe_grouped_matmul",
      benchmark,
      [](const MoeMatmulTestCase &c, const auto &handle, auto &stream) {
        const auto type = c.outputs[0].data_type;
        const double relative = type == FLAGDNN_DATA_BFLOAT16  ? 4e-3
                                : type == FLAGDNN_DATA_FLOAT16 ? 5e-4
                                                               : 1e-3;
        const double absolute =
            type == FLAGDNN_DATA_FLOAT32
                ? 2e-3 * std::sqrt(std::max(
                             1.0, double(c.inputs[0].dimensions.back()) / 128))
                : relative;
        // Compare the GPU result directly with the FP64-accumulated oracle.
        // Rounding that oracle to FP16 first creates a full-ULP discrepancy
        // when an FP32 dot product lands on the opposite side of a midpoint.
        host::run_case(
            c.name, c.inputs, c.outputs, moe_matmul_inputs(c), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_moe_matmul(h, c); },
            [&](const host::Values &v) {
              return cpu::evaluate_moe_matmul(
                  {host::shapes(c.inputs), c.outputs[0].dimensions, c.offsets,
                   c.token_index, c.token_ks, c.mode, c.top_k, c.backward},
                  v);
            },
            absolute, relative, {}, kPairedComparison,
            host::ReferenceOutput::kFullPrecision);
      });
}
int run_causal_convolution_functional_test(
    int argc, char **argv, std::span<const CausalConvolutionTestCase> cases,
    bool benchmark) {
  return host::run_suite(
      argc, argv, cases, "causal_conv1d", benchmark,
      [](const CausalConvolutionTestCase &c, const auto &handle, auto &stream) {
        const auto type = c.outputs[0].data_type;
        const double limit = type == FLAGDNN_DATA_BFLOAT16  ? 8e-3
                             : type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                                            : 2e-5;
        host::run_case(
            c.name, c.inputs, c.outputs, causal_convolution_inputs(c), handle,
            stream,
            [&](flagdnn::Handle &h) {
              return build_flagdnn_causal_convolution(h, c);
            },
            [&](const host::Values &v) {
              return cpu::evaluate_causal_convolution(
                  {c.inputs[0].dimensions, c.inputs[1].dimensions[1],
                   c.dilation, c.precision, c.silu},
                  v);
            },
            limit, limit, {}, kPairedComparison);
      });
}
int run_resample_functional_test(int argc, char **argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark) {
  return host::run_suite(
      argc, argv, cases, "resample", benchmark,
      [](const ResampleTestCase &c, const auto &handle, auto &stream) {
        const auto type = c.outputs[0].data_type;
        const double limit = type == FLAGDNN_DATA_BFLOAT16  ? 4e-3
                             : type == FLAGDNN_DATA_FLOAT16 ? 5e-4
                                                            : 1e-5;
        std::vector<std::pair<double, double>> limits(c.outputs.size(), {0, 0});
        limits[0] = {limit, limit};
        host::run_case(
            c.name, c.inputs, c.outputs, resample_inputs(c), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_resample(h, c); },
            [&](const host::Values &v) {
              return cpu::evaluate_resample(
                  {c.inputs[0].dimensions, c.outputs[0].dimensions, c.window,
                   c.stride, c.pre, c.post, c.outputs.size(), c.mode, c.padding,
                   c.align_corners},
                  v);
            },
            -1, -1, limits, kPairedComparison);
      });
}
int run_rng_functional_test(int argc, char **argv,
                            std::span<const RngTestCase> cases,
                            bool benchmark) {
  return host::run_suite(
      argc, argv, cases, "rng", benchmark,
      [](const RngTestCase &c, const auto &handle, auto &stream) {
        host::run_case(
            c.name, c.inputs, c.outputs, host::Values{}, handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_rng(h, c); },
            [&](const host::Values &) {
              return cpu::evaluate_rng(
                  {c.outputs[0].dimensions, c.seed, c.offset, c.distribution,
                   c.outputs[0].data_type == FLAGDNN_DATA_FLOAT32   ? 24
                   : c.outputs[0].data_type == FLAGDNN_DATA_FLOAT16 ? 11
                                                                    : 8,
                   c.probability});
            },
            -1, -1, {}, kPairedComparison);
      });
}
int run_rope_functional_test(int argc, char **argv,
                             std::span<const RoPETestCase> cases,
                             bool benchmark) {
  return host::run_suite(
      argc, argv, cases, cases.empty() ? "extended" : cases.front().operation,
      benchmark, [](const RoPETestCase &c, const auto &handle, auto &stream) {
        host::run_case(
            c.name, c.inputs, c.outputs, rope_inputs(c), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_rope(h, c); },
            [&](const host::Values &v) {
              return cpu::evaluate_rope(c.inputs[0].dimensions, c.rope_dim,
                                        c.output_scale,
                                        c.operation == "rope_backward", v);
            },
            -1, -1, {}, kPairedComparison);
      });
}
int run_extended_normalization_functional_test(
    int argc, char **argv, std::span<const ExtendedNormalizationTestCase> cases,
    bool benchmark) {
  return host::run_suite(
      argc, argv, cases, cases.empty() ? "extended" : cases.front().operation,
      benchmark,
      [](const ExtendedNormalizationTestCase &c, const auto &handle,
         auto &stream) {
        auto values = host::default_inputs(c.inputs);
        if (c.operation.ends_with("_backward"))
          for (auto &v : values.back())
            v = 0.5F + std::abs(v);
        host::run_case(
            c.name, c.inputs, c.outputs, values, handle, stream,
            [&](flagdnn::Handle &h) {
              return build_flagdnn_extended_normalization(h, c);
            },
            [&](const host::Values &v) {
              return cpu::evaluate_normalization(
                  {c.operation,
                   c.inputs[c.operation.ends_with("_backward") ? 1 : 0]
                       .dimensions,
                   c.inputs[c.operation.ends_with("_backward") ? 2 : 1]
                       .dimensions,
                   c.axes, c.epsilon},
                  v);
            },
            -1, -1, {}, kPairedComparison);
      });
}
int run_fp8_matmul_functional_test(int argc, char **argv,
                                   std::span<const Fp8MatmulTestCase> cases,
                                   bool benchmark) {
  const char *api = std::getenv("FLAGDNN_FP8_MATMUL_API");
  if (api && std::string_view(api) != "matmul" &&
      std::string_view(api) != "matmul_fp8")
    throw std::invalid_argument("unknown FP8 matmul API selection");
  std::vector<Fp8MatmulTestCase> selected;
  for (const auto &c : cases)
    if (c.plain_matmul == (api && std::string_view(api) == "matmul"))
      selected.push_back(c);
  cases = selected;
  return host::run_suite(
      argc, argv, cases,
      !cases.empty() && cases.front().plain_matmul ? "matmul" : "matmul_fp8",
      benchmark,
      [](const Fp8MatmulTestCase &c, const auto &handle, auto &stream) {
        const auto type = c.outputs[0].data_type;
        const double absolute =
            c.scale_mode == 1
                ? 2e-3 * std::sqrt(std::max(
                             1.0, double(c.inputs[0].dimensions.back()) / 128))
                : 2e-5;
        const double relative = type == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3;
        host::run_case(
            c.name, c.inputs, c.outputs, fp8_matmul_inputs(c), handle, stream,
            [&](flagdnn::Handle &h) { return build_flagdnn_fp8_matmul(h, c); },
            [&](const host::Values &v) {
              return cpu::evaluate_fp8_matmul({host::shapes(c.inputs),
                                               c.outputs[0].dimensions,
                                               c.scale_mode},
                                              v);
            },
            absolute, relative, {}, kPairedComparison);
      });
}
} // namespace flagdnn::testing
