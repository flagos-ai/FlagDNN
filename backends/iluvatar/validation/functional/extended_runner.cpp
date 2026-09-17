// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "common/causal_convolution.hpp"
#include "common/index.hpp"
#include "common/normalization_extended.hpp"
#include "common/position_embedding.hpp"
#include "common/random.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include "extended_reference.hpp"
#include "functional/raw_reference.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/causal_convolution.hpp"
#include "reference/cpu/index.hpp"
#include "reference/cpu/normalization.hpp"
#include "reference/cpu/position_embedding.hpp"
#include "reference/cpu/random.hpp"
#include "reference/cpu/resample.hpp"
#include "reference/cpu/statistics.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace flagdnn::testing {
namespace {
namespace f = iluvatar::validation::functional;
namespace cpu = reference::cpu;
using Values = std::vector<std::vector<float>>;
std::string marker(std::string operation, bool benchmark) {
  std::transform(operation.begin(), operation.end(), operation.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return "FLAGDNN_" + operation + (benchmark ? "_BENCHMARK" : "_FUNCTIONAL");
}
Values inputs(const std::vector<TestTensor> &tensors, bool backward = false) {
  Values result;
  for (const auto &tensor : tensors) {
    std::vector<float> values(f::element_count(tensor));
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<float>(
                      static_cast<int>((i * 17 + tensor.uid * 7) % 61) - 30) /
                  19.0F;
      if (backward && result.size() == tensors.size() - 1)
        values[i] = 0.5F + std::abs(values[i]);
    }
    result.push_back(std::move(values));
  }
  return result;
}
template <class Case, class Build, class Input, class Reference>
int run(int argc, char **argv, std::span<const Case> cases,
        std::string operation, bool benchmark, Build build, Input input,
        Reference reference) {
  if (cases.empty())
    throw std::invalid_argument("empty extended suite");
  f::FunctionalSuite suite(argc, argv, operation, marker(operation, benchmark));
  for (const auto &c : cases) {
    f::CasePlan plan{.operation = operation,
                     .case_name = c.name,
                     .inputs = {},
                     .outputs = {}};
    const auto reference_skip =
        iluvatar::validation::extended_reference_skip(c);
    if (benchmark && !reference_skip.empty()) {
      const auto &tensor =
          c.inputs.empty() ? c.outputs.front() : c.inputs.front();
      plan.outputs.push_back({tensor, 0, 0, "skip"});
      suite.skip_benchmark_case(plan, reference_skip);
      continue;
    }
    const auto values = input(c);
    for (std::size_t i = 0; i < c.inputs.size(); ++i)
      plan.inputs.push_back({c.inputs[i], f::InputDomain::kReal, values[i]});
    for (const auto &tensor : c.outputs) {
      const double tolerance = tensor.data_type == FLAGDNN_DATA_BFLOAT16  ? 8e-3
                               : tensor.data_type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                               : tensor.data_type == FLAGDNN_DATA_INT32   ? 0.0
                                                                        : 2e-4;
      plan.outputs.push_back({tensor, tolerance, tolerance, "output"});
    }
    suite.run(
        plan, [&] { return build(suite.handle(), c); },
        [&] { return iluvatar::validation::build_extended_reference(c); },
        [&](const Values &actual) { return reference(c, actual); },
        reference_skip.empty());
  }
  return suite.finish();
}
} // namespace

int run_extended_normalization_functional_test(
    int argc, char **argv, std::span<const ExtendedNormalizationTestCase> cases,
    bool benchmark) {
  return run(
      argc, argv, cases, cases.front().operation, benchmark,
      build_flagdnn_extended_normalization,
      [](const auto &c) {
        return inputs(c.inputs, c.operation.ends_with("_backward"));
      },
      [](const auto &c, const Values &values) {
        const bool backward = c.operation.ends_with("_backward");
        return cpu::evaluate_normalization(
            {c.operation, c.inputs[backward ? 1 : 0].dimensions,
             c.inputs[backward ? 2 : 1].dimensions, c.axes, c.epsilon},
            values);
      });
}
int run_causal_convolution_functional_test(
    int argc, char **argv, std::span<const CausalConvolutionTestCase> cases,
    bool benchmark) {
  return run(argc, argv, cases, "causal_conv1d", benchmark,
             build_flagdnn_causal_convolution, causal_convolution_inputs,
             [](const auto &c, const Values &values) {
               return cpu::evaluate_causal_convolution(
                   {c.inputs[0].dimensions, c.inputs[1].dimensions[1],
                    c.dilation, c.precision, c.silu},
                   values);
             });
}
int run_rope_functional_test(int argc, char **argv,
                             std::span<const RoPETestCase> cases,
                             bool benchmark) {
  return run(argc, argv, cases, cases.front().operation, benchmark,
             build_flagdnn_rope, rope_inputs,
             [](const auto &c, const Values &values) {
               return cpu::evaluate_rope(
                   c.inputs[0].dimensions, c.rope_dim, c.output_scale,
                   c.operation == "rope_backward", values);
             });
}
int run_rng_functional_test(int argc, char **argv,
                            std::span<const RngTestCase> cases,
                            bool benchmark) {
  return run(
      argc, argv, cases, "rng", benchmark, build_flagdnn_rng,
      [](const auto &) { return Values{}; },
      [](const auto &c, const Values &) {
        const int bits = c.outputs[0].data_type == FLAGDNN_DATA_FLOAT32   ? 24
                         : c.outputs[0].data_type == FLAGDNN_DATA_FLOAT16 ? 11
                                                                          : 8;
        return cpu::evaluate_rng({c.outputs[0].dimensions, c.seed, c.offset,
                                  c.distribution, bits, c.probability});
      });
}
int run_resample_functional_test(int argc, char **argv,
                                 std::span<const ResampleTestCase> cases,
                                 bool benchmark) {
  return run(argc, argv, cases, "resample", benchmark, build_flagdnn_resample,
             resample_inputs, [](const auto &c, const Values &values) {
               return cpu::evaluate_resample(
                   {c.inputs[0].dimensions, c.outputs[0].dimensions, c.window,
                    c.stride, c.pre, c.post, c.outputs.size(), c.mode,
                    c.padding, c.align_corners},
                   values);
             });
}
int run_statistics_functional_test(int argc, char **argv,
                                   std::span<const StatisticsTestCase> cases,
                                   bool benchmark) {
  return run(
      argc, argv, cases, cases.front().operation, benchmark,
      build_flagdnn_statistics,
      [](const auto &c) {
        Values values;
        for (std::size_t i = 0; i < c.inputs.size(); ++i) {
          std::vector<float> v(f::element_count(c.inputs[i]));
          for (std::size_t j = 0; j < v.size(); ++j)
            v[j] = statistics_input_value(c, i, j);
          values.push_back(std::move(v));
        }
        return values;
      },
      [](const auto &c, const Values &values) {
        return cpu::evaluate_statistics(c.operation, c.inputs[0].dimensions,
                                        c.outputs.size(), c.epsilon,
                                        c.accum_count, c.momentum, values);
      });
}
int run_index_functional_test(int argc, char **argv,
                              std::span<const IndexTestCase> cases,
                              bool benchmark) {
  const auto operation = cases.front().operation;
  f::FunctionalSuite suite(argc, argv, operation, marker(operation, benchmark));
  for (const auto &c : cases) {
    f::CasePlan plan{operation, c.name, {}, {{c.output, 0, 0, "output"}}};
    const auto reason = iluvatar::validation::extended_reference_skip(c);
    if (benchmark && !reason.empty()) {
      suite.skip_benchmark_case(plan, reason);
      continue;
    }
    if (reason.empty()) {
      for (const auto &tensor : c.inputs) {
        std::vector<float> values(f::element_count(tensor));
        for (std::size_t i = 0; i < values.size(); ++i)
          values[i] = static_cast<float>(static_cast<int>(i % 61) - 30);
        plan.inputs.push_back(
            {tensor, f::InputDomain::kReal, std::move(values)});
      }
      suite.run(
          plan, [&] { return build_flagdnn_index(suite.handle(), c); },
          [&] { return iluvatar::validation::build_extended_reference(c); },
          [&](const Values &values) {
            cpu::IndexParameters p{operation, {}, c.output.dimensions, c.axis};
            for (const auto &t : c.inputs)
              p.input_shapes.push_back(t.dimensions);
            std::vector<float> result(f::element_count(c.output));
            for (std::size_t i = 0; i < result.size(); ++i) {
              const auto [tensor, index] = cpu::index_source(p, i);
              result[i] = values[tensor][index];
            }
            return Values{result};
          },
          true);
      if (benchmark)
        continue;
      plan.inputs.clear();
      plan.case_name += "_bits";
    }
    std::vector<f::Bytes> source;
    cpu::IndexParameters parameters{operation, {}, c.output.dimensions, c.axis};
    for (const auto &tensor : c.inputs) {
      plan.inputs.push_back({tensor, f::InputDomain::kReal, {}});
      parameters.input_shapes.push_back(tensor.dimensions);
      source.push_back(f::raw_pattern(tensor, source.size()));
    }
    const auto width = f::data_type_size(c.output.data_type);
    f::Bytes expected(f::element_count(c.output) * width);
    for (std::size_t i = 0; i < f::element_count(c.output); ++i) {
      const auto [tensor, index] = cpu::index_source(parameters, i);
      if (operation == "concatenate")
        std::copy_n(source[tensor].data() + index * width, width,
                    expected.data() + i * width);
      else if (c.output.data_type == FLAGDNN_DATA_INT32) {
        const auto value = static_cast<std::int32_t>(index);
        std::memcpy(expected.data() + i * width, &value, width);
      } else {
        const auto value = static_cast<float>(index);
        std::memcpy(expected.data() + i * width, &value, width);
      }
    }
    suite.run_raw(plan, [&] { return build_flagdnn_index(suite.handle(), c); },
                  source, {expected});
  }
  return suite.finish();
}
} // namespace flagdnn::testing
