/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "common/causal_convolution.hpp"
#include "common/fp8_matmul.hpp"
#include "common/moe_matmul.hpp"
#include "common/normalization_extended.hpp"
#include "common/resample.hpp"
#include "common/statistics.hpp"
#include "equivalent_baseline.hpp"
#include "native.hpp"
#include "reference/cpu/causal_convolution.hpp"
#include "reference/cpu/fp8_matmul.hpp"
#include "reference/cpu/moe_matmul.hpp"
#include "reference/cpu/normalization.hpp"
#include "reference/cpu/resample.hpp"
#include "reference/cpu/statistics.hpp"
#include "tensor_io.hpp"

namespace flagdnn::testing {
namespace {
namespace bm = benchmarking;
namespace cpu = reference::cpu;
namespace io = validation::hygon::tensor_io;
using Json = native::json::Value;
using Values = std::vector<std::vector<float>>;

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}
void keys(const Json& value, std::initializer_list<std::string_view> names) {
  require(value.as_object().size() == names.size(),
          "native attribute count differs");
  for (auto name : names) (void)value.at(name);
}
void number(const Json& value, std::string_view key, double expected) {
  require(value.at(key).as_double() == expected,
          "native numeric attribute differs");
}
template <class T>
void array(const Json& value, std::string_view key,
           const std::vector<T>& expected) {
  const auto& actual = value.at(key).as_array();
  require(actual.size() == expected.size(),
          "native attribute array size differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    require(actual[i].as_int() == static_cast<std::int64_t>(expected[i]),
            "native attribute array differs");
}

class Executable final : public bm::BenchmarkExecutable {
 public:
  explicit Executable(std::unique_ptr<TestExecutable> executable)
      : executable_(std::move(executable)) {}
  void prepare(std::span<const flagdnnBinding_t> bindings,
               flagdnnStream_t stream) override {
    executable_->prepare(bindings, stream);
  }
  std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }
  void execute(std::span<const flagdnnBinding_t> bindings, void* workspace,
               std::size_t bytes, flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, bytes, stream);
  }

 private:
  std::unique_ptr<TestExecutable> executable_;
};

std::vector<std::vector<std::int64_t>> shapes(
    const std::vector<TestTensor>& tensors) {
  std::vector<std::vector<std::int64_t>> result;
  for (const auto& t : tensors) result.push_back(t.dimensions);
  return result;
}

Values quantize(Values values, const std::vector<TestTensor>& tensors) {
  require(values.size() == tensors.size(), "native value arity differs");
  for (std::size_t i = 0; i < values.size(); ++i) {
    require(values[i].size() == io::element_count(tensors[i]),
            "native value size differs");
    values[i] = io::decode(io::encode(values[i], tensors[i].data_type),
                           tensors[i].data_type, values[i].size());
  }
  return values;
}

template <class T, class Builder, class Generator, class Oracle,
          class Attributes>
int run(int argc, char** argv, std::string_view operation,
        const std::vector<T>& catalog, Builder builder, Generator generator,
        Oracle oracle, Attributes attributes, double atol, double rtol) {
  std::vector<bm::BenchmarkCase> cases;
  std::map<std::string, T, std::less<>> native_cases;
  for (const auto& value : catalog) {
    bm::BenchmarkCase spec;
    spec.name = value.name;
    spec.operation = bm::Operation::kGraph;
    for (const auto& tensor : value.inputs)
      spec.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                              tensor.strides, tensor.binding_byte_offset});
    for (const auto& tensor : value.outputs)
      spec.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                              tensor.strides, tensor.binding_byte_offset});
    spec.output_count = value.outputs.size();
    const auto dtype = value.outputs.front().data_type;
    double limit = dtype == FLAGDNN_DATA_BFLOAT16  ? 8e-3
                   : dtype == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                                   : 2e-4;
    if constexpr (std::is_same_v<T, ResampleTestCase>)
      limit = dtype == FLAGDNN_DATA_BFLOAT16  ? 4e-3
              : dtype == FLAGDNN_DATA_FLOAT16 ? 5e-4
                                              : 1e-5;
    if constexpr (std::is_same_v<T, CausalConvolutionTestCase>)
      limit = dtype == FLAGDNN_DATA_BFLOAT16  ? 8e-3
              : dtype == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                              : 2e-5;
    spec.absolute_tolerance = atol < 0 ? limit : atol;
    spec.relative_tolerance = rtol < 0 ? limit : rtol;
    if constexpr (std::is_same_v<T, Fp8MatmulTestCase>) {
      spec.absolute_tolerance =
          value.scale_mode == 1
              ? 2e-3 *
                    std::sqrt(std::max(
                        1.0, double(value.inputs[0].dimensions.back()) / 128))
              : 2e-5;
      spec.relative_tolerance = dtype == FLAGDNN_DATA_BFLOAT16 ? 8e-3 : 1e-3;
    }
    if constexpr (std::is_same_v<T, MoeMatmulTestCase>) {
      spec.relative_tolerance = dtype == FLAGDNN_DATA_BFLOAT16  ? 4e-3
                                : dtype == FLAGDNN_DATA_FLOAT16 ? 5e-4
                                                                : 1e-3;
      spec.absolute_tolerance =
          dtype == FLAGDNN_DATA_FLOAT32
              ? 2e-3 *
                    std::sqrt(std::max(
                        1.0, double(value.inputs[0].dimensions.back()) / 128))
              : spec.relative_tolerance;
    }
    // Match each accepted NVIDIA native record, including graph warmup.
    spec.benchmark = {10, 20, 50};
    cases.push_back(std::move(spec));
    require(native_cases.emplace(value.name, value).second,
            "duplicate native case");
  }
  std::string suite = "FLAGDNN_";
  for (char c : operation)
    suite += static_cast<char>(c >= 'a' && c <= 'z' ? c - 'a' + 'A' : c);
  suite += "_BENCHMARK";
  return bm::run_hygon_benchmark_suite(
      argc, argv, cases, suite,
      [&](flagdnn::Handle& handle, const bm::BenchmarkCase& spec) {
        return std::make_unique<Executable>(
            builder(handle, native_cases.at(spec.name)));
      },
      [&](const bm::BenchmarkCase& spec, const Values& inputs) {
        const auto& value = native_cases.at(spec.name);
        if constexpr (std::is_same_v<T, MoeMatmulTestCase>)
          return oracle(value,
                        inputs);  // Match the functional full-precision oracle.
        else
          return quantize(oracle(value, inputs), value.outputs);
      },
      [&](const bm::BenchmarkCase& spec) {
        const auto value = native_cases.at(spec.name);
        bm::NativeBaselineMetadata metadata;
        metadata.operation = operation;
        metadata.input_count = value.inputs.size();
        metadata.output_count = value.outputs.size();
        metadata.validate_attributes = [value, attributes](const Json& entry) {
          attributes(value, entry);
        };
        metadata.generate_inputs = [value, generator] {
          return quantize(generator(value), value.inputs);
        };
        return metadata;
      });
}
}  // namespace

int run_native_extended_benchmark(int argc, char** argv,
                                  std::string_view operation) {
  try {
    if (operation == "matmul_fp8") {
      const char* api = std::getenv("FLAGDNN_FP8_MATMUL_API");
      if (api && std::string_view(api) != "matmul_fp8" &&
          std::string_view(api) != "matmul")
        throw std::invalid_argument("unknown native FP8 matmul API");
      const bool plain = api && std::string_view(api) == "matmul";
      auto catalog = make_fp8_matmul_cases();
      std::erase_if(catalog,
                    [plain](const auto& c) { return c.plain_matmul != plain; });
      operation = plain ? "matmul" : "matmul_fp8";
      return run(
          argc, argv, operation, catalog, build_flagdnn_fp8_matmul,
          fp8_matmul_inputs,
          [](const Fp8MatmulTestCase& c, const Values& values) {
            return cpu::evaluate_fp8_matmul(
                {shapes(c.inputs), c.outputs[0].dimensions, c.scale_mode},
                values);
          },
          [](const Fp8MatmulTestCase& c, const Json& a) {
            keys(a, {"scale_mode", "scale_variant", "plain_matmul"});
            number(a, "scale_mode", c.scale_mode);
            number(a, "scale_variant", c.scale_variant);
            require(a.at("plain_matmul").as_bool() == c.plain_matmul,
                    "FP8 matmul API differs");
          },
          -1, -1);
    }
    if (operation == "moe_grouped_matmul")
      return run(
          argc, argv, operation, make_moe_matmul_cases(),
          build_flagdnn_moe_matmul, moe_matmul_inputs,
          [](const MoeMatmulTestCase& c, const Values& values) {
            return cpu::evaluate_moe_matmul(
                {shapes(c.inputs), c.outputs[0].dimensions, c.offsets,
                 c.token_index, c.token_ks, c.mode, c.top_k},
                values);
          },
          [](const MoeMatmulTestCase& c, const Json& a) {
            keys(a, {"mode", "top_k", "backward", "offsets", "token_index",
                     "token_ks"});
            number(a, "mode", c.mode);
            number(a, "top_k", c.top_k);
            require(!a.at("backward").as_bool(), "MoE direction differs");
            array(a, "offsets", c.offsets);
            array(a, "token_index", c.token_index);
            array(a, "token_ks", c.token_ks);
          },
          -1, -1);
    if (operation == "bn_finalize")
      return run(
          argc, argv, operation, make_bn_finalize_cases(),
          build_flagdnn_statistics,
          [](const StatisticsTestCase& c) {
            Values values;
            for (std::size_t i = 0; i < c.inputs.size(); ++i) {
              std::vector<float> v(io::element_count(c.inputs[i]));
              for (std::size_t j = 0; j < v.size(); ++j)
                v[j] = statistics_input_value(c, i, j);
              values.push_back(std::move(v));
            }
            return values;
          },
          [](const StatisticsTestCase& c, const Values& values) {
            return cpu::evaluate_bn_finalize(c.outputs.size(), c.epsilon,
                                             c.accum_count, c.momentum, values);
          },
          [](const StatisticsTestCase& c, const Json& a) {
            keys(a, {"epsilon", "accum_count", "momentum"});
            number(a, "epsilon", c.epsilon);
            number(a, "accum_count", c.accum_count);
            number(a, "momentum", c.momentum);
          },
          2e-5, 2e-4);
    if (operation == "resample")
      return run(
          argc, argv, operation, make_resample_cases(), build_flagdnn_resample,
          resample_inputs,
          [](const ResampleTestCase& c, const Values& values) {
            return cpu::evaluate_resample(
                {c.inputs[0].dimensions, c.outputs[0].dimensions, c.window,
                 c.stride, c.pre, c.post, c.outputs.size(), c.mode, c.padding,
                 c.align_corners},
                values);
          },
          [](const ResampleTestCase& c, const Json& a) {
            keys(a, {"mode", "padding", "window", "stride", "pre", "post",
                     "align_corners"});
            number(a, "mode", c.mode);
            number(a, "padding", c.padding);
            array(a, "window", c.window);
            array(a, "stride", c.stride);
            array(a, "pre", c.pre);
            array(a, "post", c.post);
            require(a.at("align_corners").as_bool() == c.align_corners,
                    "resample align_corners differs");
          },
          -1, -1);
    if (operation == "causal_conv1d")
      return run(
          argc, argv, operation, make_causal_convolution_cases(),
          build_flagdnn_causal_convolution, causal_convolution_inputs,
          [](const CausalConvolutionTestCase& c, const Values& values) {
            return cpu::evaluate_causal_convolution(
                {c.inputs[0].dimensions, c.inputs[1].dimensions[1], c.dilation,
                 c.precision, c.silu},
                values);
          },
          [](const CausalConvolutionTestCase& c, const Json& a) {
            keys(a, {"precision", "dilation", "silu"});
            number(a, "precision", c.precision);
            number(a, "dilation", c.dilation);
            require(a.at("silu").as_bool() == c.silu, "causal silu differs");
          },
          -1, -1);
    return run(
        argc, argv, operation,
        make_extended_normalization_cases(std::string(operation)),
        build_flagdnn_extended_normalization,
        [](const ExtendedNormalizationTestCase& c) {
          Values values;
          for (const auto& tensor : c.inputs) {
            std::vector<float> v(io::element_count(tensor));
            for (std::size_t i = 0; i < v.size(); ++i) {
              v[i] = float(int((i * 17 + tensor.uid * 7) % 61) - 30) / 19;
              if (c.operation.ends_with("_backward") &&
                  values.size() + 1 == c.inputs.size())
                v[i] = 0.5F + std::abs(v[i]);
            }
            values.push_back(std::move(v));
          }
          return values;
        },
        [](const ExtendedNormalizationTestCase& c, const Values& values) {
          const bool backward = c.operation.ends_with("_backward");
          return cpu::evaluate_normalization(
              {c.operation, c.inputs[backward ? 1 : 0].dimensions,
               c.inputs[backward ? 2 : 1].dimensions, c.axes, c.epsilon},
              values);
        },
        [](const ExtendedNormalizationTestCase& c, const Json& a) {
          keys(a, {"axes", "epsilon"});
          array(a, "axes", c.axes);
          number(a, "epsilon", c.epsilon);
        },
        -1, -1);
  } catch (const std::exception& error) {
    std::cerr << "native extended benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
}  // namespace flagdnn::testing
