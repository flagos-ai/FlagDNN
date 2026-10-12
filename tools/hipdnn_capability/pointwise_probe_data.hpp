/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_TOOLS_HIPDNN_CAPABILITY_POINTWISE_PROBE_DATA_HPP_
#define FLAGDNN_TOOLS_HIPDNN_CAPABILITY_POINTWISE_PROBE_DATA_HPP_

#include "backends/hygon/validation/benchmark/cases.hpp"
#include "backends/hygon/validation/benchmark/validation_guards.hpp"
#include "benchmark/common/cases.hpp"
#include "reference/cpu/benchmark.hpp"
#include "tests/common/pointwise.hpp"

#include <array>
#include <cstring>
#include <map>
#include <set>

namespace flagdnn::tools::pointwise_probe {
namespace bm = benchmarking;
namespace io = validation::hygon::tensor_io;

struct Case {
  std::string operation;
  bm::BenchmarkCase specification;
  std::string catalog;
};

inline std::vector<Case> catalog() {
  std::vector<Case> result;
  const auto append = [&](std::string operation,
                          std::vector<bm::BenchmarkCase> cases) {
    for (auto &value : bm::aligned_benchmark_cases(cases))
      result.push_back({operation, std::move(value), "benchmark_main"});
  };
  append("add", bm::add_benchmark_cases());
  append("relu", bm::relu_benchmark_cases());
  append("add_square", bm::add_square_benchmark_cases());
  for (const auto &[name, mode] :
       std::vector<std::pair<std::string, flagdnnPointwiseMode_t>>{
           {"identity", FLAGDNN_POINTWISE_IDENTITY},
           {"abs", FLAGDNN_POINTWISE_ABS},
           {"sigmoid", FLAGDNN_POINTWISE_SIGMOID_FWD},
           {"tanh", FLAGDNN_POINTWISE_TANH_FWD},
           {"elu", FLAGDNN_POINTWISE_ELU_FWD},
           {"swish", FLAGDNN_POINTWISE_SWISH_FWD}}) {
    auto attributes = bm::default_pointwise_attributes();
    if (name == "elu") {
      attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
      attributes.elu_alpha = 1;
    } else if (name == "swish") {
      attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
      attributes.swish_beta = 1.25;
    }
    append(name, bm::unary_pointwise_benchmark_cases(
                     mode, name, bm::InputDomain::kReal, attributes));
  }
  for (const auto &[name, mode] :
       std::vector<std::pair<std::string, flagdnnPointwiseMode_t>>{
           {"sub", FLAGDNN_POINTWISE_SUB},
           {"mul", FLAGDNN_POINTWISE_MUL},
           {"scale", FLAGDNN_POINTWISE_MUL},
           {"max", FLAGDNN_POINTWISE_MAX},
           {"min", FLAGDNN_POINTWISE_MIN},
           {"relu_backward", FLAGDNN_POINTWISE_RELU_BWD},
           {"sigmoid_backward", FLAGDNN_POINTWISE_SIGMOID_BWD},
           {"swish_backward", FLAGDNN_POINTWISE_SWISH_BWD},
           {"tanh_backward", FLAGDNN_POINTWISE_TANH_BWD}})
    append(name, bm::binary_pointwise_benchmark_cases(mode, name));

  testing::PointwiseCaseDefinition definition;
  definition.operation_name = "identity";
  definition.mode = FLAGDNN_POINTWISE_IDENTITY;
  for (const auto &native : testing::make_unary_pointwise_cases(definition)) {
    if (native.output.data_type == FLAGDNN_DATA_FP8_E8M0)
      continue; // Not registered in the Hygon native benchmark catalog.
    bm::BenchmarkCase value;
    value.name = native.name;
    value.operation = bm::Operation::kPointwise;
    value.pointwise_mode = native.mode;
    value.pointwise_attributes = native.attributes;
    value.add_alpha = native.alpha;
    value.absolute_tolerance = native.absolute_tolerance;
    value.relative_tolerance = native.relative_tolerance;
    const auto convert = [](const testing::TestTensor &tensor) {
      return bm::TensorSpec{tensor.uid, tensor.data_type, tensor.dimensions,
                            tensor.strides, tensor.binding_byte_offset};
    };
    for (const auto &input : native.inputs)
      value.tensors.push_back(convert(input));
    value.tensors.push_back(convert(native.output));
    result.push_back({"identity", std::move(value), "benchmark_native_copy"});
  }
  std::set<std::string> names;
  for (const auto &value : result)
    if (!names.insert(value.specification.name).second)
      throw std::runtime_error("duplicate pointwise probe case name");
  return result;
}

inline std::string dtype(flagdnnDataType_t type) {
  switch (type) {
  case FLAGDNN_DATA_FLOAT32:
    return "fp32";
  case FLAGDNN_DATA_FLOAT16:
    return "fp16";
  case FLAGDNN_DATA_BFLOAT16:
    return "bfloat16";
  case FLAGDNN_DATA_INT32:
    return "int32";
  case FLAGDNN_DATA_BOOLEAN:
    return "bool";
  case FLAGDNN_DATA_FP8_E4M3:
    return "fp8_e4m3";
  case FLAGDNN_DATA_FP8_E5M2:
    return "fp8_e5m2";
  default:
    throw std::invalid_argument("unsupported probe dtype");
  }
}

inline std::vector<float> quantize(std::span<const float> values,
                                   flagdnnDataType_t type) {
  return io::decode(io::encode(values, type), type, values.size());
}

struct Data {
  std::vector<std::vector<std::uint8_t>> inputs;
  std::vector<float> expected;
  std::vector<std::uint8_t> expected_storage;
  std::vector<std::uint8_t> poison_storage;
};

inline Data prepare(const Case &value) {
  const auto &spec = value.specification;
  Data data;
  std::vector<std::vector<float>> logical;
  for (std::size_t input = 0; input < bm::input_tensor_count(spec); ++input) {
    const auto &tensor = spec.tensors[input];
    std::vector<float> values(io::element_count(tensor));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const int centered = static_cast<int>((i * 17 + input * 11) % 41) - 20;
      values[i] = tensor.data_type == FLAGDNN_DATA_BOOLEAN
                      ? ((i * 17 + input * 11) % 3 != 0 ? 1.F : 0.F)
                      : static_cast<float>(centered) / float(13 + input);
    }
    auto encoded = io::encode(io::scatter(values, tensor), tensor.data_type);
    // Identity must preserve every integer bit, including values that cannot
    // round-trip through float. The oracle below copies these bytes directly.
    if (tensor.data_type == FLAGDNN_DATA_INT32) {
      if (value.operation != "identity")
        throw std::invalid_argument(
            "integer probe requires bit-exact identity");
      constexpr std::array<std::int32_t, 8> integers = {
          0, 1, -1, 16777217, -16777217, INT32_MAX, INT32_MIN, 0x12345678};
      for (std::size_t i = 0; i < values.size(); ++i) {
        const auto integer = integers[i % integers.size()];
        std::memcpy(encoded.data() + io::logical_offset(i, tensor) * 4,
                    &integer, 4);
      }
    }
    logical.push_back(io::gather(io::decode(encoded, tensor.data_type,
                                            io::storage_element_count(tensor)),
                                 tensor));
    data.inputs.push_back(std::move(encoded));
  }
  const auto &output = bm::output_tensor(spec);
  data.expected =
      value.operation == "identity"
          ? logical.at(0)
          : reference::cpu::evaluate_benchmark(spec, logical, quantize).at(0);
  // Do not re-encode decoded INT32 here: INT32_MAX rounds to 2^31 in float.
  // Identity's expected logical bytes are copied from the original storage.
  data.expected_storage =
      io::encode(io::scatter(value.operation == "identity"
                                 ? std::vector<float>(data.expected.size(), 0.F)
                                 : data.expected,
                             output),
                 output.data_type);
  if (value.operation == "identity") {
    const auto bytes = io::data_type_size(output.data_type);
    for (std::size_t i = 0; i < data.expected.size(); ++i)
      std::memcpy(data.expected_storage.data() +
                      io::logical_offset(i, output) * bytes,
                  data.inputs.at(0).data() +
                      io::logical_offset(i, spec.tensors.at(0)) * bytes,
                  bytes);
  }
  data.poison_storage = io::encode(
      io::scatter(bm::capture_output_poison(data.expected, output.data_type,
                                            spec.absolute_tolerance,
                                            spec.relative_tolerance),
                  output),
      output.data_type);
  return data;
}

struct Comparison {
  std::size_t mismatches = 0, unchanged = 0, nonfinite = 0, padding_errors = 0;
  std::size_t first = std::numeric_limits<std::size_t>::max();
  double max_absolute = 0, max_relative = 0;
  bool passed() const { return mismatches == 0 && padding_errors == 0; }
};

inline Comparison compare(const Case &value, const Data &data,
                          std::span<const std::uint8_t> actual) {
  const auto &spec = value.specification;
  const auto &output = bm::output_tensor(spec);
  if (actual.size() != data.poison_storage.size())
    throw std::invalid_argument("probe output storage size mismatch");
  const auto decoded =
      io::decode(actual, output.data_type, io::storage_element_count(output));
  const auto bytes = io::data_type_size(output.data_type);
  std::vector<bool> occupied(decoded.size(), false);
  Comparison result;
  for (std::size_t i = 0; i < data.expected.size(); ++i) {
    const auto offset = io::logical_offset(i, output);
    occupied[offset] = true;
    const auto position = offset * bytes;
    const bool bit_equal =
        std::memcmp(actual.data() + position,
                    data.expected_storage.data() + position, bytes) == 0;
    result.unchanged +=
        std::memcmp(actual.data() + position,
                    data.poison_storage.data() + position, bytes) == 0;
    const double actual_value = decoded[offset], expected = data.expected[i];
    const double absolute = std::abs(actual_value - expected);
    const double relative = absolute / std::max({std::abs(actual_value),
                                                 std::abs(expected), 1e-30});
    result.nonfinite += !std::isfinite(actual_value);
    if (std::isfinite(absolute))
      result.max_absolute = std::max(result.max_absolute, absolute);
    if (std::isfinite(relative))
      result.max_relative = std::max(result.max_relative, relative);
    const bool mismatch =
        value.operation == "identity"
            ? !bit_equal
            : !std::isfinite(absolute) || (absolute > spec.absolute_tolerance &&
                                           relative > spec.relative_tolerance);
    if (mismatch) {
      ++result.mismatches;
      result.first = std::min(result.first, i);
    }
  }
  for (std::size_t i = 0; i < occupied.size(); ++i)
    if (!occupied[i] &&
        std::memcmp(actual.data() + i * bytes,
                    data.poison_storage.data() + i * bytes, bytes) != 0)
      ++result.padding_errors;
  return result;
}
} // namespace flagdnn::tools::pointwise_probe
#endif
