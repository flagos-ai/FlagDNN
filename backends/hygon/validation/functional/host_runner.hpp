/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "capability_cases.hpp"
#include "host_pointwise.hpp"
#include "pointwise_runner_support.hpp"
#include <cstring>

namespace flagdnn::testing::hygon_functional::host {
namespace hv = validation::hygon;
namespace io = hv::tensor_io;
namespace pw = hygon_functional::pointwise;
using Values = std::vector<std::vector<float>>;
enum class ReferenceOutput { kStoragePrecision, kFullPrecision };

inline Values default_inputs(std::span<const TestTensor> tensors) {
  Values result;
  for (const auto &tensor : tensors) {
    std::vector<float> values(io::element_count(tensor));
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(int((i * 17 + tensor.uid * 7) % 61) - 30) / 19.0F;
    result.push_back(std::move(values));
  }
  return result;
}

inline std::vector<std::vector<std::int64_t>>
shapes(std::span<const TestTensor> tensors) {
  std::vector<std::vector<std::int64_t>> result;
  for (const auto &t : tensors)
    result.push_back(t.dimensions);
  return result;
}

struct Buffer {
  TestTensor tensor;
  std::vector<std::uint8_t> initial;
  std::unique_ptr<hv::DeviceBuffer> device;
};

inline Buffer buffer(const TestTensor &tensor, std::span<const float> values,
                     hv::Stream &stream) {
  auto encoded = io::encode(io::scatter(values, tensor), tensor.data_type);
  Buffer result{tensor,
                std::vector<std::uint8_t>(
                    tensor.binding_byte_offset + encoded.size() + 32, 0xA5),
                nullptr};
  std::copy(encoded.begin(), encoded.end(),
            result.initial.begin() + tensor.binding_byte_offset);
  result.device = std::make_unique<hv::DeviceBuffer>(result.initial.size());
  result.device->copy_from_host_at(result.initial.data(), result.initial.size(),
                                   0, stream.get());
  return result;
}

inline std::vector<float> read(Buffer &buffer, hv::Stream &stream,
                               bool output) {
  std::vector<std::uint8_t> bytes(buffer.initial.size());
  buffer.device->copy_to_host_at(bytes.data(), bytes.size(), 0, stream.get());
  stream.synchronize();
  const auto &t = buffer.tensor;
  const auto width = io::data_type_size(t.data_type);
  std::vector<bool> used(bytes.size(), false);
  if (output)
    for (std::size_t i = 0; i < io::element_count(t); ++i)
      for (std::size_t j = 0; j < width; ++j)
        used[t.binding_byte_offset + io::logical_offset(i, t) * width + j] =
            true;
  for (std::size_t i = 0; i < bytes.size(); ++i)
    if (!used[i] && bytes[i] != buffer.initial[i])
      throw std::runtime_error(output ? "Hygon modified output padding"
                                      : "Hygon modified input storage");
  const auto storage = io::storage_element_count(t);
  return io::gather(io::decode(std::span(bytes).subspan(t.binding_byte_offset,
                                                        storage * width),
                               t.data_type, storage),
                    t);
}

inline double tolerance(flagdnnDataType_t type) {
  return type == FLAGDNN_DATA_BFLOAT16  ? 8e-3
         : type == FLAGDNN_DATA_FLOAT16 ? 1e-3
                                        : 2e-4;
}

template <class Build, class Reference>
void run_case(
    std::string_view name, std::span<const TestTensor> inputs,
    std::span<const TestTensor> outputs, Values values,
    const std::function<flagdnn::Handle &()> &handle, hv::Stream &stream,
    Build build, Reference reference, double atol = -1, double rtol = -1,
    std::span<const std::pair<double, double>> output_tolerances = {},
    ComparisonRule comparison = ComparisonRule::kEitherTolerance,
    ReferenceOutput reference_output = ReferenceOutput::kStoragePrecision) {
  if (!output_tolerances.empty() && output_tolerances.size() != outputs.size())
    throw std::logic_error("output tolerance count mismatch");
  if (values.size() != inputs.size())
    throw std::logic_error("host reference input count mismatch");
  std::vector<Buffer> buffers;
  std::vector<flagdnnBinding_t> bindings;
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    buffers.push_back(buffer(inputs[i], values[i], stream));
    values[i] = io::decode(io::encode(values[i], inputs[i].data_type),
                           inputs[i].data_type, values[i].size());
  }
  for (const auto &out : outputs)
    buffers.push_back(
        buffer(out, std::vector<float>(io::element_count(out), 0), stream));
  for (const auto &b : buffers)
    bindings.push_back(
        {b.tensor.uid, static_cast<std::uint8_t *>(b.device->opaque()) +
                           b.tensor.binding_byte_offset});
  auto expected = reference(values);
  if (expected.size() != outputs.size())
    throw std::logic_error("host reference output count mismatch");
  auto executable = build(handle());
  hv::DeviceBuffer workspace(executable->workspace_size());
  pw::execute(*executable, bindings, workspace, stream);
  stream.synchronize();
  for (std::size_t i = 0; i < inputs.size(); ++i)
    (void)read(buffers[i], stream, false);
  Values first_outputs;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto &out = outputs[i];
    const auto actual = read(buffers[inputs.size() + i], stream, true);
    const auto rounded =
        reference_output == ReferenceOutput::kStoragePrecision
            ? io::decode(io::encode(expected[i], out.data_type), out.data_type,
                         expected[i].size())
            : std::vector<float>{};
    const std::span<const float> wanted =
        reference_output == ReferenceOutput::kStoragePrecision
            ? std::span<const float>(rounded)
            : std::span<const float>(expected[i]);
    const auto limits =
        output_tolerances.empty()
            ? std::pair{atol < 0 ? tolerance(out.data_type) : atol,
                        rtol < 0 ? tolerance(out.data_type) : rtol}
            : output_tolerances[i];
    compare_outputs(actual, wanted, limits.first, limits.second, name,
                    "independent CPU reference", comparison);
    if (comparison == ComparisonRule::kCombinedTolerance)
      first_outputs.push_back(actual);
  }
  if (comparison == ComparisonRule::kCombinedTolerance) {
    // Like NV's paired runner, verify deterministic re-execution with the
    // same inputs and RNG counters, including input and padding guards.
    pw::execute(*executable, bindings, workspace, stream);
    for (std::size_t i = 0; i < inputs.size(); ++i)
      (void)read(buffers[i], stream, false);
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      const auto actual = read(buffers[inputs.size() + i], stream, true);
      if (std::memcmp(actual.data(), first_outputs[i].data(),
                      actual.size() * sizeof(float)) != 0)
        throw std::runtime_error(std::string(name) +
                                 " repeated execution differs");
    }
  }
  std::cout << name << ": FlagDNN Graph vs independent CPU reference PASS"
            << std::endl;
}

inline Buffer raw_buffer(const TestTensor &tensor, hv::Stream &stream) {
  Buffer b{
      tensor,
      std::vector<std::uint8_t>(tensor.binding_byte_offset +
                                    io::storage_element_count(tensor) *
                                        io::data_type_size(tensor.data_type) +
                                    32,
                                0xA5),
      nullptr};
  b.device = std::make_unique<hv::DeviceBuffer>(b.initial.size());
  (void)stream;
  return b;
}
inline std::uint8_t *raw_element(Buffer &b, std::size_t i) {
  return b.initial.data() + b.tensor.binding_byte_offset +
         io::logical_offset(i, b.tensor) *
             io::data_type_size(b.tensor.data_type);
}

template <class Build>
void execute_raw(std::string_view name, std::vector<Buffer> &buffers,
                 std::vector<std::uint8_t> expected,
                 const std::function<flagdnn::Handle &()> &handle,
                 hv::Stream &stream, Build build) {
  std::vector<flagdnnBinding_t> bindings;
  for (auto &b : buffers) {
    b.device->copy_from_host_at(b.initial.data(), b.initial.size(), 0,
                                stream.get());
    bindings.push_back(
        {b.tensor.uid, static_cast<std::uint8_t *>(b.device->opaque()) +
                           b.tensor.binding_byte_offset});
  }
  auto executable = build(handle());
  hv::DeviceBuffer workspace(executable->workspace_size());
  pw::execute(*executable, bindings, workspace, stream);
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    auto &b = buffers[i];
    std::vector<std::uint8_t> actual(b.initial.size());
    b.device->copy_to_host_at(actual.data(), actual.size(), 0, stream.get());
    stream.synchronize();
    const auto &wanted = i + 1 == buffers.size() ? expected : b.initial;
    if (actual != wanted) {
      const auto mismatch =
          std::mismatch(actual.begin(), actual.end(), wanted.begin()).first -
          actual.begin();
      throw std::runtime_error(
          std::string(name) + " raw byte mismatch at tensor " +
          std::to_string(b.tensor.uid) + " byte " + std::to_string(mismatch));
    }
  }
  std::cout << name << ": FlagDNN Graph vs exact host bytes PASS" << std::endl;
}

template <class Source, class Build>
void run_copy(std::string_view name, std::span<const TestTensor> inputs,
              const TestTensor &output,
              const std::function<flagdnn::Handle &()> &handle,
              hv::Stream &stream, Source source, Build build) {
  std::vector<Buffer> buffers;
  constexpr std::uint32_t patterns[] = {0,           0x80000000U, 0x7fc12345U,
                                        0x3f800001U, 0x7fffffffU, 0xffffffffU};
  for (const auto &t : inputs) {
    buffers.push_back(raw_buffer(t, stream));
    for (std::size_t i = 0; i < io::element_count(t); ++i) {
      auto *ptr = raw_element(buffers.back(), i);
      const auto width = io::data_type_size(t.data_type);
      std::uint32_t bits =
          patterns[(i + t.uid) % 6] ^ (i % 3 == 0 ? std::uint32_t(i * 97) : 0U);
      if (width == 1)
        bits = t.data_type == FLAGDNN_DATA_BOOLEAN
                   ? (i + t.uid) % 2
                   : (i * 17 + t.uid * 23) % 256;
      std::memcpy(ptr, &bits, width);
    }
  }
  buffers.push_back(raw_buffer(output, stream));
  auto expected = buffers.back().initial;
  const auto width = io::data_type_size(output.data_type);
  for (std::size_t i = 0; i < io::element_count(output); ++i) {
    const auto [tensor, offset] = source(i);
    std::memcpy(expected.data() + output.binding_byte_offset +
                    io::logical_offset(i, output) * width,
                raw_element(buffers.at(tensor), offset), width);
  }
  execute_raw(name, buffers, std::move(expected), handle, stream, build);
}

template <class Build, class Evaluate>
void run_integer_case(const PointwiseTestCase &c,
                      const std::function<flagdnn::Handle &()> &handle,
                      hv::Stream &stream, Build build, Evaluate evaluate) {
  std::vector<Buffer> buffers;
  std::vector<std::vector<std::int32_t>> values;
  for (std::size_t j = 0; j < c.inputs.size(); ++j) {
    const auto &t = c.inputs[j];
    buffers.push_back(raw_buffer(t, stream));
    values.emplace_back(io::element_count(t));
    for (std::size_t i = 0; i < values.back().size(); ++i) {
      values.back()[i] = t.data_type == FLAGDNN_DATA_BOOLEAN
                             ? (i + j) % 2
                             : pointwise_integer_input(i, j, c.mode);
      std::memcpy(raw_element(buffers.back(), i), &values.back()[i],
                  io::data_type_size(t.data_type));
    }
  }
  buffers.push_back(raw_buffer(c.output, stream));
  auto expected = buffers.back().initial;
  for (std::size_t i = 0; i < io::element_count(c.output); ++i) {
    const auto input = [&](std::size_t j) {
      return j < values.size()
                 ? values[j][pw::broadcast_index(i, c.inputs[j], c.output)]
                 : 0;
    };
    const auto result = evaluate(input(0), input(1), input(2) != 0,
                                 static_cast<std::int32_t>(c.alpha));
    std::memcpy(expected.data() + c.output.binding_byte_offset +
                    io::logical_offset(i, c.output) *
                        io::data_type_size(c.output.data_type),
                &result, io::data_type_size(c.output.data_type));
  }
  execute_raw(c.name, buffers, std::move(expected), handle, stream, build);
}
inline void
run_integer_pointwise(const PointwiseTestCase &c,
                      const std::function<flagdnn::Handle &()> &handle,
                      hv::Stream &stream) {
  run_integer_case(
      c, handle, stream,
      [&](flagdnn::Handle &h) { return build_flagdnn_pointwise(h, c); },
      [&](auto a, auto b, bool p, auto alpha) {
        return reference::cpu::pointwise_integer_reference(c.mode, a, b, p,
                                                           alpha);
      });
}

template <class Case, class Run>
int run_suite(int argc, char **argv, std::span<const Case> cases,
              std::string_view operation, bool benchmark, Run run) {
  if (cases.empty())
    throw std::invalid_argument("empty host-reference case catalog");
  if constexpr (
      requires { cases.front().inputs; } || requires { cases.front().input; }) {
    if (benchmark)
      return hygon::skip_cases(
          argc, std::vector<Case>(cases.begin(), cases.end()), operation, true,
          "functional CPU reference available; no equivalent hipDNN benchmark "
          "reference");
  } else if (benchmark) {
    throw std::logic_error(
        "CPU reference cannot be used for performance comparison");
  }
  std::string marker(operation);
  std::transform(marker.begin(), marker.end(), marker.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return pw::run_suite(
      argc, argv, operation, "FLAGDNN_" + marker + "_FUNCTIONAL",
      [&](const auto &handle, hv::Stream &stream, auto &matched, auto &executed,
          auto &skipped) {
        const char *filter = std::getenv("FLAGDNN_HYGON_CASE");
        if (!filter) {
          std::string family = marker;
          if (operation == "reshape" || operation == "transpose" ||
              operation == "slice")
            family = "LAYOUT";
          else if (operation == "add_square")
            family = "COMPOSITE";
          else if (operation == "genstats" || operation == "bn_finalize")
            family = "STATISTICS";
          else if (operation == "gen_index" || operation == "concatenate")
            family = "INDEX";
          else if (operation == "moe_grouped_matmul" ||
                   operation == "moe_grouped_matmul_bwd")
            family = "MOE_MATMUL";
          else if (operation == "causal_conv1d")
            family = "CAUSAL_CONVOLUTION";
          else if (operation == "matmul_fp8" || operation == "matmul")
            family = "FP8_MATMUL";
          else if (operation == "rope" || operation == "rope_backward")
            family = "ROPE";
          else if (operation.find("norm") != std::string_view::npos)
            family = "NORMALIZATION_EXTENDED";
          filter = std::getenv(("FLAGDNN_" + family + "_CASE").c_str());
        }
        for (const auto &test_case : cases) {
          if (filter && test_case.name.find(filter) == std::string::npos)
            continue;
          ++matched;
          if constexpr (requires { test_case.precision; }) {
            if (test_case.precision == 2) {
              pw::emit_skip(operation, test_case.name,
                            "Hygon does not implement TF32 input precision",
                            [&] {
                              std::vector<hv::ReferenceTensor> tensors;
                              for (const auto &t : test_case.inputs)
                                tensors.push_back(hv::as_reference_tensor(t));
                              for (const auto &t : test_case.outputs)
                                tensors.push_back(hv::as_reference_tensor(t));
                              return tensors;
                            }());
              ++skipped;
              continue;
            }
          }
          run(test_case, handle, stream);
          ++executed;
        }
      });
}
} // namespace flagdnn::testing::hygon_functional::host
