/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// The 80 auxiliary cases in appendix I of the 2026-10-12 missing-case audit.
// --list, --catalog and --self-test never initialize CUDA. This executable
// builds only native cuDNN references; it never builds a FlagDNN executable.
#include "common/pointwise.hpp"
#include "common/layout.hpp"
#include "reference/cpu/layout.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "common/reduction.hpp"
#include "reference/cpu/matrix.hpp"
#include "validation/benchmark/timing.hpp"
#include "validation/tensor_io.hpp"

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLAGDNN_AUX_SOURCE_REVISION
#define FLAGDNN_AUX_SOURCE_REVISION "unknown"
#endif

namespace missing_aux {
using namespace flagdnn::testing;
namespace io = flagdnn::testing::cuda;
namespace timing = flagdnn::validation::nvidia::timing;
using Bytes = std::vector<std::uint8_t>;
constexpr auto kBoolean = io::BooleanEncoding::kBitPacked;
constexpr std::size_t kGuard = 256;
constexpr int kWarmup = 10, kSamples = 20, kBatch = 50;

struct Entry {
  std::string op, group, name;
  std::vector<TestTensor> inputs;
  TestTensor output;
  std::optional<PointwiseTestCase> pointwise;
  std::optional<ReductionTestCase> reduction;
  double atol = 0, rtol = 0;
  bool logical_domain = false;
  std::optional<LayoutTestCase> layout;
  bool shared_selection = false;
};

std::vector<Entry> catalog(bool all_shared = false) {
  std::vector<Entry> all;
  for (const std::string op : {"identity", "logical_and", "logical_or"}) {
    PointwiseCaseDefinition definition;
    definition.operation_name = op;
    definition.mode = op == "identity" ? FLAGDNN_POINTWISE_IDENTITY
                      : op == "logical_and" ? FLAGDNN_POINTWISE_LOGICAL_AND
                                            : FLAGDNN_POINTWISE_LOGICAL_OR;
    definition.input_domain = op == "identity" ? PointwiseInputDomain::kReal
                                                : PointwiseInputDomain::kLogical;
    const auto cases = op == "identity" ? make_unary_pointwise_cases(definition)
                                        : make_binary_pointwise_cases(definition);
    for (const auto& value : cases) {
      const auto type = value.inputs.front().data_type;
      if (op == "identity" && type != FLAGDNN_DATA_BFLOAT16 &&
          type != FLAGDNN_DATA_INT32 && type != FLAGDNN_DATA_BOOLEAN &&
          (!all_shared || type != FLAGDNN_DATA_FP8_E8M0)) continue;
      all.push_back({op, op == "identity" ? "auxiliary_copy" : "auxiliary_boolean",
                     value.name, value.inputs, value.output, value, {}, 0, 0,
                     op != "identity", {}});
    }
  }
  for (const auto& value : make_reduction_cases()) {
    if (value.name.find("_to_fp32_") == std::string::npos ||
        value.output.data_type != FLAGDNN_DATA_FLOAT32) continue;
    const bool bf16 = value.input.data_type == FLAGDNN_DATA_BFLOAT16 &&
                      value.input.dimensions.size() == 2 &&
                      value.mode == FLAGDNN_REDUCTION_ADD;
    const bool fp16 = value.input.data_type == FLAGDNN_DATA_FLOAT16 &&
        ((value.input.dimensions == std::vector<std::int64_t>{3, 5, 7} && value.axis == 1) ||
         (value.input.dimensions == std::vector<std::int64_t>{2, 7, 5, 3} && value.axis == 2));
    if (all_shared || bf16 || fp16)
      all.push_back({"reduction", "auxiliary_fp32_output", value.name,
                     {value.input}, value.output, {}, value,
                     value.absolute_tolerance, value.relative_tolerance, false, {}});
  }
  if (all_shared) {
    for (const auto& [op, mode] :
         {std::pair{"reshape", LayoutOperation::kReshape},
          std::pair{"transpose", LayoutOperation::kTranspose},
          std::pair{"slice", LayoutOperation::kSlice}}) {
      for (const auto& value : make_layout_cases(mode)) {
        Entry entry;
        entry.op = op; entry.group = "auxiliary_layout";
        entry.name = value.name; entry.inputs = {value.input};
        entry.output = value.output; entry.layout = value;
        all.push_back(std::move(entry));
      }
    }
  }
  for (auto& entry : all) entry.shared_selection = all_shared;
  std::set<std::string> names;
  for (const auto& entry : all)
    if (!names.insert(entry.name).second) throw std::runtime_error("duplicate auxiliary case");
  if (!all_shared && all.size() != 80) throw std::runtime_error("auxiliary catalog must contain exactly 80 cases");
  return all;
}

std::string quote(std::string_view value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
    else out << c;
  }
  return out.str() + '"';
}
template <class T> void array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
const char* dtype(flagdnnDataType_t type) {
  switch (type) {
    case FLAGDNN_DATA_FLOAT32: return "fp32";
    case FLAGDNN_DATA_FLOAT16: return "fp16";
    case FLAGDNN_DATA_BFLOAT16: return "bfloat16";
    case FLAGDNN_DATA_INT32: return "int32";
    case FLAGDNN_DATA_BOOLEAN: return "bool";
    case FLAGDNN_DATA_FP8_E4M3: return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2: return "fp8_e5m2";
    case FLAGDNN_DATA_FP8_E8M0: return "fp8_e8m0";
    default: throw std::invalid_argument("unexpected auxiliary dtype");
  }
}
void tensor(std::ostream& out, const TestTensor& t, std::string_view role) {
  out << "{\"uid\":" << t.uid << ",\"role\":" << quote(role)
      << ",\"dtype\":" << quote(dtype(t.data_type)) << ",\"shape\":";
  array(out, t.dimensions);
  out << ",\"strides\":"; array(out, t.strides);
  out << ",\"native_strides\":"; array(out, t.strides);
  out << ",\"reference_strides\":"; array(out, t.strides);
  out << ",\"binding_byte_offset\":" << t.binding_byte_offset
      << ",\"logical_elements\":" << io::element_count(t)
      << ",\"storage_elements\":" << io::storage_element_count(t)
      << ",\"reference_storage_bits\":" << (t.data_type == FLAGDNN_DATA_BOOLEAN ? 1 : io::data_type_size(t.data_type) * 8)
      << ",\"reference_storage_bytes\":" << io::encoded_byte_count(t, kBoolean)
      << ",\"logical_storage_bits\":" << io::data_type_size(t.data_type) * 8
      << ",\"boolean_encoding\":" << quote(t.data_type == FLAGDNN_DATA_BOOLEAN ? "bitpacked" : "not_applicable") << '}';
}
void metadata(std::ostream& out, const Entry& entry) {
  out << "\"op\":" << quote(entry.op) << ",\"case_name\":" << quote(entry.name)
      << ",\"provider\":\"cudnn\",\"group\":" << quote(entry.group)
      << ",\"source_revision\":" << quote(FLAGDNN_AUX_SOURCE_REVISION)
      << ",\"catalog_source\":" << quote(entry.layout ? "tests/common/layout.cpp" : entry.reduction ? "tests/common/reduction.cpp" : "tests/common/pointwise.cpp")
      << ",\"selection_source\":" << quote(entry.shared_selection
          ? "backends/iluvatar/validation/benchmark/shared_cases.cpp"
          : "backends/hygon/validation/benchmark/native_dtype.cpp")
      << ",\"reference_builder\":" << quote(entry.layout ? "build_layout_reference" : entry.reduction ? "build_reduction_reference" : "build_pointwise_reference")
      << ",\"reference_api\":" << quote(entry.layout ? "cuDNN Graph layout, with native cudnnTransformTensor fallback" : entry.reduction ? "cuDNN Graph, with native cudnnReduceTensor fallback" : "cuDNN Graph pointwise")
      << ",\"reference_layout_conversion\":false,\"guard_prefix_bytes\":" << kGuard
      << ",\"guard_suffix_bytes\":" << kGuard << ",\"tensors\":[";
  for (std::size_t i = 0; i < entry.inputs.size(); ++i) { if (i) out << ','; tensor(out, entry.inputs[i], "input"); }
  out << ','; tensor(out, entry.output, "output");
  out << "],\"intermediate_tensors\":[],\"attributes\":{";
  if (entry.layout) {
    const auto& layout = *entry.layout;
    out << "\"permutation\":";
    array(out, layout.permutation);
    out << ",\"slices\":[";
    for (std::size_t i = 0; i < layout.slices.size(); ++i) {
      if (i) out << ',';
      out << '[' << layout.slices[i].first << ',' << layout.slices[i].second << ']';
    }
    out << "],\"slice_strides\":";
    array(out, layout.slice_strides);
  } else if (entry.reduction) {
    const auto& r = *entry.reduction;
    out << "\"mode\":" << int(r.mode) << ",\"mode_name\":"
        << quote(r.mode == FLAGDNN_REDUCTION_ADD ? "sum" : r.mode == FLAGDNN_REDUCTION_AVG ? "avg" : "mul")
        << ",\"axis\":" << r.axis << ",\"keep_dimensions\":" << (r.keep_dimensions ? "true" : "false");
  } else {
    const auto& p = *entry.pointwise;
    out << "\"mode\":" << int(p.mode) << ",\"alpha\":" << p.alpha
        << ",\"flags\":" << p.attributes.flags
        << ",\"relu_lower_clip\":" << p.attributes.relu_lower_clip
        << ",\"relu_upper_clip\":" << p.attributes.relu_upper_clip
        << ",\"relu_lower_clip_slope\":" << p.attributes.relu_lower_clip_slope
        << ",\"swish_beta\":" << p.attributes.swish_beta
        << ",\"elu_alpha\":" << p.attributes.elu_alpha
        << ",\"softplus_beta\":" << p.attributes.softplus_beta;
  }
  out << "},\"inputgen\":{\"source\":\"backends/hygon/validation/benchmark/runner.cpp:make_input\",\"inputs\":[";
  for (std::size_t i = 0; i < entry.inputs.size(); ++i) {
    if (i) out << ',';
    out << "{\"uid\":" << entry.inputs[i].uid << ",\"input_index\":" << i
        << ",\"domain\":" << quote(entry.logical_domain ? "logical" : "real")
        << ",\"formula\":" << quote(entry.logical_domain ? "((index * 17 + input_index * 11) % 3) != 0"
          : "float(int((index * 17 + input_index * 11) % 41) - 20) / float(13 + input_index)")
        << ",\"conversion\":" << quote(entry.inputs[i].data_type == FLAGDNN_DATA_INT32
          ? "truncate toward zero; equivalent integer rational evaluation without FLOAT storage"
          : entry.inputs[i].data_type == FLAGDNN_DATA_BOOLEAN ? "nonzero to bool; pack little-endian bits"
          : "round to tensor dtype before CPU oracle") << '}';
  }
  out << "],\"padding_value\":-97}";
}

// These formulas deliberately match the benchmark runner, not the functional
// runner's separate fixtures. INT32 identity is generated and checked as raw
// integer storage; its current shared fixture contains only -1, 0 and 1.
std::vector<float> values(const Entry& entry, std::size_t input_index) {
  std::vector<float> result(io::element_count(entry.inputs.at(input_index)));
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (entry.logical_domain) result[i] = ((i * 17 + input_index * 11) % 3) != 0;
    else result[i] = float(int((i * 17 + input_index * 11) % 41) - 20) / float(13 + input_index);
  }
  return result;
}
Bytes encode_input(const Entry& entry, std::size_t index) {
  const auto& t = entry.inputs.at(index);
  if (t.data_type != FLAGDNN_DATA_INT32)
    return io::encode(io::scatter(values(entry, index), t), t.data_type, kBoolean);
  std::vector<std::int32_t> integers(io::storage_element_count(t), -97);
  for (std::size_t i = 0; i < io::element_count(t); ++i)
    integers[io::logical_offset(i, t)] =
        (int((i * 17 + index * 11) % 41) - 20) / int(13 + index);
  Bytes bytes(integers.size() * sizeof(std::int32_t));
  std::memcpy(bytes.data(), integers.data(), bytes.size());
  return bytes;
}
bool bit(const Bytes& bytes, std::size_t index) { return (bytes.at(index / 8) >> (index % 8)) & 1U; }
void set_bit(Bytes& bytes, std::size_t index, bool value) {
  auto& b = bytes.at(index / 8);
  const auto mask = std::uint8_t(1U << (index % 8));
  b = (b & std::uint8_t(~mask)) | (value ? mask : 0);
}
std::vector<double> reduction_expected(const Entry& entry, const Bytes& input) {
  const auto& r = *entry.reduction;
  const auto physical = io::decode(input, r.input.data_type, io::storage_element_count(r.input), kBoolean);
  const auto logical = io::gather(physical, r.input);
  const auto output = flagdnn::reference::cpu::evaluate_reduction(
      {r.input.dimensions, r.axis, r.mode}, logical);
  if (output.size() != io::element_count(r.output))
    throw std::runtime_error("CPU reduction oracle output size differs");
  return {output.begin(), output.end()};
}
Bytes expected_bytes(const Entry& entry, const std::vector<Bytes>& inputs,
                     std::vector<double>& numeric) {
  const auto& output = entry.output;
  Bytes expected(io::encoded_byte_count(output, kBoolean), 0xA5);
  if (entry.layout) {
    const auto& layout = *entry.layout;
    const flagdnn::reference::cpu::LayoutParameters parameters{
        entry.op, layout.input.dimensions, layout.output.dimensions,
        layout.permutation, layout.slices, layout.slice_strides};
    // Gather raw logical bytes before applying the independent CPU mapping.
    // BOOLEAN is stored as packed bits by cuDNN and copied as one logical byte
    // by the CPU oracle; all other types preserve every original storage bit.
    const auto width = output.data_type == FLAGDNN_DATA_BOOLEAN
        ? std::size_t{1} : io::data_type_size(output.data_type);
    Bytes logical(io::element_count(layout.input) * width);
    for (std::size_t i = 0; i < io::element_count(layout.input); ++i) {
      const auto offset = io::logical_offset(i, layout.input);
      if (output.data_type == FLAGDNN_DATA_BOOLEAN)
        logical[i] = bit(inputs.front(), offset);
      else
        std::copy_n(inputs.front().begin() + offset * width, width,
                    logical.begin() + i * width);
    }
    const auto transformed = flagdnn::reference::cpu::evaluate_layout_bytes(
        parameters, logical, width);
    for (std::size_t i = 0; i < io::element_count(output); ++i) {
      const auto offset = io::logical_offset(i, output);
      if (output.data_type == FLAGDNN_DATA_BOOLEAN)
        set_bit(expected, offset, transformed.at(i) != 0);
      else
        std::copy_n(transformed.begin() + i * width, width,
                    expected.begin() + offset * width);
    }
    return expected;
  }
  if (entry.reduction) {
    numeric = reduction_expected(entry, inputs.front());
    std::vector<float> logical(numeric.begin(), numeric.end());
    return io::encode(io::scatter(logical, output), output.data_type, kBoolean);
  }
  for (std::size_t i = 0; i < io::element_count(output); ++i) {
    const auto source = io::logical_offset(i, entry.inputs.front());
    const auto destination = io::logical_offset(i, output);
    if (output.data_type == FLAGDNN_DATA_BOOLEAN) {
      const bool a = bit(inputs.front(), source);
      const bool b = inputs.size() == 2 ? bit(inputs[1], io::logical_offset(i, entry.inputs[1])) : false;
      set_bit(expected, destination, entry.op == "identity" ? a : entry.op == "logical_and" ? a && b : a || b);
    } else {
      const auto width = io::data_type_size(output.data_type);
      std::copy_n(inputs.front().begin() + source * width, width, expected.begin() + destination * width);
    }
  }
  return expected;
}
struct HostBuffer {
  TestTensor tensor;
  Bytes initial, writable_mask;
  std::size_t begin = 0, payload_size = 0;
};
HostBuffer guarded(const TestTensor& t, const Bytes& payload, bool output) {
  HostBuffer result;
  result.tensor = t;
  result.begin = kGuard + t.binding_byte_offset;
  result.payload_size = payload.size();
  result.initial.assign(result.begin + payload.size() + kGuard, 0xA5);
  std::copy(payload.begin(), payload.end(), result.initial.begin() + result.begin);
  result.writable_mask.assign(result.initial.size(), 0);
  if (output) for (std::size_t i = 0; i < io::element_count(t); ++i) {
    const auto offset = io::logical_offset(i, t);
    if (t.data_type == FLAGDNN_DATA_BOOLEAN)
      result.writable_mask[result.begin + offset / 8] |= std::uint8_t(1U << (offset % 8));
    else {
      const auto width = io::data_type_size(t.data_type);
      std::fill_n(result.writable_mask.begin() + result.begin + offset * width, width, 0xFF);
    }
  }
  return result;
}
void require_guards(const HostBuffer& host, const Bytes& actual) {
  if (actual.size() != host.initial.size()) throw std::runtime_error("guard storage size differs");
  for (std::size_t i = 0; i < actual.size(); ++i)
    if (((actual[i] ^ host.initial[i]) & std::uint8_t(~host.writable_mask[i])) != 0)
      throw std::runtime_error("input, padding or allocation boundary modified: uid=" +
          std::to_string(host.tensor.uid) + " byte=" + std::to_string(i));
}
struct Phase {
  std::string name, status = "NOT_RUN";
  std::size_t checked = 0;
  double max_abs = 0, max_rel = 0;
  bool inputs_unchanged = false, guards_intact = false;
};
void validate_output(const Entry& entry, const HostBuffer& host, const Bytes& actual,
                     const Bytes& expected, const std::vector<double>& numeric, Phase& phase) {
  phase.status = "FAIL";
  require_guards(host, actual);
  phase.guards_intact = true;
  const auto& output = entry.output;
  Bytes payload(actual.begin() + host.begin, actual.begin() + host.begin + host.payload_size);
  if (entry.reduction) {
    const auto physical = io::decode(payload, output.data_type, io::storage_element_count(output), kBoolean);
    const auto logical = io::gather(physical, output);
    for (std::size_t i = 0; i < logical.size(); ++i) {
      const double error = std::abs(double(logical[i]) - numeric[i]);
      if (!std::isfinite(error))
        throw std::runtime_error("nonfinite CPU reduction comparison at element " + std::to_string(i));
      const double relative = error / std::max({std::abs(double(logical[i])), std::abs(numeric[i]), 1e-30});
      phase.max_abs = std::max(phase.max_abs, error);
      phase.max_rel = std::max(phase.max_rel, relative);
      if (error > entry.atol && relative > entry.rtol)
        throw std::runtime_error("CPU full reduction mismatch at element " + std::to_string(i));
      ++phase.checked;
    }
  } else {
    for (std::size_t i = 0; i < io::element_count(output); ++i) {
      const auto offset = io::logical_offset(i, output);
      bool equal;
      if (output.data_type == FLAGDNN_DATA_BOOLEAN) equal = bit(payload, offset) == bit(expected, offset);
      else {
        const auto width = io::data_type_size(output.data_type);
        equal = std::equal(payload.begin() + offset * width, payload.begin() + (offset + 1) * width,
                           expected.begin() + offset * width);
      }
      if (!equal) throw std::runtime_error("CPU bit-exact mismatch at element " + std::to_string(i));
      ++phase.checked;
    }
  }
  phase.status = "PASS";
}
struct Fixture {
  std::vector<HostBuffer> inputs;
  HostBuffer output;
  Bytes expected;
  std::vector<double> numeric;
};
Fixture fixture(const Entry& entry) {
  Fixture f;
  std::vector<Bytes> inputs;
  for (std::size_t i = 0; i < entry.inputs.size(); ++i) {
    inputs.push_back(encode_input(entry, i));
    f.inputs.push_back(guarded(entry.inputs[i], inputs.back(), false));
  }
  f.expected = expected_bytes(entry, inputs, f.numeric);
  f.output = guarded(entry.output, f.expected, true);
  // Every logical output begins with a value different from its CPU oracle.
  // Floating outputs are NaN; exact-copy outputs complement every output bit.
  for (std::size_t i = 0; i < f.output.initial.size(); ++i)
    f.output.initial[i] ^= f.output.writable_mask[i];
  if (entry.reduction) {
    const float poison = std::numeric_limits<float>::quiet_NaN();
    for (std::size_t i = 0; i < io::element_count(entry.output); ++i)
      std::memcpy(f.output.initial.data() + f.output.begin + io::logical_offset(i, entry.output) * sizeof(float), &poison, sizeof(float));
  }
  return f;
}
struct Result {
  std::string status = "ERROR", stage = "setup", reason;
  std::vector<double> samples, batches;
  std::vector<Phase> phases;
  double build_us = 0;
  std::size_t workspace = 0;
};
Bytes read(const DeviceBuffer& buffer, const HostBuffer& host, Stream& stream) {
  Bytes bytes(host.initial.size());
  buffer.copy_to_host(bytes.data(), bytes.size(), stream.get());
  stream.synchronize();
  return bytes;
}
void measure(const Entry& entry, Result& result) {
  result.stage = "context";
  DriverContext context;
  Stream stream;
  result.stage = "build";
  const auto start = timing::HostClock::now();
  auto executable = entry.layout ? build_layout_reference(*entry.layout)
                    : entry.reduction ? build_reduction_reference(*entry.reduction)
                                    : build_pointwise_reference(*entry.pointwise);
  result.build_us = timing::host_microseconds_since(start);
  result.workspace = executable->workspace_size();
  result.stage = "input_preparation";
  const auto host = fixture(entry);
  std::vector<std::unique_ptr<DeviceBuffer>> input_buffers;
  std::vector<flagdnnBinding_t> bindings;
  for (const auto& input : host.inputs) {
    auto buffer = std::make_unique<DeviceBuffer>(input.initial.size());
    buffer->copy_from_host(input.initial.data(), input.initial.size(), stream.get());
    bindings.push_back({input.tensor.uid, buffer->opaque_at(input.begin)});
    input_buffers.push_back(std::move(buffer));
  }
  DeviceBuffer output(host.output.initial.size()), workspace(result.workspace);
  output.copy_from_host(host.output.initial.data(), host.output.initial.size(), stream.get());
  bindings.push_back({entry.output.uid, output.opaque_at(host.output.begin)});
  executable->prepare(bindings, stream.opaque());
  stream.synchronize();
  const auto launch = [&] { executable->execute(bindings, workspace.opaque(), result.workspace, stream.opaque()); };
  const auto validate = [&](std::string name) {
    result.stage = name;
    result.phases.push_back({std::move(name)});
    auto& phase = result.phases.back();
    phase.status = "FAIL";
    for (std::size_t i = 0; i < input_buffers.size(); ++i)
      require_guards(host.inputs[i], read(*input_buffers[i], host.inputs[i], stream));
    phase.inputs_unchanged = true;
    validate_output(entry, host.output, read(output, host.output, stream), host.expected, host.numeric, phase);
  };
  result.stage = "execute";
  launch(); stream.synchronize();
  validate("before_timing");
  result.stage = "warmup";
  for (int i = 0; i < kWarmup; ++i) launch();
  stream.synchronize();
  result.stage = "capture";
  timing::CapturedExecutionBatch graph(stream.get(), kBatch, launch);
  result.stage = "graph_warmup";
  graph.launch(stream.get()); stream.synchronize();
  result.stage = "timing";
  EventTimer timer;
  for (int i = 0; i < kSamples; ++i) {
    const auto batch = timer.measure_microseconds(stream.get(), 1, [&] { graph.launch(stream.get()); });
    if (!std::isfinite(batch) || batch <= 0) throw std::runtime_error("invalid timing sample");
    result.batches.push_back(batch); result.samples.push_back(batch / kBatch);
  }
  validate("after_timing");
  result.stage = "reset_output";
  output.copy_from_host(host.output.initial.data(), host.output.initial.size(), stream.get());
  stream.synchronize();
  result.stage = "reset_replay";
  graph.launch(stream.get()); stream.synchronize();
  validate("after_reset_replay");
  result.status = "PASS"; result.stage = "complete";
}
void emit(std::ostream& out, const Entry& entry, const Result& r) {
  out << std::setprecision(17) << '{'; metadata(out, entry);
  out << ",\"status\":" << quote(r.status) << ",\"stage\":" << quote(r.stage)
      << ",\"reason\":" << quote(r.reason) << ",\"build_us\":" << r.build_us
      << ",\"workspace_bytes\":" << r.workspace << ",\"correctness\":{\"status\":"
      << quote(r.status == "PASS" ? "PASS" : r.phases.empty() ? "NOT_RUN" : "FAIL")
      << ",\"scope\":" << quote(entry.layout || entry.op == "identity" ? "bit_exact" : "cpu_full")
      << ",\"oracle\":" << quote(entry.layout ? "reference/cpu/layout.cpp:evaluate_layout_bytes; every logical storage byte"
         : entry.reduction ? "reference/cpu/matrix.cpp:evaluate_reduction; quantized inputs, double accumulation, FP32 output"
         : entry.op == "identity" ? "copy input raw encoding at every logical element" : "CPU boolean truth table at every logical element")
      << ",\"comparison\":" << quote(entry.reduction ? "finite && (absolute <= atol || absolute/max(abs(actual),abs(reference),1e-30) <= rtol)" : "exact logical bits or storage bytes")
      << ",\"absolute_tolerance\":" << entry.atol << ",\"relative_tolerance\":" << entry.rtol
      << ",\"total_elements\":" << io::element_count(entry.output)
      << ",\"checked_elements\":" << (r.phases.empty() ? 0 : r.phases.back().checked)
      << ",\"phases\":[";
  for (std::size_t i = 0; i < r.phases.size(); ++i) {
    if (i) out << ',';
    const auto& p = r.phases[i];
    out << "{\"phase\":" << quote(p.name) << ",\"status\":" << quote(p.status)
        << ",\"checked_elements\":" << p.checked << ",\"maximum_absolute_error\":" << p.max_abs
        << ",\"maximum_relative_error\":" << p.max_rel
        << ",\"inputs_unchanged\":" << (p.inputs_unchanged ? "true" : "false")
        << ",\"padding_and_boundaries_intact\":" << (p.guards_intact ? "true" : "false") << '}';
  }
  out << "]},\"timing\":{\"method\":\"cuda_event_cuda_graph_batch\",\"unit\":\"us\",\"warmup_iterations\":" << kWarmup
      << ",\"sample_count\":" << kSamples << ",\"iterations_per_sample\":" << kBatch
      << ",\"graph_warmup_launches\":1,\"percentile_method\":\"nearest_rank\",\"samples_us\":";
  array(out, r.samples); out << ",\"sample_batches_us\":"; array(out, r.batches);
  if (r.status == "PASS")
    out << ",\"median_us\":" << timing::percentile(r.samples, 0.5)
        << ",\"p90_us\":" << timing::percentile(r.samples, 0.9);
  out << "}}\n";
}

void self_test(const std::vector<Entry>& entries) {
  std::size_t checked = 0;
  for (const auto& entry : entries) {
    const auto f = fixture(entry);
    Bytes actual = f.output.initial;
    for (std::size_t i = 0; i < f.expected.size(); ++i) {
      const auto at = f.output.begin + i;
      actual[at] = (actual[at] & std::uint8_t(~f.output.writable_mask[at])) |
                   (f.expected[i] & f.output.writable_mask[at]);
    }
    Phase good{"cpu_oracle_self_test"};
    validate_output(entry, f.output, actual, f.expected, f.numeric, good);
    checked += good.checked;
    // Validate negative controls: unwritten output and input/guard corruption.
    bool rejected = false;
    try { Phase p{"unwritten"}; validate_output(entry, f.output, f.output.initial, f.expected, f.numeric, p); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("unwritten output escaped validation");
    auto corrupt = f.inputs.front().initial;
    corrupt[f.inputs.front().begin] ^= 1;
    rejected = false;
    try { require_guards(f.inputs.front(), corrupt); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("input mutation escaped validation");
    actual.back() ^= 1;
    rejected = false;
    try { require_guards(f.output, actual); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("boundary mutation escaped validation");
    if (entry.output.data_type == FLAGDNN_DATA_INT32) {
      const auto generated = encode_input(entry, 0);
      const auto hygon = io::encode(io::scatter(values(entry, 0), entry.inputs.front()), FLAGDNN_DATA_INT32, kBoolean);
      if (generated != hygon) throw std::runtime_error("integer fixture differs from Hygon");
    }
  }
  std::cout << "{\"status\":\"PASS\",\"gpu_initialized\":false,\"cases\":" << entries.size()
            << ",\"checked_elements\":" << checked << ",\"negative_controls_per_case\":3}\n";
}
}  // namespace missing_aux

int main(int argc, char** argv) {
  using namespace missing_aux;
  try {
    bool list = false, describe = false, test = false, all_shared = false;
    std::set<std::string> selected;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--all-shared") all_shared = true;
      else if (arg == "--list") list = true;
      else if (arg == "--catalog") describe = true;
      else if (arg == "--self-test") test = true;
      else if (arg == "--case" && i + 1 < argc) selected.insert(argv[++i]);
      else if (arg == "--case-file" && i + 1 < argc) {
        std::ifstream file(argv[++i]);
        if (!file) throw std::invalid_argument("cannot open case file");
        std::string line;
        while (std::getline(file, line)) {
          const auto first = line.find_first_not_of(" \t\r");
          if (first == std::string::npos || line[first] == '#') continue;
          const auto last = line.find_last_not_of(" \t\r");
          selected.insert(line.substr(first, last - first + 1));
        }
        if (selected.empty()) throw std::invalid_argument("case file contains no cases");
      } else if (arg == "--help") {
        std::cout << "Usage: benchmark_nvidia_missing_aux [--all-shared] [--list|--catalog|--self-test] [--case NAME] [--case-file PATH]\n";
        return 0;
      } else throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    if (int(list) + int(describe) + int(test) > 1) throw std::invalid_argument("choose one listing/test mode");
    auto entries = catalog(all_shared);
    std::set<std::string> known;
    for (const auto& e : entries) known.insert(e.name);
    for (const auto& name : selected)
      if (!known.contains(name)) throw std::invalid_argument("unknown auxiliary case: " + name);
    if (!selected.empty()) std::erase_if(entries, [&](const auto& e) { return !selected.contains(e.name); });
    if (test) { self_test(entries); return 0; }
    for (const auto& entry : entries) {
      if (list) { std::cout << entry.name << '\n'; continue; }
      if (describe) {
        Result r; r.status = "CATALOG"; r.stage = "catalog";
        emit(std::cout, entry, r); continue;
      }
      Result result;
      try { measure(entry, result); }
      catch (const std::exception& error) { result.reason = error.what(); }
      emit(std::cout, entry, result); std::cout.flush();
      // Stop on the first failure; the caller can isolate/retry the next case
      // in a fresh process if a CUDA failure left the context unusable.
      if (result.status != "PASS") return 3;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "auxiliary collector: " << error.what() << '\n';
    return 2;
  }
}
