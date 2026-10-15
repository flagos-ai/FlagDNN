/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// Strict native cuDNN collection for the 90 active activation-backward cases.
// --list, --catalog and --self-test never initialize CUDA. This executable
// builds only native cuDNN references; it never builds a FlagDNN executable.
#include "common/pointwise.hpp"
#include "reference/cpu/benchmark.hpp"
#include "common/case.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/cudnn_graph.hpp"
#include <bit>
#include <ctime>
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
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FLAGDNN_SUPPLEMENT_SOURCE_REVISION
#define FLAGDNN_SUPPLEMENT_SOURCE_REVISION "unknown"
#endif

namespace supplement {
using namespace flagdnn::testing;
namespace io = flagdnn::testing::cuda;
namespace timing = flagdnn::validation::nvidia::timing;
using Bytes = std::vector<std::uint8_t>;
constexpr auto kBoolean = io::BooleanEncoding::kBitPacked;
constexpr std::size_t kGuard = 256;
constexpr int kWarmup = 10, kSamples = 20, kBatch = 50;

struct Entry {
  std::string op, name;
  std::vector<TestTensor> inputs;
  TestTensor output;
  PointwiseTestCase pointwise;
  double atol = 0, rtol = 0;
};
std::vector<Entry> catalog() {
  std::vector<Entry> all;
  for (const auto& [op, mode] :
       std::vector<std::pair<std::string, flagdnnPointwiseMode_t>>{
         {"elu_backward", FLAGDNN_POINTWISE_ELU_BWD},
         {"softplus_backward", FLAGDNN_POINTWISE_SOFTPLUS_BWD}}) {
    PointwiseCaseDefinition definition;
    definition.operation_name = op; definition.mode = mode;
    for (const auto& p : make_binary_pointwise_cases(definition)) {
      all.push_back({op, p.name, p.inputs, p.output, p,
                     p.absolute_tolerance, p.relative_tolerance});
    }
  }
  std::set<std::string> names;
  for (const auto& e : all)
    if (!names.insert(e.name).second) throw std::runtime_error("duplicate case");
  if (all.size() != 90) throw std::runtime_error("supplement requires 90 active cases");
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
    default: throw std::invalid_argument("unexpected supplemental dtype");
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
      << ",\"provider\":\"cudnn\",\"source_revision\":" << quote(FLAGDNN_SUPPLEMENT_SOURCE_REVISION)
      << ",\"catalog_source\":" << quote("tests/common/pointwise.cpp")
      << ",\"guard_prefix_bytes\":" << kGuard << ",\"guard_suffix_bytes\":" << kGuard
      << ",\"reference_layout_conversion\":false,\"tensors\":[";
  for (std::size_t i = 0; i < entry.inputs.size(); ++i) {
    if (i) out << ',';
    tensor(out, entry.inputs[i], "input");
  }
  out << ','; tensor(out, entry.output, "output");
  out << "],\"attributes\":{";
  {
    const auto& p = entry.pointwise;
    out << "\"mode\":" << int(p.mode) << ",\"alpha\":" << p.alpha
        << ",\"flags\":" << p.attributes.flags
        << ",\"relu_lower_clip\":" << p.attributes.relu_lower_clip
        << ",\"relu_upper_clip\":" << p.attributes.relu_upper_clip
        << ",\"relu_lower_clip_slope\":" << p.attributes.relu_lower_clip_slope
        << ",\"swish_beta\":" << p.attributes.swish_beta
        << ",\"elu_alpha\":" << p.attributes.elu_alpha
        << ",\"softplus_beta\":" << p.attributes.softplus_beta;
  }
  out << "},\"input_domains\":[";
  for (std::size_t i = 0; i < entry.pointwise.input_domains.size(); ++i) {
    if (i) out << ',';
    out << int(entry.pointwise.input_domains[i]);
  }
  out << "],\"input_generator\":{\"version\":1,\"endianness\":"
      << quote(std::endian::native == std::endian::little ? "little" : "big")
      << ",\"source\":" << quote("backends/nvidia/validation/functional/pointwise_runner.cpp::make_input")
      << ",\"formula\":" << quote("real=float(int((i*17+k*11)%41)-20)/float(13+k); kReal=real; kScaled=real*4; encode then decode actual dtype")
      << "}";
}
bool bit(const Bytes& bytes, std::size_t index) { return (bytes.at(index / 8) >> (index % 8)) & 1U; }
Bytes encode_input(const Entry& entry, std::size_t k) {
  const auto& t = entry.inputs[k];
  std::vector<float> values(io::element_count(t));
  const auto domain = entry.pointwise.input_domains.at(k);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = float(int((i * 17 + k * 11) % 41) - 20) / float(13 + k);
    if (domain == PointwiseInputDomain::kScaled) values[i] *= 4;
    else if (domain != PointwiseInputDomain::kReal) throw std::runtime_error("unexpected input domain");
  }
  return io::encode(io::scatter(values, t), t.data_type, kBoolean);
}
Bytes expected_bytes(const Entry& entry, const std::vector<Bytes>& inputs,
                     std::vector<double>& numeric) {
  const auto& output = entry.output;
  Bytes expected;
  {
    flagdnn::benchmarking::BenchmarkCase p;
    p.name = entry.name; p.operation = flagdnn::benchmarking::Operation::kPointwise;
    p.pointwise_mode = entry.pointwise.mode; p.pointwise_attributes = entry.pointwise.attributes;
    std::vector<std::vector<float>> logical;
    for (std::size_t k = 0; k < inputs.size(); ++k) {
      const auto& t = entry.inputs[k];
      p.tensors.push_back({t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
      logical.push_back(io::gather(io::decode(inputs[k], t.data_type, io::storage_element_count(t), kBoolean), t));
    }
    p.tensors.push_back({output.uid, output.data_type, output.dimensions, output.strides, output.binding_byte_offset});
    const auto quantize = [](std::span<const float> values, flagdnnDataType_t type) {
      return io::decode(io::encode(values, type, kBoolean), type, values.size(), kBoolean);
    };
    const auto values = flagdnn::reference::cpu::evaluate_benchmark(p, logical, quantize).back();
    numeric.assign(values.begin(), values.end());
    expected = io::encode(io::scatter(values, output), output.data_type, kBoolean);
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
  if (!numeric.empty()) {
    const auto physical = io::decode(payload, output.data_type, io::storage_element_count(output), kBoolean);
    const auto logical = io::gather(physical, output);
    for (std::size_t i = 0; i < logical.size(); ++i) {
      const double error = std::abs(double(logical[i]) - numeric[i]);
      if (!std::isfinite(error))
        throw std::runtime_error("nonfinite CPU pointwise comparison at element " + std::to_string(i));
      const double relative = error / std::max({std::abs(double(logical[i])), std::abs(numeric[i]), 1e-30});
      phase.max_abs = std::max(phase.max_abs, error);
      phase.max_rel = std::max(phase.max_rel, relative);
      if (error > entry.atol && relative > entry.rtol)
        throw std::runtime_error("CPU full pointwise mismatch at element " + std::to_string(i));
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
      if (!equal) {
        std::ostringstream message;
        message << "CPU bit-exact mismatch at element " << i;
        if (output.data_type == FLAGDNN_DATA_INT32) {
          std::int32_t a, e;
          std::memcpy(&a, payload.data() + offset * sizeof(a), sizeof(a));
          std::memcpy(&e, expected.data() + offset * sizeof(e), sizeof(e));
          message << " actual=" << a << " expected=" << e;
        }
        throw std::runtime_error(message.str());
      }
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
  // Floating outputs are NaN; exact outputs complement every logical bit.
  for (std::size_t i = 0; i < f.output.initial.size(); ++i)
    f.output.initial[i] ^= f.output.writable_mask[i];
  if (!f.numeric.empty()) {
    const auto poison = io::encode(std::vector<float>(io::element_count(entry.output),
        std::numeric_limits<float>::quiet_NaN()), entry.output.data_type, kBoolean);
    const auto width = io::data_type_size(entry.output.data_type);
    for (std::size_t i = 0; i < io::element_count(entry.output); ++i)
      std::copy_n(poison.begin() + i * width, width,
          f.output.initial.begin() + f.output.begin + io::logical_offset(i, entry.output) * width);
  }
  return f;
}
struct Result {
  std::string status = "ERROR", stage = "setup", reason, route, graph_error;
  std::size_t calls = 0, special_fixtures = 0;
  cudaDeviceProp device{};
  int driver = 0, runtime = 0;
  std::string graph_capture = "NOT_RUN", graph_replay = "NOT_RUN";
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
std::unique_ptr<TestExecutable> build(const Entry& entry, Result& result) {
  result.route = "cuDNN pointwise graph FLOAT compute";
  result.calls = 1;
  return build_pointwise_reference(entry.pointwise);
}
void measure(const Entry& entry, Result& result) {
  result.stage = "context";
  DriverContext context;
  Stream stream;
  io::check_cuda_runtime(cudaGetDeviceProperties(&result.device, 0), "cudaGetDeviceProperties");
  io::check_cuda_runtime(cudaDriverGetVersion(&result.driver), "cudaDriverGetVersion");
  io::check_cuda_runtime(cudaRuntimeGetVersion(&result.runtime), "cudaRuntimeGetVersion");
  result.stage = "build";
  const auto start = timing::HostClock::now();
  auto executable = build(entry, result);
  result.build_us = timing::host_microseconds_since(start);
  result.workspace = executable->workspace_size();
  result.stage = "input_preparation";
  auto host = fixture(entry);
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
  result.graph_capture = "PASS";
  result.stage = "graph_warmup";
  graph.launch(stream.get()); stream.synchronize();
  result.graph_replay = "PASS";
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
  out << ",\"device\":" << quote(r.device.name)
      << ",\"compute_capability\":[" << r.device.major << ',' << r.device.minor << ']'
      << ",\"l2_cache_bytes\":" << r.device.l2CacheSize
      << ",\"total_memory_bytes\":" << r.device.totalGlobalMem
      << ",\"cuda_driver\":" << r.driver << ",\"cuda_runtime\":" << r.runtime;
  const auto now = std::time(nullptr);
  out << ",\"timestamp_utc\":\"" << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ")
      << "\",\"cudnn_version\":" << cudnnGetVersion()
      << ",\"reference_route\":" << quote(r.route)
      << ",\"graph_build_error\":" << quote(r.graph_error)
      << ",\"native_calls_per_execution\":" << r.calls
      << ",\"special_encoding_fixtures\":" << r.special_fixtures
      << ",\"graph_capture\":" << quote(r.graph_capture)
      << ",\"graph_replay\":" << quote(r.graph_replay);
  out << ",\"status\":" << quote(r.status) << ",\"stage\":" << quote(r.stage)
      << ",\"reason\":" << quote(r.reason) << ",\"build_us\":" << r.build_us
      << ",\"workspace_bytes\":" << r.workspace << ",\"correctness\":{\"status\":"
      << quote(r.status == "PASS" ? "PASS" : r.phases.empty() ? "NOT_RUN" : "FAIL")
      << ",\"scope\":" << quote("cpu_full")
      << ",\"oracle\":" << quote("reference/cpu/benchmark.cpp::evaluate_benchmark")
      << ",\"comparison\":" << quote("quantized inputs/output; finite && (absolute <= atol || relative <= rtol)")
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
    for (std::size_t j = 0; j < io::element_count(entry.output); ++j) {
      auto partial = actual;
      const auto offset = io::logical_offset(j, entry.output);
      if (entry.output.data_type == FLAGDNN_DATA_BOOLEAN) {
        const auto at = f.output.begin + offset / 8;
        partial[at] ^= std::uint8_t(1U << (offset % 8));
      } else {
        const auto width = io::data_type_size(entry.output.data_type);
        const auto at = f.output.begin + offset * width;
        std::copy_n(f.output.initial.begin() + at, width, partial.begin() + at);
      }
      rejected = false;
      try { Phase p{"partial_write"}; validate_output(entry, f.output, partial, f.expected, f.numeric, p); }
      catch (const std::runtime_error&) { rejected = true; }
      if (!rejected) throw std::runtime_error("single missing write escaped validation");
    }
    actual.back() ^= 1;
    rejected = false;
    try { require_guards(f.output, actual); } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("boundary mutation escaped validation");

  }
  std::cout << "{\"status\":\"PASS\",\"gpu_initialized\":false,\"cases\":" << entries.size()
            << ",\"checked_elements\":" << checked << ",\"negative_controls_per_case\":3}\n";
}
}  // namespace supplement

int main(int argc, char** argv) {
  using namespace supplement;
  try {
    bool list = false, describe = false, test = false;
    std::set<std::string> selected;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--list") list = true;
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
        std::cout << "Usage: benchmark_nvidia_supplement [--list|--catalog|--self-test] [--case NAME] [--case-file PATH]\n";
        return 0;
      } else throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
    if (int(list) + int(describe) + int(test) > 1) throw std::invalid_argument("choose one listing/test mode");
    auto entries = catalog();
    std::set<std::string> known;
    for (const auto& e : entries) known.insert(e.name);
    for (const auto& name : selected)
      if (!known.contains(name)) throw std::invalid_argument("unknown supplemental case: " + name);
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
    std::cerr << "supplemental collector: " << error.what() << '\n';
    return 2;
  }
}
