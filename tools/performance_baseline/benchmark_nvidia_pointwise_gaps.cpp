// Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0
// Real shared native cases omitted from the ordinary performance catalog.
// --list is CPU-only; --group attributes|integer|identity_fp8 selects a subset.
// --probe-first selects one existing case per operator/dtype for capability probes.
#include "../../backends/nvidia/validation/functional/pointwise_runner.cpp"
#include "common/add.hpp"
#include "common/composite.hpp"
#include "../../benchmark/common/case.hpp"
#include "reference/cpu/benchmark.hpp"
#include "reference/cpu/pointwise.hpp"
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <optional>
#include <set>

namespace gap_reference {
using namespace flagdnn::testing;
namespace io = flagdnn::testing::cuda;
namespace timing = flagdnn::validation::nvidia::timing;
namespace cpu = flagdnn::reference::cpu;

struct Entry {
  std::string op, group;
  PointwiseTestCase spec;
  std::optional<AddTestCase> add;
  std::optional<AddSquareTestCase> square;
};

std::vector<Entry> catalog() {
  std::vector<Entry> result;
  for (const std::string op : {"relu_backward", "leaky_relu_backward", "swish_backward"}) {
    PointwiseCaseDefinition definition;
    definition.operation_name = op;
    definition.mode = op == "swish_backward" ? FLAGDNN_POINTWISE_SWISH_BWD : FLAGDNN_POINTWISE_RELU_BWD;
    if (op == "leaky_relu_backward") {
      definition.attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
      definition.attributes.relu_lower_clip_slope = 0.2;
    }
    for (auto spec : make_binary_pointwise_cases(definition))
      if (spec.name.find("_attributes_") != std::string::npos)
        result.push_back({op, "attributes", std::move(spec), {}, {}});
  }
  for (const auto& [op, mode] : std::vector<std::pair<std::string, flagdnnPointwiseMode_t>>{
      {"sub", FLAGDNN_POINTWISE_SUB}, {"mul", FLAGDNN_POINTWISE_MUL},
      {"div", FLAGDNN_POINTWISE_DIV}, {"pow", FLAGDNN_POINTWISE_POW},
      {"max", FLAGDNN_POINTWISE_MAX}, {"min", FLAGDNN_POINTWISE_MIN},
      {"mod", FLAGDNN_POINTWISE_MOD}, {"scale", FLAGDNN_POINTWISE_MUL},
      {"cmp_eq", FLAGDNN_POINTWISE_CMP_EQ}, {"cmp_neq", FLAGDNN_POINTWISE_CMP_NEQ},
      {"cmp_gt", FLAGDNN_POINTWISE_CMP_GT}, {"cmp_ge", FLAGDNN_POINTWISE_CMP_GE},
      {"cmp_lt", FLAGDNN_POINTWISE_CMP_LT}, {"cmp_le", FLAGDNN_POINTWISE_CMP_LE},
      {"binary_select", FLAGDNN_POINTWISE_BINARY_SELECT}}) {
    PointwiseCaseDefinition definition;
    definition.operation_name = op;
    definition.mode = mode;
    const auto cases = mode == FLAGDNN_POINTWISE_BINARY_SELECT
        ? make_binary_select_cases(definition) : make_binary_pointwise_cases(definition);
    for (auto spec : cases)
      if (spec.inputs.front().data_type == FLAGDNN_DATA_INT32)
        result.push_back({op, "integer", std::move(spec), {}, {}});
  }
  for (const auto& source : make_add_cases()) {
    if (source.left.data_type != FLAGDNN_DATA_INT32) continue;
    PointwiseTestCase spec;
    spec.name = source.name; spec.mode = FLAGDNN_POINTWISE_ADD;
    spec.inputs = {source.left, source.right}; spec.output = source.output;
    spec.alpha = source.alpha;
    result.push_back({"add", "integer", std::move(spec), source, {}});
  }
  for (const auto& source : make_add_square_cases()) {
    if (source.left.data_type != FLAGDNN_DATA_INT32) continue;
    PointwiseTestCase spec;
    spec.name = source.name; spec.mode = FLAGDNN_POINTWISE_ADD;
    spec.inputs = {source.left, source.right}; spec.output = source.output;
    result.push_back({"add_square", "integer", std::move(spec), {}, source});
  }
  for (auto spec : make_unary_pointwise_cases(
      {.operation_name = "identity", .mode = FLAGDNN_POINTWISE_IDENTITY})) {
    if (spec.inputs.front().data_type == FLAGDNN_DATA_FP8_E4M3 ||
        spec.inputs.front().data_type == FLAGDNN_DATA_FP8_E5M2)
      result.push_back({"identity", "identity_fp8", std::move(spec), {}, {}});
  }
  return result;
}

std::string quote(std::string_view value) {
  std::ostringstream out; out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(c) << std::dec;
    else out << c;
  }
  out << '"'; return out.str();
}
const char* dtype(flagdnnDataType_t value) {
  switch (value) {
    case FLAGDNN_DATA_FLOAT32: return "fp32";
    case FLAGDNN_DATA_FLOAT16: return "fp16";
    case FLAGDNN_DATA_BFLOAT16: return "bfloat16";
    case FLAGDNN_DATA_INT32: return "int32";
    case FLAGDNN_DATA_BOOLEAN: return "boolean_bitpacked";
    case FLAGDNN_DATA_FP8_E4M3: return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2: return "fp8_e5m2";
    default: return "unsupported";
  }
}
template<class T> void array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
void tensor(std::ostream& out, const TestTensor& value, std::string_view role) {
  out << "{\"uid\":" << value.uid << ",\"role\":" << quote(role)
      << ",\"dtype\":" << quote(dtype(value.data_type)) << ",\"shape\":";
  array(out, value.dimensions); out << ",\"strides\":"; array(out, value.strides);
  out << ",\"binding_byte_offset\":" << value.binding_byte_offset << '}';
}
void emit_catalog(const Entry& entry) {
  const auto& spec = entry.spec;
  std::ostringstream out; out << std::setprecision(12);
  out << "{\"op\":" << quote(entry.op) << ",\"group\":" << quote(entry.group)
      << ",\"case_name\":" << quote(spec.name) << ",\"dtype\":" << quote(dtype(spec.inputs.front().data_type))
      << ",\"catalog_source\":" << quote(entry.add ? "tests/common/add.cpp" : entry.square ? "tests/common/composite.cpp" : "tests/common/pointwise.cpp")
      << ",\"reference_builder\":" << quote(entry.add ? "build_add_reference" : entry.square ? "build_add_square_reference" : "build_pointwise_reference")
      << ",\"alpha\":" << spec.alpha << ",\"mode\":" << static_cast<int>(spec.mode)
      << ",\"attributes\":{\"flags\":" << spec.attributes.flags
      << ",\"relu_lower_clip\":" << spec.attributes.relu_lower_clip
      << ",\"relu_upper_clip\":" << spec.attributes.relu_upper_clip
      << ",\"relu_lower_clip_slope\":" << spec.attributes.relu_lower_clip_slope
      << ",\"swish_beta\":" << spec.attributes.swish_beta << "},\"tensors\":[";
  for (std::size_t i = 0; i < spec.inputs.size(); ++i) { if (i) out << ','; tensor(out, spec.inputs[i], "input"); }
  out << ','; tensor(out, spec.output, "output");
  out << "]}"; std::cout << out.str() << '\n';
}

std::size_t broadcast_index(std::size_t index, const TestTensor& input, const TestTensor& output) {
  std::vector<std::size_t> coordinates(output.dimensions.size());
  for (std::size_t a = coordinates.size(); a-- > 0;) {
    coordinates[a] = index % output.dimensions[a]; index /= output.dimensions[a];
  }
  const auto leading = output.dimensions.size() - input.dimensions.size();
  std::size_t result = 0;
  for (std::size_t a = 0; a < input.dimensions.size(); ++a)
    result = result * input.dimensions[a] + (input.dimensions[a] == 1 ? 0 : coordinates[leading + a]);
  return result;
}
struct Result {
  std::string stage = "setup", correctness = "NOT_RUN", error;
  std::size_t checked = 0, workspace = 0;
  double max_abs = 0, max_rel = 0, build_us = 0, warmup_us = 0;
  std::vector<double> samples, batches;
};

void measure(const Entry& entry, Stream& stream, Result& result) {
  const auto& spec = entry.spec;
  result.stage = "build";
  const auto start = timing::HostClock::now();
  auto reference = entry.add ? build_add_reference(*entry.add)
      : entry.square ? build_add_square_reference(*entry.square)
      : build_pointwise_reference(spec);
  result.build_us = timing::host_microseconds_since(start);
  result.workspace = reference->workspace_size();
  result.stage = "input_preparation";
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
  std::vector<std::vector<float>> floating;
  std::vector<std::vector<std::int32_t>> integers;
  std::vector<std::vector<std::uint8_t>> input_bytes;
  for (std::size_t i = 0; i < spec.inputs.size(); ++i) {
    const auto& t = spec.inputs[i];
    std::vector<std::uint8_t> bytes;
    if (entry.group == "integer" && t.data_type == FLAGDNN_DATA_INT32) {
      std::vector<std::int32_t> logical(io::element_count(t));
      std::vector<std::int32_t> physical(io::storage_element_count(t), -97);
      for (std::size_t j = 0; j < logical.size(); ++j) {
        logical[j] = pointwise_integer_input(j, i, spec.mode);
        physical[io::logical_offset(j, t)] = logical[j];
      }
      bytes.resize(physical.size() * sizeof(std::int32_t));
      std::memcpy(bytes.data(), physical.data(), bytes.size());
      integers.push_back(std::move(logical));
      floating.emplace_back();
    } else {
      std::vector<float> logical(io::element_count(t));
      if (entry.group == "identity_fp8") {
        for (std::size_t j = 0; j < logical.size(); ++j) logical[j] = static_cast<float>(1U << (j % 6));
      } else {
        logical = flagdnn::testing::make_input(logical.size(), i, spec.input_domains.at(i));
      }
      bytes = io::encode(io::scatter(logical, t), t.data_type, io::BooleanEncoding::kBitPacked);
      floating.push_back(io::gather(io::decode(bytes, t.data_type,
          io::storage_element_count(t), io::BooleanEncoding::kBitPacked), t));
      integers.emplace_back();
    }
    auto buffer = std::make_unique<DeviceBuffer>(t.binding_byte_offset + bytes.size());
    buffer->copy_from_host_at(bytes.data(), bytes.size(), t.binding_byte_offset, stream.get());
    stream.synchronize();
    bindings.push_back({t.uid, buffer->opaque_at(t.binding_byte_offset)});
    buffers.push_back(std::move(buffer)); input_bytes.push_back(std::move(bytes));
  }
  const auto& output = spec.output;
  std::vector<std::uint8_t> initial(io::encoded_byte_count(output, io::BooleanEncoding::kBitPacked), 0xA5);
  if (entry.group == "attributes") {
    initial = io::encode(std::vector<float>(io::storage_element_count(output),
        std::numeric_limits<float>::quiet_NaN()), output.data_type,
        io::BooleanEncoding::kBitPacked);
  }
  DeviceBuffer destination(output.binding_byte_offset + initial.size());
  destination.copy_from_host_at(initial.data(), initial.size(), output.binding_byte_offset, stream.get());
  bindings.push_back({output.uid, destination.opaque_at(output.binding_byte_offset)});
  DeviceBuffer workspace(reference->workspace_size());
  stream.synchronize();
  const auto execute = [&] { reference->execute(bindings, workspace.opaque(), reference->workspace_size(), stream.opaque()); };
  result.stage = "execute";
  execute(); stream.synchronize();
  result.stage = "readback";
  std::vector<std::uint8_t> actual(initial.size());
  destination.copy_to_host_at(actual.data(), actual.size(), output.binding_byte_offset, stream.get());
  stream.synchronize();
  result.stage = "correctness"; result.correctness = "FAIL";
  if (entry.group == "integer") {
    for (std::size_t j = 0; j < io::element_count(output); ++j) {
      const auto a = integers[0][broadcast_index(j, spec.inputs[0], output)];
      const auto b = integers[1][broadcast_index(j, spec.inputs[1], output)];
      const bool predicate = spec.inputs.size() == 3 ? floating[2][broadcast_index(j, spec.inputs[2], output)] != 0 : false;
      const auto expected = entry.square
          ? cpu::pointwise_integer_reference(FLAGDNN_POINTWISE_ADD, a,
                cpu::pointwise_integer_reference(FLAGDNN_POINTWISE_MUL, b, b, false, 1), false, 1)
          : cpu::pointwise_integer_reference(spec.mode, a, b, predicate, static_cast<std::int32_t>(spec.alpha));
      const auto offset = io::logical_offset(j, output);
      std::int32_t value;
      if (output.data_type == FLAGDNN_DATA_BOOLEAN) value = (actual[offset / 8] >> (offset % 8)) & 1;
      else std::memcpy(&value, actual.data() + offset * sizeof(value), sizeof(value));
      if (value != expected) throw std::runtime_error("exact INT32 mismatch index=" + std::to_string(j) + " actual=" + std::to_string(value) + " expected=" + std::to_string(expected));
      ++result.checked;
    }
  } else if (entry.group == "identity_fp8") {
    if (actual != input_bytes[0]) throw std::runtime_error("FP8 identity changed original bits");
    result.checked = io::element_count(output);
  } else {
    flagdnn::benchmarking::BenchmarkCase cpu_spec;
    cpu_spec.name = spec.name; cpu_spec.operation = flagdnn::benchmarking::Operation::kPointwise;
    cpu_spec.pointwise_mode = spec.mode; cpu_spec.pointwise_attributes = spec.attributes;
    for (const auto& t : spec.inputs) cpu_spec.tensors.push_back({t.uid, t.data_type, t.dimensions, t.strides, t.binding_byte_offset});
    cpu_spec.tensors.push_back({output.uid, output.data_type, output.dimensions, output.strides, output.binding_byte_offset});
    const auto quantize = [](std::span<const float> values, flagdnnDataType_t type) {
      return io::decode(io::encode(values, type, io::BooleanEncoding::kByte), type, values.size(), io::BooleanEncoding::kByte);
    };
    const auto expected = cpu::evaluate_benchmark(cpu_spec, floating, quantize).back();
    const auto values = io::gather(io::decode(actual, output.data_type, io::storage_element_count(output), io::BooleanEncoding::kBitPacked), output);
    for (std::size_t j = 0; j < values.size(); ++j) {
      const double absolute = std::abs(static_cast<double>(values[j]) - expected[j]);
      const double relative = absolute / std::max({std::abs(static_cast<double>(values[j])), std::abs(static_cast<double>(expected[j])), 1.0e-30});
      result.max_abs = std::max(result.max_abs, absolute); result.max_rel = std::max(result.max_rel, relative);
      if (!std::isfinite(absolute) || (absolute > spec.absolute_tolerance && relative > spec.relative_tolerance))
        throw std::runtime_error("CPU attribute derivative mismatch index=" + std::to_string(j));
      ++result.checked;
    }
  }
  // Require all byte padding to survive, independently of the numeric oracle.
  if (output.data_type != FLAGDNN_DATA_BOOLEAN) {
    const auto width = io::data_type_size(output.data_type);
    std::vector<bool> occupied(io::storage_element_count(output));
    for (std::size_t j = 0; j < io::element_count(output); ++j) occupied[io::logical_offset(j, output)] = true;
    for (std::size_t j = 0; j < occupied.size(); ++j) if (!occupied[j])
      for (std::size_t k = 0; k < width; ++k) if (actual[j * width + k] != initial[j * width + k])
        throw std::runtime_error("cuDNN modified output padding");
  }
  result.correctness = "PASS"; result.stage = "warmup";
  EventTimer timer;
  result.warmup_us = timer.measure_microseconds(stream.get(), 1, [&] { for (int i = 0; i < 10; ++i) execute(); });
  result.stage = "capture";
  timing::CapturedExecutionBatch batch(stream.get(), 50, execute);
  batch.launch(stream.get()); stream.synchronize();
  result.stage = "sampling";
  for (int sample = 0; sample < 20; ++sample) {
    const auto elapsed = timer.measure_microseconds(stream.get(), 1, [&] { batch.launch(stream.get()); });
    if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("invalid timing sample");
    result.batches.push_back(elapsed); result.samples.push_back(elapsed / 50.0);
  }
  result.stage = "complete";
}

void emit(const Entry& entry, const Result& result, std::string_view device, int driver, int runtime) {
  const auto& spec = entry.spec;
  std::ostringstream out; out << std::setprecision(12);
  std::time_t now = std::time(nullptr);
  out << "{\"schema_version\":1,\"provider\":\"cudnn\",\"op\":" << quote(entry.op)
      << ",\"group\":" << quote(entry.group) << ",\"case_name\":" << quote(spec.name)
      << ",\"dtype\":" << quote(dtype(spec.inputs.front().data_type))
      << ",\"output_dtype\":" << quote(dtype(spec.output.data_type))
      << ",\"device\":" << quote(device) << ",\"device_ordinal\":0,\"cuda_driver\":" << driver
      << ",\"cuda_runtime\":" << runtime << ",\"cudnn_version\":" << cudnnGetVersion()
      << ",\"timestamp_utc\":\"" << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ") << '"'
      << ",\"catalog_source\":" << quote(entry.add ? "tests/common/add.cpp" : entry.square ? "tests/common/composite.cpp" : "tests/common/pointwise.cpp")
      << ",\"case_filter\":\"shared native attribute/INT32/FP8 identity cases; no changed shapes\""
      << ",\"input_generator\":" << quote(entry.group == "integer" ? "tests/common/pointwise.cpp::pointwise_integer_input (exact raw int32)" : entry.group == "identity_fp8" ? "backends/nvidia/validation/functional/raw_copy.hpp::run_raw_copy_graph" : "backends/nvidia/validation/functional/pointwise_runner.cpp::make_input")
      << ",\"mode\":" << static_cast<int>(spec.mode) << ",\"alpha\":" << spec.alpha
      << ",\"attributes\":{\"flags\":" << spec.attributes.flags << ",\"relu_lower_clip\":" << spec.attributes.relu_lower_clip
      << ",\"relu_upper_clip\":" << spec.attributes.relu_upper_clip << ",\"relu_lower_clip_slope\":" << spec.attributes.relu_lower_clip_slope
      << ",\"swish_beta\":" << spec.attributes.swish_beta << "},\"tensors\":[";
  for (std::size_t i = 0; i < spec.inputs.size(); ++i) { if (i) out << ','; tensor(out, spec.inputs[i], "input"); }
  out << ','; tensor(out, spec.output, "output");
  out << "],\"status\":" << quote(result.error.empty() ? "PASS" : "ERROR") << ",\"stage\":" << quote(result.stage)
      << ",\"error\":" << quote(result.error) << ",\"correctness\":{\"status\":" << quote(result.correctness)
      << ",\"oracle\":" << quote(entry.group == "integer" ? "reference/cpu/pointwise.cpp::pointwise_integer_reference" : entry.group == "identity_fp8" ? "bit-exact encoded shared identity inputs" : "reference/cpu/benchmark.cpp::evaluate_benchmark")
      << ",\"checked_elements\":" << result.checked << ",\"total_elements\":" << io::element_count(spec.output)
      << ",\"max_abs\":";
  if (std::isfinite(result.max_abs)) out << result.max_abs; else out << "null";
  out << ",\"max_rel\":";
  if (std::isfinite(result.max_rel)) out << result.max_rel; else out << "null";
  out << ",\"atol\":" << spec.absolute_tolerance << ",\"rtol\":" << spec.relative_tolerance << '}'
      << ",\"build_us\":" << result.build_us << ",\"workspace_bytes\":" << result.workspace
      << ",\"timing\":{\"method\":\"cuda_events_over_cuda_graph\",\"unit\":\"us\",\"warmup_iterations\":10,\"sample_count\":20,\"iterations_per_sample\":50,\"graph_warmup_launches\":1,\"warmup_total_us\":" << result.warmup_us
      << ",\"percentile_method\":\"nearest_rank\",\"sample_batch_us\":";
  array(out, result.batches); out << ",\"samples_us\":"; array(out, result.samples);
  if (result.error.empty() && result.samples.size() == 20)
    out << ",\"median_us\":" << timing::percentile(result.samples, 0.5) << ",\"p90_us\":" << timing::percentile(result.samples, 0.9);
  out << "}}"; std::cout << out.str() << std::endl;
}
}  // namespace gap_reference

int main(int argc, char** argv) {
  using namespace gap_reference;
  try {
    std::string group, name, op; bool list = false, catalog_json = false, probe_first = false;
    for (int i = 1; i < argc; ++i) {
      const std::string_view argument = argv[i];
      if (argument == "--list") list = true;
      else if (argument == "--catalog") catalog_json = true;
      else if (argument == "--probe-first") probe_first = true;
      else if (i + 1 < argc && argument == "--group") group = argv[++i];
      else if (i + 1 < argc && argument == "--case") name = argv[++i];
      else if (i + 1 < argc && argument == "--op") op = argv[++i];
      else throw std::invalid_argument("usage: benchmark_nvidia_pointwise_gaps [--list|--catalog] [--probe-first] [--group NAME] [--op NAME] [--case EXACT]");
    }
    std::vector<Entry> selected; std::set<std::string> seen;
    for (auto entry : catalog()) {
      if ((!group.empty() && entry.group != group) || (!name.empty() && entry.spec.name != name) || (!op.empty() && entry.op != op)) continue;
      if (probe_first && !seen.insert(entry.op + ":" + dtype(entry.spec.inputs.front().data_type)).second) continue;
      selected.push_back(std::move(entry));
    }
    if (selected.empty()) throw std::runtime_error("case filter matched no cases");
    if (catalog_json) { for (const auto& entry : selected) emit_catalog(entry); return 0; }
    if (list) {
      for (const auto& entry : selected) std::cout << entry.group << '\t' << entry.op << '\t' << entry.spec.name << '\n';
      return 0;
    }
    DriverContext context; Stream stream;
    CUdevice gpu; cuDeviceGet(&gpu, 0); char device[256] = {};
    cuDeviceGetName(device, sizeof(device), gpu); int driver = 0, runtime = 0;
    cuDriverGetVersion(&driver); cudaRuntimeGetVersion(&runtime);
    std::size_t passed = 0, errors = 0;
    for (const auto& entry : selected) {
      Result result;
      try { measure(entry, stream, result); ++passed; }
      catch (const std::exception& error) { result.error = error.what(); ++errors; }
      emit(entry, result, device, driver, runtime);
      std::cerr << (result.error.empty() ? "PASS " : "ERROR ") << entry.spec.name << " stage=" << result.stage << '\n';
      const bool context_error = result.error.find("CUDA_ERROR_") != std::string::npos ||
          result.error.find("CUDNN_STATUS_EXECUTION_FAILED") != std::string::npos ||
          result.error.find("CUDNN_STATUS_INTERNAL_ERROR") != std::string::npos ||
          result.error.find("illegal memory access") != std::string::npos;
      if (!result.error.empty() && (context_error ||
          (result.stage != "build" && result.stage != "correctness"))) {
        std::cerr << "Stopping group after a possible CUDA execution/context error; remaining cases require a fresh process.\n";
        return 1;
      }
    }
    std::cerr << "pass=" << passed << " error=" << errors << '\n';
    return errors ? 1 : 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
}
