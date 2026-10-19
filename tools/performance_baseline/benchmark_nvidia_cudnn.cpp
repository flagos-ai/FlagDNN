/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
/*
Standalone cuDNN measurements of the actual core benchmark/test_<op>.cpp catalog.
Build: bash tools/performance_baseline/build_benchmark_nvidia_cudnn.sh /tmp/benchmark_nvidia_cudnn
List without using the GPU: /tmp/benchmark_nvidia_cudnn --list
Run all new catalog entries while retaining old results:
  /tmp/benchmark_nvidia_cudnn --exclude-file previous-case-names.txt \
    > core.jsonl 2> core.log
Filters: --op NAME[,NAME...] --dtype fp32|fp16|bfloat16 --case EXACT_NAME
Timing is 10 warmups, 20 samples, 50 CUDA Graph executions per sample, except
training BatchNorm, which retains the shared catalog's 5/10/20 configuration.
Matmul checks 257 deterministic CPU outputs and convolution checks 64; every
other operation checks every output with the shared CPU benchmark oracle.
Only CudnnProvider is instantiated; FlagDNN is never used as the reference.
*/
#include "../../backends/nvidia/validation/benchmark/runner.cpp"
#include "common/cases.hpp"
#include <ctime>
#include <fstream>
#include <set>
#include "benchmark_nvidia_core_catalog.hpp"
#include "benchmark_nvidia_core_validation.hpp"
#include "reference/cpu/benchmark.hpp"

namespace standalone_cudnn {
namespace bench = flagdnn::benchmarking;
}
#include "benchmark_nvidia_sampled_reference.hpp"
namespace standalone_cudnn {

const char* dtype(flagdnnDataType_t value) {
  switch (value) {
    case FLAGDNN_DATA_FLOAT32: return "fp32";
    case FLAGDNN_DATA_FLOAT16: return "fp16";
    case FLAGDNN_DATA_BFLOAT16: return "bfloat16";
    case FLAGDNN_DATA_BOOLEAN: return "boolean_bitpacked";
    default: throw std::invalid_argument("unexpected catalog data type");
  }
}

std::string json_number(double value) {
  if (!std::isfinite(value)) return "null";
  std::ostringstream result;
  result << std::setprecision(17) << value;
  return result.str();
}

std::string quote(std::string_view value) {
  std::ostringstream result;
  result << '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': result << "\\\""; break;
      case '\\': result << "\\\\"; break;
      case '\n': result << "\\n"; break;
      case '\r': result << "\\r"; break;
      case '\t': result << "\\t"; break;
      default:
        if (c < 0x20) {
          result << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(c) << std::dec;
        } else {
          result << c;
        }
    }
  }
  result << '"';
  return result.str();
}

template <typename T>
void array(std::ostream& out, const std::vector<T>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i) out << ',';
    out << values[i];
  }
  out << ']';
}

struct OutputValidation {
  std::int64_t uid = 0;
  std::string status = "FAIL";
  std::size_t total_elements = 0, checked_elements = 0;
  std::size_t protected_bytes_checked = 0;
  bool complete_write = false;
  ValidationAccuracy accuracy;
};
struct PhaseValidation {
  std::string name, status = "FAIL";
  std::size_t input_bytes_checked = 0;
  bool inputs_unchanged = false;
  std::vector<OutputValidation> outputs;
};

struct Measurements {
  std::string stage = "setup";
  std::string correctness = "NOT_RUN";
  ValidationAccuracy accuracy;
  std::size_t checked_elements = 0;
  std::size_t total_elements = 0;
  std::string validation_scope = "full";
  std::vector<std::size_t> checked_indices;
  std::vector<PhaseValidation> validation_phases;
  std::size_t workspace_bytes = 0;
  double build_us = 0;
  double warmup_total_us = 0;
  std::vector<double> samples_us;
  std::vector<double> sample_batch_us;
};

std::vector<float> quantize_output(std::span<const float> values, flagdnnDataType_t type) {
  const auto encoded = encode_cudnn_values(values, type);
  return decode_cudnn_values(encoded, type, values.size());
}

using InputGenerator = std::function<std::vector<float>(std::size_t, std::size_t, InputDomain)>;
using SampledOracle = std::function<std::vector<float>(const bench::BenchmarkCase&,
    const std::vector<std::vector<float>>&, const std::vector<std::size_t>&)>;

template <typename Provider>
void measure(const bench::BenchmarkCase& spec, Provider& provider,
             bench::Stream& stream, Measurements& result,
             InputGenerator generate = make_input,
             SampledOracle sample_oracle = sampled_reference<>,
             bool probe_native = false) {
  result.stage = "capability";
  const auto capability = provider.capability(spec);
  if (!probe_native && !capability.supported)
    throw bench::BenchmarkUnsupportedError(capability.reason);
  result.stage = "input_preparation";
  std::vector<std::unique_ptr<DeviceBuffer>> buffers;
  std::vector<CoreTensorStorage> storage;
  std::vector<flagdnnBinding_t> bindings;
  std::vector<std::vector<float>> quantized_inputs;
  const auto input_count = bench::input_tensor_count(spec);
  const auto add_buffer = [&](const bench::TensorSpec& tensor,
                              const std::vector<std::uint8_t>& encoded, bool input) {
    storage.push_back(make_core_tensor_storage(
        tensor, encoded, input, data_type_size(tensor.data_type)));
    const auto& host = storage.back();
    auto buffer = std::make_unique<DeviceBuffer>(host.initial.size());
    buffer->copy_from_host(host.initial.data(), host.initial.size(), stream.get());
    stream.synchronize();
    bindings.push_back({tensor.uid, buffer->opaque_at(host.binding_start)});
    buffers.push_back(std::move(buffer));
  };
  for (std::size_t i = 0; i < input_count; ++i) {
    const auto tensor = cudnn_tensor_specification(spec, i);
    const InputDomain domain = spec.input_domains.empty()
        ? spec.input_domain : spec.input_domains.at(i);
    const auto physical = scatter_logical_values(
        generate(element_count(tensor), i, domain), tensor);
    const auto encoded = encode_cudnn_values(physical, tensor.data_type);
    quantized_inputs.push_back(gather_logical_values(
        decode_cudnn_values(encoded, tensor.data_type, storage_element_count(tensor)), tensor));
    add_buffer(tensor, encoded, true);
  }

  result.stage = "cpu_reference";
  const bool sampled = spec.operation == bench::Operation::kMatmul ||
      spec.operation == bench::Operation::kConvolutionFprop ||
      spec.operation == bench::Operation::kConvolutionDgrad ||
      spec.operation == bench::Operation::kConvolutionWgrad;
  std::vector<std::vector<float>> expected;
  if (sampled) {
    result.validation_scope = "sampled";
    result.checked_indices = sample_indices(element_count(spec.tensors.back()),
        spec.operation == bench::Operation::kMatmul ? 257 : 64);
    expected = {quantize_output(sample_oracle(spec, quantized_inputs, result.checked_indices),
                               spec.tensors.back().data_type)};
  } else {
    expected = flagdnn::reference::cpu::evaluate_benchmark(spec, quantized_inputs, quantize_output);
  }
  if (expected.size() != spec.output_count)
    throw std::runtime_error("CPU oracle output count mismatch");
  for (std::size_t oi = 0; oi < spec.output_count; ++oi) {
    const auto tensor = cudnn_tensor_specification(spec, input_count + oi);
    const auto count = element_count(tensor);
    result.total_elements += count;
    if (!sampled && expected[oi].size() != count)
      throw std::runtime_error("CPU oracle output element count mismatch");
    std::vector<float> physical(storage_element_count(tensor), kPaddingSentinel);
    for (std::size_t i = 0; i < count; ++i) {
      // Complement every expected BOOLEAN bit, so a missing write cannot pass.
      physical[tensor_io::logical_offset(i, tensor)] =
          tensor.data_type == FLAGDNN_DATA_BOOLEAN
              ? (expected[oi][i] == 0.0F ? 1.0F : 0.0F)
              : (sampled ? std::numeric_limits<float>::quiet_NaN()
                         : core_output_poison(expected[oi][i],
                               spec.absolute_tolerance, spec.relative_tolerance));
    }
    add_buffer(tensor, encode_cudnn_values(physical, tensor.data_type), false);
  }
  result.stage = "build";
  const auto build_begin = HostClock::now();
  auto executable = provider.build(spec);
  result.build_us = host_microseconds_since(build_begin);
  result.workspace_bytes = executable->workspace_size();
  auto workspace = make_workspace(*executable);
  const auto run = [&] { execute(*executable, bindings, *workspace, stream); };
  const auto read_allocation = [&](std::size_t index) {
    std::vector<std::uint8_t> bytes(storage[index].initial.size());
    buffers[index]->copy_to_host(bytes.data(), bytes.size(), stream.get());
    stream.synchronize();
    return bytes;
  };
  const auto validate = [&](const char* phase_name) {
    result.stage = phase_name;
    result.correctness = "FAIL";
    result.validation_phases.push_back({phase_name});
    auto& phase = result.validation_phases.back();
    for (std::size_t i = 0; i < input_count; ++i) {
      const auto bytes = read_allocation(i);
      phase.input_bytes_checked += validate_core_storage(storage[i], bytes);
    }
    phase.inputs_unchanged = true;
    std::size_t checked = 0;
    for (std::size_t oi = 0; oi < spec.output_count; ++oi) {
      const auto ti = input_count + oi;
      const auto output = cudnn_tensor_specification(spec, ti);
      phase.outputs.push_back({output.uid});
      auto& check = phase.outputs.back();
      check.total_elements = element_count(output);
      const auto allocation = read_allocation(ti);
      check.protected_bytes_checked = validate_core_storage(storage[ti], allocation);
      const auto encoded = std::span<const std::uint8_t>(allocation).subspan(
          storage[ti].binding_start, storage[ti].payload_bytes);
      const auto physical = decode_cudnn_values(encoded, output.data_type,
                                               storage_element_count(output));
      auto actual = gather_logical_values(physical, output);
      for (std::size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]))
          throw std::runtime_error("unwritten/nonfinite output uid=" +
              std::to_string(output.uid) + " index=" + std::to_string(i));
      }
      if (sampled) {
        std::vector<float> selected;
        for (auto index : result.checked_indices) selected.push_back(actual.at(index));
        actual = std::move(selected);
      }
      const bool identity = spec.operation == bench::Operation::kPointwise &&
          spec.pointwise_mode == FLAGDNN_POINTWISE_IDENTITY;
      compare_core_values(actual, expected.at(oi),
          identity ? 0.0 : spec.absolute_tolerance,
          identity ? 0.0 : spec.relative_tolerance,
          check.accuracy, check.checked_elements);
      check.complete_write = true;
      check.status = "PASS";
      checked += check.checked_elements;
      result.accuracy.maximum_absolute = std::max(
          result.accuracy.maximum_absolute, check.accuracy.maximum_absolute);
      result.accuracy.maximum_relative = std::max(
          result.accuracy.maximum_relative, check.accuracy.maximum_relative);
    }
    result.checked_elements = checked;
    phase.status = "PASS";
  };

  result.stage = "execute";
  run();
  stream.synchronize();
  validate("before_timing");
  result.stage = "warmup";
  EventTimer timer;
  if (spec.benchmark.warmup_iterations) {
    result.warmup_total_us = timer.measure_microseconds(stream.get(), 1, [&] {
      for (int i = 0; i < spec.benchmark.warmup_iterations; ++i) run();
    });
  }
  result.stage = "capture";
  CapturedExecutionBatch batch(stream.get(), spec.benchmark.iterations_per_sample, run);
  batch.launch(stream.get());
  stream.synchronize();
  result.stage = "sampling";
  for (int i = 0; i < spec.benchmark.sample_count; ++i) {
    const double batch_us = timer.measure_microseconds(
        stream.get(), 1, [&] { batch.launch(stream.get()); });
    if (!std::isfinite(batch_us) || batch_us <= 0.0)
      throw std::runtime_error("invalid CUDA timing sample");
    result.sample_batch_us.push_back(batch_us);
    result.samples_us.push_back(batch_us / batch.execution_count());
  }
  validate("after_timing");
  result.stage = "reset_outputs";
  for (std::size_t i = input_count; i < storage.size(); ++i) {
    buffers[i]->copy_from_host(storage[i].initial.data(), storage[i].initial.size(),
                               stream.get());
  }
  stream.synchronize();
  result.stage = "reset_output_graph_replay";
  batch.launch(stream.get());
  stream.synchronize();
  validate("after_reset_graph_replay");
  result.correctness = "PASS";
  result.stage = "complete";
}

std::string tensor_semantic_role(const bench::BenchmarkCase& spec, std::size_t index) {
  if (spec.operation == bench::Operation::kBatchnorm) {
    static constexpr const char* names[] = {
        "x", "scale", "bias", "previous_running_mean", "previous_running_variance",
        "y", "saved_mean", "saved_inverse_variance", "next_running_mean",
        "next_running_variance"};
    return names[index];
  }
  const auto inputs = bench::input_tensor_count(spec);
  return index < inputs ? "input_" + std::to_string(index)
                        : "output_" + std::to_string(index - inputs);
}

void emit_tensor_metadata(std::ostream& out, const bench::TensorSpec& tensor,
                          const bench::TensorSpec& reference,
                          std::string_view role, std::string_view semantic_role,
                          bool virtual_tensor, bool flattened_graph) {
  out << "{\"uid\":" << tensor.uid << ",\"role\":" << quote(role)
      << ",\"semantic_role\":" << quote(semantic_role)
      << ",\"dtype\":" << quote(dtype(tensor.data_type)) << ",\"shape\":";
  array(out, tensor.dimensions);
  out << ",\"strides\":"; array(out, tensor.strides);
  out << ",\"native_strides\":"; array(out, reference.strides);
  out << ",\"reference_strides\":"; array(out, reference.strides);
  out << ",\"reference_shape\":"; array(out, reference.dimensions);
  out << ",\"reference_layout_conversion_timed\":false"
      << ",\"reference_layout_conversion\":"
      << (tensor.strides == reference.strides ? "false" : "true")
      << ",\"binding_byte_offset\":" << tensor.binding_byte_offset
      << ",\"storage_bytes\":" << cudnn_encoded_byte_count(tensor)
      << ",\"reference_storage_bytes\":" << cudnn_encoded_byte_count(reference)
      << ",\"logical_elements\":" << element_count(tensor)
      << ",\"boolean_encoding\":"
      << (tensor.data_type == FLAGDNN_DATA_BOOLEAN ? "\"bitpacked\"" : "null")
      << ",\"virtual\":" << (virtual_tensor ? "true" : "false");
  if (flattened_graph) {
    const auto count = static_cast<std::int64_t>(element_count(tensor));
    out << ",\"reference_descriptor_shape\":";
    array(out, std::vector<std::int64_t>{1, count, 1, 1});
    out << ",\"reference_descriptor_strides\":";
    array(out, std::vector<std::int64_t>{count, 1, 1, 1});
  }
  out << '}';
}

void emit_pointwise_attributes(std::ostream& out,
                               const flagdnnPointwiseAttributes_t& attributes) {
  out << "\"pointwise_flags\":" << attributes.flags
      << ",\"relu_lower_clip\":" << attributes.relu_lower_clip
      << ",\"relu_upper_clip\":" << attributes.relu_upper_clip
      << ",\"relu_lower_clip_slope\":" << attributes.relu_lower_clip_slope
      << ",\"swish_beta\":" << attributes.swish_beta
      << ",\"elu_alpha\":" << attributes.elu_alpha
      << ",\"softplus_beta\":" << attributes.softplus_beta;
}

void emit_convolution_attributes(std::ostream& out,
                                 const bench::ConvolutionAttributes& convolution) {
  out << "\"convolution_spatial_rank\":" << convolution.spatial_rank
      << ",\"convolution_groups\":" << convolution.groups
      << ",\"convolution_mode\":" << static_cast<int>(convolution.mode)
      << ",\"convolution_pre_padding\":"; array(out, convolution.pre_padding);
  out << ",\"convolution_post_padding\":"; array(out, convolution.post_padding);
  out << ",\"convolution_stride\":"; array(out, convolution.stride);
  out << ",\"convolution_dilation\":"; array(out, convolution.dilation);
}

void emit_case_metadata(std::ostream& out, const bench::BenchmarkCase& spec) {
  const auto input_count = bench::input_tensor_count(spec);
  out << ",\"operation_kind\":" << static_cast<int>(spec.operation)
      << ",\"input_domain\":" << static_cast<int>(spec.input_domain)
      << ",\"input_domains\":[";
  for (std::size_t i = 0; i < input_count; ++i) {
    if (i) out << ',';
    out << static_cast<int>(spec.input_domains.empty()
        ? spec.input_domain : spec.input_domains.at(i));
  }
  out << "],\"input_count\":" << input_count
      << ",\"output_count\":" << spec.output_count
      << ",\"reference_uses_quantized_inputs\":true,\"tensors\":[";
  const bool flattened_graph = spec.operation == bench::Operation::kGraph &&
      std::none_of(spec.graph.nodes.begin(), spec.graph.nodes.end(), [](const auto& node) {
        return node.operation == bench::Operation::kConvolutionFprop;
      });
  for (std::size_t i = 0; i < spec.tensors.size(); ++i) {
    if (i) out << ',';
    emit_tensor_metadata(out, spec.tensors[i], cudnn_tensor_specification(spec, i),
        i < input_count ? "input" : "output", tensor_semantic_role(spec, i),
        false, flattened_graph);
  }
  for (const auto& tensor : spec.graph.intermediates) {
    if (!spec.tensors.empty() || &tensor != &spec.graph.intermediates.front()) out << ',';
    emit_tensor_metadata(out, tensor, tensor, "intermediate",
                         "intermediate_" + std::to_string(tensor.uid), true, flattened_graph);
  }
  out << "],\"attributes\":{\"pointwise_mode\":" << static_cast<int>(spec.pointwise_mode)
      << ",\"alpha\":" << spec.add_alpha << ',';
  emit_pointwise_attributes(out, spec.pointwise_attributes);
  out << ",\"epsilon\":" << spec.normalization.epsilon
      << ",\"momentum\":" << spec.normalization.momentum
      << ",\"reduction_mode\":" << static_cast<int>(spec.reduction_mode)
      << ",\"reduction_axis\":" << spec.reduction_axis
      << ",\"keep_dimensions\":" << (spec.keep_dimensions ? "true" : "false") << ',';
  emit_convolution_attributes(out, spec.convolution);
  if (spec.operation == bench::Operation::kReshape ||
      spec.operation == bench::Operation::kTranspose ||
      spec.operation == bench::Operation::kSlice) {
    out << ",\"reshape_logical\":" << (spec.reshape.logical ? "true" : "false")
        << ",\"reshape_dimensions\":"; array(out, spec.reshape.dimensions);
    out << ",\"reshape_strides\":"; array(out, spec.reshape.strides);
    out << ",\"permutation\":"; array(out, spec.transpose.permutation);
    out << ",\"slice_strides\":"; array(out, spec.slice.strides);
    out << ",\"slices\":[";
    for (std::size_t i = 0; i < spec.slice.slices.size(); ++i) {
      if (i) out << ',';
      out << '[' << spec.slice.slices[i].first << ','
          << spec.slice.slices[i].second << ']';
    }
    out << ']';
  }
  out << "},\"graph_nodes\":[";
  for (std::size_t i = 0; i < spec.graph.nodes.size(); ++i) {
    if (i) out << ',';
    const auto& node = spec.graph.nodes[i];
    out << "{\"name\":" << quote(node.name)
        << ",\"operation_kind\":" << static_cast<int>(node.operation)
        << ",\"input_uids\":"; array(out, node.input_uids);
    out << ",\"output_uid\":" << node.output_uid
        << ",\"pointwise_mode\":" << static_cast<int>(node.pointwise_mode)
        << ",\"alpha\":" << node.alpha << ',';
    emit_pointwise_attributes(out, node.pointwise_attributes);
    out << ','; emit_convolution_attributes(out, node.convolution);
    out << '}';
  }
  out << ']' << ",\"allocation_guard_bytes_per_side\":" << kCoreGuardBytes;
}

void emit_validation_phases(std::ostream& out, const Measurements& result) {
  out << ",\"phases\":{";
  for (std::size_t i = 0; i < result.validation_phases.size(); ++i) {
    if (i) out << ',';
    const auto& phase = result.validation_phases[i];
    out << quote(phase.name) << ":{\"status\":" << quote(phase.status)
        << ",\"inputs_unchanged\":" << (phase.inputs_unchanged ? "true" : "false")
        << ",\"input_bytes_checked\":" << phase.input_bytes_checked << ",\"outputs\":[";
    for (std::size_t j = 0; j < phase.outputs.size(); ++j) {
      if (j) out << ',';
      const auto& output = phase.outputs[j];
      out << "{\"uid\":" << output.uid << ",\"status\":" << quote(output.status)
          << ",\"total_elements\":" << output.total_elements
          << ",\"checked_elements\":" << output.checked_elements
          << ",\"complete_write\":" << (output.complete_write ? "true" : "false")
          << ",\"protected_bytes_checked\":" << output.protected_bytes_checked
          << ",\"max_abs_error\":" << json_number(output.accuracy.maximum_absolute)
          << ",\"max_rel_error\":" << json_number(output.accuracy.maximum_relative) << '}';
    }
    out << "]}";
  }
  out << '}';
}

void emit_catalog(const bench::BenchmarkCase& spec, std::string_view operation) {
  std::ostringstream out;
  out << std::setprecision(12) << "{\"schema_version\":5,\"status\":\"catalog\""
      << ",\"provider\":\"cudnn\",\"op\":" << quote(operation)
      << ",\"case_name\":" << quote(spec.name)
      << ",\"dtype\":" << quote(dtype(spec.tensors[0].data_type))
      << ",\"catalog_source\":" << quote("benchmark/test_" + std::string(operation) + ".cpp")
      << ",\"input_generator\":\"backends/nvidia/validation/benchmark/runner.cpp::make_input\"";
  emit_case_metadata(out, spec);
  out << ",\"absolute_tolerance\":" << spec.absolute_tolerance
      << ",\"relative_tolerance\":" << spec.relative_tolerance
      << ",\"timing\":{\"method\":\"cuda_events_over_cuda_graph\",\"unit\":\"us\""
      << ",\"warmup_iterations\":" << spec.benchmark.warmup_iterations
      << ",\"sample_count\":" << spec.benchmark.sample_count
      << ",\"iterations_per_sample\":" << spec.benchmark.iterations_per_sample
      << ",\"graph_warmup_launches\":1,\"percentile_method\":\"nearest_rank\"}}";
  std::cout << out.str() << '\n';
}

void emit(const bench::BenchmarkCase& spec, std::string_view operation,
          const std::string& device, int driver, int runtime,
          const Measurements& result, const std::string& error,
          std::string catalog_source = {}, std::string input_generator = {},
          int input_precision = 0) {
  const std::time_t now = std::time(nullptr);
  std::ostringstream out;
  out << std::setprecision(12)
      << "{\"schema_version\":4,\"provider\":\"cudnn\",\"case_name\":" << quote(spec.name)
      << ",\"op\":" << quote(operation) << ",\"dtype\":" << quote(dtype(spec.tensors[0].data_type))
      << ",\"output_dtype\":" << quote(dtype(bench::output_tensor(spec).data_type))
      << ",\"timestamp_utc\":\"" << std::put_time(std::gmtime(&now), "%Y-%m-%dT%H:%M:%SZ") << '"'
      << ",\"device\":" << quote(device) << ",\"device_ordinal\":0"
      << ",\"cuda_driver\":" << driver << ",\"cuda_runtime\":" << runtime
      << ",\"cudnn_version\":" << cudnnGetVersion()
      << ",\"catalog_source\":" << quote(catalog_source.empty() ? "benchmark/test_" + std::string(operation) + ".cpp" : catalog_source)
      << ",\"case_filter_source\":" << quote(input_precision == 2 ?
          "backends/nvidia/validation/benchmark/native_matrix.cpp::run_native_precision_benchmark" :
          "backends/nvidia/validation/benchmark/cases.hpp")
      << ",\"input_generator\":" << quote(input_generator.empty() ?
          "backends/nvidia/validation/benchmark/runner.cpp::make_input" : input_generator)
      << ",\"input_precision\":" << input_precision
      << ",\"reference_input_precision\":" << input_precision
      << ",\"reference_quantizes_outputs\":true"
      << ",\"reference_tf32_rounding\":" << (input_precision == 2 ? "\"nearest_even\"" : "null")
      ;
  emit_case_metadata(out, spec);
  out << ",\"timing\":{\"method\":\"cuda_events_over_cuda_graph\",\"unit\":\"us\""
      << ",\"warmup_iterations\":" << spec.benchmark.warmup_iterations
      << ",\"sample_count\":" << spec.benchmark.sample_count
      << ",\"iterations_per_sample\":" << spec.benchmark.iterations_per_sample
      << ",\"graph_warmup_launches\":1,\"warmup_total_us\":" << result.warmup_total_us
      << ",\"percentile_method\":\"nearest_rank\",\"sample_batch_us\":";
  array(out, result.sample_batch_us);
  out << ",\"samples_us\":";
  array(out, result.samples_us);
  if (error.empty() && result.correctness == "PASS" &&
      result.samples_us.size() == static_cast<std::size_t>(spec.benchmark.sample_count)) {
    out << ",\"median_us\":" << percentile(result.samples_us, 0.5)
        << ",\"p90_us\":" << percentile(result.samples_us, 0.9)
        << ",\"p95_us\":" << percentile(result.samples_us, 0.95)
        << ",\"min_us\":" << *std::min_element(result.samples_us.begin(), result.samples_us.end());
  }
  out << "},\"workspace_bytes\":" << result.workspace_bytes << ",\"build_us\":" << result.build_us
      << ",\"correctness\":{\"status\":" << quote(result.correctness)
      << ",\"reference\":" << quote(result.validation_scope == "sampled" ?
          "tools/performance_baseline/benchmark_nvidia_sampled_reference.hpp" : "reference/cpu/benchmark.cpp::evaluate_benchmark")
      << ",\"scope\":" << quote(result.validation_scope)
      << ",\"total_elements\":" << result.total_elements
      << ",\"checked_elements\":" << result.checked_elements
      << ",\"sample_indices\":"; array(out, result.checked_indices);
  out
      << ",\"absolute_tolerance\":" << spec.absolute_tolerance
      << ",\"relative_tolerance\":" << spec.relative_tolerance
      << ",\"comparison\":\"abs <= atol OR abs/max(abs(actual),abs(reference),1e-30) <= rtol\""
      << ",\"max_abs_error\":" << (result.correctness == "PASS" ? json_number(result.accuracy.maximum_absolute) : "null")
      << ",\"max_rel_error\":" << (result.correctness == "PASS" ? json_number(result.accuracy.maximum_relative) : "null");
  emit_validation_phases(out, result);
  out << '}'
      << ",\"status\":" << quote(error.empty() ? "PASS" : "ERROR")
      << ",\"stage\":" << quote(result.stage) << ",\"error\":"
      << (error.empty() ? "null" : quote(error)) << '}';
  std::cout << out.str() << std::endl;
}
}  // namespace standalone_cudnn

#ifndef FLAGDNN_CUDNN_NO_MAIN
int main(int argc, char** argv) {
  using namespace standalone_cudnn;
  try {
    std::string op = "all", case_name, dtype_filter, exclude_file, case_file;
    bool list = false, catalog_json = false, probe_native = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "--help") {
        std::cout << "--list|--catalog --probe-native --op all|NAME[,NAME...] --dtype fp32|fp16|bfloat16 --case EXACT_NAME --case-file PATH --exclude-file PATH\n";
        return 0;
      }
      if (argument == "--probe-native") { probe_native = true; continue; }
      if (argument == "--list") { list = true; continue; }
      if (argument == "--catalog") { catalog_json = true; continue; }
      if (++i == argc) throw std::invalid_argument("missing value for " + argument);
      if (argument == "--op") op = argv[i];
      else if (argument == "--case") case_name = argv[i];
      else if (argument == "--dtype") dtype_filter = argv[i];
      else if (argument == "--exclude-file") exclude_file = argv[i];
      else if (argument == "--case-file") case_file = argv[i];
      else throw std::invalid_argument("unknown argument: " + argument);
    }
    std::set<std::string> excluded;
    if (!exclude_file.empty()) {
      std::ifstream input(exclude_file);
      if (!input) throw std::invalid_argument("cannot read exclude file");
      for (std::string line; std::getline(input, line);) excluded.insert(line);
    }
    std::set<std::string> requested;
    if (!case_file.empty()) {
      std::ifstream input(case_file);
      if (!input) throw std::invalid_argument("cannot read case file");
      for (std::string line; std::getline(input, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) requested.insert(line);
      }
      if (requested.empty()) throw std::invalid_argument("case file is empty");
    }
    const auto matches = [&](const char* operation) {
      return op == "all" || ("," + op + ",").find("," + std::string(operation) + ",") != std::string::npos;
    };
    std::vector<std::pair<std::string, bench::BenchmarkCase>> selected;
    for (const auto& operation : catalog_entries) {
      if (!matches(operation.name)) continue;
      operation.entry(0, nullptr);
      // Explicit support probes bypass historical catalog/capability exclusions.
      const auto cases = probe_native ? bench::captured_standalone_cases
          : bench::cudnn_benchmark_cases(bench::captured_standalone_cases);
      for (auto spec : cases) {
        if (!case_name.empty() && spec.name != case_name) continue;
        if (!dtype_filter.empty() && dtype_filter != dtype(spec.tensors[0].data_type)) continue;
        if (excluded.contains(spec.name)) continue;
        if (!case_file.empty() && !requested.contains(spec.name)) continue;
        spec.benchmark = core_benchmark_configuration(spec);
        selected.emplace_back(operation.name, std::move(spec));
      }
    }
    if (selected.empty()) throw std::invalid_argument("operation/case filter matched no shared case");
    if (!case_file.empty()) {
      for (const auto& [operation, spec] : selected) requested.erase(spec.name);
      if (!requested.empty())
        throw std::invalid_argument("requested case missing from selected catalog: " + *requested.begin());
    }
    if (catalog_json) {
      for (const auto& [operation, spec] : selected) emit_catalog(spec, operation);
      return 0;
    }
    if (list) {
      for (const auto& [operation, spec] : selected)
        std::cout << operation << '\t' << dtype(spec.tensors[0].data_type) << '\t' << spec.name << '\n';
      return 0;
    }
    bench::DriverContext context;
    bench::Stream stream;
    bench::CudnnProvider provider;
    CUdevice cuda_device;
    bench::check_cuda(cuCtxGetDevice(&cuda_device), "cuCtxGetDevice");
    char device_name[256]{};
    bench::check_cuda(cuDeviceGetName(device_name, sizeof(device_name), cuda_device), "cuDeviceGetName");
    int driver = 0, runtime = 0;
    bench::check_cuda(cuDriverGetVersion(&driver), "cuDriverGetVersion");
    const auto runtime_status = cudaRuntimeGetVersion(&runtime);
    if (runtime_status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(runtime_status));
    std::size_t total = 0, failed = 0;
    for (const auto& [operation, spec] : selected) {
        ++total;
        Measurements result;
        std::string error;
        try {
          measure(spec, provider, stream, result, make_input,
                  sampled_reference<>, probe_native);
        } catch (const std::exception& exception) {
          error = exception.what();
          ++failed;
        }
        emit(spec, operation, device_name, driver, runtime, result, error);
        std::cerr << total << ' ' << spec.name << ' ' << (error.empty() ? "PASS" : "ERROR")
                  << (error.empty() ? "" : ": " + error) << std::endl;
        // cuDNN failures may leave asynchronous CUDA errors hidden in status
        // messages. Let the wrapper recover subsequent cases in a fresh process.
        if (!error.empty()) return 3;
    }
    if (!total) throw std::invalid_argument("operation/case filter matched no shared case");
    std::cerr << "cuDNN catalog complete: cases=" << total << " passed=" << total-failed
              << " failed=" << failed << std::endl;
    return failed ? 1 : 0;
  } catch (const std::exception& exception) {
    std::cerr << "ERROR: " << exception.what() << std::endl;
    return 1;
  }
}

#endif  // FLAGDNN_CUDNN_NO_MAIN
