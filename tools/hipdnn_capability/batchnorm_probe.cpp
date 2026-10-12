/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
// Standalone correctness probe; never measures performance or modifies a
// baseline. Reuses the exact shared training catalog and CPU oracle. --list is
// GPU-free.
#include "backends/hygon/validation/benchmark/validation_guards.hpp"
#include "backends/hygon/validation/tensor_io.hpp"
#include "benchmark/common/cases.hpp"
#include "reference/cpu/benchmark.hpp"

#include <hip/hip_runtime_api.h>
#include <hipdnn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace bm = flagdnn::benchmarking;
namespace io = flagdnn::validation::hygon::tensor_io;
namespace cpu = flagdnn::reference::cpu;

namespace {
std::string quote(std::string_view value) {
  std::string result = "\"";
  for (const char c : value) {
    if (c == '"' || c == '\\')
      result += '\\';
    if (c == '\n')
      result += "\\n";
    else if (c == '\r')
      result += "\\r";
    else
      result += c;
  }
  return result + '"';
}
const char *dtype(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32)
    return "fp32";
  if (type == FLAGDNN_DATA_FLOAT16)
    return "fp16";
  if (type == FLAGDNN_DATA_BFLOAT16)
    return "bfloat16";
  throw std::invalid_argument("unsupported probe dtype");
}
hipdnnDataType_t vendor_dtype(flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FLOAT32)
    return HIPDNN_DATA_FLOAT;
  if (type == FLAGDNN_DATA_FLOAT16)
    return HIPDNN_DATA_HALF;
  if (type == FLAGDNN_DATA_BFLOAT16)
    return HIPDNN_DATA_BFLOAT16;
  throw std::invalid_argument("unsupported probe descriptor dtype");
}
void hip_check(hipError_t status, std::string_view stage) {
  std::cout << "{\"event\":\"hip_status\",\"stage\":" << quote(stage)
            << ",\"status\":" << int(status) << "}" << std::endl;
  if (status != hipSuccess)
    throw std::runtime_error(std::string(stage) + ": " +
                             hipGetErrorString(status));
}
void dnn_check(hipdnnStatus_t status, std::string_view stage) {
  std::cout << "{\"event\":\"hipdnn_status\",\"stage\":" << quote(stage)
            << ",\"status\":" << int(status) << "}" << std::endl;
  if (status != HIPDNN_STATUS_SUCCESS)
    throw std::runtime_error(std::string(stage) + ": " +
                             hipdnnGetErrorString(status));
}
std::vector<float> quantize(std::span<const float> values,
                            flagdnnDataType_t type) {
  return io::decode(io::encode(values, type), type, values.size());
}
std::vector<std::vector<float>> inputs(const bm::BenchmarkCase &spec) {
  std::vector<std::vector<float>> result;
  for (std::size_t ti = 0; ti < 5; ++ti) {
    std::vector<float> values(io::element_count(spec.tensors[ti]));
    for (std::size_t i = 0; i < values.size(); ++i) {
      const int centered = int((i * 17 + ti * 11) % 41) - 20;
      const float real = float(centered) / float(13 + ti);
      values[i] = spec.input_domains.at(ti) == bm::InputDomain::kPositive
                      ? std::abs(real) + 0.5F
                      : real;
    }
    result.push_back(quantize(values, spec.tensors[ti].data_type));
  }
  return result;
}

// Allocate a guarded FP32-sized backing store even for low-precision data.
// A vendor that wrongly interprets HALF scale as FLOAT cannot read past the
// allocation. Logical bytes and every byte outside them are checked separately.
struct Buffer {
  void *pointer = nullptr;
  std::size_t logical_bytes = 0;
  std::size_t allocated_bytes = 0;
  Buffer(std::size_t bytes, std::size_t elements)
      : logical_bytes(bytes),
        allocated_bytes(std::max(bytes, elements * 4) + 256) {
    hip_check(hipMalloc(&pointer, allocated_bytes), "allocate");
  }
  ~Buffer() {
    if (pointer)
      (void)hipFree(pointer);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  void upload(std::span<const std::uint8_t> bytes, hipStream_t stream) {
    if (bytes.size() != logical_bytes)
      throw std::logic_error("upload byte count");
    std::vector<std::uint8_t> guarded(allocated_bytes, 0xA5);
    std::copy(bytes.begin(), bytes.end(), guarded.begin());
    hip_check(hipMemcpyAsync(pointer, guarded.data(), guarded.size(),
                             hipMemcpyHostToDevice, stream),
              "upload");
    hip_check(hipStreamSynchronize(stream), "upload_sync");
  }
  std::vector<std::uint8_t> read(hipStream_t stream) const {
    std::vector<std::uint8_t> bytes(allocated_bytes);
    hip_check(hipMemcpyAsync(bytes.data(), pointer, bytes.size(),
                             hipMemcpyDeviceToHost, stream),
              "read");
    hip_check(hipStreamSynchronize(stream), "read_sync");
    return bytes;
  }
};

struct Resources {
  hipStream_t stream = nullptr;
  hipdnnHandle_t handle = nullptr;
  std::array<hipdnnTensorDescriptor_t, 4> descriptors{};
  hipGraph_t graph = nullptr;
  hipGraphExec_t executable = nullptr;
  Resources() = default;
  ~Resources() {
    if (stream)
      (void)hipStreamSynchronize(stream);
    if (executable)
      (void)hipGraphExecDestroy(executable);
    if (graph)
      (void)hipGraphDestroy(graph);
    for (auto descriptor : descriptors)
      if (descriptor)
        (void)hipdnnDestroyTensorDescriptor(descriptor);
    if (handle)
      (void)hipdnnDestroy(handle);
    if (stream)
      (void)hipStreamDestroy(stream);
  }
  void initialize(const bm::BenchmarkCase &spec, bool transform_parameters) {
    hip_check(hipStreamCreate(&stream), "stream_create");
    dnn_check(hipdnnCreate(&handle), "handle_create");
    dnn_check(hipdnnSetStream(handle, stream), "set_stream");
    const std::array<std::size_t, 4> indices{0, 5, 1, 1};
    for (std::size_t i = 0; i < (transform_parameters ? 4U : 3U); ++i) {
      auto tensor = spec.tensors[indices[i]];
      if (i == 2 && transform_parameters)
        tensor.data_type = FLAGDNN_DATA_FLOAT32;
      dnn_check(hipdnnCreateTensorDescriptor(&descriptors[i]),
                "descriptor_create");
      std::vector<int> dims(tensor.dimensions.begin(), tensor.dimensions.end());
      std::vector<int> strides(tensor.strides.begin(), tensor.strides.end());
      dnn_check(hipdnnSetTensorNdDescriptor(
                    descriptors[i], vendor_dtype(tensor.data_type),
                    int(dims.size()), dims.data(), strides.data()),
                "descriptor_set");
    }
  }
};

struct Check {
  std::size_t errors = 0;
  std::size_t first = std::numeric_limits<std::size_t>::max();
  std::size_t unwritten = 0;
  double max_abs = 0;
  double max_rel = 0;
};

int probe(const bm::BenchmarkCase &original, bool fp32_parameters,
          bool transform_parameters, std::string_view api, int device) {
  bm::BenchmarkCase spec = original;
  const auto logical_inputs = inputs(original);
  const auto expected =
      cpu::evaluate_benchmark(original, logical_inputs, quantize);
  if (fp32_parameters) {
    spec.tensors[1].data_type = FLAGDNN_DATA_FLOAT32;
    spec.tensors[2].data_type = FLAGDNN_DATA_FLOAT32;
  }
  std::cout << "{\"kind\":\"batchnorm_probe\",\"case_name\":"
            << quote(original.name) << ",\"api\":" << quote(api)
            << ",\"data_dtype\":" << quote(dtype(spec.tensors[0].data_type))
            << ",\"scale_bias_dtype\":"
            << quote(dtype(spec.tensors[1].data_type))
            << ",\"api_parameter_dtype\":"
            << quote(transform_parameters ? "fp32"
                                          : dtype(spec.tensors[1].data_type))
            << ",\"parameter_mapping\":"
            << quote(transform_parameters ? "hipdnn_transform_per_execute"
                     : fp32_parameters    ? "host_promoted_diagnostic"
                                          : "direct_exact")
            << ",\"stats_dtype\":\"fp32\",\"exact_catalog_semantics\":"
            << (fp32_parameters ? "false" : "true")
            << ",\"epsilon\":" << spec.normalization.epsilon
            << ",\"momentum\":" << spec.normalization.momentum
            << ",\"graph_batch_iterations\":20,\"timed\":false}" << std::endl;
  hip_check(hipSetDevice(device), "set_device");
  Resources resources;
  resources.initialize(spec, transform_parameters);
  std::unique_ptr<Buffer> ex_workspace, ex_reserve;
  std::size_t workspace_bytes = 0, reserve_bytes = 0;
  if (api == "ex") {
    // BN-only has neither residual z nor activation. Query both scratch sizes
    // before execution, and distinguish unsupported query APIs from an actual
    // Ex execution or numerical failure. Never guess a workspace size.
    workspace_bytes = reserve_bytes = std::numeric_limits<std::size_t>::max();
    const auto workspace_status =
        hipdnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
            resources.handle, HIPDNN_BATCHNORM_SPATIAL, HIPDNN_BATCHNORM_OPS_BN,
            resources.descriptors[0], nullptr, resources.descriptors[1],
            resources.descriptors[2], nullptr, &workspace_bytes);
    const auto reserve_status =
        hipdnnGetBatchNormalizationTrainingExReserveSpaceSize(
            resources.handle, HIPDNN_BATCHNORM_SPATIAL, HIPDNN_BATCHNORM_OPS_BN,
            nullptr, resources.descriptors[0], &reserve_bytes);
    std::cout << "{\"kind\":\"ex_storage_query\",\"workspace_status\":"
              << int(workspace_status)
              << ",\"reserve_status\":" << int(reserve_status)
              << ",\"workspace_bytes\":" << workspace_bytes
              << ",\"reserve_bytes\":" << reserve_bytes << "}" << std::endl;
    dnn_check(workspace_status, "ex_workspace_query");
    dnn_check(reserve_status, "ex_reserve_query");
    constexpr std::size_t maximum_probe_scratch = 512U * 1024U * 1024U;
    if (workspace_bytes > maximum_probe_scratch ||
        reserve_bytes > maximum_probe_scratch)
      throw std::runtime_error(
          "Ex scratch size was not initialized or exceeds the 512 MiB "
          "per-buffer functional-probe allocation limit");
    if (workspace_bytes)
      ex_workspace = std::make_unique<Buffer>(workspace_bytes, 0);
    if (reserve_bytes)
      ex_reserve = std::make_unique<Buffer>(reserve_bytes, 0);
  }
  std::vector<std::unique_ptr<Buffer>> buffers;
  std::vector<std::vector<std::uint8_t>> encoded_inputs;
  std::vector<std::vector<float>> poison;
  std::vector<std::unique_ptr<Buffer>> parameter_scratch;
  std::vector<std::vector<float>> scratch_poison;
  for (std::size_t i = 0; i < 10; ++i) {
    const auto &tensor = spec.tensors[i];
    buffers.push_back(
        std::make_unique<Buffer>(io::storage_element_count(tensor) *
                                     io::data_type_size(tensor.data_type),
                                 io::storage_element_count(tensor)));
    if (i < 5) {
      encoded_inputs.push_back(
          io::encode(io::scatter(logical_inputs[i], tensor), tensor.data_type));
      buffers.back()->upload(encoded_inputs.back(), resources.stream);
    } else {
      poison.push_back(bm::capture_output_poison(
          expected[i - 5], tensor.data_type, original.absolute_tolerance,
          original.relative_tolerance));
    }
  }
  if (transform_parameters) {
    for (std::size_t i = 1; i <= 2; ++i) {
      const auto count = logical_inputs[i].size();
      parameter_scratch.push_back(
          std::make_unique<Buffer>(count * sizeof(float), count));
      scratch_poison.push_back(bm::capture_output_poison(
          logical_inputs[i], FLAGDNN_DATA_FLOAT32, 0, 0));
    }
  }
  const auto reset = [&] {
    for (std::size_t i = 5; i < 10; ++i)
      buffers[i]->upload(io::encode(io::scatter(poison[i - 5], spec.tensors[i]),
                                    spec.tensors[i].data_type),
                         resources.stream);
    for (std::size_t i = 0; i < parameter_scratch.size(); ++i)
      parameter_scratch[i]->upload(
          io::encode(scratch_poison[i], FLAGDNN_DATA_FLOAT32),
          resources.stream);
    for (auto *scratch : {ex_workspace.get(), ex_reserve.get()})
      if (scratch)
        scratch->upload(std::vector<std::uint8_t>(scratch->logical_bytes, 0),
                        resources.stream);
  };
  const auto execute = [&](std::string_view phase) {
    float alpha = 1, beta = 0;
    std::array<void *, 2> parameters{buffers[1]->pointer, buffers[2]->pointer};
    for (std::size_t i = 0; i < parameter_scratch.size(); ++i) {
      parameters[i] = parameter_scratch[i]->pointer;
      dnn_check(hipdnnTransformTensor(resources.handle, &alpha,
                                      resources.descriptors[3],
                                      buffers[i + 1]->pointer, &beta,
                                      resources.descriptors[2], parameters[i]),
                std::string(phase) + "_transform_parameter_" +
                    std::to_string(i));
    }
    // This copy is part of the native adapter, not an external dtype
    // conversion. It restores fixed previous statistics before every training
    // invocation.
    for (std::size_t i : {std::size_t{3}, std::size_t{4}})
      hip_check(hipMemcpyAsync(buffers[i + 5]->pointer, buffers[i]->pointer,
                               buffers[i]->logical_bytes,
                               hipMemcpyDeviceToDevice, resources.stream),
                std::string(phase) + "_restore_stats");
    if (api == "ex") {
      dnn_check(
          hipdnnBatchNormalizationForwardTrainingEx(
              resources.handle, HIPDNN_BATCHNORM_SPATIAL,
              HIPDNN_BATCHNORM_OPS_BN, &alpha, &beta, resources.descriptors[0],
              buffers[0]->pointer, nullptr, nullptr, resources.descriptors[1],
              buffers[5]->pointer, resources.descriptors[2], parameters[0],
              parameters[1], spec.normalization.momentum, buffers[8]->pointer,
              buffers[9]->pointer, spec.normalization.epsilon,
              buffers[6]->pointer, buffers[7]->pointer, nullptr,
              ex_workspace ? ex_workspace->pointer : nullptr, workspace_bytes,
              ex_reserve ? ex_reserve->pointer : nullptr, reserve_bytes),
          std::string(phase) + "_api_ex");
    } else {
      dnn_check(hipdnnBatchNormalizationForwardTraining(
                    resources.handle, HIPDNN_BATCHNORM_SPATIAL, &alpha, &beta,
                    resources.descriptors[0], buffers[0]->pointer,
                    resources.descriptors[1], buffers[5]->pointer,
                    resources.descriptors[2], parameters[0], parameters[1],
                    spec.normalization.momentum, buffers[8]->pointer,
                    buffers[9]->pointer, spec.normalization.epsilon,
                    buffers[6]->pointer, buffers[7]->pointer),
                std::string(phase) + "_api");
    }
  };
  const auto validate = [&](std::string_view phase) {
    std::size_t failures = 0;
    for (const auto &[name, scratch] :
         {std::pair{"workspace", ex_workspace.get()},
          std::pair{"reserve", ex_reserve.get()}}) {
      if (!scratch)
        continue;
      const auto bytes = scratch->read(resources.stream);
      const auto guard_errors =
          std::count_if(bytes.begin() + scratch->logical_bytes, bytes.end(),
                        [](auto value) { return value != 0xA5; });
      if (guard_errors)
        ++failures;
      std::cout << "{\"kind\":\"ex_scratch_guard_check\",\"phase\":"
                << quote(phase) << ",\"buffer\":" << quote(name)
                << ",\"guard_errors\":" << guard_errors << "}" << std::endl;
    }
    for (std::size_t i = 0; i < parameter_scratch.size(); ++i) {
      const auto bytes = parameter_scratch[i]->read(resources.stream);
      const auto logical_size = parameter_scratch[i]->logical_bytes;
      const auto guard_errors =
          std::count_if(bytes.begin() + logical_size, bytes.end(),
                        [](auto b) { return b != 0xA5; });
      const auto actual =
          io::decode(std::span<const std::uint8_t>(bytes.data(), logical_size),
                     FLAGDNN_DATA_FLOAT32, logical_inputs[i + 1].size());
      std::size_t errors = 0, unwritten = 0;
      for (std::size_t j = 0; j < actual.size(); ++j) {
        if (!std::isfinite(actual[j]) || actual[j] != logical_inputs[i + 1][j])
          ++errors;
        if (actual[j] == scratch_poison[i][j])
          ++unwritten;
      }
      if (errors || unwritten || guard_errors)
        ++failures;
      std::cout << "{\"kind\":\"parameter_scratch_check\",\"phase\":"
                << quote(phase) << ",\"parameter_index\":" << i
                << ",\"elements\":" << actual.size() << ",\"errors\":" << errors
                << ",\"poison_remaining\":" << unwritten
                << ",\"guard_errors\":" << guard_errors << "}" << std::endl;
    }
    for (std::size_t ti = 0; ti < 10; ++ti) {
      const auto bytes = buffers[ti]->read(resources.stream);
      const auto logical_size = buffers[ti]->logical_bytes;
      const auto guard_errors =
          std::count_if(bytes.begin() + logical_size, bytes.end(),
                        [](auto b) { return b != 0xA5; });
      if (guard_errors)
        ++failures;
      if (ti < 5) {
        const bool unchanged =
            std::equal(encoded_inputs[ti].begin(), encoded_inputs[ti].end(),
                       bytes.begin());
        if (!unchanged)
          ++failures;
        std::cout << "{\"kind\":\"input_check\",\"phase\":" << quote(phase)
                  << ",\"input_index\":" << ti
                  << ",\"unchanged\":" << (unchanged ? "true" : "false")
                  << ",\"guard_errors\":" << guard_errors << "}" << std::endl;
        continue;
      }
      const auto actual = io::gather(
          io::decode(std::span<const std::uint8_t>(bytes.data(), logical_size),
                     spec.tensors[ti].data_type,
                     io::storage_element_count(spec.tensors[ti])),
          spec.tensors[ti]);
      Check check;
      for (std::size_t i = 0; i < actual.size(); ++i) {
        const double value = actual[i], reference = expected[ti - 5][i];
        const double absolute = std::abs(value - reference);
        const double relative =
            absolute / std::max({std::abs(value), std::abs(reference), 1e-30});
        if (!std::isfinite(value) || !std::isfinite(absolute) ||
            (absolute > original.absolute_tolerance &&
             relative > original.relative_tolerance)) {
          if (check.errors == 0)
            check.first = i;
          ++check.errors;
        }
        if (actual[i] == poison[ti - 5][i])
          ++check.unwritten;
        if (std::isfinite(absolute))
          check.max_abs = std::max(check.max_abs, absolute);
        if (std::isfinite(relative))
          check.max_rel = std::max(check.max_rel, relative);
      }
      if (check.errors || check.unwritten)
        ++failures;
      std::cout << "{\"kind\":\"output_check\",\"phase\":" << quote(phase)
                << ",\"output_index\":" << ti - 5
                << ",\"elements\":" << actual.size()
                << ",\"errors\":" << check.errors
                << ",\"poison_remaining\":" << check.unwritten
                << ",\"guard_errors\":" << guard_errors << ",\"first_error\":";
      if (check.errors)
        std::cout << check.first;
      else
        std::cout << "null";
      std::cout << ",\"max_abs\":" << check.max_abs
                << ",\"max_rel\":" << check.max_rel << "}" << std::endl;
    }
    return failures == 0;
  };
  bool direct = false, graph = false;
  try {
    reset();
    execute("direct");
    hip_check(hipStreamSynchronize(resources.stream), "direct_sync");
    direct = validate("direct");
  } catch (const std::exception &error) {
    std::cout << "{\"kind\":\"phase_error\",\"phase\":\"direct\",\"error\":"
              << quote(error.what()) << "}" << std::endl;
  }
  bool capturing = false;
  try {
    hip_check(hipStreamSynchronize(resources.stream), "pre_capture_sync");
    hip_check(
        hipStreamBeginCapture(resources.stream, hipStreamCaptureModeGlobal),
        "capture_begin");
    capturing = true;
    for (int i = 0; i < 20; ++i)
      execute("capture");
    const auto end_status =
        hipStreamEndCapture(resources.stream, &resources.graph);
    capturing = false;
    hip_check(end_status, "capture_end");
    std::size_t nodes = 0;
    hip_check(hipGraphGetNodes(resources.graph, nullptr, &nodes),
              "graph_nodes");
    std::cout << "{\"kind\":\"graph_metadata\",\"node_count\":" << nodes << "}"
              << std::endl;
    hip_check(hipGraphInstantiate(&resources.executable, resources.graph,
                                  nullptr, nullptr, 0),
              "graph_instantiate");
    reset();
    hip_check(hipGraphLaunch(resources.executable, resources.stream),
              "fresh_graph_launch");
    hip_check(hipStreamSynchronize(resources.stream), "fresh_graph_sync");
    graph = validate("fresh_graph");
  } catch (const std::exception &error) {
    if (capturing) {
      hipGraph_t discarded = nullptr;
      (void)hipStreamEndCapture(resources.stream, &discarded);
      if (discarded)
        (void)hipGraphDestroy(discarded);
    }
    std::cout
        << "{\"kind\":\"phase_error\",\"phase\":\"fresh_graph\",\"error\":"
        << quote(error.what()) << "}" << std::endl;
  }
  std::cout << "{\"kind\":\"batchnorm_result\",\"case_name\":"
            << quote(original.name) << ",\"api\":" << quote(api)
            << ",\"exact_catalog_semantics\":"
            << (fp32_parameters ? "false" : "true") << ",\"parameter_mapping\":"
            << quote(transform_parameters ? "hipdnn_transform_per_execute"
                     : fp32_parameters    ? "host_promoted_diagnostic"
                                          : "direct_exact")
            << ",\"direct_passed\":" << (direct ? "true" : "false")
            << ",\"graph_passed\":" << (graph ? "true" : "false") << "}"
            << std::endl;
  return direct && graph ? 0 : 1;
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::string name;
    std::string api = "standard";
    int device = 0;
    bool list = false, fp32_parameters = false, transform_parameters = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--list")
        list = true;
      else if (arg == "--fp32-parameters")
        fp32_parameters = true;
      else if (arg == "--transform-parameters")
        transform_parameters = true;
      else if (arg == "--case" && i + 1 < argc)
        name = argv[++i];
      else if (arg == "--device" && i + 1 < argc)
        device = std::stoi(argv[++i]);
      else if (arg == "--api" && i + 1 < argc)
        api = argv[++i];
      else
        throw std::invalid_argument(
            "usage: batchnorm_probe --list | --case NAME [--device N] "
            "[--api standard|ex] "
            "[--fp32-parameters | --transform-parameters]");
    }
    if (fp32_parameters && transform_parameters)
      throw std::invalid_argument(
          "parameter mapping modes are mutually exclusive");
    if (api != "standard" && api != "ex")
      throw std::invalid_argument("--api must be standard or ex");
    std::cout << std::setprecision(17);
    for (const auto &spec : bm::batchnorm_benchmark_cases()) {
      if (list)
        std::cout << spec.name << '\n';
      else if (spec.name == name)
        return probe(spec, fp32_parameters, transform_parameters, api, device);
    }
    if (list)
      return 0;
    throw std::invalid_argument(
        "--case must match exactly one shared BatchNorm benchmark case");
  } catch (const std::exception &error) {
    std::cout << "{\"kind\":\"probe_error\",\"error\":" << quote(error.what())
              << "}" << std::endl;
    return 1;
  }
}
