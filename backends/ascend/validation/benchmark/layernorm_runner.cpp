/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"

#include "common/flagdnn_provider.hpp"
#include "runtime/sha256.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/benchmark/aclnn_layernorm_provider.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_layernorm.hpp"
#include "validation/layernorm_validation.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef FLAGDNN_ASCEND_VALIDATION_CANN_VERSION
#define FLAGDNN_ASCEND_VALIDATION_CANN_VERSION "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID
#define FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID
#define FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256
#define FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256 "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256
#define FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256 "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256
#define FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256 "unknown"
#endif
#ifndef FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256
#define FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256 "unknown"
#endif

namespace flagdnn::benchmarking {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kGuard = 0xDDU;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

struct Samples {
  std::vector<double> stream;
  std::vector<double> submit;
  std::vector<double> end_to_end;

  void push(const acl::TimingSample& sample) {
    stream.push_back(sample.stream_us);
    submit.push_back(sample.submit_us);
    end_to_end.push_back(sample.end_to_end_us);
  }
};

struct ArtifactIdentity {
  std::string compiler;
  std::string request;
  std::string abi;
  std::string candidate;
};

using CacheSnapshot = std::map<std::string, std::string>;

std::string read_text(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("cannot read LayerNorm benchmark identity: " +
                             path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string json_string(std::string_view document, std::string_view key) {
  const std::string marker = "\"" + std::string(key) + "\"";
  std::size_t position = document.find(marker);
  if (position == std::string_view::npos) {
    throw std::runtime_error("LayerNorm benchmark identity is missing " +
                             std::string(key));
  }
  position = document.find(':', position + marker.size());
  position = document.find('"', position + 1);
  const std::size_t end = document.find('"', position + 1);
  if (position == std::string_view::npos || end == std::string_view::npos ||
      end == position + 1) {
    throw std::runtime_error("LayerNorm benchmark identity JSON is malformed");
  }
  return std::string(document.substr(position + 1, end - position - 1));
}

ArtifactIdentity artifact_identity(const std::filesystem::path& cache,
                                   std::string_view case_name) {
  const std::string marker = "\"name\":\"" + std::string(case_name) + "\"";
  std::filesystem::path directory;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error), end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (!iterator->is_regular_file() ||
        iterator->path().filename() != "request.json") {
      continue;
    }
    if (read_text(iterator->path()).find(marker) == std::string::npos) {
      continue;
    }
    if (!directory.empty()) {
      throw std::runtime_error(
          "multiple cached artifacts match LayerNorm benchmark");
    }
    directory = iterator->path().parent_path();
  }
  if (error || directory.empty()) {
    throw std::runtime_error("cannot find cached LayerNorm benchmark artifact");
  }
  const std::string manifest = read_text(directory / "manifest.json");
  return {json_string(manifest, "identity_sha256"),
          json_string(manifest, "request_sha256"),
          json_string(manifest, "launch_abi"),
          json_string(manifest, "candidate_id")};
}

CacheSnapshot snapshot_cache(const std::filesystem::path& cache) {
  CacheSnapshot result;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error), end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (iterator->is_regular_file()) {
      result.emplace(
          std::filesystem::relative(iterator->path(), cache).generic_string(),
          flagdnn::native::sha256_file(iterator->path()));
    }
  }
  if (error) {
    throw std::runtime_error("cannot snapshot LayerNorm benchmark cache");
  }
  return result;
}

std::vector<std::vector<float>> make_inputs(
    const BenchmarkCase& specification) {
  const acl::LayernormPlan plan = acl::plan_layernorm(specification);
  const std::size_t normalized =
      static_cast<std::size_t>(plan.normalized_elements);
  const std::size_t elements = tensor_io::element_count(specification.tensors[0]);
  std::vector<std::vector<float>> result(3);
  result[0].resize(elements);
  result[1].resize(normalized);
  result[2].resize(normalized);
  for (std::size_t column = 0; column < normalized; ++column) {
    const int scale_code = static_cast<int>((column * 7U) % 17U) - 8;
    const int bias_code = static_cast<int>((column * 11U) % 19U) - 9;
    result[1][column] =
        (column % 5U == 0U ? -1.0F : 1.0F) *
        (0.5F + static_cast<float>(std::abs(scale_code)) / 8.0F);
    result[2][column] = static_cast<float>(bias_code) / 16.0F;
  }
  for (std::size_t index = 0; index < elements; ++index) {
    const std::size_t row = index / normalized;
    const std::size_t column = index % normalized;
    const int centered =
        static_cast<int>((index * 17U + row * 13U + column * 3U) % 61U) - 30;
    result[0][index] = static_cast<float>(centered) / 16.0F;
  }
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = tensor_io::quantize(
        result[index], specification.tensors[index].data_type);
  }
  return result;
}

GuardedBuffer make_buffer(const TensorSpec& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  const std::size_t size = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "LayerNorm benchmark guarded allocation");
  std::vector<std::uint8_t> initial(size, kGuard);
  std::copy(encoded.begin(),
            encoded.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(size);
  device->copy_from_host_at(initial.data(), initial.size(), 0, stream.get());
  return {std::move(device), std::move(initial)};
}

std::vector<std::uint8_t> read_all(const GuardedBuffer& buffer,
                                   acl::Stream& stream) {
  std::vector<std::uint8_t> result(buffer.initial.size());
  buffer.device->copy_to_host_at(result.data(), result.size(), 0, stream.get());
  stream.synchronize();
  return result;
}

std::vector<float> read_output(const GuardedBuffer& buffer,
                               const TensorSpec& tensor,
                               acl::Stream& stream,
                               std::string_view provider,
                               std::string_view name) {
  const std::vector<std::uint8_t> all = read_all(buffer, stream);
  const std::size_t begin = tensor.binding_byte_offset;
  const std::size_t end = begin + tensor_io::encoded_byte_count(tensor);
  if (end > all.size() ||
      !std::equal(all.begin(),
                  all.begin() + static_cast<std::ptrdiff_t>(begin),
                  buffer.initial.begin()) ||
      !std::equal(all.begin() + static_cast<std::ptrdiff_t>(end),
                  all.end(),
                  buffer.initial.begin() + static_cast<std::ptrdiff_t>(end))) {
    throw std::runtime_error(std::string(provider) + " modified LayerNorm " +
                             std::string(name) + " allocation guards");
  }
  const std::vector<float> physical = tensor_io::decode_storage(
      provider,
      std::span<const std::uint8_t>(all.data() + begin, end - begin),
      tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  return tensor_io::gather(physical, tensor);
}

void execute(BenchmarkExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             acl::DeviceBuffer& workspace,
             acl::Stream& stream) {
  executable.execute(bindings,
                     executable.workspace_size() == 0
                         ? nullptr
                         : workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1;
  return values.at(std::min(index, values.size() - 1));
}

std::string escape(std::string_view value) {
  std::string result;
  for (const char character : value) {
    if (character == '"' || character == '\\') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return result;
}

void emit_metric(std::string_view name, const std::vector<double>& values) {
  std::cout << '"' << name << "\":{\"median\":"
            << percentile(values, 0.5) << ",\"p90\":"
            << percentile(values, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << values[index];
  }
  std::cout << "]}";
}

void emit_record(const BenchmarkCase& specification,
                 std::string_view provider,
                 std::string_view fingerprint,
                 const ArtifactIdentity& artifact,
                 const Samples& samples) {
  std::cout << "{\"schema_version\":2,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << escape(specification.name) << "\",\"environment\":{"
            << "\"soc_fingerprint\":\"" << escape(fingerprint) << "\","
            << "\"cann_package_version\":\""
            << FLAGDNN_ASCEND_VALIDATION_CANN_VERSION << "\","
            << "\"ascendcl_build_id\":\""
            << FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID << "\","
            << "\"runtime_build_id\":\""
            << FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID << "\"},"
            << "\"provider_identity\":";
  if (provider == "flagdnn") {
    std::cout << "{\"libtriton_jit_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256
              << "\",\"compiler_identity_sha256\":\"" << artifact.compiler
              << "\",\"artifact_request_sha256\":\"" << artifact.request
              << "\",\"launch_abi\":\"" << artifact.abi
              << "\",\"selected_candidate\":\"" << artifact.candidate
              << "\"},";
  } else {
    std::cout << "{\"libnnopbase_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256
              << "\",\"libopapi_math_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256
              << "\",\"libopapi_nn_sha256\":\""
              << FLAGDNN_ASCEND_VALIDATION_OPAPI_NN_SHA256 << "\"},";
  }
  std::cout << "\"benchmark_config\":{\"warmup_iterations\":"
            << specification.benchmark.warmup_iterations
            << ",\"sample_count\":" << specification.benchmark.sample_count
            << ",\"iterations_per_sample\":"
            << specification.benchmark.iterations_per_sample << "},";
  emit_metric("stream_us", samples.stream);
  std::cout << ',';
  emit_metric("submit_us", samples.submit);
  std::cout << ',';
  emit_metric("end_to_end_us", samples.end_to_end);
  std::cout << "}\n";
}

void validate_outputs(const BenchmarkCase& specification,
                      std::span<GuardedBuffer> flagdnn_outputs,
                      std::span<GuardedBuffer> aclnn_outputs,
                      acl::Stream& stream) {
  static constexpr std::array<std::string_view, 3> names = {
      "Y", "mean", "inverse variance"};
  for (std::size_t index = 0; index < names.size(); ++index) {
    const TensorSpec& tensor = specification.tensors[index + 3];
    const std::vector<float> flagdnn = read_output(
        flagdnn_outputs[index], tensor, stream, "FlagDNN", names[index]);
    const std::vector<float> aclnn = read_output(
        aclnn_outputs[index], tensor, stream, "ACLNN", names[index]);
    const double tolerance = index == 0
                                 ? specification.absolute_tolerance
                                 : std::max(2.0e-4,
                                            specification.absolute_tolerance);
    acl::compare_exact_reference(
        flagdnn,
        aclnn,
        tolerance,
        tolerance,
        specification.name + ":" + std::string(names[index]));
  }
}

void run_case(const BenchmarkCase& specification,
              flagdnn::Handle& handle,
              acl::Stream& stream,
              const std::filesystem::path& cache) {
  (void)acl::plan_layernorm(specification);
  const std::vector<std::vector<float>> logical = make_inputs(specification);
  auto make_buffers = [&] {
    std::vector<GuardedBuffer> result;
    result.reserve(6);
    for (std::size_t index = 0; index < 3; ++index) {
      result.push_back(make_buffer(
          specification.tensors[index],
          tensor_io::scatter(logical[index], specification.tensors[index]),
          stream));
    }
    for (std::size_t index = 3; index < 6; ++index) {
      result.push_back(make_buffer(
          specification.tensors[index],
          std::vector<float>(
              tensor_io::storage_element_count(specification.tensors[index]),
              tensor_io::kPaddingSentinel),
          stream));
    }
    return result;
  };
  std::vector<GuardedBuffer> flagdnn_buffers = make_buffers();
  std::vector<GuardedBuffer> aclnn_buffers = make_buffers();
  auto make_bindings = [&](std::vector<GuardedBuffer>& buffers) {
    std::vector<flagdnnBinding_t> result;
    result.reserve(6);
    for (std::size_t index = 0; index < 6; ++index) {
      result.push_back({
          specification.tensors[index].uid,
          buffers[index].device->opaque_at(
              specification.tensors[index].binding_byte_offset),
      });
    }
    return result;
  };
  std::vector<flagdnnBinding_t> flagdnn_bindings =
      make_bindings(flagdnn_buffers);
  std::vector<flagdnnBinding_t> aclnn_bindings = make_bindings(aclnn_buffers);

  FlagdnnProvider flagdnn_provider(handle);
  flagdnn_provider.set_autotune(false);
  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  std::unique_ptr<BenchmarkExecutable> aclnn =
      build_aclnn_layernorm(specification);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    aclnn->prepare(aclnn_bindings, stream.opaque());
    aclnn->prepare(aclnn_bindings, stream.opaque());
  } catch (const flagdnn::testing::AclnnLayernormUnsupportedError& error) {
    aclnn.reset();
    flagdnn.reset();
    throw BenchmarkUnsupportedError(
        specification.name + ": ACLNN_UNSUPPORTED status=" +
        std::to_string(error.status()) + " reason=" +
        single_line_reason(error.what()));
  }
  acl::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  acl::DeviceBuffer aclnn_workspace(aclnn->workspace_size());

  execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
  execute(*aclnn, aclnn_bindings, aclnn_workspace, stream);
  stream.synchronize();
  validate_outputs(specification,
                   std::span<GuardedBuffer>(flagdnn_buffers).subspan(3),
                   std::span<GuardedBuffer>(aclnn_buffers).subspan(3),
                   stream);
  for (std::size_t index = 0; index < 3; ++index) {
    if (read_all(flagdnn_buffers[index], stream) !=
            flagdnn_buffers[index].initial ||
        read_all(aclnn_buffers[index], stream) !=
            aclnn_buffers[index].initial) {
      throw std::runtime_error(
          "LayerNorm benchmark provider modified an input or guard");
    }
  }

  for (int index = 0; index < specification.benchmark.warmup_iterations;
       ++index) {
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    execute(*aclnn, aclnn_bindings, aclnn_workspace, stream);
  }
  stream.synchronize();
  const ArtifactIdentity artifact = artifact_identity(cache, specification.name);
  const CacheSnapshot before = snapshot_cache(cache);
  Samples flagdnn_samples;
  Samples aclnn_samples;
  acl::EventTimer timer;
  for (int sample = 0; sample < specification.benchmark.sample_count;
       ++sample) {
    const auto measure = [&](BenchmarkExecutable& executable,
                             std::span<const flagdnnBinding_t> bindings,
                             acl::DeviceBuffer& workspace,
                             Samples& samples) {
      samples.push(timer.measure(
          stream.get(),
          specification.benchmark.iterations_per_sample,
          [&] { execute(executable, bindings, workspace, stream); }));
    };
    if (sample % 2 == 0) {
      measure(*flagdnn, flagdnn_bindings, flagdnn_workspace, flagdnn_samples);
      measure(*aclnn, aclnn_bindings, aclnn_workspace, aclnn_samples);
    } else {
      measure(*aclnn, aclnn_bindings, aclnn_workspace, aclnn_samples);
      measure(*flagdnn, flagdnn_bindings, flagdnn_workspace, flagdnn_samples);
    }
  }
  stream.synchronize();
  if (before != snapshot_cache(cache)) {
    throw std::runtime_error(
        "Triton cache changed during LayerNorm timing");
  }
  validate_outputs(specification,
                   std::span<GuardedBuffer>(flagdnn_buffers).subspan(3),
                   std::span<GuardedBuffer>(aclnn_buffers).subspan(3),
                   stream);
  for (std::size_t index = 0; index < 3; ++index) {
    if (read_all(flagdnn_buffers[index], stream) !=
            flagdnn_buffers[index].initial ||
        read_all(aclnn_buffers[index], stream) !=
            aclnn_buffers[index].initial) {
      throw std::runtime_error(
          "LayerNorm benchmark timing modified an input or guard");
    }
  }
  emit_record(specification,
              "flagdnn",
              handle.target_fingerprint(),
              artifact,
              flagdnn_samples);
  emit_record(specification,
              "aclnn",
              handle.target_fingerprint(),
              artifact,
              aclnn_samples);
}

}  // namespace

int run_benchmark_suite(int argc,
                        char** argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    if (cases.empty()) {
      throw std::invalid_argument(
          "common LayerNorm benchmark must contain cases");
    }
    const std::vector<BenchmarkCase> ascend_cases =
        acl::make_ascend_layernorm_benchmark_cases(cases);
    acl::DevelopmentEnvironment development("layernorm-benchmark");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';
    std::cout << std::setprecision(9);
    const char* filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::vector<const BenchmarkCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const BenchmarkCase& specification : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          specification.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&specification);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "LayerNorm benchmark filter matched no cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      flagdnn::Handle handle("ascend", 0);
      handle.set_compiler(
          argv[1], argv[2], development.graph_cache().string());
      for (const BenchmarkCase* specification : selected) {
        try {
          run_case(*specification, handle, stream, development.graph_cache());
          coverage.record_pass(specification->name);
        } catch (const BenchmarkUnsupportedError& error) {
          coverage.record_skip(
              specification->name, single_line_reason(error.what()));
        }
      }
      stream.synchronize();
    }
    runtime.finalize();
    development.cleanup();
    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << suite_name << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << ascend_cases.size()
              << " schema_v2_records=" << coverage.passed() * 2U << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
