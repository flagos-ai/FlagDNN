/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"

#include "common/flagdnn_provider.hpp"
#include "runtime/sha256.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/batchnorm_validation.hpp"
#include "validation/benchmark/aclnn_batchnorm_provider.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_batchnorm.hpp"
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
namespace io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kGuard = 0xDBU;

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
    throw std::runtime_error("cannot read benchmark identity: " +
                             path.string());
  }
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

std::string json_string(std::string_view document, std::string_view key) {
  const std::string marker = "\"" + std::string(key) + "\"";
  std::size_t position = document.find(marker);
  if (position == std::string_view::npos) {
    throw std::runtime_error("benchmark identity is missing " +
                             std::string(key));
  }
  position = document.find(':', position + marker.size());
  position = document.find('"', position + 1);
  const std::size_t end = document.find('"', position + 1);
  if (position == std::string_view::npos || end == std::string_view::npos ||
      end == position + 1) {
    throw std::runtime_error("benchmark identity JSON is malformed");
  }
  return std::string(document.substr(position + 1, end - position - 1));
}

ArtifactIdentity artifact_identity(const std::filesystem::path& cache,
                                   std::string_view case_name) {
  const std::string marker =
      "\"name\":\"" + std::string(case_name) + "\"";
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
      throw std::runtime_error("multiple cached artifacts match benchmark");
    }
    directory = iterator->path().parent_path();
  }
  if (error || directory.empty()) {
    throw std::runtime_error("cannot find cached benchmark artifact");
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
    throw std::runtime_error("cannot snapshot benchmark cache");
  }
  return result;
}

std::vector<std::vector<float>> make_inputs(
    const BenchmarkCase& specification) {
  const TensorSpec& x = specification.tensors[0];
  const std::size_t channels = static_cast<std::size_t>(x.dimensions[1]);
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < x.dimensions.size(); ++axis) {
    spatial = io::checked_multiply(
        spatial,
        static_cast<std::size_t>(x.dimensions[axis]),
        "BatchNorm benchmark spatial elements");
  }
  std::vector<std::vector<float>> result(5);
  for (std::size_t index = 1; index < result.size(); ++index) {
    result[index].resize(channels);
  }
  for (std::size_t channel = 0; channel < channels; ++channel) {
    result[1][channel] = channel % 3U == 0U
                             ? -1.5F
                             : 0.75F + 0.03125F * channel;
    result[2][channel] =
        static_cast<float>(static_cast<int>(channel % 7U) - 3) * 0.125F;
    result[3][channel] =
        static_cast<float>(static_cast<int>(channel % 17U) - 8);
    result[4][channel] = 0.5F + static_cast<float>(channel % 19U) * 0.25F;
  }
  result[0].resize(io::element_count(x));
  for (std::size_t index = 0; index < result[0].size(); ++index) {
    const std::size_t channel = (index / spatial) % channels;
    const float center = channel % 2U == 0U ? 32.0F : -16.0F;
    const int code =
        static_cast<int>((index * 17U + channel * 13U) % 29U) - 14;
    result[0][index] = center + static_cast<float>(code) * 0.125F;
  }
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] =
        io::quantize(result[index], specification.tensors[index].data_type);
  }
  return result;
}

GuardedBuffer make_buffer(const TensorSpec& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      io::encode(physical, tensor.data_type);
  const std::size_t size = io::checked_add(
      io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "BatchNorm benchmark guarded allocation");
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
                               std::string_view provider) {
  const std::vector<std::uint8_t> all = read_all(buffer, stream);
  const std::size_t begin = tensor.binding_byte_offset;
  const std::size_t end = begin + io::encoded_byte_count(tensor);
  if (end > all.size() ||
      !std::equal(all.begin(),
                  all.begin() + static_cast<std::ptrdiff_t>(begin),
                  buffer.initial.begin()) ||
      !std::equal(all.begin() + static_cast<std::ptrdiff_t>(end),
                  all.end(),
                  buffer.initial.begin() + static_cast<std::ptrdiff_t>(end))) {
    throw std::runtime_error(std::string(provider) +
                             " modified benchmark allocation guards");
  }
  const std::vector<float> physical = io::decode_storage(
      provider,
      std::span<const std::uint8_t>(all.data() + begin, end - begin),
      tensor);
  io::require_padding_unchanged(provider, physical, tensor);
  return io::gather(physical, tensor);
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
    if (character == '"' || character == '\\') result.push_back('\\');
    result.push_back(character);
  }
  return result;
}

void emit_metric(std::string_view name, const std::vector<double>& values) {
  std::cout << '"' << name << "\":{\"median\":"
            << percentile(values, 0.5) << ",\"p90\":"
            << percentile(values, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) std::cout << ',';
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

std::vector<GuardedBuffer> make_buffers(
    const BenchmarkCase& specification,
    const std::vector<std::vector<float>>& logical,
    acl::Stream& stream) {
  std::vector<GuardedBuffer> result;
  for (std::size_t index = 0; index < 5; ++index) {
    result.push_back(make_buffer(specification.tensors[index],
                                 io::scatter(logical[index],
                                             specification.tensors[index]),
                                 stream));
  }
  for (std::size_t index = 5; index < 10; ++index) {
    result.push_back(make_buffer(
        specification.tensors[index],
        std::vector<float>(io::storage_element_count(
                               specification.tensors[index]),
                           io::kPaddingSentinel),
        stream));
  }
  return result;
}

std::vector<flagdnnBinding_t> make_bindings(
    const BenchmarkCase& specification,
    std::vector<GuardedBuffer>& buffers) {
  std::vector<flagdnnBinding_t> result;
  for (std::size_t index = 0; index < specification.tensors.size(); ++index) {
    result.push_back(
        {specification.tensors[index].uid,
         buffers[index].device->opaque_at(
             specification.tensors[index].binding_byte_offset)});
  }
  return result;
}

std::array<std::vector<float>, 5> read_outputs(
    std::string_view provider,
    const BenchmarkCase& specification,
    const std::vector<GuardedBuffer>& buffers,
    acl::Stream& stream) {
  std::array<std::vector<float>, 5> result;
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = read_output(buffers[index + 5],
                                specification.tensors[index + 5],
                                stream,
                                provider);
  }
  return result;
}

void run_case(const BenchmarkCase& specification,
              flagdnn::Handle& handle,
              acl::Stream& stream,
              const std::filesystem::path& cache) {
  (void)acl::plan_batchnorm(specification);
  const BenchmarkCase reference_specification =
      acl::batchnorm_reference_benchmark_case(specification);
  const std::vector<std::vector<float>> logical = make_inputs(specification);
  std::vector<GuardedBuffer> flagdnn_buffers =
      make_buffers(specification, logical, stream);
  std::vector<GuardedBuffer> aclnn_buffers =
      make_buffers(reference_specification, logical, stream);
  std::vector<flagdnnBinding_t> flagdnn_bindings =
      make_bindings(specification, flagdnn_buffers);
  std::vector<flagdnnBinding_t> aclnn_bindings =
      make_bindings(reference_specification, aclnn_buffers);

  FlagdnnProvider flagdnn_provider(handle);
  flagdnn_provider.set_autotune(false);
  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  std::unique_ptr<BenchmarkExecutable> aclnn =
      build_aclnn_batchnorm(reference_specification);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    aclnn->prepare(aclnn_bindings, stream.opaque());
    aclnn->prepare(aclnn_bindings, stream.opaque());
  } catch (const flagdnn::testing::AclnnBatchnormUnsupportedError& error) {
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
  const auto flagdnn_output =
      read_outputs("FlagDNN", specification, flagdnn_buffers, stream);
  auto aclnn_output =
      read_outputs("ACLNN", reference_specification, aclnn_buffers, stream);
  acl::batchnorm_reference_variance_to_invstd(
      aclnn_output[2], specification.normalization.epsilon);
  static constexpr std::array<std::string_view, 5> output_names = {
      "Y", "mean", "invstd", "next mean", "next variance"};
  for (std::size_t index = 0; index < output_names.size(); ++index) {
    acl::compare_exact_reference(
        flagdnn_output[index],
        aclnn_output[index],
        specification.absolute_tolerance,
        specification.relative_tolerance,
        specification.name + ":" + std::string(output_names[index]));
  }
  for (std::size_t index = 0; index < 5; ++index) {
    if (read_all(flagdnn_buffers[index], stream) !=
            flagdnn_buffers[index].initial ||
        read_all(aclnn_buffers[index], stream) != aclnn_buffers[index].initial) {
      throw std::runtime_error("benchmark provider modified an input/guard");
    }
  }

  for (int index = 0; index < specification.benchmark.warmup_iterations;
       ++index) {
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    execute(*aclnn, aclnn_bindings, aclnn_workspace, stream);
  }
  stream.synchronize();
  const ArtifactIdentity artifact =
      artifact_identity(cache, specification.name);
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
    throw std::runtime_error("Triton cache changed during timing");
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
  aclnn.reset();
  flagdnn.reset();
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
    if (cases.size() != 24) {
      throw std::invalid_argument(
          "common BatchNorm benchmark must contain 24 cases");
    }
    const std::vector<BenchmarkCase> ascend_cases =
        acl::make_ascend_batchnorm_benchmark_cases(cases);
    if (ascend_cases.size() != 24) {
      throw std::logic_error(
          "Ascend BatchNorm benchmark must contain 24 cases");
    }
    acl::DevelopmentEnvironment development("batchnorm-benchmark");
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
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
      throw std::runtime_error("benchmark filter matched no cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      flagdnn::Handle handle("ascend", 0);
      handle.set_compiler(argv[1], argv[2], development.graph_cache().string());
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
