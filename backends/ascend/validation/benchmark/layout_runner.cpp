/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"

#include "common/flagdnn_provider.hpp"
#include "runtime/sha256.hpp"
#include "validation/acl_runtime.hpp"
#include "validation/benchmark/aclnn_layout_provider.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_flatten.h>

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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

namespace flagdnn::benchmarking {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kAllocationGuard = 0xD3U;

std::string read_text_file(const std::filesystem::path& path,
                           std::size_t maximum_size = 4U * 1024U * 1024U) {
  std::error_code error;
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size > maximum_size) {
    throw std::runtime_error(
        "layout benchmark identity file is missing or too large: " +
        path.string());
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error(
        "cannot read layout benchmark identity: " + path.string());
  }
  std::string result((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
  if (input.bad()) {
    throw std::runtime_error(
        "cannot finish reading layout benchmark identity");
  }
  return result;
}

std::string extract_json_string(std::string_view document,
                                std::string_view key) {
  const std::string marker = "\"" + std::string(key) + "\"";
  std::size_t position = document.find(marker);
  if (position == std::string_view::npos ||
      (position = document.find(':', position + marker.size())) ==
          std::string_view::npos ||
      (position = document.find('"', position + 1)) ==
          std::string_view::npos) {
    throw std::runtime_error(
        "layout benchmark identity is missing JSON key " +
        std::string(key));
  }
  ++position;
  std::string result;
  bool escaped = false;
  for (; position < document.size(); ++position) {
    const char character = document[position];
    if (escaped) {
      if (character != '"' && character != '\\' && character != '/') {
        throw std::runtime_error(
            "layout benchmark identity has unsupported JSON escape");
      }
      result.push_back(character);
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      if (result.empty()) {
        throw std::runtime_error(
            "layout benchmark identity JSON string is empty");
      }
      return result;
    } else {
      result.push_back(character);
    }
  }
  throw std::runtime_error(
      "layout benchmark identity JSON string is unterminated");
}

struct FlagdnnArtifactIdentity {
  std::string compiler_identity_sha256;
  std::string artifact_request_sha256;
  std::string launch_abi;
  std::string selected_candidate;
};

FlagdnnArtifactIdentity find_artifact_identity(
    const std::filesystem::path& cache,
    std::string_view case_name) {
  const std::string marker =
      "\"graph\":{\"name\":\"" + std::string(case_name) + "\"";
  std::filesystem::path directory;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error),
       end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (!iterator->is_regular_file() ||
        iterator->path().filename() != "request.json" ||
        read_text_file(iterator->path()).find(marker) == std::string::npos) {
      continue;
    }
    if (!directory.empty()) {
      throw std::runtime_error(
          "multiple cached artifacts match layout benchmark case " +
          std::string(case_name));
    }
    directory = iterator->path().parent_path();
  }
  if (error || directory.empty()) {
    throw std::runtime_error(
        "cannot find cached artifact for layout benchmark case " +
        std::string(case_name));
  }
  const std::string manifest = read_text_file(directory / "manifest.json");
  return {extract_json_string(manifest, "identity_sha256"),
          extract_json_string(manifest, "request_sha256"),
          extract_json_string(manifest, "launch_abi"),
          extract_json_string(manifest, "candidate_id")};
}

using CacheSnapshot = std::map<std::string, std::string>;

CacheSnapshot snapshot_cache(const std::filesystem::path& cache) {
  CacheSnapshot result;
  std::error_code error;
  for (std::filesystem::recursive_directory_iterator iterator(cache, error),
       end;
       iterator != end && !error;
       iterator.increment(error)) {
    if (!iterator->is_regular_file()) {
      continue;
    }
    const std::filesystem::path relative =
        std::filesystem::relative(iterator->path(), cache, error);
    if (error) {
      break;
    }
    result.emplace(relative.generic_string(),
                   flagdnn::native::sha256_file(iterator->path()));
  }
  if (error) {
    throw std::runtime_error(
        "cannot snapshot layout benchmark cache: " + error.message());
  }
  return result;
}

std::string operation_name(Operation operation) {
  switch (operation) {
    case Operation::kReshape:
      return "reshape";
    case Operation::kTranspose:
      return "transpose";
    case Operation::kSlice:
      return "slice";
    default:
      break;
  }
  throw std::invalid_argument("Ascend layout benchmark operation is invalid");
}

BenchmarkCase make_special_case(const BenchmarkCase& prototype) {
  BenchmarkCase result;
  result.name = operation_name(prototype.operation) +
                "_ascend_gapped_offsets_padding_perf_fp32";
  result.operation = prototype.operation;
  result.benchmark = prototype.benchmark;
  constexpr std::int64_t kInputUid = 76000;
  constexpr std::int64_t kOutputUid = 76001;
  switch (result.operation) {
    case Operation::kReshape:
      result.tensors = {
          {kInputUid, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {47, 13, 2}, 32},
          {kOutputUid, FLAGDNN_DATA_FLOAT32, {6, 4}, {11, 2}, 64},
      };
      result.reshape.dimensions = result.tensors[1].dimensions;
      result.reshape.strides = result.tensors[1].strides;
      result.reshape.logical = true;
      break;
    case Operation::kTranspose:
      result.tensors = {
          {kInputUid, FLAGDNN_DATA_FLOAT32, {2, 3, 4}, {47, 13, 2}, 32},
          {kOutputUid,
           FLAGDNN_DATA_FLOAT32,
           {4, 2, 3},
           {29, 11, 3},
           64},
      };
      result.transpose.permutation = {2, 0, 1};
      break;
    case Operation::kSlice:
      result.tensors = {
          {kInputUid, FLAGDNN_DATA_FLOAT32, {3, 5, 7}, {101, 17, 2}, 32},
          {kOutputUid,
           FLAGDNN_DATA_FLOAT32,
           {2, 3, 3},
           {43, 11, 3},
           64},
      };
      result.slice.slices = {{1, 3}, {0, 5}, {1, 7}};
      result.slice.strides = {1, 2, 2};
      break;
    default:
      throw std::invalid_argument(
          "Ascend layout benchmark special-case operation is invalid");
  }
  return result;
}

std::vector<float> make_input(const TensorSpec& tensor) {
  constexpr std::array<float, 16> kValues = {
      -0.0F,
      0.0F,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      std::bit_cast<float>(std::uint32_t{0x7FC12345U}),
      std::bit_cast<float>(std::uint32_t{0xFFC54321U}),
      1.0F,
      -1.0F,
      0.5F,
      -2.25F,
      3.5F,
      -4.0F,
      5.25F,
      -6.5F,
      7.0F,
      -8.75F};
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = kValues[index % kValues.size()];
  }
  return tensor_io::quantize(result, tensor.data_type);
}

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

GuardedBuffer make_buffer(const TensorSpec& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "layout benchmark guarded allocation");
  std::vector<std::uint8_t> initial(allocation, kAllocationGuard);
  std::copy(encoded.begin(),
            encoded.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(initial.data(), initial.size(), 0, stream.get());
  return {std::move(device), std::move(initial)};
}

GuardedBuffer make_input_buffer(const TensorSpec& tensor,
                                std::span<const float> logical,
                                acl::Stream& stream) {
  return make_buffer(tensor, tensor_io::scatter(logical, tensor), stream);
}

GuardedBuffer make_output_buffer(const TensorSpec& tensor,
                                 acl::Stream& stream) {
  const std::vector<float> physical(tensor_io::storage_element_count(tensor),
                                    tensor_io::kPaddingSentinel);
  return make_buffer(tensor, physical, stream);
}

std::vector<std::uint8_t> read_all(const GuardedBuffer& buffer,
                                   acl::Stream& stream) {
  std::vector<std::uint8_t> result(buffer.initial.size());
  buffer.device->copy_to_host_at(result.data(), result.size(), 0, stream.get());
  stream.synchronize();
  return result;
}

void require_input_unchanged(std::string_view provider,
                             const GuardedBuffer& buffer,
                             acl::Stream& stream) {
  if (read_all(buffer, stream) != buffer.initial) {
    throw std::runtime_error(std::string(provider) +
                             " modified layout benchmark input/guards");
  }
}

std::vector<float> read_output(
    std::string_view provider,
    const GuardedBuffer& buffer,
    const TensorSpec& tensor,
    acl::Stream& stream) {
  const std::vector<std::uint8_t> all = read_all(buffer, stream);
  const std::size_t begin = tensor.binding_byte_offset;
  const std::size_t end = tensor_io::checked_add(
      begin,
      tensor_io::encoded_byte_count(tensor),
      "layout benchmark output payload");
  if (all.size() != buffer.initial.size() || end > all.size() ||
      !std::equal(all.begin(),
                  all.begin() + static_cast<std::ptrdiff_t>(begin),
                  buffer.initial.begin()) ||
      !std::equal(all.begin() + static_cast<std::ptrdiff_t>(end),
                  all.end(),
                  buffer.initial.begin() + static_cast<std::ptrdiff_t>(end))) {
    throw std::runtime_error(std::string(provider) +
                             " modified layout benchmark offset/tail guard");
  }
  const std::span<const std::uint8_t> payload(all.data() + begin, end - begin);
  const std::vector<float> physical =
      tensor_io::decode_storage(provider, payload, tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  return tensor_io::gather(physical, tensor);
}

void require_exact(std::string_view provider,
                   std::span<const float> actual,
                   std::span<const float> expected,
                   flagdnnDataType_t data_type) {
  const std::vector<std::uint8_t> actual_bytes =
      tensor_io::encode(actual, data_type);
  const std::vector<std::uint8_t> expected_bytes =
      tensor_io::encode(expected, data_type);
  if (actual_bytes != expected_bytes) {
    throw std::runtime_error(std::string(provider) +
                             " differs from the exact ACLNN layout result");
  }
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

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::invalid_argument(
        "cannot summarize empty layout benchmark samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t index = static_cast<std::size_t>(
      std::ceil(fraction * static_cast<double>(values.size()))) - 1;
  return values[std::min(index, values.size() - 1)];
}

struct MetricSamples {
  std::vector<double> stream_us;
  std::vector<double> submit_us;
  std::vector<double> end_to_end_us;

  void push(const acl::TimingSample& sample) {
    stream_us.push_back(sample.stream_us);
    submit_us.push_back(sample.submit_us);
    end_to_end_us.push_back(sample.end_to_end_us);
  }
};

std::string json_escape(std::string_view input) {
  std::string result;
  for (const char character : input) {
    if (character == '"' || character == '\\') {
      result.push_back('\\');
      result.push_back(character);
    } else if (static_cast<unsigned char>(character) < 0x20U) {
      throw std::invalid_argument(
          "layout benchmark JSON string has control byte");
    } else {
      result.push_back(character);
    }
  }
  return result;
}

void emit_metric(std::string_view name,
                 const std::vector<double>& samples) {
  std::cout << '"' << name << "\":{\"median\":"
            << percentile(samples, 0.5) << ",\"p90\":"
            << percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}";
}

struct EnvironmentIdentity {
  std::string soc_fingerprint;
  std::string cann_package_version;
  std::string ascendcl_build_id;
  std::string runtime_build_id;
};

struct AclnnLibraryIdentity {
  std::string libnnopbase_sha256;
  std::string libopapi_math_sha256;
};

template <typename Function>
std::string loaded_symbol_object_sha256(Function* symbol,
                                        std::string_view expected_name) {
  Dl_info information{};
  if (dladdr(reinterpret_cast<void*>(symbol), &information) == 0 ||
      information.dli_fname == nullptr) {
    throw std::runtime_error(
        "cannot resolve loaded layout object for " +
        std::string(expected_name));
  }
  std::error_code error;
  const std::filesystem::path path =
      std::filesystem::canonical(information.dli_fname, error);
  if (error || path.filename().string().find(expected_name) ==
                   std::string::npos) {
    throw std::runtime_error(
        "loaded layout ACLNN object identity is invalid");
  }
  return flagdnn::native::sha256_file(path);
}

struct LoadedObjectQuery {
  std::string_view expected_name;
  std::vector<std::filesystem::path> matches;
};

int collect_loaded_object(dl_phdr_info* information,
                          std::size_t,
                          void* opaque) {
  auto& query = *static_cast<LoadedObjectQuery*>(opaque);
  if (information == nullptr || information->dlpi_name == nullptr ||
      information->dlpi_name[0] == '\0') {
    return 0;
  }
  std::error_code error;
  const std::filesystem::path path =
      std::filesystem::canonical(information->dlpi_name, error);
  if (!error && path.filename().string().find(query.expected_name) !=
                    std::string::npos &&
      std::find(query.matches.begin(), query.matches.end(), path) ==
          query.matches.end()) {
    query.matches.push_back(path);
  }
  return 0;
}

std::string loaded_named_object_sha256(std::string_view expected_name) {
  LoadedObjectQuery query{expected_name, {}};
  (void)dl_iterate_phdr(&collect_loaded_object, &query);
  if (query.matches.size() != 1) {
    throw std::runtime_error(
        "expected exactly one loaded layout object matching " +
        std::string(expected_name));
  }
  return flagdnn::native::sha256_file(query.matches.front());
}

AclnnLibraryIdentity aclnn_library_identity() {
  AclnnLibraryIdentity result{
      loaded_symbol_object_sha256(&aclCreateTensor, "libnnopbase.so"),
      loaded_symbol_object_sha256(&aclnnFlatten, "libopapi_math.so")};
  if (result.libnnopbase_sha256 !=
          FLAGDNN_ASCEND_VALIDATION_NNOPBASE_SHA256 ||
      result.libopapi_math_sha256 !=
          FLAGDNN_ASCEND_VALIDATION_OPAPI_MATH_SHA256) {
    throw std::runtime_error(
        "loaded layout ACLNN libraries differ from configured libraries");
  }
  return result;
}

EnvironmentIdentity environment_identity(const flagdnn::Handle& handle) {
  const std::string runtime_version = acl::runtime_package_version();
  if (runtime_version != FLAGDNN_ASCEND_VALIDATION_CANN_VERSION) {
    throw std::runtime_error(
        "loaded CANN runtime differs from configured layout benchmark");
  }
  return {std::string(handle.target_fingerprint()),
          runtime_version,
          FLAGDNN_ASCEND_VALIDATION_ASCENDCL_BUILD_ID,
          FLAGDNN_ASCEND_VALIDATION_RUNTIME_BUILD_ID};
}

void emit_prefix(std::string_view provider,
                 const BenchmarkCase& specification,
                 const EnvironmentIdentity& environment) {
  std::cout << "{\"schema_version\":2,\"kind\":\"steady_state\","
            << "\"provider\":\"" << json_escape(provider) << "\","
            << "\"case\":\"" << json_escape(specification.name) << "\","
            << "\"environment\":{\"soc_fingerprint\":\""
            << json_escape(environment.soc_fingerprint)
            << "\",\"cann_package_version\":\""
            << json_escape(environment.cann_package_version)
            << "\",\"ascendcl_build_id\":\""
            << json_escape(environment.ascendcl_build_id)
            << "\",\"runtime_build_id\":\""
            << json_escape(environment.runtime_build_id) << "\"},";
}

void emit_config(const BenchmarkConfig& config) {
  std::cout << "\"benchmark_config\":{\"warmup_iterations\":"
            << config.warmup_iterations << ",\"sample_count\":"
            << config.sample_count << ",\"iterations_per_sample\":"
            << config.iterations_per_sample << "},";
}

void emit_metrics(const MetricSamples& samples) {
  emit_metric("stream_us", samples.stream_us);
  std::cout << ',';
  emit_metric("submit_us", samples.submit_us);
  std::cout << ',';
  emit_metric("end_to_end_us", samples.end_to_end_us);
  std::cout << "}\n";
}

void emit_flagdnn_record(const BenchmarkCase& specification,
                         const EnvironmentIdentity& environment,
                         const FlagdnnArtifactIdentity& identity,
                         std::string_view libtriton_jit_sha256,
                         const MetricSamples& samples) {
  emit_prefix("flagdnn", specification, environment);
  std::cout << "\"provider_identity\":{\"libtriton_jit_sha256\":\""
            << json_escape(libtriton_jit_sha256)
            << "\",\"compiler_identity_sha256\":\""
            << json_escape(identity.compiler_identity_sha256)
            << "\",\"artifact_request_sha256\":\""
            << json_escape(identity.artifact_request_sha256)
            << "\",\"launch_abi\":\"" << json_escape(identity.launch_abi)
            << "\",\"selected_candidate\":\""
            << json_escape(identity.selected_candidate) << "\"},";
  emit_config(specification.benchmark);
  emit_metrics(samples);
}

void emit_aclnn_record(const BenchmarkCase& specification,
                       const EnvironmentIdentity& environment,
                       const AclnnLibraryIdentity& identity,
                       const MetricSamples& samples) {
  emit_prefix("aclnn", specification, environment);
  std::cout << "\"provider_identity\":{\"libnnopbase_sha256\":\""
            << json_escape(identity.libnnopbase_sha256)
            << "\",\"libopapi_math_sha256\":\""
            << json_escape(identity.libopapi_math_sha256) << "\"},";
  emit_config(specification.benchmark);
  emit_metrics(samples);
}

void validate_benchmark_config(const BenchmarkConfig& config) {
  if (config.warmup_iterations < 0 || config.sample_count <= 0 ||
      config.iterations_per_sample <= 0) {
    throw std::invalid_argument(
        "Ascend layout benchmark configuration is invalid");
  }
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void verify_results(const BenchmarkCase& specification,
                    GuardedBuffer& flagdnn_input,
                    GuardedBuffer& flagdnn_output,
                    GuardedBuffer& aclnn_input,
                    GuardedBuffer& aclnn_output,
                    acl::Stream& stream) {
  require_input_unchanged("FlagDNN", flagdnn_input, stream);
  require_input_unchanged("ACLNN", aclnn_input, stream);
  const TensorSpec& output = specification.tensors[1];
  const std::vector<float> flagdnn = read_output(
      "FlagDNN", flagdnn_output, output, stream);
  const std::vector<float> reference = read_output(
      "ACLNN", aclnn_output, output, stream);
  require_exact("FlagDNN", flagdnn, reference, output.data_type);
}

void run_case(const BenchmarkCase& specification,
              FlagdnnProvider& flagdnn_provider,
              AclnnLayoutProvider& aclnn_provider,
              acl::Stream& stream,
              const std::filesystem::path& cache,
              const EnvironmentIdentity& environment,
              std::string_view libtriton_jit_sha256,
              const AclnnLibraryIdentity& aclnn_identity) {
  validate_benchmark_config(specification.benchmark);
  const ProviderCapability capability =
      aclnn_provider.capability(specification);
  if (!capability.supported) {
    throw BenchmarkUnsupportedError(
        specification.name + ": ACLNN_UNSUPPORTED: " + capability.reason);
  }
  const AclnnLayoutBenchmarkPlan plan =
      plan_aclnn_layout_benchmark(specification);
  const std::vector<float> logical_input = make_input(plan.input);

  GuardedBuffer flagdnn_input =
      make_input_buffer(plan.input, logical_input, stream);
  GuardedBuffer aclnn_input =
      make_input_buffer(plan.input, logical_input, stream);
  GuardedBuffer flagdnn_output = make_output_buffer(plan.output, stream);
  GuardedBuffer aclnn_output = make_output_buffer(plan.output, stream);
  const std::array<flagdnnBinding_t, 2> flagdnn_bindings = {
      flagdnnBinding_t{plan.input.uid,
                       flagdnn_input.device->opaque_at(
                           plan.input.binding_byte_offset)},
      flagdnnBinding_t{plan.output.uid,
                       flagdnn_output.device->opaque_at(
                           plan.output.binding_byte_offset)}};
  const std::array<flagdnnBinding_t, 2> aclnn_bindings = {
      flagdnnBinding_t{plan.input.uid,
                       aclnn_input.device->opaque_at(
                           plan.input.binding_byte_offset)},
      flagdnnBinding_t{plan.output.uid,
                       aclnn_output.device->opaque_at(
                           plan.output.binding_byte_offset)}};

  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  std::unique_ptr<BenchmarkExecutable> aclnn =
      aclnn_provider.build(specification);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    aclnn->prepare(aclnn_bindings, stream.opaque());
    aclnn->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnLayoutBenchmarkUnsupportedError& error) {
    aclnn.reset();
    flagdnn.reset();
    throw BenchmarkUnsupportedError(
        specification.name + ": ACLNN_UNSUPPORTED status=" +
        std::to_string(error.status()) + " reason=" + error.what());
  }
  auto flagdnn_workspace =
      std::make_unique<acl::DeviceBuffer>(flagdnn->workspace_size());
  auto aclnn_workspace =
      std::make_unique<acl::DeviceBuffer>(aclnn->workspace_size());

  try {
    stream.synchronize();
    execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
    execute(*aclnn, aclnn_bindings, *aclnn_workspace, stream);
    stream.synchronize();
    verify_results(specification,
                   flagdnn_input,
                   flagdnn_output,
                   aclnn_input,
                   aclnn_output,
                   stream);
    std::cout << specification.name
              << ": correctness PASS exact_reference=ACLNN repeatable=2\n";

    for (int index = 0; index < specification.benchmark.warmup_iterations;
         ++index) {
      execute(*flagdnn, flagdnn_bindings, *flagdnn_workspace, stream);
    }
    stream.synchronize();
    for (int index = 0; index < specification.benchmark.warmup_iterations;
         ++index) {
      execute(*aclnn, aclnn_bindings, *aclnn_workspace, stream);
    }
    stream.synchronize();

    const FlagdnnArtifactIdentity artifact =
        find_artifact_identity(cache, specification.name);
    const CacheSnapshot before = snapshot_cache(cache);
    MetricSamples flagdnn_samples;
    MetricSamples aclnn_samples;
    flagdnn_samples.stream_us.reserve(specification.benchmark.sample_count);
    flagdnn_samples.submit_us.reserve(specification.benchmark.sample_count);
    flagdnn_samples.end_to_end_us.reserve(
        specification.benchmark.sample_count);
    aclnn_samples.stream_us.reserve(specification.benchmark.sample_count);
    aclnn_samples.submit_us.reserve(specification.benchmark.sample_count);
    aclnn_samples.end_to_end_us.reserve(
        specification.benchmark.sample_count);
    acl::EventTimer timer;
    const auto measure_flagdnn = [&]() {
      flagdnn_samples.push(timer.measure(
          stream.get(),
          specification.benchmark.iterations_per_sample,
          [&]() {
            execute(*flagdnn,
                    flagdnn_bindings,
                    *flagdnn_workspace,
                    stream);
          }));
    };
    const auto measure_aclnn = [&]() {
      aclnn_samples.push(timer.measure(
          stream.get(),
          specification.benchmark.iterations_per_sample,
          [&]() {
            execute(*aclnn, aclnn_bindings, *aclnn_workspace, stream);
          }));
    };
    for (int sample = 0; sample < specification.benchmark.sample_count;
         ++sample) {
      if (sample % 2 == 0) {
        measure_flagdnn();
        measure_aclnn();
      } else {
        measure_aclnn();
        measure_flagdnn();
      }
    }
    stream.synchronize();
    if (before != snapshot_cache(cache)) {
      throw std::runtime_error(
          specification.name +
          ": Triton cache changed during layout sample window");
    }
    verify_results(specification,
                   flagdnn_input,
                   flagdnn_output,
                   aclnn_input,
                   aclnn_output,
                   stream);
    emit_flagdnn_record(specification,
                        environment,
                        artifact,
                        libtriton_jit_sha256,
                        flagdnn_samples);
    emit_aclnn_record(
        specification, environment, aclnn_identity, aclnn_samples);
    std::cout << specification.name
              << ": stream_speedup="
              << percentile(aclnn_samples.stream_us, 0.5) /
                     percentile(flagdnn_samples.stream_us, 0.5)
              << " end_to_end_speedup="
              << percentile(aclnn_samples.end_to_end_us, 0.5) /
                     percentile(flagdnn_samples.end_to_end_us, 0.5)
              << '\n';
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    aclnn.reset();
    flagdnn.reset();
    throw;
  }
  stream.synchronize();
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
    std::cout << std::setprecision(9);
    if (cases.empty()) {
      throw std::invalid_argument("layout benchmark suite has no cases");
    }
    const Operation operation = cases.front().operation;
    const std::size_t expected_common =
        operation == Operation::kSlice ? 12U : 15U;
    if (cases.size() != expected_common ||
        std::any_of(cases.begin(), cases.end(), [&](const auto& item) {
          return item.operation != operation;
        })) {
      throw std::invalid_argument(
          "Ascend layout benchmark common catalog count/mode is invalid");
    }
    std::vector<BenchmarkCase> ascend_cases(cases.begin(), cases.end());
    ascend_cases.push_back(make_special_case(cases.front()));
    const std::size_t expected_catalog = expected_common + 1U;
    if (ascend_cases.size() != expected_catalog) {
      throw std::logic_error(
          "Ascend layout benchmark extended catalog count is invalid");
    }

    acl::DevelopmentEnvironment development(
        operation_name(operation) + "-benchmark");
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
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
          "FLAGDNN_BENCHMARK_CASE matched no layout cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        const EnvironmentIdentity environment = environment_identity(handle);
        const std::string libtriton_jit_sha256 =
            loaded_named_object_sha256("libtriton_jit.so");
        if (libtriton_jit_sha256 !=
            FLAGDNN_ASCEND_VALIDATION_TRITON_JIT_SHA256) {
          throw std::runtime_error(
              "loaded libtriton_jit differs from production target");
        }
        const AclnnLibraryIdentity aclnn_identity =
            aclnn_library_identity();
        FlagdnnProvider flagdnn_provider(handle);
        flagdnn_provider.set_autotune(false);
        AclnnLayoutProvider aclnn_provider;
        for (const BenchmarkCase* specification : selected) {
          try {
            run_case(*specification,
                     flagdnn_provider,
                     aclnn_provider,
                     stream,
                     development.graph_cache(),
                     environment,
                     libtriton_jit_sha256,
                     aclnn_identity);
            coverage.record_pass(specification->name);
          } catch (const BenchmarkUnsupportedError& error) {
            coverage.record_skip(
                specification->name, single_line_reason(error.what()));
          }
        }
        stream.synchronize();
      }
    }
    runtime.finalize();
    development.cleanup();
    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << suite_name << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << expected_catalog
              << " schema_v2_records=" << coverage.passed() * 2U << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
