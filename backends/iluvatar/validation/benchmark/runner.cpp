// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/runner.hpp"
#include "benchmark/shared_cases.hpp"

#include "benchmark/corex_cudnn_provider.hpp"
#include "benchmark/cuda_graph.hpp"
#include "benchmark/runner_contract.hpp"
#include "common/flagdnn_provider.hpp"
#include "corex_cudnn_status.hpp"
#include "functional/runner_support.hpp"
#include "functional/capability_skips.hpp"

#include <cuda_runtime_api.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace iv = flagdnn::iluvatar::validation;
namespace ivb = flagdnn::iluvatar::validation::benchmark;
namespace fr = flagdnn::iluvatar::validation::functional;

constexpr int kSkipReturnCode = 77;
constexpr std::size_t kGuardBytes = 64;
constexpr std::uint8_t kGuardValue = 0xd3U;

enum class CaseResult { kComparable, kReferenceSkipped };

class BenchmarkCache final {
public:
  BenchmarkCache() {
    const char *configured = std::getenv("FLAGDNN_CACHE_PATH");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
      std::filesystem::create_directories(path_);
      return;
    }
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-iluvatar-benchmark-XXXXXX")
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed for benchmark cache");
    }
    path_ = created;
    owned_ = true;
  }

  ~BenchmarkCache() noexcept {
    if (owned_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
  bool owned_ = false;
};

flagdnn::testing::TestTensor test_tensor(const TensorSpec &tensor) {
  return {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
          tensor.binding_byte_offset};
}

std::size_t logical_offset(std::size_t logical_index,
                           const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    result += coordinate * static_cast<std::size_t>(tensor.strides[current]);
  }
  return result;
}

InputDomain input_domain(const BenchmarkCase &specification,
                         std::size_t input_index) {
  return input_index < specification.input_domains.size()
             ? specification.input_domains[input_index]
             : specification.input_domain;
}

std::vector<float> make_logical_input(const BenchmarkCase &specification,
                                      std::size_t input_index) {
  const auto tensor = test_tensor(specification.tensors.at(input_index));
  const std::size_t count = fr::element_count(tensor);
  std::vector<float> result(count);
  const InputDomain domain = input_domain(specification, input_index);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const float real =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
    switch (domain) {
    case InputDomain::kReal:
      result[index] = real;
      break;
    case InputDomain::kPositive:
      result[index] = std::abs(real) + 0.5F;
      break;
    case InputDomain::kScaled:
      result[index] = real * 4.0F;
      break;
    case InputDomain::kTan:
      result[index] = static_cast<float>(centered) / 40.0F;
      break;
    case InputDomain::kDivisor:
    case InputDomain::kModulo:
      result[index] = input_index == 1 ? std::abs(real) + 0.5F : real;
      break;
    case InputDomain::kPower:
      result[index] = input_index == 0
                          ? std::abs(real) + 0.5F
                          : std::fmod(std::abs(real), 2.0F) + 0.125F;
      break;
    case InputDomain::kModuloSigned: {
      constexpr float left[] = {-3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
      constexpr float right[] = {2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
      result[index] = input_index == 0 ? left[index % 6] : right[index % 6];
      break;
    }
    case InputDomain::kComparison: {
      const float base =
          static_cast<float>(static_cast<int>((index * 17) % 41) - 20) / 13.0F;
      result[index] = input_index == 0 || index % 3 == 0
                          ? base
                          : (index % 3 == 1 ? base + 0.25F : base - 0.25F);
      break;
    }
    case InputDomain::kLogical:
      result[index] = ((index * 17 + input_index * 11) % 3) == 0 ? 0.0F : 1.0F;
      break;
    }
  }
  return result;
}

struct BoundTensor {
  flagdnn::testing::TestTensor specification;
  std::unique_ptr<ivb::CudaDeviceBuffer> buffer;
  std::vector<std::uint8_t> initial_allocation;
  std::size_t storage_bytes = 0;
};

struct PreparedBuffers {
  std::vector<BoundTensor> tensors;
  std::vector<flagdnnBinding_t> bindings;
  std::size_t input_count = 0;
};

BoundTensor make_bound_tensor(const TensorSpec &specification,
                              std::optional<std::span<const float>> logical,
                              ivb::CudaStream &stream) {
  BoundTensor result;
  result.specification = test_tensor(specification);
  std::vector<float> physical;
  if (logical.has_value()) {
    physical = fr::scatter(*logical, result.specification);
  } else {
    physical.assign(fr::storage_element_count(result.specification),
                    fr::kPaddingSentinel);
  }
  const std::vector<std::uint8_t> encoded =
      fr::encode(physical, result.specification.data_type);
  result.storage_bytes = encoded.size();
  const std::size_t allocation_bytes =
      result.specification.binding_byte_offset + result.storage_bytes +
      kGuardBytes;
  result.initial_allocation.assign(allocation_bytes, kGuardValue);
  std::copy(encoded.begin(), encoded.end(),
            result.initial_allocation.begin() +
                static_cast<std::ptrdiff_t>(
                    result.specification.binding_byte_offset));
  result.buffer = std::make_unique<ivb::CudaDeviceBuffer>(allocation_bytes);
  result.buffer->copy_from_host(result.initial_allocation.data(),
                                result.initial_allocation.size(), 0,
                                stream.get());
  return result;
}

PreparedBuffers prepare_buffers(const BenchmarkCase &specification,
                                ivb::CudaStream &stream) {
  PreparedBuffers result;
  result.input_count = input_tensor_count(specification);
  result.tensors.reserve(specification.tensors.size());
  std::vector<std::vector<float>> logical_inputs;
  logical_inputs.reserve(result.input_count);
  for (std::size_t index = 0; index < result.input_count; ++index) {
    logical_inputs.push_back(make_logical_input(specification, index));
    result.tensors.push_back(make_bound_tensor(specification.tensors[index],
                                               logical_inputs.back(), stream));
  }
  for (std::size_t index = result.input_count;
       index < specification.tensors.size(); ++index) {
    result.tensors.push_back(
        make_bound_tensor(specification.tensors[index], std::nullopt, stream));
  }
  result.bindings.reserve(result.tensors.size());
  for (const BoundTensor &tensor : result.tensors) {
    result.bindings.push_back(
        {tensor.specification.uid,
         tensor.buffer->at(tensor.specification.binding_byte_offset)});
  }
  stream.synchronize();
  return result;
}

void reset_outputs(PreparedBuffers &buffers, ivb::CudaStream &stream) {
  for (std::size_t index = buffers.input_count; index < buffers.tensors.size();
       ++index) {
    const BoundTensor &tensor = buffers.tensors[index];
    tensor.buffer->copy_from_host(tensor.initial_allocation.data(),
                                  tensor.initial_allocation.size(), 0,
                                  stream.get());
  }
  stream.synchronize();
}

void require_bytes(std::span<const std::uint8_t> bytes, std::uint8_t expected,
                   std::string_view description) {
  if (!std::all_of(bytes.begin(), bytes.end(), [expected](std::uint8_t value) {
        return value == expected;
      })) {
    throw std::runtime_error(std::string(description) + " guard was modified");
  }
}

std::vector<float> read_output(PreparedBuffers &buffers,
                               std::size_t output_index,
                               ivb::CudaStream &stream,
                               std::string_view provider, bool require_finite) {
  BoundTensor &output = buffers.tensors.at(buffers.input_count + output_index);
  std::vector<std::uint8_t> allocation(output.buffer->size());
  output.buffer->copy_to_host(allocation.data(), allocation.size(), 0,
                              stream.get());
  stream.synchronize();
  const std::size_t entrance = output.specification.binding_byte_offset;
  require_bytes(std::span(allocation).first(entrance), kGuardValue,
                std::string(provider) + " prefix");
  require_bytes(std::span(allocation).subspan(entrance + output.storage_bytes),
                kGuardValue, std::string(provider) + " suffix");

  const std::vector<float> physical =
      fr::decode(std::span(allocation).subspan(entrance, output.storage_bytes),
                 output.specification.data_type,
                 fr::storage_element_count(output.specification));
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < fr::element_count(output.specification);
       ++index) {
    occupied[logical_offset(index, output.specification)] = true;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    if (!occupied[index] && physical[index] != fr::kPaddingSentinel) {
      throw std::runtime_error(std::string(provider) +
                               " modified output padding for uid " +
                               std::to_string(output.specification.uid));
    }
  }
  std::vector<float> logical = fr::gather(physical, output.specification);
  if (require_finite &&
      !std::all_of(logical.begin(), logical.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error(std::string(provider) +
                             " produced a nonfinite output");
  }
  return logical;
}

void compare_outputs(std::span<const float> actual,
                     std::span<const float> reference,
                     const BenchmarkCase &specification,
                     std::size_t output_index, std::string_view actual_name,
                     std::string_view reference_name) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("benchmark reference output size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    if (left == right) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    if (!std::isfinite(absolute) ||
        (absolute > specification.absolute_tolerance &&
         relative > specification.relative_tolerance)) {
      std::ostringstream message;
      message << specification.name << " output=" << output_index
              << " differs at element " << index << ": " << actual_name << '='
              << left << ' ' << reference_name << '=' << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << specification.absolute_tolerance
              << " rtol=" << specification.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

void execute(BenchmarkExecutable &executable,
             std::span<const flagdnnBinding_t> bindings,
             ivb::CudaDeviceBuffer &workspace, ivb::CudaStream &stream) {
  executable.execute(bindings, workspace.at(), executable.workspace_size(),
                     stream.opaque());
}

void warmup(BenchmarkExecutable &executable,
            std::span<const flagdnnBinding_t> bindings,
            ivb::CudaDeviceBuffer &workspace, ivb::CudaStream &stream,
            int iterations) {
  if (iterations < 0) {
    throw std::invalid_argument("benchmark warmup count is invalid");
  }
  for (int iteration = 0; iteration < iterations; ++iteration) {
    execute(executable, bindings, workspace, stream);
  }
  stream.synchronize();
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    return "int32";

  case FLAGDNN_DATA_FP8_E8M0:
    return "fp8_e8m0";
  case FLAGDNN_DATA_FLOAT32:
    return "fp32";
  case FLAGDNN_DATA_FLOAT16:
    return "fp16";
  case FLAGDNN_DATA_BFLOAT16:
    return "bf16";
  case FLAGDNN_DATA_BOOLEAN:
    return "bool";
  case FLAGDNN_DATA_FP8_E4M3:
    return "fp8_e4m3";
  case FLAGDNN_DATA_FP8_E5M2:
    return "fp8_e5m2";
  }
  return "unknown";
}

bool is_contiguous(const TensorSpec &tensor) {
  std::int64_t stride = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != stride) {
      return false;
    }
    stride *= tensor.dimensions[axis - 1];
  }
  return true;
}

std::string shape_name(const TensorSpec &tensor) {
  if (tensor.dimensions.empty()) {
    return "scalar";
  }
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result;
}

void emit_skip(const ivb::CorexCudnnProvider &provider,
               const BenchmarkCase &specification, std::string_view reason) {
  if (reason.empty() || specification.tensors.empty()) {
    throw std::runtime_error("CoreX cuDNN benchmark SKIP is malformed");
  }
  const TensorSpec &representative = specification.tensors.front();
  std::cout << "[SKIP][corex-cudnn]"
            << " op=" << provider.operation_name(specification)
            << " case=" << specification.name << " reason=" << reason
            << " cudnn_header=7605"
            << " cudnn_runtime=7605"
            << " corex=4.4.0"
            << " target=corex_71"
            << " dtype=" << data_type_name(representative.data_type)
            << " layout="
            << (is_contiguous(representative) ? "contiguous" : "strided")
            << " shape=" << shape_name(representative) << std::endl;
}

std::string json_escape(std::string_view input) {
  std::string output;
  output.reserve(input.size());
  for (const char value : input) {
    switch (value) {
    case '\\':
      output += "\\\\";
      break;
    case '"':
      output += "\\\"";
      break;
    case '\n':
      output += "\\n";
      break;
    case '\r':
      output += "\\r";
      break;
    case '\t':
      output += "\\t";
      break;
    default:
      output.push_back(value);
      break;
    }
  }
  return output;
}

void emit_samples(std::string_view provider, const BenchmarkCase &specification,
                  std::span<const double> samples) {
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << json_escape(specification.name)
            << "\",\"unit\":\"us\",\"median\":" << ivb::percentile(samples, 0.5)
            << ",\"p90\":" << ivb::percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}" << std::endl;
}

CaseResult run_case(const BenchmarkCase &specification,
                    ivb::CorexCudnnProvider &corex_cudnn_provider,
                    FlagdnnProvider &flagdnn_provider,
                    ivb::CudaStream &flagdnn_stream,
                    ivb::CudaStream &corex_cudnn_stream) {
  const BenchmarkConfig &config = specification.benchmark;
  if (config.warmup_iterations < 0 || config.sample_count <= 0 ||
      config.iterations_per_sample <= 0) {
    throw std::invalid_argument("benchmark sample configuration is invalid");
  }

  // The functional suite validates production execution independently.
  // A known missing reference cannot produce a paired benchmark result;
  // account for that case before allocating buffers or invoking the JIT.
  // Shared cases use exact catalog evidence, including qualified BOOL cases.
  const ProviderCapability capability =
      corex_cudnn_provider.capability(specification);
  if (!capability.supported) {
    emit_skip(corex_cudnn_provider, specification, capability.reason);
    return CaseResult::kReferenceSkipped;
  }

  std::unique_ptr<BenchmarkExecutable> flagdnn =
      flagdnn_provider.build(specification);
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(specification, flagdnn_stream);
  ivb::CudaDeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace,
          flagdnn_stream);
  flagdnn_stream.synchronize();

  std::vector<std::vector<float>> production_outputs;
  production_outputs.reserve(specification.output_count);
  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    production_outputs.push_back(read_output(flagdnn_buffers, output_index,
                                             flagdnn_stream, "FlagDNN", true));
  }

  std::unique_ptr<BenchmarkExecutable> reference;
  try {
    reference = corex_cudnn_provider.build(specification);
  } catch (const BenchmarkUnsupportedError &error) {
    emit_skip(corex_cudnn_provider, specification, error.what());
    return CaseResult::kReferenceSkipped;
  } catch (const iv::CorexCudnnStatusError &error) {
    if (!iv::cudnn_status_is_runtime_capability(error.status())) {
      throw;
    }
    emit_skip(corex_cudnn_provider, specification,
              "CUDNN_STATUS_NOT_SUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  PreparedBuffers reference_buffers =
      prepare_buffers(specification, corex_cudnn_stream);
  ivb::CudaDeviceBuffer reference_workspace(reference->workspace_size());
  try {
    execute(*reference, reference_buffers.bindings, reference_workspace,
            corex_cudnn_stream);
    corex_cudnn_stream.synchronize();
  } catch (const iv::CorexCudnnStatusError &error) {
    if (!iv::cudnn_status_is_runtime_capability(error.status())) {
      throw;
    }
    emit_skip(corex_cudnn_provider, specification,
              "CUDNN_STATUS_NOT_SUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  std::vector<std::vector<float>> reference_outputs;
  reference_outputs.reserve(specification.output_count);
  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    reference_outputs.push_back(read_output(reference_buffers, output_index,
                                            corex_cudnn_stream, "CoreX cuDNN",
                                            true));
    compare_outputs(production_outputs[output_index], reference_outputs.back(),
                    specification, output_index, "FlagDNN", "CoreX-cuDNN");
  }
  std::cout << specification.name
            << ": FlagDNN Graph vs CoreX cuDNN correctness PASS" << std::endl;

  reset_outputs(flagdnn_buffers, flagdnn_stream);
  reset_outputs(reference_buffers, corex_cudnn_stream);
  warmup(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, flagdnn_stream,
         config.warmup_iterations);
  try {
    warmup(*reference, reference_buffers.bindings, reference_workspace,
           corex_cudnn_stream, config.warmup_iterations);
  } catch (const iv::CorexCudnnStatusError &error) {
    if (!iv::cudnn_status_is_runtime_capability(error.status())) {
      throw;
    }
    emit_skip(corex_cudnn_provider, specification,
              "CUDNN_STATUS_NOT_SUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  std::unique_ptr<ivb::CapturedExecutionBatch> flagdnn_batch;
  try {
    flagdnn_batch = std::make_unique<ivb::CapturedExecutionBatch>(
        flagdnn_stream.get(), config.iterations_per_sample, [&] {
          execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace,
                  flagdnn_stream);
        });
  } catch (const ivb::StreamCaptureUnsupported &) {
    emit_skip(corex_cudnn_provider, specification, "CAPTURE_UNSUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  std::unique_ptr<ivb::CapturedExecutionBatch> reference_batch;
  try {
    reference_batch = std::make_unique<ivb::CapturedExecutionBatch>(
        corex_cudnn_stream.get(), config.iterations_per_sample, [&] {
          execute(*reference, reference_buffers.bindings, reference_workspace,
                  corex_cudnn_stream);
        });
  } catch (const iv::CorexCudnnStatusError &error) {
    if (!iv::cudnn_status_is_runtime_capability(error.status())) {
      throw;
    }
    emit_skip(corex_cudnn_provider, specification,
              "CUDNN_STATUS_NOT_SUPPORTED");
    return CaseResult::kReferenceSkipped;
  } catch (const ivb::StreamCaptureUnsupported &) {
    emit_skip(corex_cudnn_provider, specification, "CAPTURE_UNSUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  if (ivb::validate_capture_pair(true, ivb::CaptureCapability::kSupported,
                                 ivb::CaptureCapability::kSupported,
                                 flagdnn_batch->execution_count(),
                                 reference_batch->execution_count()) !=
      ivb::PairDisposition::kComparable) {
    throw std::logic_error("complete CUDA Graph pair was not comparable");
  }

  try {
    flagdnn_batch->launch(flagdnn_stream.get());
    flagdnn_stream.synchronize();
    reference_batch->launch(corex_cudnn_stream.get());
    corex_cudnn_stream.synchronize();
  } catch (const ivb::StreamCaptureUnsupported &) {
    emit_skip(corex_cudnn_provider, specification, "CAPTURE_UNSUPPORTED");
    return CaseResult::kReferenceSkipped;
  }

  ivb::CudaEventTimer flagdnn_timer;
  ivb::CudaEventTimer reference_timer;
  std::vector<double> flagdnn_samples;
  std::vector<double> reference_samples;
  flagdnn_samples.reserve(static_cast<std::size_t>(config.sample_count));
  reference_samples.reserve(static_cast<std::size_t>(config.sample_count));
  for (int sample = 0; sample < config.sample_count; ++sample) {
    const auto measure_flagdnn = [&] {
      flagdnn_samples.push_back(
          flagdnn_timer.measure_microseconds_per_execution(flagdnn_stream.get(),
                                                           *flagdnn_batch));
    };
    const auto measure_reference = [&] {
      reference_samples.push_back(
          reference_timer.measure_microseconds_per_execution(
              corex_cudnn_stream.get(), *reference_batch));
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_reference();
    } else {
      measure_reference();
      measure_flagdnn();
    }
  }

  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    const std::vector<float> post_flagdnn =
        read_output(flagdnn_buffers, output_index, flagdnn_stream,
                    "FlagDNN postcheck", true);
    const std::vector<float> post_reference =
        read_output(reference_buffers, output_index, corex_cudnn_stream,
                    "CoreX cuDNN postcheck", true);
    compare_outputs(post_flagdnn, reference_outputs[output_index],
                    specification, output_index, "FlagDNN postcheck",
                    "CoreX-cuDNN direct oracle");
    compare_outputs(post_reference, reference_outputs[output_index],
                    specification, output_index, "CoreX-cuDNN postcheck",
                    "CoreX-cuDNN direct oracle");
  }

  ivb::PairSampleCollector pair;
  pair.add(ivb::BenchmarkProviderKind::kFlagdnn, std::move(flagdnn_samples));
  pair.add(ivb::BenchmarkProviderKind::kCorexCudnn,
           std::move(reference_samples));
  pair.validate(static_cast<std::size_t>(config.sample_count));
  emit_samples("flagdnn", specification, pair.flagdnn());
  emit_samples("corex_cudnn", specification, pair.corex_cudnn());
  std::cout << specification.name
            << ": median_us flagdnn=" << ivb::percentile(pair.flagdnn(), 0.5)
            << " corex_cudnn=" << ivb::percentile(pair.corex_cudnn(), 0.5)
            << " speedup="
            << ivb::percentile(pair.corex_cudnn(), 0.5) /
                   ivb::percentile(pair.flagdnn(), 0.5)
            << std::endl;
  return CaseResult::kComparable;
}

} // namespace

int run_benchmark_suite(int argc, char **argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << suite_name
              << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    ivb::check_cuda_runtime(cudaSetDevice(0), "cudaSetDevice");
    ivb::check_cuda_runtime(cudaFree(nullptr), "cudaFree(context init)");
    BenchmarkCache cache;
    flagdnn::Handle handle("iluvatar", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    FlagdnnProvider flagdnn_provider(handle);
    flagdnn_provider.set_autotune(true);
    ivb::CorexCudnnProvider corex_cudnn_provider;
    ivb::CudaStream flagdnn_stream;
    ivb::CudaStream corex_cudnn_stream;

    const char *case_filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t matched = 0;
    std::size_t comparable_executed = 0;
    std::size_t reference_skipped = 0;
    std::vector<BenchmarkCase> workloads(cases.begin(), cases.end());
    auto supplemental = ivb::shared_benchmark_cases(suite_name);
    workloads.insert(workloads.end(), supplemental.begin(), supplemental.end());
    for (const BenchmarkCase &specification : workloads) {
      if (case_filter != nullptr && case_filter[0] != '\0' &&
          specification.name != case_filter) {
        continue;
      }
      ++matched;
      const CaseResult result =
          run_case(specification, corex_cudnn_provider, flagdnn_provider,
                   flagdnn_stream, corex_cudnn_stream);
      if (result == CaseResult::kComparable) {
        ++comparable_executed;
      } else {
        ++reference_skipped;
      }
    }
    if (matched == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    if (matched != comparable_executed + reference_skipped) {
      throw std::runtime_error("benchmark suite accounting invariant failed");
    }
    if (suite_name == "FLAGDNN_MATMUL_BENCHMARK" &&
        (case_filter == nullptr || case_filter[0] == '\0'))
      iv::emit_plain_fp8_matmul_skips();
    const bool all_skipped = comparable_executed == 0;
    std::cout << suite_name << ": " << (all_skipped ? "SKIP" : "PASS")
              << " cases=" << matched
              << " comparable_executed=" << comparable_executed
              << " reference_skipped=" << reference_skipped << std::endl;
    return all_skipped ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

} // namespace flagdnn::benchmarking
