#include "common/runner.hpp"

#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <flagdnn/flagdnn.hpp>
#include <iomanip>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/flagdnn_provider.hpp"
#include "validation/benchmark/cases.hpp"
#include "validation/benchmark/cudnn_common.hpp"
#include "validation/benchmark/cudnn_provider.hpp"
#include "validation/benchmark/timing.hpp"
#include "validation/cuda_driver.hpp"
#include "validation/tensor_io.hpp"

namespace {

namespace tensor_io = flagdnn::validation::nvidia::tensor_io;
using flagdnn::validation::nvidia::timing::CapturedExecutionBatch;
using flagdnn::validation::nvidia::timing::emit_samples;
using flagdnn::validation::nvidia::timing::host_microseconds_since;
using flagdnn::validation::nvidia::timing::HostClock;
using flagdnn::validation::nvidia::timing::percentile;
using flagdnn::validation::nvidia::timing::RuntimeMeasurements;
constexpr float kPaddingSentinel = tensor_io::kPaddingSentinel;

using flagdnn::benchmarking::BenchmarkCase;
using flagdnn::benchmarking::BenchmarkExecutable;
using flagdnn::benchmarking::BenchmarkUnsupportedError;
using flagdnn::benchmarking::DeviceBuffer;
using flagdnn::benchmarking::EventTimer;
using flagdnn::benchmarking::InputDomain;
using flagdnn::validation::nvidia::check_cuda;

class BenchmarkCache {
 public:
  BenchmarkCache() {
    const char* configured = std::getenv("FLAGDNN_BENCHMARK_CACHE_DIRECTORY");
    if (configured != nullptr && configured[0] != '\0') {
      path_ = configured;
    } else {
      path_ = std::filesystem::temp_directory_path() /
              ("flagdnn-benchmark-cache-" + std::to_string(getuid()));
    }
    std::error_code error;
    std::filesystem::create_directories(path_, error);
    if (error) {
      throw std::runtime_error("cannot create benchmark cache directory: " +
                               error.message());
    }
    const char* fresh = std::getenv("FLAGDNN_BENCHMARK_FRESH_CACHE");
    if (fresh != nullptr && std::string_view(fresh) == "1") {
      const std::string pattern = (path_ / "run-XXXXXX").string();
      std::vector<char> name(pattern.begin(), pattern.end());
      name.push_back('\0');
      const char* created = mkdtemp(name.data());
      if (created == nullptr) {
        throw std::runtime_error(
            "cannot create fresh benchmark artifact cache");
      }
      path_ = created;
      owned_ = true;
    }
  }

  ~BenchmarkCache() {
    if (owned_) {
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }

  BenchmarkCache(const BenchmarkCache&) = delete;
  BenchmarkCache& operator=(const BenchmarkCache&) = delete;

  [[nodiscard]] bool fresh() const noexcept { return owned_; }

  [[nodiscard]] const std::filesystem::path& path() const noexcept {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool owned_ = false;
};

std::size_t element_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::element_count(specification);
}

std::size_t storage_element_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::storage_element_count(specification);
}

flagdnn::benchmarking::TensorSpec cudnn_tensor_specification(
    const BenchmarkCase& specification, std::size_t tensor_index) {
  const auto& tensor = specification.tensors.at(tensor_index);
  if ((specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnorm ||
       specification.operation ==
           flagdnn::benchmarking::Operation::kBatchnormInference) &&
      (tensor_index == 0 ||
       tensor_index == input_tensor_count(specification))) {
    return flagdnn::benchmarking::cudnn_detail::batchnorm_inference_nhwc_tensor(
        tensor);
  }
  return tensor;
}

std::vector<float> scatter_logical_values(
    std::span<const float> logical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::scatter(logical, specification);
}

std::vector<float> gather_logical_values(
    std::span<const float> physical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::gather(physical, specification);
}

void require_output_padding_unchanged(
    std::string_view provider, std::span<const float> physical,
    const flagdnn::benchmarking::TensorSpec& specification) {
  tensor_io::require_padding_unchanged(provider, physical, specification);
}

std::size_t data_type_size(flagdnnDataType_t data_type) {
  return tensor_io::data_type_size(data_type);
}

std::vector<std::uint8_t> encode_values(std::span<const float> values,
                                        flagdnnDataType_t data_type) {
  return tensor_io::encode(values, data_type,
                           tensor_io::BooleanEncoding::kByte);
}

std::vector<float> decode_values(std::span<const std::uint8_t> bytes,
                                 flagdnnDataType_t data_type) {
  const std::size_t count = bytes.size() / data_type_size(data_type);
  return tensor_io::decode(bytes, data_type, count,
                           tensor_io::BooleanEncoding::kByte);
}

std::size_t cudnn_encoded_byte_count(
    const flagdnn::benchmarking::TensorSpec& specification) {
  return tensor_io::encoded_byte_count(specification,
                                       tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<std::uint8_t> encode_cudnn_values(std::span<const float> physical,
                                              flagdnnDataType_t data_type) {
  return tensor_io::encode(physical, data_type,
                           tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<float> decode_cudnn_values(std::span<const std::uint8_t> bytes,
                                       flagdnnDataType_t data_type,
                                       std::size_t storage_count) {
  return tensor_io::decode(bytes, data_type, storage_count,
                           tensor_io::BooleanEncoding::kBitPacked);
}

std::vector<float> make_input(std::size_t count, std::size_t tensor_index,
                              InputDomain domain) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 11) % 41) - 20;
    const float real_value =
        static_cast<float>(centered) / static_cast<float>(13 + tensor_index);
    switch (domain) {
      case InputDomain::kReal:
        result[index] = real_value;
        break;
      case InputDomain::kPositive:
        result[index] = std::abs(real_value) + 0.5F;
        break;
      case InputDomain::kScaled:
        result[index] = real_value * 4.0F;
        break;
      case InputDomain::kTan:
        result[index] = static_cast<float>(centered) / 40.0F;
        break;
      case InputDomain::kDivisor:
      case InputDomain::kModulo:
        result[index] =
            tensor_index == 1 ? std::abs(real_value) + 0.5F : real_value;
        break;
      case InputDomain::kPower:
        result[index] = tensor_index == 0
                            ? std::abs(real_value) + 0.5F
                            : std::fmod(std::abs(real_value), 2.0F) + 0.125F;
        break;
      case InputDomain::kModuloSigned: {
        constexpr std::array<float, 6> kLeft = {-3.0F, -3.0F, 3.0F,
                                                3.0F,  -5.5F, 5.5F};
        constexpr std::array<float, 6> kRight = {2.0F,  -2.0F, 2.0F,
                                                 -2.0F, 2.25F, -2.25F};
        result[index] = tensor_index == 0 ? kLeft[index % kLeft.size()]
                                          : kRight[index % kRight.size()];
        break;
      }
      case InputDomain::kComparison: {
        const int base_centered = static_cast<int>((index * 17) % 41) - 20;
        const float base = static_cast<float>(base_centered) / 13.0F;
        if (tensor_index == 0 || index % 3 == 0) {
          result[index] = base;
        } else if (index % 3 == 1) {
          result[index] = base + 0.25F;
        } else {
          result[index] = base - 0.25F;
        }
        break;
      }
      case InputDomain::kLogical:
        result[index] =
            ((index * 17 + tensor_index * 11) % 3) != 0 ? 1.0F : 0.0F;
        break;
    }
  }
  return result;
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare_outputs(std::span<const float> actual,
                         std::span<const float> reference,
                         double absolute_tolerance, double relative_tolerance) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("output sizes do not match");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (!std::isfinite(absolute) ||
        (absolute > absolute_tolerance && relative > relative_tolerance)) {
      std::ostringstream message;
      message << "FlagDNN output differs from reference at element " << index
              << ": actual=" << left << ", reference=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << absolute_tolerance
              << ", rtol=" << relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

std::unique_ptr<BenchmarkExecutable> build_profiled(
    flagdnn::benchmarking::BenchmarkProvider& provider,
    const BenchmarkCase& specification, RuntimeMeasurements& measurements) {
  auto begin = HostClock::now();
  auto executable = provider.build(specification);
  measurements.build_us = host_microseconds_since(begin);
  measurements.workspace_bytes = executable->workspace_size();
  begin = HostClock::now();
  auto warm_executable = provider.build(specification);
  measurements.warm_build_us = host_microseconds_since(begin);
  return executable;
}

struct CaseBuffers {
  std::vector<std::unique_ptr<DeviceBuffer>> inputs;
  std::vector<std::vector<float>> host_inputs;
  std::vector<std::unique_ptr<DeviceBuffer>> flagdnn_outputs;
  std::vector<std::unique_ptr<DeviceBuffer>> cudnn_outputs;
  std::vector<flagdnnBinding_t> flagdnn_bindings;
  std::vector<flagdnnBinding_t> cudnn_bindings;
};

CaseBuffers make_buffers(const BenchmarkCase& specification,
                         flagdnn::benchmarking::Stream& stream) {
  if (specification.tensors.size() < 2) {
    throw std::invalid_argument("case has too few tensors");
  }
  CaseBuffers result;
  const std::size_t input_count = input_tensor_count(specification);
  result.inputs.reserve(input_count * 2);
  result.host_inputs.reserve(input_count);
  result.flagdnn_bindings.reserve(specification.tensors.size());
  result.cudnn_bindings.reserve(specification.tensors.size());
  for (std::size_t index = 0; index < input_count; ++index) {
    const auto& tensor = specification.tensors[index];
    const std::size_t count = element_count(tensor);
    const InputDomain domain = specification.input_domains.empty()
                                   ? specification.input_domain
                                   : specification.input_domains.at(index);
    std::vector<float> logical = make_input(count, index, domain);
    std::vector<float> physical = scatter_logical_values(logical, tensor);
    std::vector<std::uint8_t> encoded =
        encode_values(physical, tensor.data_type);
    result.host_inputs.push_back(gather_logical_values(
        decode_values(encoded, tensor.data_type), tensor));
    auto flagdnn_buffer = std::make_unique<DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    flagdnn_buffer->copy_from_host_at(encoded.data(), encoded.size(),
                                      tensor.binding_byte_offset, stream.get());
    void* flagdnn_binding =
        flagdnn_buffer->opaque_at(tensor.binding_byte_offset);
    result.flagdnn_bindings.push_back({tensor.uid, flagdnn_binding});
    const auto cudnn_tensor = cudnn_tensor_specification(specification, index);
    const bool separate_cudnn_buffer =
        tensor.data_type == FLAGDNN_DATA_BOOLEAN ||
        cudnn_tensor.dimensions != tensor.dimensions ||
        cudnn_tensor.strides != tensor.strides;
    if (separate_cudnn_buffer) {
      const std::vector<float> cudnn_physical =
          scatter_logical_values(logical, cudnn_tensor);
      const std::vector<std::uint8_t> cudnn_encoded =
          encode_cudnn_values(cudnn_physical, tensor.data_type);
      auto cudnn_buffer = std::make_unique<DeviceBuffer>(
          tensor.binding_byte_offset + cudnn_encoded.size());
      cudnn_buffer->copy_from_host_at(cudnn_encoded.data(),
                                      cudnn_encoded.size(),
                                      tensor.binding_byte_offset, stream.get());
      result.cudnn_bindings.push_back(
          {tensor.uid, cudnn_buffer->opaque_at(tensor.binding_byte_offset)});
      result.inputs.push_back(std::move(cudnn_buffer));
    } else {
      result.cudnn_bindings.push_back({tensor.uid, flagdnn_binding});
    }
    result.inputs.push_back(std::move(flagdnn_buffer));
  }

  result.flagdnn_outputs.reserve(specification.output_count);
  result.cudnn_outputs.reserve(specification.output_count);
  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    const std::size_t tensor_index = input_count + output_index;
    const auto& output = specification.tensors[tensor_index];
    const auto cudnn_output_specification =
        cudnn_tensor_specification(specification, tensor_index);
    const std::size_t output_bytes =
        storage_element_count(output) * data_type_size(output.data_type);
    auto flagdnn_output = std::make_unique<DeviceBuffer>(
        output.binding_byte_offset + output_bytes);
    const std::size_t cudnn_output_bytes =
        cudnn_encoded_byte_count(cudnn_output_specification);
    auto cudnn_output = std::make_unique<DeviceBuffer>(
        output.binding_byte_offset + cudnn_output_bytes);
    const std::vector<float> initial_output(storage_element_count(output),
                                            kPaddingSentinel);
    const std::vector<float> initial_cudnn_output(
        storage_element_count(cudnn_output_specification), kPaddingSentinel);
    const std::vector<std::uint8_t> encoded_output =
        encode_values(initial_output, output.data_type);
    const std::vector<std::uint8_t> cudnn_encoded_output =
        encode_cudnn_values(initial_cudnn_output, output.data_type);
    flagdnn_output->copy_from_host_at(encoded_output.data(),
                                      encoded_output.size(),
                                      output.binding_byte_offset, stream.get());
    cudnn_output->copy_from_host_at(cudnn_encoded_output.data(),
                                    cudnn_encoded_output.size(),
                                    output.binding_byte_offset, stream.get());
    result.flagdnn_bindings.push_back(
        {output.uid, flagdnn_output->opaque_at(output.binding_byte_offset)});
    result.cudnn_bindings.push_back(
        {output.uid, cudnn_output->opaque_at(output.binding_byte_offset)});
    result.flagdnn_outputs.push_back(std::move(flagdnn_output));
    result.cudnn_outputs.push_back(std::move(cudnn_output));
  }
  return result;
}

std::unique_ptr<DeviceBuffer> make_workspace(
    const BenchmarkExecutable& executable) {
  return std::make_unique<DeviceBuffer>(executable.workspace_size());
}

void execute(BenchmarkExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             DeviceBuffer& workspace, flagdnn::benchmarking::Stream& stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

Accuracy compare_provider_outputs(const BenchmarkCase& specification,
                                  flagdnn::benchmarking::Stream& stream,
                                  CaseBuffers& buffers) {
  const std::size_t input_count = input_tensor_count(specification);
  Accuracy aggregate;
  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    const std::size_t tensor_index = input_count + output_index;
    const auto& flagdnn_output = specification.tensors[tensor_index];
    const auto cudnn_output =
        cudnn_tensor_specification(specification, tensor_index);

    std::vector<std::uint8_t> flagdnn_encoded(
        storage_element_count(flagdnn_output) *
        data_type_size(flagdnn_output.data_type));
    std::vector<std::uint8_t> cudnn_encoded(
        cudnn_encoded_byte_count(cudnn_output));
    buffers.flagdnn_outputs[output_index]->copy_to_host_at(
        flagdnn_encoded.data(), flagdnn_encoded.size(),
        flagdnn_output.binding_byte_offset, stream.get());
    buffers.cudnn_outputs[output_index]->copy_to_host_at(
        cudnn_encoded.data(), cudnn_encoded.size(),
        cudnn_output.binding_byte_offset, stream.get());
    stream.synchronize();

    const std::vector<float> flagdnn_physical =
        decode_values(flagdnn_encoded, flagdnn_output.data_type);
    const std::vector<float> cudnn_physical =
        decode_cudnn_values(cudnn_encoded, cudnn_output.data_type,
                            storage_element_count(cudnn_output));
    require_output_padding_unchanged("FlagDNN", flagdnn_physical,
                                     flagdnn_output);
    if (cudnn_output.data_type != FLAGDNN_DATA_BOOLEAN) {
      require_output_padding_unchanged("cuDNN", cudnn_physical, cudnn_output);
    }
    const std::vector<float> flagdnn_logical =
        gather_logical_values(flagdnn_physical, flagdnn_output);
    const std::vector<float> cudnn_logical =
        gather_logical_values(cudnn_physical, cudnn_output);
    Accuracy accuracy;
    try {
      accuracy = compare_outputs(flagdnn_logical, cudnn_logical,
                                 specification.absolute_tolerance,
                                 specification.relative_tolerance);
    } catch (const std::runtime_error& error) {
      throw std::runtime_error(specification.name +
                               " output_index=" + std::to_string(output_index) +
                               ": " + error.what());
    }
    aggregate.maximum_absolute =
        std::max(aggregate.maximum_absolute, accuracy.maximum_absolute);
    aggregate.maximum_relative =
        std::max(aggregate.maximum_relative, accuracy.maximum_relative);
  }
  std::cout << specification.name
            << ": FlagDNN vs cuDNN PASS max_abs=" << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;
  return aggregate;
}

void warmup(BenchmarkExecutable& executable,
            std::span<const flagdnnBinding_t> bindings, DeviceBuffer& workspace,
            flagdnn::benchmarking::Stream& stream, int iterations) {
  for (int index = 0; index < iterations; ++index) {
    execute(executable, bindings, workspace, stream);
  }
  stream.synchronize();
}

std::vector<double> measure_host_submission(
    BenchmarkExecutable& executable, std::span<const flagdnnBinding_t> bindings,
    DeviceBuffer& workspace, flagdnn::benchmarking::Stream& stream,
    const flagdnn::benchmarking::BenchmarkConfig& benchmark) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(benchmark.sample_count));
  for (int sample = 0; sample < benchmark.sample_count; ++sample) {
    stream.synchronize();
    const auto begin = HostClock::now();
    for (int i = 0; i < benchmark.iterations_per_sample; ++i) {
      execute(executable, bindings, workspace, stream);
    }
    samples.push_back(host_microseconds_since(begin) /
                      static_cast<double>(benchmark.iterations_per_sample));
    // Completion waits are deliberately outside the host submission interval.
    stream.synchronize();
  }
  return samples;
}

void run_case(const BenchmarkCase& specification,
              flagdnn::benchmarking::FlagdnnProvider& flagdnn_provider,
              flagdnn::benchmarking::CudnnProvider& cudnn_provider,
              flagdnn::benchmarking::Stream& stream,
              bool fresh_artifact_cache) {
  RuntimeMeasurements flagdnn_runtime;
  flagdnn_runtime.build_cache =
      fresh_artifact_cache ? "fresh_artifact_cache" : "reuse_allowed";
  std::unique_ptr<BenchmarkExecutable> flagdnn_executable =
      build_profiled(flagdnn_provider, specification, flagdnn_runtime);
  CaseBuffers buffers = make_buffers(specification, stream);
  stream.synchronize();

  const auto capability = cudnn_provider.capability(specification);
  if (!capability.supported) {
    throw std::runtime_error(
        specification.name +
        ": NVIDIA benchmarks require a cuDNN reference: " + capability.reason);
  }

  std::unique_ptr<BenchmarkExecutable> cudnn_executable;
  RuntimeMeasurements cudnn_runtime;
  try {
    cudnn_executable =
        build_profiled(cudnn_provider, specification, cudnn_runtime);
  } catch (const BenchmarkUnsupportedError& error) {
    throw std::runtime_error(
        specification.name +
        ": NVIDIA benchmarks require a cuDNN reference: " + error.what());
  }

  auto flagdnn_workspace = make_workspace(*flagdnn_executable);
  auto cudnn_workspace = make_workspace(*cudnn_executable);
  execute(*flagdnn_executable, buffers.flagdnn_bindings, *flagdnn_workspace,
          stream);
  execute(*cudnn_executable, buffers.cudnn_bindings, *cudnn_workspace, stream);
  stream.synchronize();
  compare_provider_outputs(specification, stream, buffers);

  warmup(*flagdnn_executable, buffers.flagdnn_bindings, *flagdnn_workspace,
         stream, specification.benchmark.warmup_iterations);
  warmup(*cudnn_executable, buffers.cudnn_bindings, *cudnn_workspace, stream,
         specification.benchmark.warmup_iterations);

  CapturedExecutionBatch flagdnn_batch(
      stream.get(), specification.benchmark.iterations_per_sample, [&] {
        execute(*flagdnn_executable, buffers.flagdnn_bindings,
                *flagdnn_workspace, stream);
      });
  CapturedExecutionBatch cudnn_batch(
      stream.get(), specification.benchmark.iterations_per_sample, [&] {
        execute(*cudnn_executable, buffers.cudnn_bindings, *cudnn_workspace,
                stream);
      });
  flagdnn_batch.launch(stream.get());
  cudnn_batch.launch(stream.get());
  stream.synchronize();

  EventTimer flagdnn_timer;
  EventTimer cudnn_timer;
  std::vector<double> flagdnn_samples;
  std::vector<double> cudnn_samples;
  flagdnn_samples.reserve(
      static_cast<std::size_t>(specification.benchmark.sample_count));
  cudnn_samples.reserve(
      static_cast<std::size_t>(specification.benchmark.sample_count));
  for (int sample = 0; sample < specification.benchmark.sample_count;
       ++sample) {
    const auto measure_flagdnn = [&] {
      const double batch_microseconds = flagdnn_timer.measure_microseconds(
          stream.get(), 1, [&] { flagdnn_batch.launch(stream.get()); });
      flagdnn_samples.push_back(
          batch_microseconds /
          static_cast<double>(flagdnn_batch.execution_count()));
    };
    const auto measure_cudnn = [&] {
      const double batch_microseconds = cudnn_timer.measure_microseconds(
          stream.get(), 1, [&] { cudnn_batch.launch(stream.get()); });
      cudnn_samples.push_back(
          batch_microseconds /
          static_cast<double>(cudnn_batch.execution_count()));
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_cudnn();
    } else {
      measure_cudnn();
      measure_flagdnn();
    }
  }
  flagdnn_runtime.host_submit_us = measure_host_submission(
      *flagdnn_executable, buffers.flagdnn_bindings, *flagdnn_workspace, stream,
      specification.benchmark);
  cudnn_runtime.host_submit_us = measure_host_submission(
      *cudnn_executable, buffers.cudnn_bindings, *cudnn_workspace, stream,
      specification.benchmark);
  emit_samples("flagdnn", specification.name, flagdnn_samples, flagdnn_runtime);
  emit_samples("cudnn", specification.name, cudnn_samples, cudnn_runtime);
  const double flagdnn_median = percentile(flagdnn_samples, 0.5);
  const double cudnn_median = percentile(cudnn_samples, 0.5);
  std::cout << specification.name << ": median_us flagdnn=" << flagdnn_median
            << " cudnn=" << cudnn_median
            << " speedup=" << cudnn_median / flagdnn_median << std::endl;
}

}  // namespace

namespace flagdnn::benchmarking {

int run_benchmark_suite(int argc, char** argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name) {
  if (argc != 3) {
    std::cerr << "usage: " << suite_name
              << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    DriverContext driver;
    Stream stream;
    BenchmarkCache cache;
    flagdnn::Handle handle(FLAGDNN_BACKEND_NVIDIA, 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    FlagdnnProvider flagdnn_provider(handle);
    CudnnProvider cudnn_provider;
    const auto selected_cases = cudnn_benchmark_cases(cases);
    const char* case_filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t executed_cases = 0;
    for (const BenchmarkCase& specification : selected_cases) {
      if (case_filter != nullptr && case_filter[0] != '\0' &&
          specification.name != case_filter) {
        continue;
      }
      flagdnn_provider.set_autotune(true);
      run_case(specification, flagdnn_provider, cudnn_provider, stream,
               cache.fresh());
      ++executed_cases;
    }
    if (executed_cases == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    std::cout << suite_name << ": PASS cases=" << executed_cases << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

}  // namespace flagdnn::benchmarking
