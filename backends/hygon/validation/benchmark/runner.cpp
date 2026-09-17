/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/runner.hpp"
#include "cases.hpp"
#include "native.hpp"

#include "benchmark_phase.hpp"
#include "common/flagdnn_provider.hpp"
#include "hip_driver.hpp"
#include "hip_graph.hpp"
#include "hipdnn_provider.hpp"
#include "tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking {
namespace {

namespace hv = validation::hygon;
namespace io = validation::hygon::tensor_io;

constexpr int kSkipReturnCode = 77;

template <typename Function>
decltype(auto) run_benchmark_phase(BenchmarkProviderKind provider,
                                   BenchmarkPhase phase, Function &&function) {
  const std::string context = benchmark_phase_context(provider, phase);
  try {
    if constexpr (std::is_void_v<std::invoke_result_t<Function>>) {
      std::invoke(std::forward<Function>(function));
      return;
    } else {
      return std::invoke(std::forward<Function>(function));
    }
  } catch (const std::exception &error) {
    throw std::runtime_error(context + ": " + error.what());
  } catch (...) {
    throw std::runtime_error(context + ": unknown failure");
  }
}

void require_no_pending_hip_error(BenchmarkProviderKind provider,
                                  BenchmarkPhase phase) {
  const std::string operation =
      benchmark_phase_context(provider, phase) + " hipGetLastError";
  hv::check_hip(hipGetLastError(), operation.c_str());
}

void synchronize_phase(hv::Stream &stream, BenchmarkProviderKind provider,
                       BenchmarkPhase phase) {
  const std::string context = benchmark_phase_context(provider, phase);
  const hipError_t synchronize_status = hipStreamSynchronize(stream.get());
  const hipError_t pending_status = hipGetLastError();

  // Query both statuses at every boundary, then fail immediately on either.
  // hipGetLastError is never used to clear an error and continue execution.
  const std::string synchronize_operation = context + " hipStreamSynchronize";
  hv::check_hip(synchronize_status, synchronize_operation.c_str());
  const std::string pending_operation = context + " hipGetLastError";
  hv::check_hip(pending_status, pending_operation.c_str());
}

class BenchmarkCache final {
public:
  BenchmarkCache() {
    const char *configured = std::getenv("FLAGDNN_BENCHMARK_CACHE_DIRECTORY");
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
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

std::size_t logical_element_count(const TensorSpec &tensor) {
  return io::element_count(tensor);
}

std::size_t storage_element_count(const TensorSpec &tensor) {
  return tensor.dimensions.empty() ? 1 : io::storage_element_count(tensor);
}

std::vector<float> scatter(std::span<const float> logical,
                           const TensorSpec &tensor) {
  if (tensor.dimensions.empty()) {
    if (logical.size() != 1) {
      throw std::invalid_argument("scalar logical tensor must have one value");
    }
    return {logical.front()};
  }
  return io::scatter(logical, tensor);
}

std::vector<float> gather(std::span<const float> physical,
                          const TensorSpec &tensor) {
  if (tensor.dimensions.empty()) {
    if (physical.empty()) {
      throw std::invalid_argument("scalar physical tensor is empty");
    }
    return {physical.front()};
  }
  return io::gather(physical, tensor);
}

void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const TensorSpec &tensor) {
  if (!tensor.dimensions.empty()) {
    io::require_padding_unchanged(provider, physical, tensor);
  }
}

std::vector<InputDomain> input_domains(const BenchmarkCase &specification) {
  const std::size_t count = input_tensor_count(specification);
  if (!specification.input_domains.empty()) {
    if (specification.input_domains.size() != count) {
      throw std::invalid_argument("benchmark input domain count is invalid");
    }
    return specification.input_domains;
  }
  return std::vector<InputDomain>(count, specification.input_domain);
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
      result[index] = ((index * 17 + tensor_index * 11) % 3) != 0 ? 1.0F : 0.0F;
      break;
    }
  }
  return result;
}

std::vector<std::vector<float>>
make_logical_inputs(const BenchmarkCase &specification) {
  const std::size_t count = input_tensor_count(specification);
  const std::vector<InputDomain> domains = input_domains(specification);
  std::vector<std::vector<float>> result;
  result.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const TensorSpec &tensor = specification.tensors[index];
    const std::vector<float> physical = scatter(
        make_input(logical_element_count(tensor), index, domains[index]),
        tensor);
    const std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    result.push_back(
        gather(io::decode(encoded, tensor.data_type, physical.size()), tensor));
  }
  return result;
}

enum class BindingAddress {
  kStorageBase,
  kTensorEntrance,
};

struct PreparedBuffers {
  std::vector<TensorSpec> tensors;
  std::size_t input_count = 0;
  std::vector<std::unique_ptr<hv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

PreparedBuffers
prepare_buffers(std::vector<TensorSpec> tensors, std::size_t input_count,
                const std::vector<std::vector<float>> &logical_inputs,
                BindingAddress binding_address, hv::Stream &stream) {
  if (input_count > tensors.size() || logical_inputs.size() != input_count) {
    throw std::invalid_argument("benchmark buffer tensor arity is invalid");
  }
  PreparedBuffers result;
  result.tensors = std::move(tensors);
  result.input_count = input_count;
  result.buffers.reserve(result.tensors.size());
  result.bindings.reserve(result.tensors.size());
  for (std::size_t index = 0; index < result.tensors.size(); ++index) {
    const TensorSpec &tensor = result.tensors[index];
    std::vector<float> physical;
    if (index < input_count) {
      physical = scatter(logical_inputs[index], tensor);
    } else {
      physical.assign(storage_element_count(tensor), io::kPaddingSentinel);
    }
    const std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    auto buffer = std::make_unique<hv::DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    buffer->copy_from_host_at(encoded.data(), encoded.size(),
                              tensor.binding_byte_offset, stream.get());
    void *pointer = binding_address == BindingAddress::kStorageBase
                        ? buffer->opaque()
                        : buffer->opaque_at(tensor.binding_byte_offset);
    result.bindings.push_back({tensor.uid, pointer});
    result.buffers.push_back(std::move(buffer));
  }
  return result;
}

void reset_outputs(PreparedBuffers &prepared, hv::Stream &stream,
                   BenchmarkProviderKind provider, BenchmarkPhase phase) {
  for (std::size_t tensor_index = prepared.input_count;
       tensor_index < prepared.tensors.size(); ++tensor_index) {
    const TensorSpec &output = prepared.tensors[tensor_index];
    const std::vector<float> physical(storage_element_count(output),
                                      io::kPaddingSentinel);
    const std::vector<std::uint8_t> encoded =
        io::encode(physical, output.data_type);
    run_benchmark_phase(provider, phase, [&] {
      prepared.buffers.at(tensor_index)
          ->copy_from_host_at(encoded.data(), encoded.size(),
                              output.binding_byte_offset, stream.get());
    });
    synchronize_phase(stream, provider, phase);
  }
}

std::vector<float> read_output(const PreparedBuffers &prepared,
                               std::size_t output_index, hv::Stream &stream,
                               BenchmarkProviderKind provider,
                               BenchmarkPhase phase,
                               std::string_view display_provider) {
  const std::size_t tensor_index = prepared.input_count + output_index;
  const TensorSpec &output = prepared.tensors.at(tensor_index);
  const std::size_t storage = storage_element_count(output);
  std::vector<std::uint8_t> encoded(storage *
                                    io::data_type_size(output.data_type));
  run_benchmark_phase(provider, phase, [&] {
    prepared.buffers.at(tensor_index)
        ->copy_to_host_at(encoded.data(), encoded.size(),
                          output.binding_byte_offset, stream.get());
  });
  synchronize_phase(stream, provider, phase);
  const std::vector<float> physical =
      io::decode(encoded, output.data_type, storage);
  require_padding_unchanged(display_provider, physical, output);
  return gather(physical, output);
}

void execute(BenchmarkExecutable &executable,
             std::span<const flagdnnBinding_t> bindings,
             hv::DeviceBuffer &workspace, hv::Stream &stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
  std::optional<std::string> mismatch;
};

Accuracy compare_outputs(std::span<const float> actual,
                         std::span<const float> reference,
                         const BenchmarkCase &specification,
                         std::size_t output_index,
                         std::string_view actual_provider,
                         std::string_view reference_provider) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(std::string(actual_provider) + " and " +
                             std::string(reference_provider) +
                             " output sizes do not match");
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
        (absolute > specification.absolute_tolerance &&
         relative > specification.relative_tolerance)) {
      if (result.mismatch.has_value()) {
        continue;
      }
      std::ostringstream message;
      message << specification.name << " output " << output_index
              << " differs at element " << index << ": " << actual_provider
              << '=' << left << ", " << reference_provider << '=' << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << specification.absolute_tolerance
              << ", rtol=" << specification.relative_tolerance;
      result.mismatch = message.str();
    }
  }
  return result;
}

void merge_accuracy(Accuracy &aggregate, const Accuracy &value) {
  aggregate.maximum_absolute =
      std::max(aggregate.maximum_absolute, value.maximum_absolute);
  aggregate.maximum_relative =
      std::max(aggregate.maximum_relative, value.maximum_relative);
  if (!aggregate.mismatch.has_value() && value.mismatch.has_value()) {
    aggregate.mismatch = value.mismatch;
  }
}

void require_accuracy(const Accuracy &accuracy) {
  if (accuracy.mismatch.has_value()) {
    throw std::runtime_error(*accuracy.mismatch);
  }
}

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty benchmark samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(values.size()))) -
      1;
  return values[std::min(index, values.size() - 1)];
}

void emit_samples(std::string_view provider, const BenchmarkCase &specification,
                  const std::vector<double> &samples) {
  std::cout << "{\"schema_version\":1,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\""
            << specification.name << "\",\"unit\":\"us\","
            << "\"median\":" << percentile(samples, 0.5)
            << ",\"p90\":" << percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "]}\n";
}

void emit_skip(const HipdnnProvider &provider,
               const BenchmarkCase &specification, std::string_view reason,
               std::span<const hv::ReferenceTensor> tensors) {
  std::cout << "[SKIP][hipdnn] op=" << provider.operation_name(specification)
            << " case=" << specification.name << " reason=" << reason << ' '
            << hv::hipdnn_environment() << ' '
            << hv::describe_reference_tensors(tensors) << std::endl;
}

void warmup(BenchmarkExecutable &executable,
            std::span<const flagdnnBinding_t> bindings,
            hv::DeviceBuffer &workspace, hv::Stream &stream, int iterations,
            BenchmarkProviderKind provider) {
  if (iterations < 0) {
    throw std::invalid_argument("benchmark warmup iterations are negative");
  }
  run_benchmark_phase(provider, BenchmarkPhase::kWarmup, [&] {
    for (int index = 0; index < iterations; ++index) {
      execute(executable, bindings, workspace, stream);
    }
  });
  synchronize_phase(stream, provider, BenchmarkPhase::kWarmup);
}

double measure_batch(hv::EventTimer &timer, const CapturedExecutionBatch &batch,
                     hv::Stream &stream, BenchmarkProviderKind provider) {
  const double batch_microseconds =
      run_benchmark_phase(provider, BenchmarkPhase::kTiming, [&] {
        return timer.measure_microseconds(stream.get(), 1,
                                          [&] { batch.launch(stream.get()); });
      });
  require_no_pending_hip_error(provider, BenchmarkPhase::kTiming);
  return batch_microseconds / static_cast<double>(batch.execution_count());
}

enum class CaseResult {
  kExecuted,
  kSkipped,
};

constexpr bool
hipdnn_capture_status_is_capability(hipdnnStatus_t status) noexcept {
  /* Some hipDNN primitives execute correctly but cannot be launched while a
   * HIP stream is being captured on specific shapes.  This classification is
   * deliberately local to the benchmark capture probe: EXECUTION_FAILED
   * remains a hard failure for descriptor setup, direct execution, warmup,
   * replay, timing, and every functional reference path. */
  return status == HIPDNN_STATUS_NOT_SUPPORTED ||
         status == HIPDNN_STATUS_EXECUTION_FAILED;
}

static_assert(hipdnn_capture_status_is_capability(HIPDNN_STATUS_NOT_SUPPORTED));
static_assert(
    hipdnn_capture_status_is_capability(HIPDNN_STATUS_EXECUTION_FAILED));
static_assert(!hipdnn_capture_status_is_capability(HIPDNN_STATUS_BAD_PARAM));
static_assert(
    !hipdnn_capture_status_is_capability(HIPDNN_STATUS_INTERNAL_ERROR));
static_assert(
    !hipdnn_capture_status_is_capability(HIPDNN_STATUS_ARCH_MISMATCH));

CaseResult
run_case(const BenchmarkCase &specification, HipdnnProvider &hipdnn_provider,
         const std::function<FlagdnnProvider &()> &get_flagdnn_provider,
         hv::Stream &stream,
         const std::function<std::unique_ptr<BenchmarkExecutable>(
             const BenchmarkCase&)>& native_builder) {
  const std::vector<hv::ReferenceTensor> diagnostics =
      hipdnn_provider.diagnostic_tensors(specification);
  const ProviderCapability capability =
      hipdnn_provider.capability(specification);
  if (!capability.supported) {
    emit_skip(hipdnn_provider, specification, capability.reason, diagnostics);
    return CaseResult::kSkipped;
  }

  const std::size_t inputs = input_tensor_count(specification);
  const std::vector<std::vector<float>> logical_inputs =
      make_logical_inputs(specification);
  const bool accuracy_gated_performance =
      hipdnn_provider.requires_accuracy_gated_performance(specification);
  const bool fp32_correctness_oracle =
      hipdnn_provider.uses_fp32_correctness_oracle(specification);
  std::unique_ptr<HipdnnExecutable> reference;
  try {
    reference = hipdnn_provider.build_reference(
        specification, HipdnnReferencePolicy::kCorrectnessOracle);
  } catch (const hv::HipdnnStatusError &error) {
    if (!hv::hipdnn_status_is_capability(error.status())) {
      throw;
    }
    emit_skip(hipdnn_provider, specification,
              std::string("hipDNN build capability: ") + error.what(),
              diagnostics);
    return CaseResult::kSkipped;
  }
  PreparedBuffers reference_buffers = run_benchmark_phase(
      BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kProbe, [&] {
        return prepare_buffers(
            hipdnn_provider.reference_specs(
                specification, HipdnnReferencePolicy::kCorrectnessOracle),
            inputs, logical_inputs, BindingAddress::kStorageBase, stream);
      });
  hv::DeviceBuffer reference_workspace(reference->workspace_size());
  synchronize_phase(stream, BenchmarkProviderKind::kHipdnn,
                    BenchmarkPhase::kProbe);
  const ProviderCapability runtime_capability = run_benchmark_phase(
      BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kProbe, [&] {
        return reference->probe(reference_buffers.bindings,
                                reference_workspace.opaque(),
                                reference->workspace_size(), stream.opaque());
      });
  synchronize_phase(stream, BenchmarkProviderKind::kHipdnn,
                    BenchmarkPhase::kProbe);
  if (!runtime_capability.supported) {
    emit_skip(hipdnn_provider, specification, runtime_capability.reason,
              diagnostics);
    return CaseResult::kSkipped;
  }
  const std::string runtime_description = reference->runtime_description();
  if (!runtime_description.empty()) {
    std::cout << "[hipdnn] op=" << hipdnn_provider.operation_name(specification)
              << " case=" << specification.name << ' ' << runtime_description
              << std::endl;
  }

  FlagdnnProvider &flagdnn_provider = get_flagdnn_provider();
  flagdnn_provider.set_autotune(true);
  std::unique_ptr<BenchmarkExecutable> flagdnn =
      native_builder ? native_builder(specification)
                     : flagdnn_provider.build(specification);
  PreparedBuffers flagdnn_buffers = run_benchmark_phase(
      BenchmarkProviderKind::kFlagdnn, BenchmarkPhase::kProbe, [&] {
        return prepare_buffers(specification.tensors, inputs, logical_inputs,
                               BindingAddress::kTensorEntrance, stream);
      });
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  run_benchmark_phase(
      BenchmarkProviderKind::kFlagdnn, BenchmarkPhase::kProbe, [&] {
        execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
      });
  synchronize_phase(stream, BenchmarkProviderKind::kFlagdnn,
                    BenchmarkPhase::kProbe);

  Accuracy aggregate;
  std::vector<std::vector<float>> oracle_outputs;
  oracle_outputs.reserve(specification.output_count);
  for (std::size_t output_index = 0; output_index < specification.output_count;
       ++output_index) {
    oracle_outputs.push_back(read_output(reference_buffers, output_index,
                                         stream, BenchmarkProviderKind::kHipdnn,
                                         BenchmarkPhase::kProbe, "hipDNN"));
    const Accuracy accuracy =
        compare_outputs(read_output(flagdnn_buffers, output_index, stream,
                                    BenchmarkProviderKind::kFlagdnn,
                                    BenchmarkPhase::kProbe, "FlagDNN"),
                        oracle_outputs.back(), specification, output_index,
                        "FlagDNN", "hipDNN correctness oracle");
    merge_accuracy(aggregate, accuracy);
  }
  require_accuracy(aggregate);
  std::cout << specification.name << ": FlagDNN Graph vs hipDNN primitive"
            << (hipdnn_provider.uses_sequence(specification) ? " sequence" : "")
            << " PASS max_abs=" << aggregate.maximum_absolute
            << " max_rel=" << aggregate.maximum_relative << std::endl;

  const ProviderCapability capture_capability =
      hipdnn_provider.capture_capability(specification);
  if (!capture_capability.supported) {
    emit_skip(hipdnn_provider, specification, capture_capability.reason,
              diagnostics);
    return CaseResult::kSkipped;
  }

  std::unique_ptr<HipdnnExecutable> performance_reference;
  std::unique_ptr<PreparedBuffers> performance_buffers;
  std::unique_ptr<hv::DeviceBuffer> performance_workspace;
  HipdnnExecutable *timed_reference = reference.get();
  PreparedBuffers *timed_reference_buffers = &reference_buffers;
  hv::DeviceBuffer *timed_reference_workspace = &reference_workspace;
  if (accuracy_gated_performance) {
    try {
      performance_reference = hipdnn_provider.build_reference(
          specification, HipdnnReferencePolicy::kPerformance);
    } catch (const hv::HipdnnStatusError &error) {
      throw std::runtime_error(
          "hipDNN performance plan became unavailable after its correctness "
          "oracle succeeded: " +
          std::string(error.what()));
    }
    performance_workspace = std::make_unique<hv::DeviceBuffer>(
        performance_reference->workspace_size());
    PreparedBuffers *candidate_buffers = &reference_buffers;
    if (fp32_correctness_oracle) {
      performance_buffers = std::make_unique<PreparedBuffers>(
          run_benchmark_phase(
              BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kProbe, [&] {
                return prepare_buffers(
                    hipdnn_provider.reference_specs(
                        specification, HipdnnReferencePolicy::kPerformance),
                    inputs, logical_inputs, BindingAddress::kStorageBase,
                    stream);
              }));
      candidate_buffers = performance_buffers.get();
    }

    bool rejected_numerical_candidate = false;
    for (;;) {
      /* A rejected candidate must not leave values that could mask a partial
       * write by the next candidate. */
      reset_outputs(*candidate_buffers, stream,
                    BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kProbe);
      const ProviderCapability candidate_capability = run_benchmark_phase(
          BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kProbe, [&] {
            return performance_reference->probe(
                candidate_buffers->bindings, performance_workspace->opaque(),
                performance_reference->workspace_size(), stream.opaque());
          });
      synchronize_phase(stream, BenchmarkProviderKind::kHipdnn,
                        BenchmarkPhase::kProbe);
      if (!candidate_capability.supported) {
        if (fp32_correctness_oracle && rejected_numerical_candidate) {
          emit_skip(
              hipdnn_provider, specification,
              "provider=hipdnn phase=probe "
              "status=HIPDNN_PERFORMANCE_ACCURACY_EXHAUSTED all executable "
              "FP16 performance candidates failed the source-FP16-quantized "
              "FP32 hipDNN correctness oracle: " +
                  candidate_capability.reason,
              diagnostics);
          return CaseResult::kSkipped;
        }
        throw std::runtime_error(
            "hipDNN performance candidates became unavailable after the "
            "same primitive correctness oracle succeeded: " +
            candidate_capability.reason);
      }

      const std::string candidate_description =
          performance_reference->runtime_description();
      if (candidate_description.empty()) {
        throw std::logic_error(
            "hipDNN convolution performance candidate has no description");
      }
      Accuracy candidate_accuracy;
      for (std::size_t output_index = 0;
           output_index < specification.output_count; ++output_index) {
        merge_accuracy(
            candidate_accuracy,
            compare_outputs(read_output(*candidate_buffers, output_index,
                                        stream, BenchmarkProviderKind::kHipdnn,
                                        BenchmarkPhase::kProbe,
                                        "hipDNN performance candidate"),
                            oracle_outputs.at(output_index), specification,
                            output_index, "hipDNN performance candidate",
                            "hipDNN correctness oracle"));
      }
      if (!candidate_accuracy.mismatch.has_value()) {
        std::cout << "[hipdnn][selected] op="
                  << hipdnn_provider.operation_name(specification)
                  << " case=" << specification.name << ' '
                  << candidate_description
                  << " max_abs=" << candidate_accuracy.maximum_absolute
                  << " max_rel=" << candidate_accuracy.maximum_relative
                  << std::endl;
        break;
      }

      std::cout << "[hipdnn][reject] op="
                << hipdnn_provider.operation_name(specification)
                << " case=" << specification.name << ' '
                << candidate_description
                << " max_abs=" << candidate_accuracy.maximum_absolute
                << " max_rel=" << candidate_accuracy.maximum_relative
                << " reason=" << *candidate_accuracy.mismatch << std::endl;
      rejected_numerical_candidate = true;
      performance_reference->reject_selected_candidate(
          *candidate_accuracy.mismatch);
    }
    timed_reference = performance_reference.get();
    timed_reference_buffers = candidate_buffers;
    timed_reference_workspace = performance_workspace.get();
  }

  const BenchmarkConfig &config = specification.benchmark;
  if (config.sample_count <= 0 || config.iterations_per_sample <= 0) {
    throw std::invalid_argument("benchmark sample configuration is invalid");
  }
  warmup(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream,
         config.warmup_iterations, BenchmarkProviderKind::kFlagdnn);
  warmup(*timed_reference, timed_reference_buffers->bindings,
         *timed_reference_workspace, stream, config.warmup_iterations,
         BenchmarkProviderKind::kHipdnn);

  std::unique_ptr<CapturedExecutionBatch> flagdnn_batch = run_benchmark_phase(
      BenchmarkProviderKind::kFlagdnn, BenchmarkPhase::kCaptureBuild, [&] {
        return std::make_unique<CapturedExecutionBatch>(
            stream.get(), config.iterations_per_sample, [&] {
              execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace,
                      stream);
            });
      });
  require_no_pending_hip_error(BenchmarkProviderKind::kFlagdnn,
                               BenchmarkPhase::kCaptureBuild);
  std::unique_ptr<CapturedExecutionBatch> reference_batch;
  try {
    reference_batch = std::make_unique<CapturedExecutionBatch>(
        stream.get(), config.iterations_per_sample, [&] {
          execute(*timed_reference, timed_reference_buffers->bindings,
                  *timed_reference_workspace, stream);
        });
  } catch (const hv::HipdnnStatusError &error) {
    const std::string context = benchmark_phase_context(
        BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kCaptureBuild);
    if (!hipdnn_capture_status_is_capability(error.status())) {
      throw std::runtime_error(context + ": " + error.what());
    }
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusActive;
    const std::string capture_query = context + " hipStreamIsCapturing";
    hv::check_hip(hipStreamIsCapturing(stream.get(), &capture_status),
                  capture_query.c_str());
    if (capture_status != hipStreamCaptureStatusNone) {
      throw std::runtime_error(
          context + ": failed hipDNN capability probe left the stream in "
                    "capture state");
    }
    synchronize_phase(stream, BenchmarkProviderKind::kHipdnn,
                      BenchmarkPhase::kCaptureBuild);
    emit_skip(hipdnn_provider, specification,
              context + " status=" +
                  std::string(hv::hipdnn_status_name(error.status())) +
                  " HIP Graph capture capability: " + std::string(error.what()),
              diagnostics);
    return CaseResult::kSkipped;
  } catch (const std::exception &error) {
    throw std::runtime_error(
        benchmark_phase_context(BenchmarkProviderKind::kHipdnn,
                                BenchmarkPhase::kCaptureBuild) +
        ": " + error.what());
  }
  require_no_pending_hip_error(BenchmarkProviderKind::kHipdnn,
                               BenchmarkPhase::kCaptureBuild);

  run_benchmark_phase(BenchmarkProviderKind::kFlagdnn,
                      BenchmarkPhase::kCaptureReplay,
                      [&] { flagdnn_batch->launch(stream.get()); });
  synchronize_phase(stream, BenchmarkProviderKind::kFlagdnn,
                    BenchmarkPhase::kCaptureReplay);
  run_benchmark_phase(BenchmarkProviderKind::kHipdnn,
                      BenchmarkPhase::kCaptureReplay,
                      [&] { reference_batch->launch(stream.get()); });
  synchronize_phase(stream, BenchmarkProviderKind::kHipdnn,
                    BenchmarkPhase::kCaptureReplay);

  std::unique_ptr<hv::EventTimer> flagdnn_timer = run_benchmark_phase(
      BenchmarkProviderKind::kFlagdnn, BenchmarkPhase::kTiming,
      [] { return std::make_unique<hv::EventTimer>(); });
  require_no_pending_hip_error(BenchmarkProviderKind::kFlagdnn,
                               BenchmarkPhase::kTiming);
  std::unique_ptr<hv::EventTimer> hipdnn_timer = run_benchmark_phase(
      BenchmarkProviderKind::kHipdnn, BenchmarkPhase::kTiming,
      [] { return std::make_unique<hv::EventTimer>(); });
  require_no_pending_hip_error(BenchmarkProviderKind::kHipdnn,
                               BenchmarkPhase::kTiming);
  std::vector<double> flagdnn_samples;
  std::vector<double> hipdnn_samples;
  flagdnn_samples.reserve(static_cast<std::size_t>(config.sample_count));
  hipdnn_samples.reserve(static_cast<std::size_t>(config.sample_count));
  for (int sample = 0; sample < config.sample_count; ++sample) {
    const auto measure_flagdnn = [&] {
      flagdnn_samples.push_back(measure_batch(*flagdnn_timer, *flagdnn_batch,
                                              stream,
                                              BenchmarkProviderKind::kFlagdnn));
    };
    const auto measure_hipdnn = [&] {
      hipdnn_samples.push_back(measure_batch(*hipdnn_timer, *reference_batch,
                                             stream,
                                             BenchmarkProviderKind::kHipdnn));
    };
    if (sample % 2 == 0) {
      measure_flagdnn();
      measure_hipdnn();
    } else {
      measure_hipdnn();
      measure_flagdnn();
    }
  }

  {
    Accuracy post_flagdnn_accuracy;
    Accuracy post_hipdnn_accuracy;
    for (std::size_t output_index = 0;
         output_index < specification.output_count; ++output_index) {
      merge_accuracy(
          post_flagdnn_accuracy,
          compare_outputs(
              read_output(flagdnn_buffers, output_index, stream,
                          BenchmarkProviderKind::kFlagdnn,
                          BenchmarkPhase::kTiming, "FlagDNN post-timing"),
              oracle_outputs.at(output_index), specification, output_index,
              "FlagDNN post-timing", "hipDNN correctness oracle"));
      merge_accuracy(
          post_hipdnn_accuracy,
          compare_outputs(
              read_output(*timed_reference_buffers, output_index, stream,
                          BenchmarkProviderKind::kHipdnn,
                          BenchmarkPhase::kTiming, "hipDNN post-timing"),
              oracle_outputs.at(output_index), specification, output_index,
              "hipDNN post-timing", "hipDNN correctness oracle"));
    }
    require_accuracy(post_flagdnn_accuracy);
    require_accuracy(post_hipdnn_accuracy);
    std::string timed_description;
    if (accuracy_gated_performance) {
      timed_description = timed_reference->runtime_description();
      if (timed_description.empty()) {
        throw std::logic_error(
            "timed hipDNN convolution candidate lost its description");
      }
    }
    std::cout << "[postcheck] op="
              << hipdnn_provider.operation_name(specification)
              << " case=" << specification.name;
    if (!timed_description.empty()) {
      std::cout << ' ' << timed_description;
    }
    std::cout << " flagdnn_max_abs=" << post_flagdnn_accuracy.maximum_absolute
              << " flagdnn_max_rel=" << post_flagdnn_accuracy.maximum_relative
              << " hipdnn_max_abs=" << post_hipdnn_accuracy.maximum_absolute
              << " hipdnn_max_rel=" << post_hipdnn_accuracy.maximum_relative
              << std::endl;
  }

  emit_samples("flagdnn", specification, flagdnn_samples);
  emit_samples("hipdnn", specification, hipdnn_samples);
  const double flagdnn_median = percentile(flagdnn_samples, 0.5);
  const double hipdnn_median = percentile(hipdnn_samples, 0.5);
  std::cout << specification.name << ": median_us flagdnn=" << flagdnn_median
            << " hipdnn=" << hipdnn_median
            << " speedup=" << hipdnn_median / flagdnn_median << std::endl;
  return CaseResult::kExecuted;
}

} // namespace

int run_hygon_benchmark_suite(int argc, char **argv,
                        std::span<const BenchmarkCase> cases,
                        std::string_view suite_name,
                        const NativeBuilder& builder) {
  if (argc != 3) {
    std::cerr << "usage: " << suite_name
              << " COMPILER_EXECUTABLE COMPILER_ENTRY" << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    BenchmarkCache cache;
    HipdnnProvider hipdnn_provider;
    std::unique_ptr<flagdnn::Handle> handle;
    std::unique_ptr<FlagdnnProvider> flagdnn_provider;
    const std::function<FlagdnnProvider &()> get_flagdnn_provider =
        [&]() -> FlagdnnProvider & {
      if (flagdnn_provider == nullptr) {
        handle = std::make_unique<flagdnn::Handle>("hygon", 0);
        handle->set_compiler(argv[1], argv[2], cache.path().string());
        flagdnn_provider = std::make_unique<FlagdnnProvider>(*handle);
      }
      return *flagdnn_provider;
    };

    std::function<std::unique_ptr<BenchmarkExecutable>(const BenchmarkCase&)> native_builder;
    if (builder) native_builder = [&](const BenchmarkCase& specification) {
      (void)get_flagdnn_provider();
      return builder(*handle, specification);
    };
    const char *case_filter = std::getenv("FLAGDNN_BENCHMARK_CASE");
    std::size_t matched = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const BenchmarkCase &specification : cases) {
      if (case_filter != nullptr && case_filter[0] != '\0' &&
          specification.name != case_filter) {
        continue;
      }
      ++matched;
      const CaseResult result = run_case(specification, hipdnn_provider,
                                         get_flagdnn_provider, stream, native_builder);
      if (result == CaseResult::kExecuted) {
        ++executed;
      } else {
        ++skipped;
      }
    }
    if (matched == 0) {
      throw std::invalid_argument(
          "FLAGDNN_BENCHMARK_CASE did not match any case");
    }
    std::cout << suite_name << ": " << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << matched << " executed=" << executed
              << " skipped=" << skipped << std::endl;
    return executed == 0 ? kSkipReturnCode : 0;
  } catch (const std::exception &error) {
    std::cerr << suite_name << "_FAILED: " << error.what() << std::endl;
    return 1;
  }
}

int run_benchmark_suite(int argc, char** argv,
    std::span<const BenchmarkCase> cases, std::string_view suite_name) {
  const auto selected = aligned_benchmark_cases(cases);
  return run_hygon_benchmark_suite(argc, argv, selected, suite_name);
}

} // namespace flagdnn::benchmarking
