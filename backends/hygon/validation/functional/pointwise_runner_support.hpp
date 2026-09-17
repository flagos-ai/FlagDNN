/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_

#include "common/common.hpp"
#include "common/pointwise.hpp"
#include "functional/accuracy.hpp"
#include "functional/binding_address.hpp"
#include "hip_driver.hpp"
#include "hipdnn_reference.hpp"
#include "pointwise_reference.hpp"
#include "reference/cpu/pointwise.hpp"
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
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing::hygon_functional::pointwise {

namespace hv = validation::hygon;
namespace io = validation::hygon::tensor_io;
namespace cpu_reference = ::flagdnn::reference::cpu;

inline constexpr int kSkipReturnCode = 77;

class TemporaryCache {
public:
  explicit TemporaryCache(std::string_view family) {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           ("flagdnn-hygon-" + std::string(family) + "-XXXXXX"))
                              .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char *created = mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error("mkdtemp failed");
    }
    path_ = created;
  }

  ~TemporaryCache() {
    std::error_code ignored;
    std::filesystem::remove_all(path_, ignored);
  }

  [[nodiscard]] const std::filesystem::path &path() const noexcept {
    return path_;
  }

private:
  std::filesystem::path path_;
};

class ReferenceExecutable final : public TestExecutable {
public:
  ReferenceExecutable(hv::HipdnnPointwiseOperation operation,
                      std::vector<hv::ReferenceTensor> tensors)
      : plan_(std::move(operation), std::move(tensors)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnPointwisePlan plan_;
};

inline hv::HipdnnPointwiseOperation
fixed_operation(hv::HipdnnPointwiseKind kind, double alpha = 1.0) {
  hv::HipdnnPointwiseOperation result;
  result.kind = kind;
  result.alpha = alpha;
  result.unavailable_reason.clear();
  return result;
}

inline std::string case_operation_name(std::string_view case_name) {
  constexpr std::array<std::string_view, 7> markers = {
      "_strided_", "_perf_",     "_fp32_", "_fp16_",
      "_bf16_",    "_bfloat16_", "_bool_"};
  std::size_t end = std::string_view::npos;
  for (std::string_view marker : markers) {
    const std::size_t position = case_name.find(marker);
    if (position != std::string_view::npos) {
      end = std::min(end, position);
    }
  }
  return std::string(end == std::string_view::npos ? case_name
                                                   : case_name.substr(0, end));
}

inline std::vector<hv::ReferenceTensor>
reference_tensors(std::span<const TestTensor> inputs,
                  const TestTensor &output) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(inputs.size() + 1);
  for (const TestTensor &input : inputs) {
    result.push_back(hv::as_reference_tensor(input));
  }
  result.push_back(hv::as_reference_tensor(output));
  return result;
}

inline std::vector<float> make_input(std::size_t count, std::size_t input_index,
                                     PointwiseInputDomain domain) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const float real =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
    switch (domain) {
    case PointwiseInputDomain::kReal:
      result[index] = real;
      break;
    case PointwiseInputDomain::kPositive:
      result[index] = std::abs(real) + 0.5F;
      break;
    case PointwiseInputDomain::kScaled:
      result[index] = real * 4.0F;
      break;
    case PointwiseInputDomain::kTan:
      result[index] = static_cast<float>(centered) / 40.0F;
      break;
    case PointwiseInputDomain::kDivisor:
    case PointwiseInputDomain::kModulo:
      result[index] = input_index == 1 ? std::abs(real) + 0.5F : real;
      break;
    case PointwiseInputDomain::kPower:
      result[index] = input_index == 0
                          ? std::abs(real) + 0.5F
                          : std::fmod(std::abs(real), 2.0F) + 0.125F;
      break;
    case PointwiseInputDomain::kModuloSigned: {
      constexpr std::array<float, 6> left = {-3.0F, -3.0F, 3.0F,
                                             3.0F,  -5.5F, 5.5F};
      constexpr std::array<float, 6> right = {2.0F,  -2.0F, 2.0F,
                                              -2.0F, 2.25F, -2.25F};
      result[index] = input_index == 0 ? left[index % left.size()]
                                       : right[index % right.size()];
      break;
    }
    case PointwiseInputDomain::kComparison: {
      const int base_centered = static_cast<int>((index * 17) % 41) - 20;
      const float base = static_cast<float>(base_centered) / 13.0F;
      result[index] = input_index == 0 || index % 3 == 0
                          ? base
                          : (index % 3 == 1 ? base + 0.25F : base - 0.25F);
      break;
    }
    case PointwiseInputDomain::kLogical:
      result[index] = ((index * 17 + input_index * 11) % 3) != 0 ? 1.0F : 0.0F;
      break;
    }
  }
  return result;
}

struct PreparedBuffers {
  std::vector<TestTensor> inputs;
  TestTensor output;
  std::vector<std::vector<float>> logical_inputs;
  std::vector<std::unique_ptr<hv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

enum class LogicalInputCapture { kDisabled, kEnabled };

inline PreparedBuffers
prepare_buffers(std::span<const TestTensor> inputs, const TestTensor &output,
                std::span<const PointwiseInputDomain> domains,
                BindingAddress binding_address,
                LogicalInputCapture logical_input_capture, hv::Stream &stream) {
  if (inputs.size() != domains.size()) {
    throw std::invalid_argument("functional input domains do not match arity");
  }
  PreparedBuffers result;
  result.inputs.assign(inputs.begin(), inputs.end());
  result.output = output;
  if (logical_input_capture == LogicalInputCapture::kEnabled) {
    result.logical_inputs.reserve(inputs.size());
  }
  result.buffers.reserve(inputs.size() + 1);
  result.bindings.reserve(inputs.size() + 1);
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const TestTensor &tensor = inputs[index];
    const std::vector<float> logical =
        make_input(io::element_count(tensor), index, domains[index]);
    const std::vector<float> physical = io::scatter(logical, tensor);
    const std::vector<std::uint8_t> encoded =
        io::encode(physical, tensor.data_type);
    if (logical_input_capture == LogicalInputCapture::kEnabled) {
      result.logical_inputs.push_back(io::gather(
          io::decode(encoded, tensor.data_type, physical.size()), tensor));
    }
    auto buffer = std::make_unique<hv::DeviceBuffer>(
        tensor.binding_byte_offset + encoded.size());
    buffer->copy_from_host_at(encoded.data(), encoded.size(),
                              tensor.binding_byte_offset, stream.get());
    result.bindings.push_back(
        {tensor.uid,
         binding_pointer(buffer->opaque(), tensor.binding_byte_offset,
                         binding_address)});
    result.buffers.push_back(std::move(buffer));
  }
  const std::vector<float> initial(io::storage_element_count(output),
                                   io::kPaddingSentinel);
  const std::vector<std::uint8_t> encoded =
      io::encode(initial, output.data_type);
  auto buffer = std::make_unique<hv::DeviceBuffer>(output.binding_byte_offset +
                                                   encoded.size());
  buffer->copy_from_host_at(encoded.data(), encoded.size(),
                            output.binding_byte_offset, stream.get());
  result.bindings.push_back(
      {output.uid, binding_pointer(buffer->opaque(), output.binding_byte_offset,
                                   binding_address)});
  result.buffers.push_back(std::move(buffer));
  return result;
}

inline std::vector<float>
quantize_logical_output(std::span<const float> logical,
                        const TestTensor &output) {
  const std::vector<float> physical = io::scatter(logical, output);
  const std::vector<std::uint8_t> encoded =
      io::encode(physical, output.data_type);
  return io::gather(io::decode(encoded, output.data_type, physical.size()),
                    output);
}

inline std::vector<float> read_output(const PreparedBuffers &prepared,
                                      hv::Stream &stream,
                                      std::string_view provider) {
  const std::size_t storage = io::storage_element_count(prepared.output);
  std::vector<std::uint8_t> bytes(
      storage * io::data_type_size(prepared.output.data_type));
  prepared.buffers.back()->copy_to_host_at(bytes.data(), bytes.size(),
                                           prepared.output.binding_byte_offset,
                                           stream.get());
  stream.synchronize();
  const std::vector<float> physical =
      io::decode(bytes, prepared.output.data_type, storage);
  io::require_padding_unchanged(provider, physical, prepared.output);
  return io::gather(physical, prepared.output);
}

inline void execute(TestExecutable &executable,
                    std::span<const flagdnnBinding_t> bindings,
                    hv::DeviceBuffer &workspace, hv::Stream &stream) {
  executable.execute(bindings, workspace.opaque(), executable.workspace_size(),
                     stream.opaque());
}

inline void emit_skip(std::string_view operation, std::string_view case_name,
                      std::string_view reason,
                      std::span<const hv::ReferenceTensor> tensors) {
  std::cout << "[SKIP][hipdnn] op=" << operation << " case=" << case_name
            << " reason=" << reason << ' ' << hv::hipdnn_environment() << ' '
            << hv::describe_reference_tensors(tensors) << std::endl;
}

enum class CaseResult { kExecuted, kSkipped };

inline constexpr bool
uses_hygon_cpu_reference(flagdnnPointwiseMode_t mode) noexcept {
  switch (mode) {
  case FLAGDNN_POINTWISE_DIV:
  case FLAGDNN_POINTWISE_POW:
  case FLAGDNN_POINTWISE_MOD:
  case FLAGDNN_POINTWISE_CMP_EQ:
    return true;
  default:
    return false;
  }
}

template <typename BuildFlagdnn>
CaseResult run_cpu_case(std::string_view operation, std::string_view case_name,
                        std::span<const TestTensor> inputs,
                        const TestTensor &output,
                        std::span<const PointwiseInputDomain> domains,
                        flagdnnPointwiseMode_t mode, double absolute_tolerance,
                        double relative_tolerance,
                        const std::function<flagdnn::Handle &()> &get_handle,
                        hv::Stream &stream, BuildFlagdnn &&build_flagdnn) {
  if (!uses_hygon_cpu_reference(mode) || inputs.size() != 2) {
    throw std::invalid_argument(
        "CPU pointwise reference requires a supported binary operation");
  }
  if (!cpu_reference::supports_binary_pointwise(mode)) {
    throw std::logic_error(
        "Hygon CPU-reference policy has no matching CPU implementation");
  }

  auto flagdnn = build_flagdnn(get_handle());
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(inputs, output, domains, BindingAddress::kTensorEntrance,
                      LogicalInputCapture::kEnabled, stream);
  if (flagdnn_buffers.logical_inputs.size() != 2) {
    throw std::logic_error("CPU pointwise reference input preparation failed");
  }
  std::vector<float> expected = cpu_reference::evaluate_binary_pointwise(
      mode, flagdnn_buffers.logical_inputs[0], inputs[0].dimensions,
      flagdnn_buffers.logical_inputs[1], inputs[1].dimensions,
      output.dimensions);
  expected = quantize_logical_output(expected, output);

  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();

  const Accuracy accuracy = compare_outputs(
      read_output(flagdnn_buffers, stream, "FlagDNN"), expected,
      absolute_tolerance, relative_tolerance, case_name, "CPU semantic oracle");
  std::cout << case_name << ": FlagDNN Graph vs CPU semantic oracle PASS"
            << " op=" << operation << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

template <typename BuildFlagdnn, typename BuildReference>
CaseResult run_case(std::string_view operation, std::string_view case_name,
                    std::span<const TestTensor> inputs,
                    const TestTensor &output,
                    std::span<const PointwiseInputDomain> domains,
                    const hv::HipdnnPointwiseOperation &reference_operation,
                    double absolute_tolerance, double relative_tolerance,
                    const std::function<flagdnn::Handle &()> &get_handle,
                    hv::Stream &stream, BuildFlagdnn &&build_flagdnn,
                    BuildReference &&build_reference) {
  const std::vector<hv::ReferenceTensor> tensors =
      reference_tensors(inputs, output);
  const hv::HipdnnCapability capability =
      hv::hipdnn_pointwise_capability(reference_operation, tensors, true);
  hv::require_valid_hipdnn_adapter_contract(capability, operation);
  if (!capability.supported) {
    return CaseResult::kSkipped;
  }

  std::unique_ptr<TestExecutable> reference;
  PreparedBuffers reference_buffers;
  try {
    reference = build_reference();
    reference_buffers =
        prepare_buffers(inputs, output, domains, BindingAddress::kStorageBase,
                        LogicalInputCapture::kDisabled, stream);
    hv::DeviceBuffer reference_workspace(reference->workspace_size());
    stream.synchronize();
    execute(*reference, reference_buffers.bindings, reference_workspace,
            stream);
    stream.synchronize();
  } catch (const hv::HipdnnStatusError &error) {
    if (!hv::hipdnn_status_is_capability(error.status())) {
      throw;
    }
    return CaseResult::kSkipped;
  }

  auto flagdnn = build_flagdnn(get_handle());
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(inputs, output, domains, BindingAddress::kTensorEntrance,
                      LogicalInputCapture::kDisabled, stream);
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();

  const Accuracy accuracy =
      compare_outputs(read_output(flagdnn_buffers, stream, "FlagDNN"),
                      read_output(reference_buffers, stream, "hipDNN"),
                      absolute_tolerance, relative_tolerance, case_name);
  std::cout << case_name << ": FlagDNN Graph vs hipDNN primitive"
            << (hv::hipdnn_pointwise_uses_sequence(reference_operation.kind)
                    ? " sequence"
                    : "")
            << " PASS max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

template <typename Run>
int run_suite(int argc, char **argv, std::string_view family,
              std::string_view suite_name, Run &&run) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    TemporaryCache cache(family);
    std::unique_ptr<flagdnn::Handle> handle;
    const std::function<flagdnn::Handle &()> get_handle =
        [&]() -> flagdnn::Handle & {
      if (handle == nullptr) {
        handle = std::make_unique<flagdnn::Handle>("hygon", 0);
        handle->set_compiler(argv[1], argv[2], cache.path().string());
      }
      return *handle;
    };
    std::size_t matched = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    run(get_handle, stream, matched, executed, skipped);
    if (matched == 0) {
      throw std::runtime_error("functional case filter matched no cases");
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

} // namespace flagdnn::testing::hygon_functional::pointwise

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_POINTWISE_RUNNER_SUPPORT_HPP_
