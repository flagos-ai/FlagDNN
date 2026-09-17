/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_CONVOLUTION_RUNNER_SUPPORT_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_CONVOLUTION_RUNNER_SUPPORT_HPP_

#include "common/composite.hpp"
#include "common/convolution.hpp"
#include "convolution_reference.hpp"
#include "hip_driver.hpp"
#include "host_runner.hpp"
#include "reference/cpu/matrix.hpp"
#include "tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

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
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing::hygon_functional::convolution {

namespace hv = validation::hygon;
namespace io = validation::hygon::tensor_io;

inline constexpr int kSkipReturnCode = 77;

class TemporaryCache {
public:
  TemporaryCache() {
    std::string pattern = (std::filesystem::temp_directory_path() /
                           "flagdnn-hygon-convolution-functional-XXXXXX")
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
  ReferenceExecutable(hv::HipdnnConvolutionOperation operation,
                      std::vector<hv::ReferenceTensor> tensors)
      : plan_(std::move(operation), std::move(tensors),
              hv::HipdnnConvolutionAlgorithmPolicy::kCorrectnessOracle) {
    const hv::HipdnnCapability capability = plan_.capability();
    if (!capability.supported) {
      throw std::invalid_argument("hipDNN convolution reference unavailable: " +
                                  capability.reason);
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    if (!probed_) {
      const hv::HipdnnCapability capability =
          plan_.probe_execute(bindings, workspace, workspace_size, stream);
      if (!capability.supported) {
        throw std::runtime_error(
            "hipDNN convolution actual execute unavailable: " +
            capability.reason);
      }
      probed_ = true;
      return;
    }
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnConvolutionPlan plan_;
  bool probed_ = false;
};

inline hv::HipdnnConvolutionMode reference_mode(ConvolutionMode mode) noexcept {
  return mode == ConvolutionMode::kConvolution
             ? hv::HipdnnConvolutionMode::kConvolution
             : hv::HipdnnConvolutionMode::kCrossCorrelation;
}

inline hv::HipdnnConvolutionKind
reference_kind(ConvolutionDirection direction) noexcept {
  switch (direction) {
  case ConvolutionDirection::kFprop:
    return hv::HipdnnConvolutionKind::kFprop;
  case ConvolutionDirection::kDgrad:
    return hv::HipdnnConvolutionKind::kDgrad;
  case ConvolutionDirection::kWgrad:
    return hv::HipdnnConvolutionKind::kWgrad;
  }
  return hv::HipdnnConvolutionKind::kUnavailable;
}

inline hv::HipdnnConvolutionOperation
reference_operation(const ConvolutionTestCase &test_case) {
  return hv::make_hipdnn_convolution_operation(
      reference_kind(test_case.direction), test_case.pre_padding,
      test_case.post_padding, test_case.stride, test_case.dilation,
      test_case.groups, reference_mode(test_case.mode));
}

inline hv::HipdnnConvolutionOperation
reference_operation(const ConvBiasReluTestCase &test_case) {
  return hv::make_hipdnn_convolution_operation(
      hv::HipdnnConvolutionKind::kConvBiasRelu, test_case.padding,
      test_case.padding, test_case.stride, test_case.dilation, 1);
}

inline std::vector<TestTensor>
semantic_tensors(const ConvolutionTestCase &test_case) {
  return {test_case.x, test_case.w, test_case.y};
}

inline std::vector<TestTensor>
semantic_tensors(const ConvBiasReluTestCase &test_case) {
  return {test_case.x, test_case.w, test_case.output, test_case.bias};
}

inline std::vector<hv::ReferenceTensor>
reference_tensors(std::span<const TestTensor> tensors) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(tensors.size());
  for (const TestTensor &tensor : tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

inline const TestTensor &output_tensor(const ConvolutionTestCase &test_case) {
  return convolution_output_tensor(test_case);
}

inline const TestTensor &output_tensor(const ConvBiasReluTestCase &test_case) {
  return test_case.output;
}

inline std::string_view operation_name(const ConvolutionTestCase &test_case) {
  switch (test_case.direction) {
  case ConvolutionDirection::kFprop:
    return "conv_fprop";
  case ConvolutionDirection::kDgrad:
    return "conv_dgrad";
  case ConvolutionDirection::kWgrad:
    return "conv_wgrad";
  }
  return "convolution";
}

inline std::string_view operation_name(const ConvBiasReluTestCase &) {
  return "conv_bias_relu";
}

inline void validate_case(const ConvolutionTestCase &test_case) {
  validate_convolution_case(test_case);
}

inline void validate_case(const ConvBiasReluTestCase &test_case) {
  validate_composite_case(test_case);
}

inline std::unique_ptr<TestExecutable>
build_flagdnn(flagdnn::Handle &handle, const ConvolutionTestCase &test_case) {
  return build_flagdnn_convolution(handle, test_case);
}

inline std::unique_ptr<TestExecutable>
build_flagdnn(flagdnn::Handle &handle, const ConvBiasReluTestCase &test_case) {
  return build_flagdnn_conv_bias_relu(handle, test_case);
}

inline std::vector<float> make_input(std::size_t count,
                                     std::size_t tensor_index) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + tensor_index * 13) % 29) - 14;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(31 + tensor_index);
  }
  return result;
}

struct PreparedBuffers {
  std::vector<TestTensor> tensors;
  std::size_t output_index = 0;
  std::vector<std::unique_ptr<hv::DeviceBuffer>> buffers;
  std::vector<flagdnnBinding_t> bindings;
};

enum class BindingAddress { kStorageBase, kTensorEntrance };

inline PreparedBuffers prepare_buffers(std::vector<TestTensor> tensors,
                                       std::int64_t output_uid,
                                       BindingAddress binding_address,
                                       hv::Stream &stream) {
  PreparedBuffers result;
  result.tensors = std::move(tensors);
  result.buffers.reserve(result.tensors.size());
  result.bindings.reserve(result.tensors.size());
  bool found_output = false;
  for (std::size_t index = 0; index < result.tensors.size(); ++index) {
    const TestTensor &tensor = result.tensors[index];
    const bool is_output = tensor.uid == output_uid;
    if (is_output) {
      if (found_output) {
        throw std::invalid_argument("convolution output UID is duplicated");
      }
      found_output = true;
      result.output_index = index;
    }
    const std::vector<float> physical =
        is_output
            ? std::vector<float>(io::storage_element_count(tensor),
                                 io::kPaddingSentinel)
            : io::scatter(make_input(io::element_count(tensor), index), tensor);
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
  if (!found_output) {
    throw std::invalid_argument("convolution output UID is not bound");
  }
  return result;
}

inline std::vector<float> read_output(const PreparedBuffers &prepared,
                                      hv::Stream &stream,
                                      std::string_view provider) {
  const TestTensor &output = prepared.tensors[prepared.output_index];
  const std::size_t storage = io::storage_element_count(output);
  std::vector<std::uint8_t> bytes(storage *
                                  io::data_type_size(output.data_type));
  prepared.buffers[prepared.output_index]->copy_to_host_at(
      bytes.data(), bytes.size(), output.binding_byte_offset, stream.get());
  stream.synchronize();
  const std::vector<float> physical =
      io::decode(bytes, output.data_type, storage);
  io::require_padding_unchanged(provider, physical, output);
  return io::gather(physical, output);
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

template <typename Case>
Accuracy compare(std::span<const float> actual,
                 std::span<const float> reference, const Case &test_case) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error(
        "FlagDNN and hipDNN convolution outputs differ in size");
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
    if (!std::isfinite(absolute) || (absolute > test_case.absolute_tolerance &&
                                     relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << ", hipDNN=" << right
              << ", abs=" << absolute << ", rel=" << relative
              << ", atol=" << test_case.absolute_tolerance
              << ", rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
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

inline reference::cpu::ConvolutionParameters
cpu_parameters(const ConvolutionTestCase &c) {
  using Direction = reference::cpu::ConvolutionDirection;
  return {c.x.dimensions,
          c.w.dimensions,
          c.y.dimensions,
          c.stride,
          c.pre_padding,
          c.dilation,
          c.direction == ConvolutionDirection::kFprop ? Direction::kForward
          : c.direction == ConvolutionDirection::kDgrad
              ? Direction::kInputGradient
              : Direction::kWeightGradient,
          c.groups,
          c.input_precision,
          c.mode == ConvolutionMode::kConvolution};
}
inline reference::cpu::ConvolutionParameters
cpu_parameters(const ConvBiasReluTestCase &c) {
  return {c.x.dimensions,
          c.w.dimensions,
          c.output.dimensions,
          c.stride,
          c.padding,
          c.dilation,
          reference::cpu::ConvolutionDirection::kForward,
          1,
          0,
          false};
}
template <class Case>
CaseResult
run_host_convolution(const Case &c,
                     const std::function<flagdnn::Handle &()> &handle,
                     hv::Stream &stream) {
  namespace host = hygon_functional::host;
  const auto all = semantic_tensors(c);
  const auto output = output_tensor(c);
  std::vector<TestTensor> inputs;
  host::Values values;
  for (std::size_t i = 0; i < all.size(); ++i)
    if (all[i].uid != output.uid) {
      inputs.push_back(all[i]);
      values.push_back(make_input(io::element_count(all[i]), i));
    }
  host::run_case(
      c.name, inputs, std::vector<TestTensor>{output}, values, handle, stream,
      [&](flagdnn::Handle &h) { return build_flagdnn(h, c); },
      [&](const host::Values &v) {
        host::Values semantic;
        std::size_t next = 0;
        for (const auto &t : all)
          semantic.push_back(t.uid == output.uid
                                 ? std::vector<float>(io::element_count(t), 0)
                                 : v[next++]);
        auto result =
            reference::cpu::evaluate_convolution(cpu_parameters(c), semantic);
        if constexpr (requires { c.bias; }) {
          // The frontend graph materializes its convolution and bias-add
          // tensors in output storage precision before the following pointwise
          // node.
          result = io::decode(io::encode(result, output.data_type),
                              output.data_type, result.size());
          for (std::size_t i = 0; i < result.size(); ++i)
            result[i] +=
                semantic[3][(i / (result.size() / output.dimensions[0] /
                                  output.dimensions[1])) %
                            output.dimensions[1]];
          result = io::decode(io::encode(result, output.data_type),
                              output.data_type, result.size());
          for (auto &x : result)
            x = std::max(x, 0.0F);
        }
        return host::Values{result};
      },
      c.absolute_tolerance, c.relative_tolerance);
  return CaseResult::kExecuted;
}

template <typename Case>
CaseResult run_case(const Case &test_case,
                    const std::function<flagdnn::Handle &()> &get_handle,
                    hv::Stream &stream) {
  validate_case(test_case);
  const std::vector<TestTensor> tensors = semantic_tensors(test_case);
  const std::vector<hv::ReferenceTensor> references =
      reference_tensors(tensors);
  if constexpr (requires { test_case.input_precision; }) {
    if (test_case.input_precision == 2) {
      emit_skip(operation_name(test_case), test_case.name,
                "Hygon implements IEEE dot products; TF32 is unsupported",
                references);
      return CaseResult::kSkipped;
    }
  }
  const hv::HipdnnConvolutionOperation operation =
      reference_operation(test_case);
  const hv::HipdnnCapability structural =
      hv::hipdnn_convolution_capability(operation, references);
  hv::require_valid_hipdnn_adapter_contract(structural,
                                            operation_name(test_case));
  if (!structural.supported) {
    return run_host_convolution(test_case, get_handle, stream);
  }

  hv::HipdnnConvolutionPlan reference(
      operation, references,
      hv::HipdnnConvolutionAlgorithmPolicy::kCorrectnessOracle);
  const hv::HipdnnCapability setup = reference.capability();
  hv::require_valid_hipdnn_adapter_contract(setup, operation_name(test_case));
  if (!setup.supported) {
    return run_host_convolution(test_case, get_handle, stream);
  }

  PreparedBuffers reference_buffers =
      prepare_buffers(tensors, output_tensor(test_case).uid,
                      BindingAddress::kStorageBase, stream);
  hv::DeviceBuffer reference_workspace(reference.workspace_size());
  stream.synchronize();
  const hv::HipdnnCapability actual = reference.probe_execute(
      reference_buffers.bindings, reference_workspace.opaque(),
      reference.workspace_size(), stream.opaque());
  hv::require_valid_hipdnn_adapter_contract(actual, operation_name(test_case));
  if (!actual.supported) {
    return run_host_convolution(test_case, get_handle, stream);
  }
  const std::string_view selected_algorithm =
      reference.selected_algorithm_name();
  if (selected_algorithm.empty()) {
    throw std::logic_error(
        "hipDNN convolution oracle probe selected no algorithm");
  }
  std::cout << "[hipdnn] op=" << operation_name(test_case)
            << " case=" << test_case.name << " policy="
            << hv::hipdnn_convolution_algorithm_policy_name(reference.policy())
            << " algorithm=" << selected_algorithm << std::endl;

  flagdnn::Handle &handle = get_handle();
  std::unique_ptr<TestExecutable> flagdnn = build_flagdnn(handle, test_case);
  PreparedBuffers flagdnn_buffers =
      prepare_buffers(tensors, output_tensor(test_case).uid,
                      BindingAddress::kTensorEntrance, stream);
  hv::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  stream.synchronize();
  execute(*flagdnn, flagdnn_buffers.bindings, flagdnn_workspace, stream);
  stream.synchronize();
  const Accuracy accuracy =
      compare(read_output(flagdnn_buffers, stream, "FlagDNN"),
              read_output(reference_buffers, stream, "hipDNN"), test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs hipDNN public primitive"
            << (operation.kind == hv::HipdnnConvolutionKind::kConvBiasRelu
                    ? " sequence"
                    : "")
            << " PASS max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << std::endl;
  return CaseResult::kExecuted;
}

template <typename Case, typename ValidateSuiteCase>
int run_suite(int argc, char **argv, std::span<const Case> cases,
              std::string_view suite_name, const char *filter_environment,
              ValidateSuiteCase validate_suite_case) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0] << " COMPILER_EXECUTABLE COMPILER_ENTRY"
              << std::endl;
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    hv::DeviceGuard device;
    hv::Stream stream;
    std::unique_ptr<TemporaryCache> cache;
    std::unique_ptr<flagdnn::Handle> handle;
    const std::function<flagdnn::Handle &()> get_handle =
        [&]() -> flagdnn::Handle & {
      if (handle == nullptr) {
        cache = std::make_unique<TemporaryCache>();
        handle = std::make_unique<flagdnn::Handle>("hygon", 0);
        handle->set_compiler(argv[1], argv[2], cache->path().string());
      }
      return *handle;
    };

    const char *filter = std::getenv(filter_environment);
    std::size_t matched = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    for (const Case &test_case : cases) {
      validate_suite_case(test_case);
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++matched;
      const CaseResult result = run_case(test_case, get_handle, stream);
      result == CaseResult::kExecuted ? ++executed : ++skipped;
    }
    if (matched == 0) {
      throw std::runtime_error(std::string(filter_environment) +
                               " matched no test cases");
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

} // namespace flagdnn::testing::hygon_functional::convolution

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_FUNCTIONAL_CONVOLUTION_RUNNER_SUPPORT_HPP_
