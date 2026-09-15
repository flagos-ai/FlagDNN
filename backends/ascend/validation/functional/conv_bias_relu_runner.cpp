/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/composite.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_conv_bias_relu.hpp"
#include "validation/tensor_io.hpp"

#include <flagdnn/flagdnn.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace tensor_io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kTailGuardBytes = 32U;
constexpr std::uint8_t kAllocationGuard = 0xC7U;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

struct OutputData {
  std::vector<float> logical;
  std::vector<std::uint8_t> allocation;
};

std::vector<float> make_input(const TestTensor& tensor,
                              std::size_t tensor_index) {
  std::vector<float> result(tensor_io::element_count(tensor));
  for (std::size_t index = 0U; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 29U + tensor_index * 13U) % 53U) - 26;
    result[index] = static_cast<float>(centered) /
                    static_cast<float>(31U + tensor_index * 3U);
  }
  return tensor_io::quantize(result, tensor.data_type);
}

std::vector<std::uint8_t> encode_input(std::span<const float> logical,
                                       const TestTensor& tensor) {
  if (logical.size() != tensor_io::element_count(tensor)) {
    throw std::logic_error(
        "ConvBiasRelu provider input logical size differs");
  }
  return tensor_io::encode(
      tensor_io::scatter(logical, tensor), tensor.data_type);
}

std::vector<std::uint8_t> initial_output(const TestTensor& tensor) {
  return tensor_io::encode(
      std::vector<float>(tensor_io::storage_element_count(tensor),
                         tensor_io::kPaddingSentinel),
      tensor.data_type);
}

GuardedBuffer make_buffer(const TestTensor& tensor,
                          std::span<const std::uint8_t> payload,
                          acl::Stream& stream) {
  if (payload.size() != tensor_io::encoded_byte_count(tensor)) {
    throw std::logic_error(
        "ConvBiasRelu encoded tensor byte count differs");
  }
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "ConvBiasRelu guarded allocation");
  std::vector<std::uint8_t> initial(allocation, kAllocationGuard);
  std::copy(payload.begin(),
            payload.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(
      initial.data(), initial.size(), 0U, stream.get());
  return {std::move(device), std::move(initial)};
}

void restore(GuardedBuffer& buffer, acl::Stream& stream) {
  buffer.device->copy_from_host_at(
      buffer.initial.data(), buffer.initial.size(), 0U, stream.get());
}

std::vector<std::uint8_t> read_all(const GuardedBuffer& buffer,
                                   acl::Stream& stream) {
  std::vector<std::uint8_t> result(buffer.initial.size());
  buffer.device->copy_to_host_at(
      result.data(), result.size(), 0U, stream.get());
  stream.synchronize();
  return result;
}

void require_input_unchanged(std::string_view provider,
                             std::string_view role,
                             const GuardedBuffer& buffer,
                             acl::Stream& stream) {
  if (read_all(buffer, stream) != buffer.initial) {
    throw std::runtime_error(
        std::string(provider) + " modified ConvBiasRelu " +
        std::string(role) + " storage or guards");
  }
}

OutputData read_output(std::string_view provider,
                       const GuardedBuffer& buffer,
                       const TestTensor& tensor,
                       acl::Stream& stream) {
  OutputData result;
  result.allocation = read_all(buffer, stream);
  const std::size_t begin = tensor.binding_byte_offset;
  const std::size_t end = tensor_io::checked_add(
      begin,
      tensor_io::encoded_byte_count(tensor),
      "ConvBiasRelu guarded payload");
  if (end > result.allocation.size() ||
      !std::equal(result.allocation.begin(),
                  result.allocation.begin() +
                      static_cast<std::ptrdiff_t>(begin),
                  buffer.initial.begin()) ||
      !std::equal(result.allocation.begin() +
                      static_cast<std::ptrdiff_t>(end),
                  result.allocation.end(),
                  buffer.initial.begin() +
                      static_cast<std::ptrdiff_t>(end))) {
    throw std::runtime_error(
        std::string(provider) +
        " modified a ConvBiasRelu binding offset/tail guard");
  }
  const std::span<const std::uint8_t> payload(
      result.allocation.data() + begin, end - begin);
  const std::vector<float> physical =
      tensor_io::decode_storage(provider, payload, tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  result.logical = tensor_io::gather(physical, tensor);
  return result;
}

void execute(CompositeExecutable& executable,
             std::span<const flagdnnBinding_t> bindings,
             acl::DeviceBuffer& workspace,
             acl::Stream& stream) {
  executable.execute(bindings,
                     executable.workspace_size() == 0U
                         ? nullptr
                         : workspace.opaque(),
                     executable.workspace_size(),
                     stream.opaque());
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  if (result.empty()) {
    result = "exact ACLNN composition is unsupported";
  }
  return result;
}

std::optional<std::string> run_case(const ConvBiasReluTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  validate_composite_case(test_case);
  const AclnnConvBiasReluPlan reference_plan =
      plan_aclnn_conv_bias_relu(test_case);

  const std::vector<float> x_logical = make_input(test_case.x, 0U);
  const std::vector<float> w_logical = make_input(test_case.w, 1U);
  const std::vector<float> bias_logical = make_input(test_case.bias, 2U);

  GuardedBuffer flagdnn_x =
      make_buffer(test_case.x, encode_input(x_logical, test_case.x), stream);
  GuardedBuffer flagdnn_w =
      make_buffer(test_case.w, encode_input(w_logical, test_case.w), stream);
  GuardedBuffer flagdnn_bias = make_buffer(
      test_case.bias, encode_input(bias_logical, test_case.bias), stream);
  GuardedBuffer aclnn_x = make_buffer(
      reference_plan.x, encode_input(x_logical, reference_plan.x), stream);
  GuardedBuffer aclnn_w = make_buffer(
      reference_plan.w, encode_input(w_logical, reference_plan.w), stream);
  GuardedBuffer aclnn_bias = make_buffer(
      reference_plan.bias,
      encode_input(bias_logical, reference_plan.bias),
      stream);
  GuardedBuffer flagdnn_output =
      make_buffer(test_case.output, initial_output(test_case.output), stream);
  GuardedBuffer aclnn_output = make_buffer(
      reference_plan.output, initial_output(reference_plan.output), stream);

  const std::array<flagdnnBinding_t, 4> flagdnn_bindings = {
      flagdnnBinding_t{
          test_case.x.uid,
          flagdnn_x.device->opaque_at(test_case.x.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.w.uid,
          flagdnn_w.device->opaque_at(test_case.w.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.bias.uid,
          flagdnn_bias.device->opaque_at(test_case.bias.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.output.uid,
          flagdnn_output.device->opaque_at(
              test_case.output.binding_byte_offset)}};
  const std::array<flagdnnBinding_t, 4> aclnn_bindings = {
      flagdnnBinding_t{
          reference_plan.x.uid,
          aclnn_x.device->opaque_at(reference_plan.x.binding_byte_offset)},
      flagdnnBinding_t{
          reference_plan.w.uid,
          aclnn_w.device->opaque_at(reference_plan.w.binding_byte_offset)},
      flagdnnBinding_t{
          reference_plan.bias.uid,
          aclnn_bias.device->opaque_at(
              reference_plan.bias.binding_byte_offset)},
      flagdnnBinding_t{
          reference_plan.output.uid,
          aclnn_output.device->opaque_at(
              reference_plan.output.binding_byte_offset)}};

  std::unique_ptr<CompositeExecutable> flagdnn =
      build_flagdnn_conv_bias_relu(handle, test_case);
  std::unique_ptr<CompositeExecutable> reference =
      build_conv_bias_relu_reference(test_case);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnConvBiasReluUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }

  acl::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  acl::DeviceBuffer aclnn_workspace(reference->workspace_size());
  std::vector<std::uint8_t> previous_flagdnn;
  std::vector<std::uint8_t> previous_aclnn;
  try {
    for (int pass = 0; pass < 2; ++pass) {
      restore(flagdnn_x, stream);
      restore(flagdnn_w, stream);
      restore(flagdnn_bias, stream);
      restore(flagdnn_output, stream);
      restore(aclnn_x, stream);
      restore(aclnn_w, stream);
      restore(aclnn_bias, stream);
      restore(aclnn_output, stream);

      execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
      execute(*reference, aclnn_bindings, aclnn_workspace, stream);
      stream.synchronize();

      require_input_unchanged(
          "FlagDNN", "x", flagdnn_x, stream);
      require_input_unchanged(
          "FlagDNN", "w", flagdnn_w, stream);
      require_input_unchanged(
          "FlagDNN", "bias", flagdnn_bias, stream);
      require_input_unchanged("ACLNN", "x", aclnn_x, stream);
      require_input_unchanged("ACLNN", "w", aclnn_w, stream);
      require_input_unchanged("ACLNN", "bias", aclnn_bias, stream);

      OutputData flagdnn_result = read_output(
          "FlagDNN", flagdnn_output, test_case.output, stream);
      OutputData aclnn_result = read_output(
          "ACLNN", aclnn_output, reference_plan.output, stream);
      acl::compare_exact_reference(flagdnn_result.logical,
                                   aclnn_result.logical,
                                   test_case.absolute_tolerance,
                                   test_case.relative_tolerance,
                                   test_case.name);
      if (pass == 0) {
        previous_flagdnn = flagdnn_result.allocation;
        previous_aclnn = aclnn_result.allocation;
      } else if (flagdnn_result.allocation != previous_flagdnn ||
                 aclnn_result.allocation != previous_aclnn) {
        throw std::runtime_error(
            test_case.name +
            " repeatable execution changed ConvBiasRelu output bytes");
      }
      std::cout << test_case.name << ": pass=" << (pass + 1)
                << " exact_reference=ACLNN_CONV_ADD_RELU\n";
    }
    stream.synchronize();
  } catch (...) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    throw;
  }
  reference.reset();
  flagdnn.reset();
  std::cout << test_case.name
            << ": PASS repeatable=2 input_guard=PASS output_padding=PASS "
               "exact_reference=ACLNN_CONV_ADD_RELU\n";
  return std::nullopt;
}

}  // namespace

int run_conv_bias_relu_functional_test(
    int argc,
    char** argv,
    std::span<const ConvBiasReluTestCase> cases) {
  constexpr std::string_view kSuite =
      "FLAGDNN_CONV_BIAS_RELU_FUNCTIONAL";
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    if (cases.empty()) {
      throw std::logic_error(
          "Ascend ConvBiasRelu functional catalog must contain cases");
    }
    const char* filter = std::getenv("FLAGDNN_ASCEND_COMPOSITE_CASE");
    std::vector<const ConvBiasReluTestCase*> selected;
    for (const ConvBiasReluTestCase& test_case : cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "FLAGDNN_ASCEND_COMPOSITE_CASE matched no ConvBiasRelu cases");
    }

    std::cout << std::setprecision(9);
    acl::DevelopmentEnvironment development("conv-bias-relu-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';

    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const ConvBiasReluTestCase* test_case : selected) {
          const std::optional<std::string> skip =
              run_case(*test_case, handle, stream);
          if (skip.has_value()) {
            coverage.record_skip(test_case->name, *skip);
          } else {
            coverage.record_pass(test_case->name);
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
    std::cout << kSuite << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << kSuite << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
