/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/convolution.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_convolution.hpp"
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
constexpr std::uint8_t kAllocationGuard = 0xD3U;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

struct InputData {
  std::vector<std::uint8_t> encoded;
};

struct OutputData {
  std::vector<float> logical;
  std::vector<std::uint8_t> allocation;
};

std::vector<float> make_input(std::size_t count, std::size_t tensor_index) {
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17U + tensor_index * 11U) % 41U) - 20;
    result[index] =
        static_cast<float>(centered) / static_cast<float>(29U + tensor_index);
  }
  return result;
}

InputData make_input_data(const TestTensor& tensor,
                          std::size_t tensor_index) {
  const std::vector<float> raw =
      make_input(tensor_io::element_count(tensor), tensor_index);
  InputData result;
  result.encoded =
      tensor_io::encode(tensor_io::scatter(raw, tensor), tensor.data_type);
  return result;
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
        "convolution encoded tensor byte count differs");
  }
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "convolution guarded allocation");
  std::vector<std::uint8_t> initial(allocation, kAllocationGuard);
  std::copy(payload.begin(),
            payload.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(initial.data(), initial.size(), 0U, stream.get());
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
        std::string(provider) + " modified convolution " +
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
      "convolution guarded payload");
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
        " modified a convolution binding offset/tail guard");
  }
  const std::span<const std::uint8_t> payload(
      result.allocation.data() + begin, end - begin);
  const std::vector<float> physical =
      tensor_io::decode_storage(provider, payload, tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  result.logical = tensor_io::gather(physical, tensor);
  return result;
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void execute(ConvolutionExecutable& executable,
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

OutputData verify_provider(std::string_view provider,
                           GuardedBuffer& input,
                           GuardedBuffer& filter,
                           GuardedBuffer& output,
                           const TestTensor& output_tensor,
                           acl::Stream& stream) {
  require_input_unchanged(provider, "input", input, stream);
  require_input_unchanged(provider, "filter", filter, stream);
  return read_output(provider, output, output_tensor, stream);
}

std::optional<std::string> run_case(const ConvolutionTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  if (test_case.input_precision == 2)
    return "Ascend 910B ACLNN exposes HF32, not the CUDA TF32 input-precision "
           "contract";
  validate_convolution_case(test_case);
  const InputData input_data = make_input_data(test_case.x, 0U);
  const InputData filter_data = make_input_data(test_case.w, 1U);
  const AclnnConvolutionFpropPlan reference_plan =
      plan_aclnn_convolution_fprop(test_case);

  GuardedBuffer flagdnn_input =
      make_buffer(test_case.x, input_data.encoded, stream);
  GuardedBuffer flagdnn_filter =
      make_buffer(test_case.w, filter_data.encoded, stream);
  GuardedBuffer aclnn_input =
      make_buffer(reference_plan.input, input_data.encoded, stream);
  GuardedBuffer aclnn_filter =
      make_buffer(reference_plan.filter, filter_data.encoded, stream);
  const std::vector<std::uint8_t> output_payload =
      initial_output(test_case.y);
  GuardedBuffer flagdnn_output =
      make_buffer(test_case.y, output_payload, stream);
  GuardedBuffer aclnn_output =
      make_buffer(reference_plan.output, output_payload, stream);

  const std::array<flagdnnBinding_t, 3> flagdnn_bindings = {
      flagdnnBinding_t{
          test_case.x.uid,
          flagdnn_input.device->opaque_at(test_case.x.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.w.uid,
          flagdnn_filter.device->opaque_at(test_case.w.binding_byte_offset)},
      flagdnnBinding_t{
          test_case.y.uid,
          flagdnn_output.device->opaque_at(test_case.y.binding_byte_offset)}};
  const std::array<flagdnnBinding_t, 3> aclnn_bindings = {
      flagdnnBinding_t{
          reference_plan.input.uid,
          aclnn_input.device->opaque_at(
              reference_plan.input.binding_byte_offset)},
      flagdnnBinding_t{
          reference_plan.filter.uid,
          aclnn_filter.device->opaque_at(
              reference_plan.filter.binding_byte_offset)},
      flagdnnBinding_t{
          reference_plan.output.uid,
          aclnn_output.device->opaque_at(
              reference_plan.output.binding_byte_offset)}};

  std::unique_ptr<ConvolutionExecutable> flagdnn =
      build_flagdnn_convolution(handle, test_case);
  std::unique_ptr<ConvolutionExecutable> reference =
      build_convolution_reference(test_case);
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnConvolutionUnsupportedError& error) {
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
      restore(flagdnn_input, stream);
      restore(flagdnn_filter, stream);
      restore(flagdnn_output, stream);
      restore(aclnn_input, stream);
      restore(aclnn_filter, stream);
      restore(aclnn_output, stream);
      execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
      execute(*reference, aclnn_bindings, aclnn_workspace, stream);
      stream.synchronize();
      OutputData flagdnn_result = verify_provider("FlagDNN",
                                                   flagdnn_input,
                                                   flagdnn_filter,
                                                   flagdnn_output,
                                                   test_case.y,
                                                   stream);
      OutputData aclnn_result = verify_provider("ACLNN",
                                                aclnn_input,
                                                aclnn_filter,
                                                aclnn_output,
                                                reference_plan.output,
                                                stream);
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
            test_case.name + " repeatable execution changed output bytes");
      }
      std::cout << test_case.name << ": pass=" << (pass + 1)
                << " exact_reference=ACLNN_CONVOLUTION\n";
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
            << ": PASS repeatable=2 input_guard=PASS "
               "output_padding=PASS\n";
  return std::nullopt;
}

}  // namespace

int run_convolution_functional_test(
    int argc,
    char** argv,
    std::span<const ConvolutionTestCase> cases,
    ConvolutionDirection expected_direction) {
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    std::cout << std::setprecision(9);
    if (expected_direction != ConvolutionDirection::kFprop || cases.empty()) {
      throw std::logic_error(
          "Ascend convolution fprop functional catalog must contain cases");
    }
    acl::DevelopmentEnvironment development("convolution-fprop-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';

    const char* filter = std::getenv("FLAGDNN_CONVOLUTION_CASE");
    std::vector<const ConvolutionTestCase*> selected;
    selected.reserve(cases.size());
    for (const ConvolutionTestCase& test_case : cases) {
      if (test_case.direction != expected_direction) {
        throw std::invalid_argument(
            "convolution suite contains the wrong direction");
      }
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "convolution case filter matched no test cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      {
        flagdnn::Handle handle("ascend", 0);
        handle.set_compiler(
            argv[1], argv[2], development.graph_cache().string());
        for (const ConvolutionTestCase* test_case : selected) {
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
    std::cout << "FLAGDNN_CONVOLUTION_FUNCTIONAL: PASS cases="
              << coverage.passed() << " catalog_cases=" << cases.size()
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FLAGDNN_CONVOLUTION_FUNCTIONAL_FAILED: " << error.what()
              << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
