/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/batchnorm_inference_validation.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_batchnorm_inference.hpp"
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

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kAllocationGuard = 0xD7U;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

struct Inputs {
  std::vector<float> x;
  std::vector<float> mean;
  std::vector<float> invstd;
  std::vector<float> scale;
  std::vector<float> bias;
};

std::vector<float> quantize(std::span<const float> logical,
                            const TestTensor& tensor) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(tensor_io::scatter(logical, tensor), tensor.data_type);
  return tensor_io::gather(
      tensor_io::decode(encoded,
                        tensor.data_type,
                        tensor_io::storage_element_count(tensor)),
      tensor);
}

Inputs make_inputs(const BatchnormInferenceTestCase& test_case) {
  const std::size_t channels =
      static_cast<std::size_t>(test_case.x.dimensions[1]);
  Inputs result;
  result.mean.resize(channels);
  result.invstd.resize(channels);
  result.scale.resize(channels);
  result.bias.resize(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    switch (channel % 5) {
      case 0:
        result.mean[channel] = 1024.0F;
        result.invstd[channel] = 1.0e-5F;
        break;
      case 1:
        result.mean[channel] = -512.0F;
        result.invstd[channel] = 64.0F;
        break;
      case 2:
        result.mean[channel] = 3.0F;
        result.invstd[channel] = 0.5F;
        break;
      case 3:
        result.mean[channel] = 0.25F;
        result.invstd[channel] = 2048.0F;
        break;
      case 4:
        result.mean[channel] = -11.0F;
        result.invstd[channel] = 3.0F;
        break;
    }
    result.scale[channel] =
        static_cast<float>(static_cast<int>(channel % 7) - 3) * 0.25F + 1.0F;
    result.bias[channel] =
        static_cast<float>(static_cast<int>(channel % 11) - 5) * 0.125F;
  }
  result.x.resize(tensor_io::element_count(test_case.x));
  std::size_t inner = 1;
  for (std::size_t axis = 2; axis < test_case.x.dimensions.size(); ++axis) {
    inner *= static_cast<std::size_t>(test_case.x.dimensions[axis]);
  }
  for (std::size_t index = 0; index < result.x.size(); ++index) {
    const std::size_t channel = (index / inner) % channels;
    const int centered = static_cast<int>((index * 17 + channel * 13) % 31) - 15;
    result.x[index] = result.mean[channel] +
                      static_cast<float>(centered) * 0.0625F;
  }
  result.x = quantize(result.x, test_case.x);
  result.mean = quantize(result.mean, test_case.mean);
  result.invstd = quantize(result.invstd, test_case.inv_variance);
  result.scale = quantize(result.scale, test_case.scale);
  result.bias = quantize(result.bias, test_case.bias);
  return result;
}

GuardedBuffer make_buffer(const TestTensor& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      tensor_io::encode(physical, tensor.data_type);
  const std::size_t allocation = tensor_io::checked_add(
      tensor_io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "BatchNorm inference guarded allocation");
  std::vector<std::uint8_t> initial(allocation, kAllocationGuard);
  std::copy(encoded.begin(),
            encoded.end(),
            initial.begin() +
                static_cast<std::ptrdiff_t>(tensor.binding_byte_offset));
  auto device = std::make_unique<acl::DeviceBuffer>(allocation);
  device->copy_from_host_at(initial.data(), initial.size(), 0, stream.get());
  return {std::move(device), std::move(initial)};
}

GuardedBuffer make_input_buffer(const TestTensor& tensor,
                                std::span<const float> logical,
                                acl::Stream& stream) {
  return make_buffer(tensor, tensor_io::scatter(logical, tensor), stream);
}

GuardedBuffer make_output_buffer(const TestTensor& tensor,
                                 acl::Stream& stream) {
  return make_buffer(
      tensor,
      std::vector<float>(tensor_io::storage_element_count(tensor),
                         tensor_io::kPaddingSentinel),
      stream);
}

std::vector<std::uint8_t> read_all(const GuardedBuffer& buffer,
                                   acl::Stream& stream) {
  std::vector<std::uint8_t> result(buffer.initial.size());
  buffer.device->copy_to_host_at(result.data(), result.size(), 0, stream.get());
  stream.synchronize();
  return result;
}

void require_unchanged(std::string_view provider,
                       std::string_view name,
                       const GuardedBuffer& buffer,
                       acl::Stream& stream) {
  if (read_all(buffer, stream) != buffer.initial) {
    throw std::runtime_error(std::string(provider) + " modified BatchNorm " +
                             std::string(name) + " input or guards");
  }
}

std::vector<float> read_output(std::string_view provider,
                               const GuardedBuffer& buffer,
                               const TestTensor& tensor,
                               acl::Stream& stream) {
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
    throw std::runtime_error(std::string(provider) +
                             " modified BatchNorm binding guards");
  }
  const std::vector<float> physical = tensor_io::decode_storage(
      provider,
      std::span<const std::uint8_t>(all.data() + begin, end - begin),
      tensor);
  tensor_io::require_padding_unchanged(provider, physical, tensor);
  return tensor_io::gather(physical, tensor);
}

std::string single_line_reason(std::string_view reason) {
  std::string result(reason);
  std::replace(result.begin(), result.end(), '\r', ' ');
  std::replace(result.begin(), result.end(), '\n', ' ');
  return result;
}

void execute(NormalizationExecutable& executable,
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

void expect_failure(const auto& callback, std::string_view needle) {
  try {
    callback();
  } catch (const std::exception& error) {
    if (std::string_view(error.what()).find(needle) != std::string_view::npos) {
      return;
    }
    throw;
  }
  throw std::runtime_error(
      "ACLNN BatchNorm inference lifecycle violation was accepted");
}

std::optional<std::string> run_case(
    const BatchnormInferenceTestCase& test_case,
    flagdnn::Handle& handle,
    acl::Stream& stream) {
  const Inputs inputs = make_inputs(test_case);
  BatchnormInferenceTestCase reference_case = test_case;
  reference_case.x =
      acl::batchnorm_inference_reference_data_tensor(test_case.x);
  reference_case.y =
      acl::batchnorm_inference_reference_data_tensor(test_case.y);
  auto make_all_buffers = [&](const BatchnormInferenceTestCase& buffer_case) {
    std::vector<GuardedBuffer> result;
    result.push_back(make_input_buffer(buffer_case.x, inputs.x, stream));
    result.push_back(make_input_buffer(buffer_case.mean, inputs.mean, stream));
    result.push_back(
        make_input_buffer(buffer_case.inv_variance, inputs.invstd, stream));
    result.push_back(
        make_input_buffer(buffer_case.scale, inputs.scale, stream));
    result.push_back(make_input_buffer(buffer_case.bias, inputs.bias, stream));
    result.push_back(make_output_buffer(buffer_case.y, stream));
    return result;
  };
  std::vector<GuardedBuffer> flagdnn_buffers = make_all_buffers(test_case);
  std::vector<GuardedBuffer> aclnn_buffers = make_all_buffers(reference_case);
  auto make_bindings = [&](std::vector<GuardedBuffer>& buffers,
                           const BatchnormInferenceTestCase& binding_case) {
    const std::array<const TestTensor*, 6> tensors = {
        &binding_case.x,
        &binding_case.mean,
        &binding_case.inv_variance,
        &binding_case.scale,
        &binding_case.bias,
        &binding_case.y,
    };
    std::vector<flagdnnBinding_t> result;
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      result.push_back({
          tensors[index]->uid,
          buffers[index].device->opaque_at(tensors[index]->binding_byte_offset),
      });
    }
    return result;
  };
  std::vector<flagdnnBinding_t> flagdnn_bindings =
      make_bindings(flagdnn_buffers, test_case);
  std::vector<flagdnnBinding_t> aclnn_bindings =
      make_bindings(aclnn_buffers, reference_case);
  std::unique_ptr<NormalizationExecutable> flagdnn =
      build_flagdnn_batchnorm_inference(handle, test_case);
  std::unique_ptr<NormalizationExecutable> reference =
      build_batchnorm_inference_reference(reference_case);
  std::unique_ptr<NormalizationExecutable> unprepared =
      build_batchnorm_inference_reference(reference_case);
  expect_failure(
      [&] { unprepared->execute(aclnn_bindings, nullptr, 0, stream.opaque()); },
      "before prepare");
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnBatchnormInferenceUnsupportedError& error) {
    try {
      stream.synchronize();
    } catch (...) {
    }
    reference.reset();
    flagdnn.reset();
    return "ACLNN status=" + std::to_string(error.status()) + " " +
           single_line_reason(error.what());
  }
  flagdnn->prepare(flagdnn_bindings, stream.opaque());
  acl::DeviceBuffer flagdnn_workspace(flagdnn->workspace_size());
  acl::DeviceBuffer aclnn_workspace(reference->workspace_size());

  std::vector<flagdnnBinding_t> changed = aclnn_bindings;
  changed[0].device_pointer = aclnn_bindings[5].device_pointer;
  expect_failure(
      [&] {
        reference->execute(changed,
                           reference->workspace_size() == 0
                               ? nullptr
                               : aclnn_workspace.opaque(),
                           reference->workspace_size(),
                           stream.opaque());
      },
      "address changed");
  expect_failure(
      [&] {
        reference->execute(aclnn_bindings,
                           reference->workspace_size() == 0
                               ? nullptr
                               : aclnn_workspace.opaque(),
                           reference->workspace_size(),
                           nullptr);
      },
      "stream changed");
  if (reference->workspace_size() != 0) {
    expect_failure(
        [&] {
          reference->execute(aclnn_bindings,
                             aclnn_workspace.opaque(),
                             reference->workspace_size() - 1,
                             stream.opaque());
        },
        "workspace is too small");
  }
  std::vector<flagdnnBinding_t> bad_prepare = aclnn_bindings;
  bad_prepare[0].device_pointer = static_cast<void*>(
      static_cast<std::uint8_t*>(bad_prepare[0].device_pointer) + 1);
  expect_failure(
      [&] { reference->prepare(bad_prepare, stream.opaque()); },
      "not 32-byte aligned");

  for (int pass = 0; pass < 2; ++pass) {
    execute(*flagdnn,
            flagdnn_bindings,
            flagdnn_workspace,
            stream);
    execute(*reference,
            aclnn_bindings,
            aclnn_workspace,
            stream);
  }
  stream.synchronize();
  const std::vector<float> flagdnn_output =
      read_output("FlagDNN", flagdnn_buffers[5], test_case.y, stream);
  const std::vector<float> aclnn_output =
      read_output("ACLNN", aclnn_buffers[5], reference_case.y, stream);
  for (std::size_t index = 0; index < 5; ++index) {
    static constexpr std::array<std::string_view, 5> names = {
        "X", "mean", "invstd", "scale", "bias"};
    require_unchanged("FlagDNN", names[index], flagdnn_buffers[index], stream);
    require_unchanged("ACLNN", names[index], aclnn_buffers[index], stream);
  }
  acl::compare_exact_reference(flagdnn_output,
                               aclnn_output,
                               test_case.absolute_tolerance,
                               test_case.relative_tolerance,
                               test_case.name);
  std::cout << test_case.name
            << ": PASS exact_reference=ACLNN_BATCHNORM_INFERENCE\n";
  stream.synchronize();
  reference.reset();
  flagdnn.reset();
  return std::nullopt;
}

}  // namespace

int run_batchnorm_inference_functional_test(
    int argc,
    char** argv,
    std::span<const BatchnormInferenceTestCase> cases) {
  constexpr std::string_view suite =
      "FLAGDNN_BATCHNORM_INFERENCE_FUNCTIONAL";
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    if (cases.empty()) {
      throw std::invalid_argument(
          "common BatchNorm inference catalog must contain cases");
    }
    const std::vector<BatchnormInferenceTestCase> ascend_cases =
        acl::make_ascend_batchnorm_inference_cases(cases);
    acl::DevelopmentEnvironment development("batchnorm-inference-functional");
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
    std::cout << std::setprecision(9);
    const char* filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::vector<const BatchnormInferenceTestCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const BatchnormInferenceTestCase& test_case : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error(
          "BatchNorm inference filter matched no cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      flagdnn::Handle handle("ascend", 0);
      handle.set_compiler(
          argv[1], argv[2], development.graph_cache().string());
      for (const BatchnormInferenceTestCase* test_case : selected) {
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
    runtime.finalize();
    development.cleanup();
    coverage.require_complete();
    for (const std::string& line : coverage.skip_lines()) {
      std::cout << line << '\n';
    }
    std::cout << coverage.summary() << '\n';
    std::cout << suite << ": PASS cases=" << coverage.passed()
              << " catalog_cases=" << ascend_cases.size() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << suite << "_FAILED: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
