/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/batchnorm_validation.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_batchnorm.hpp"
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
#include <vector>

namespace flagdnn::testing {
namespace {

namespace acl = flagdnn::validation::ascend;
namespace io = flagdnn::validation::ascend::tensor_io;

constexpr std::size_t kTailGuardBytes = 32;
constexpr std::uint8_t kAllocationGuard = 0xD7U;

struct GuardedBuffer {
  std::unique_ptr<acl::DeviceBuffer> device;
  std::vector<std::uint8_t> initial;
};

struct Inputs {
  std::vector<float> x;
  std::vector<float> scale;
  std::vector<float> bias;
  std::vector<float> previous_mean;
  std::vector<float> previous_variance;
};

struct Outputs {
  std::vector<float> y;
  std::vector<float> mean;
  std::vector<float> invstd;
  std::vector<float> next_mean;
  std::vector<float> next_variance;
};

std::vector<float> quantize(std::span<const float> logical,
                            const TestTensor& tensor) {
  const std::vector<std::uint8_t> encoded =
      io::encode(io::scatter(logical, tensor), tensor.data_type);
  return io::gather(io::decode(encoded,
                               tensor.data_type,
                               io::storage_element_count(tensor)),
                    tensor);
}

Inputs make_inputs(const BatchnormTestCase& test_case) {
  const std::size_t channels =
      static_cast<std::size_t>(test_case.x.dimensions[1]);
  std::size_t spatial = 1;
  for (std::size_t axis = 2; axis < test_case.x.dimensions.size(); ++axis) {
    spatial = io::checked_multiply(
        spatial,
        static_cast<std::size_t>(test_case.x.dimensions[axis]),
        "BatchNorm functional spatial elements");
  }
  Inputs result;
  result.scale.resize(channels);
  result.bias.resize(channels);
  result.previous_mean.resize(channels);
  result.previous_variance.resize(channels);
  for (std::size_t channel = 0; channel < channels; ++channel) {
    result.scale[channel] =
        channel % 3U == 0U ? -1.5F : 0.75F + 0.125F * channel;
    result.bias[channel] =
        static_cast<float>(static_cast<int>(channel % 5U) - 2) * 0.25F;
    result.previous_mean[channel] =
        static_cast<float>(static_cast<int>(channel) - 4) * 2.0F;
    result.previous_variance[channel] = 0.5F + 0.75F * channel;
  }
  result.x.resize(io::element_count(test_case.x));
  for (std::size_t index = 0; index < result.x.size(); ++index) {
    const std::size_t channel = (index / spatial) % channels;
    const float center = channel % 2U == 0U ? 32.0F : -16.0F;
    const int code =
        static_cast<int>((index * 17U + channel * 13U) % 29U) - 14;
    result.x[index] = center + static_cast<float>(code) * 0.125F;
  }
  result.x = quantize(result.x, test_case.x);
  result.scale = quantize(result.scale, test_case.scale);
  result.bias = quantize(result.bias, test_case.bias);
  result.previous_mean =
      quantize(result.previous_mean, test_case.previous_running_mean);
  result.previous_variance =
      quantize(result.previous_variance, test_case.previous_running_variance);
  return result;
}

GuardedBuffer make_buffer(const TestTensor& tensor,
                          std::span<const float> physical,
                          acl::Stream& stream) {
  const std::vector<std::uint8_t> encoded =
      io::encode(physical, tensor.data_type);
  const std::size_t allocation = io::checked_add(
      io::allocation_byte_count(tensor),
      kTailGuardBytes,
      "BatchNorm functional guarded allocation");
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
  return make_buffer(tensor, io::scatter(logical, tensor), stream);
}

GuardedBuffer make_output_buffer(const TestTensor& tensor,
                                 acl::Stream& stream) {
  return make_buffer(
      tensor,
      std::vector<float>(io::storage_element_count(tensor),
                         io::kPaddingSentinel),
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
    throw std::runtime_error(std::string(provider) +
                             " modified BatchNorm " + std::string(name) +
                             " input or guards");
  }
}

std::vector<float> read_output(std::string_view provider,
                               const GuardedBuffer& buffer,
                               const TestTensor& tensor,
                               acl::Stream& stream) {
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
                             " modified BatchNorm binding guards");
  }
  const std::vector<float> physical = io::decode_storage(
      provider,
      std::span<const std::uint8_t>(all.data() + begin, end - begin),
      tensor);
  io::require_padding_unchanged(provider, physical, tensor);
  return io::gather(physical, tensor);
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
  throw std::runtime_error("ACLNN BatchNorm lifecycle violation was accepted");
}

std::vector<GuardedBuffer> make_buffers(const BatchnormTestCase& test_case,
                                        const Inputs& inputs,
                                        acl::Stream& stream) {
  std::vector<GuardedBuffer> result;
  result.push_back(make_input_buffer(test_case.x, inputs.x, stream));
  result.push_back(make_input_buffer(test_case.scale, inputs.scale, stream));
  result.push_back(make_input_buffer(test_case.bias, inputs.bias, stream));
  result.push_back(make_input_buffer(
      test_case.previous_running_mean, inputs.previous_mean, stream));
  result.push_back(make_input_buffer(
      test_case.previous_running_variance, inputs.previous_variance, stream));
  result.push_back(make_output_buffer(test_case.y, stream));
  result.push_back(make_output_buffer(test_case.mean, stream));
  result.push_back(make_output_buffer(test_case.inv_variance, stream));
  result.push_back(make_output_buffer(test_case.next_running_mean, stream));
  result.push_back(make_output_buffer(test_case.next_running_variance, stream));
  return result;
}

std::vector<flagdnnBinding_t> make_bindings(
    std::vector<GuardedBuffer>& buffers,
    const BatchnormTestCase& test_case) {
  const std::array<const TestTensor*, 10> tensors = {
      &test_case.x,
      &test_case.scale,
      &test_case.bias,
      &test_case.previous_running_mean,
      &test_case.previous_running_variance,
      &test_case.y,
      &test_case.mean,
      &test_case.inv_variance,
      &test_case.next_running_mean,
      &test_case.next_running_variance,
  };
  std::vector<flagdnnBinding_t> result;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    result.push_back({
        tensors[index]->uid,
        buffers[index].device->opaque_at(tensors[index]->binding_byte_offset),
    });
  }
  return result;
}

Outputs read_outputs(std::string_view provider,
                     const std::vector<GuardedBuffer>& buffers,
                     const BatchnormTestCase& test_case,
                     acl::Stream& stream) {
  return {
      read_output(provider, buffers[5], test_case.y, stream),
      read_output(provider, buffers[6], test_case.mean, stream),
      read_output(provider, buffers[7], test_case.inv_variance, stream),
      read_output(provider, buffers[8], test_case.next_running_mean, stream),
      read_output(provider, buffers[9], test_case.next_running_variance, stream),
  };
}

std::optional<std::string> run_case(const BatchnormTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  const Inputs inputs = make_inputs(test_case);
  BatchnormTestCase reference_case = test_case;
  reference_case.x = acl::batchnorm_reference_data_tensor(test_case.x);
  reference_case.y = acl::batchnorm_reference_data_tensor(test_case.y);
  std::vector<GuardedBuffer> flagdnn_buffers =
      make_buffers(test_case, inputs, stream);
  std::vector<GuardedBuffer> aclnn_buffers =
      make_buffers(reference_case, inputs, stream);
  std::vector<flagdnnBinding_t> flagdnn_bindings =
      make_bindings(flagdnn_buffers, test_case);
  std::vector<flagdnnBinding_t> aclnn_bindings =
      make_bindings(aclnn_buffers, reference_case);

  std::unique_ptr<NormalizationExecutable> flagdnn =
      build_flagdnn_batchnorm(handle, test_case);
  std::unique_ptr<NormalizationExecutable> reference =
      build_batchnorm_reference(reference_case);
  std::unique_ptr<NormalizationExecutable> unprepared =
      build_batchnorm_reference(reference_case);
  expect_failure(
      [&] { unprepared->execute(aclnn_bindings, nullptr, 0, stream.opaque()); },
      "before prepare");
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnBatchnormUnsupportedError& error) {
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

  for (int pass = 0; pass < 2; ++pass) {
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    execute(*reference, aclnn_bindings, aclnn_workspace, stream);
  }
  stream.synchronize();
  const Outputs flagdnn_output =
      read_outputs("FlagDNN", flagdnn_buffers, test_case, stream);
  Outputs aclnn_output =
      read_outputs("ACLNN", aclnn_buffers, reference_case, stream);
  acl::batchnorm_reference_variance_to_invstd(
      aclnn_output.invstd, test_case.epsilon);
  static constexpr std::array<std::string_view, 5> input_names = {
      "X", "scale", "bias", "previous mean", "previous variance"};
  for (std::size_t index = 0; index < input_names.size(); ++index) {
    require_unchanged(
        "FlagDNN", input_names[index], flagdnn_buffers[index], stream);
    require_unchanged(
        "ACLNN", input_names[index], aclnn_buffers[index], stream);
  }

  static constexpr std::array<std::string_view, 5> output_names = {
      "Y", "mean", "invstd", "next mean", "next variance"};
  const std::array<const std::vector<float>*, 5> flagdnn_values = {
      &flagdnn_output.y,
      &flagdnn_output.mean,
      &flagdnn_output.invstd,
      &flagdnn_output.next_mean,
      &flagdnn_output.next_variance,
  };
  const std::array<const std::vector<float>*, 5> aclnn_values = {
      &aclnn_output.y,
      &aclnn_output.mean,
      &aclnn_output.invstd,
      &aclnn_output.next_mean,
      &aclnn_output.next_variance,
  };
  for (std::size_t index = 0; index < output_names.size(); ++index) {
    acl::compare_exact_reference(
        *flagdnn_values[index],
        *aclnn_values[index],
        test_case.absolute_tolerance,
        test_case.relative_tolerance,
        test_case.name + ":" + std::string(output_names[index]));
  }
  std::cout << test_case.name
            << ": PASS exact_reference=ACLNN_BATCHNORM outputs=5\n";
  stream.synchronize();
  reference.reset();
  flagdnn.reset();
  return std::nullopt;
}

}  // namespace

int run_batchnorm_functional_test(int argc,
                                  char** argv,
                                  std::span<const BatchnormTestCase> cases) {
  constexpr std::string_view suite = "FLAGDNN_BATCHNORM_FUNCTIONAL";
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    if (cases.empty()) {
      throw std::invalid_argument(
          "common BatchNorm functional catalog must contain cases");
    }
    const std::vector<BatchnormTestCase> ascend_cases =
        acl::make_ascend_batchnorm_cases(cases);
    acl::DevelopmentEnvironment development("batchnorm-functional");
    acl::AclRuntime runtime;
    development.prepare_target(acl::soc_name());
    std::cout << std::setprecision(9);
    const char* filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::vector<const BatchnormTestCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const BatchnormTestCase& test_case : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error("BatchNorm filter matched no cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      flagdnn::Handle handle("ascend", 0);
      handle.set_compiler(argv[1], argv[2], development.graph_cache().string());
      for (const BatchnormTestCase* test_case : selected) {
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
