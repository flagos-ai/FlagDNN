/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "common/normalization.hpp"

#include "validation/acl_runtime.hpp"
#include "validation/development_environment.hpp"
#include "validation/exact_reference.hpp"
#include "validation/functional/aclnn_rmsnorm.hpp"
#include "validation/rmsnorm_validation.hpp"
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

Inputs make_inputs(const RmsnormTestCase& test_case) {
  const acl::RmsnormPlan plan = acl::plan_rmsnorm(test_case);
  const std::size_t elements = tensor_io::element_count(test_case.x);
  const std::size_t normalized =
      static_cast<std::size_t>(plan.normalized_elements);
  Inputs result;
  result.x.resize(elements);
  result.scale.resize(normalized);
  result.bias.resize(normalized);
  for (std::size_t column = 0; column < normalized; ++column) {
    const int scale_code = static_cast<int>((column * 7U) % 17U) - 8;
    const int bias_code = static_cast<int>((column * 11U) % 19U) - 9;
    result.scale[column] =
        (column % 5U == 0U ? -1.0F : 1.0F) *
        (0.5F + static_cast<float>(std::abs(scale_code)) / 8.0F);
    result.bias[column] = static_cast<float>(bias_code) / 16.0F;
  }
  for (std::size_t index = 0; index < elements; ++index) {
    const std::size_t row = index / normalized;
    const std::size_t column = index % normalized;
    const int centered =
        static_cast<int>((index * 17U + row * 13U + column * 3U) % 61U) - 30;
    result.x[index] = static_cast<float>(centered) / 16.0F;
    if (column == 0U && row % 3U == 0U) {
      result.x[index] = 0.0F;
    }
  }
  result.x = quantize(result.x, test_case.x);
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
      "RMSNorm guarded allocation");
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
    throw std::runtime_error(std::string(provider) + " modified RMSNorm " +
                             std::string(name) + " input or guards");
  }
}

std::vector<float> read_output(std::string_view provider,
                               std::string_view name,
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
    throw std::runtime_error(std::string(provider) + " modified RMSNorm " +
                             std::string(name) + " binding guards");
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
  throw std::runtime_error("ACLNN RMSNorm lifecycle violation was accepted");
}

std::optional<std::string> run_case(const RmsnormTestCase& test_case,
                                    flagdnn::Handle& handle,
                                    acl::Stream& stream) {
  const Inputs inputs = make_inputs(test_case);
  auto make_buffers = [&] {
    std::vector<GuardedBuffer> result;
    result.push_back(make_input_buffer(test_case.x, inputs.x, stream));
    result.push_back(make_input_buffer(test_case.scale, inputs.scale, stream));
    result.push_back(make_input_buffer(test_case.bias, inputs.bias, stream));
    result.push_back(make_output_buffer(test_case.y, stream));
    result.push_back(make_output_buffer(test_case.inv_variance, stream));
    return result;
  };
  std::vector<GuardedBuffer> flagdnn_buffers = make_buffers();
  std::vector<GuardedBuffer> aclnn_buffers = make_buffers();
  auto make_bindings = [&](std::vector<GuardedBuffer>& buffers) {
    const std::array<const TestTensor*, 5> tensors = {
        &test_case.x,
        &test_case.scale,
        &test_case.bias,
        &test_case.y,
        &test_case.inv_variance,
    };
    std::vector<flagdnnBinding_t> result;
    result.reserve(tensors.size());
    for (std::size_t index = 0; index < tensors.size(); ++index) {
      result.push_back({
          tensors[index]->uid,
          buffers[index].device->opaque_at(tensors[index]->binding_byte_offset),
      });
    }
    return result;
  };
  std::vector<flagdnnBinding_t> flagdnn_bindings =
      make_bindings(flagdnn_buffers);
  std::vector<flagdnnBinding_t> aclnn_bindings = make_bindings(aclnn_buffers);

  std::unique_ptr<NormalizationExecutable> flagdnn =
      build_flagdnn_rmsnorm(handle, test_case);
  std::unique_ptr<NormalizationExecutable> reference =
      build_rmsnorm_reference(test_case);
  std::unique_ptr<NormalizationExecutable> unprepared =
      build_rmsnorm_reference(test_case);
  expect_failure(
      [&] { unprepared->execute(aclnn_bindings, nullptr, 0, stream.opaque()); },
      "before prepare");
  try {
    reference->prepare(aclnn_bindings, stream.opaque());
    reference->prepare(aclnn_bindings, stream.opaque());
  } catch (const AclnnRmsnormUnsupportedError& error) {
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
  changed[0].device_pointer = aclnn_bindings[3].device_pointer;
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
    execute(*flagdnn, flagdnn_bindings, flagdnn_workspace, stream);
    execute(*reference, aclnn_bindings, aclnn_workspace, stream);
  }
  stream.synchronize();

  const std::vector<float> flagdnn_y = read_output(
      "FlagDNN", "Y", flagdnn_buffers[3], test_case.y, stream);
  const std::vector<float> flagdnn_inv =
      read_output("FlagDNN",
                  "inverse variance",
                  flagdnn_buffers[4],
                  test_case.inv_variance,
                  stream);
  const std::vector<float> aclnn_y =
      read_output("ACLNN", "Y", aclnn_buffers[3], test_case.y, stream);
  const std::vector<float> aclnn_inv =
      read_output("ACLNN",
                  "inverse variance",
                  aclnn_buffers[4],
                  test_case.inv_variance,
                  stream);
  static constexpr std::array<std::string_view, 3> input_names = {
      "X", "scale", "bias"};
  for (std::size_t index = 0; index < input_names.size(); ++index) {
    require_unchanged(
        "FlagDNN", input_names[index], flagdnn_buffers[index], stream);
    require_unchanged(
        "ACLNN", input_names[index], aclnn_buffers[index], stream);
  }

  acl::compare_exact_reference(flagdnn_y,
                               aclnn_y,
                               test_case.absolute_tolerance,
                               test_case.relative_tolerance,
                               test_case.name + ":Y");
  const double statistic_tolerance =
      std::max(2.0e-4, test_case.absolute_tolerance);
  acl::compare_exact_reference(flagdnn_inv,
                               aclnn_inv,
                               statistic_tolerance,
                               statistic_tolerance,
                               test_case.name + ":inverse_variance");
  std::cout << test_case.name
            << ": PASS exact_reference=ACLNN_RMSNORM_PLUS_ADD outputs=2\n";
  stream.synchronize();
  reference.reset();
  flagdnn.reset();
  return std::nullopt;
}

}  // namespace

int run_rmsnorm_functional_test(int argc,
                                char** argv,
                                std::span<const RmsnormTestCase> cases) {
  constexpr std::string_view suite = "FLAGDNN_RMSNORM_FUNCTIONAL";
  if (argc != 3) {
    std::cerr << "usage: " << argv[0]
              << " COMPILER_EXECUTABLE COMPILER_ENTRY\n";
    return 2;
  }
  try {
    if (cases.empty()) {
      throw std::invalid_argument(
          "common RMSNorm functional catalog must contain cases");
    }
    const std::vector<RmsnormTestCase> ascend_cases =
        acl::make_ascend_rmsnorm_cases(cases);

    acl::DevelopmentEnvironment development("rmsnorm-functional");
    acl::AclRuntime runtime;
    const std::string soc = acl::soc_name();
    development.prepare_target(soc);
    std::cout << "ASCEND_VALIDATION_DEVELOPMENT_ROOT root="
              << development.root() << '\n';
    std::cout << "ACLNN_CAPABILITY_IDENTITY soc=" << soc
              << " cann_runtime=" << acl::runtime_package_version() << '\n';
    std::cout << std::setprecision(9);

    const char* filter = std::getenv("FLAGDNN_NORMALIZATION_CASE");
    std::vector<const RmsnormTestCase*> selected;
    selected.reserve(ascend_cases.size());
    for (const RmsnormTestCase& test_case : ascend_cases) {
      if (filter != nullptr && filter[0] != '\0' &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      selected.push_back(&test_case);
    }
    if (selected.empty()) {
      throw std::runtime_error("RMSNorm filter matched no cases");
    }
    acl::ExactReferenceCoverage coverage(selected.size());
    {
      acl::Stream stream;
      flagdnn::Handle handle("ascend", 0);
      handle.set_compiler(
          argv[1], argv[2], development.graph_cache().string());
      for (const RmsnormTestCase* test_case : selected) {
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
