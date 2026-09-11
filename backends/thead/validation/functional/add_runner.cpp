// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/add.hpp"
#include "functional/pointwise_runner_support.hpp"

#include "acdnn_reference.hpp"
#include "capability.hpp"
#include "numeric_types.hpp"
#include "ppu_driver.hpp"

#include <acdnn.h>
#include <cuda.h>
#include <flagdnn/flagdnn.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG
#define FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG "capability.json"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_VERSION
#define FLAGDNN_THEAD_PPU_SDK_VERSION "unknown"
#endif

namespace flagdnn::validation::thead::functional {
namespace {

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(std::string(description) + " overflows size_t");
  }
  return left * right;
}

std::size_t logical_offset(
    std::size_t logical_index,
    const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    const std::size_t stride =
        static_cast<std::size_t>(tensor.strides[current]);
    const std::size_t term = checked_multiply(coordinate, stride,
                                              "logical tensor offset");
    if (result > std::numeric_limits<std::size_t>::max() - term) {
      throw std::overflow_error("logical tensor offset overflows size_t");
    }
    result += term;
  }
  return result;
}

std::vector<float> scatter(std::span<const float> logical,
                           const flagdnn::testing::TestTensor &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("logical pointwise input size mismatch");
  }
  std::vector<float> result(storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    result.at(logical_offset(index, tensor)) = logical[index];
  }
  return result;
}

std::vector<std::uint8_t> scatter_boolean(
    std::span<const std::uint8_t> logical,
    const flagdnn::testing::TestTensor &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("logical boolean input size mismatch");
  }
  std::vector<std::uint8_t> result(storage_element_count(tensor), 0x7fU);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    result.at(logical_offset(index, tensor)) = logical[index];
  }
  return result;
}

std::vector<float> make_input(
    const flagdnn::testing::TestTensor &tensor, std::size_t input_index,
    flagdnn::testing::PointwiseInputDomain domain) {
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    switch (domain) {
      case flagdnn::testing::PointwiseInputDomain::kPositive:
      case flagdnn::testing::PointwiseInputDomain::kDivisor:
      case flagdnn::testing::PointwiseInputDomain::kModulo:
      case flagdnn::testing::PointwiseInputDomain::kPower:
        result[index] =
            static_cast<float>((index * 17 + input_index * 11) % 41 + 1) /
            static_cast<float>(13 + input_index);
        break;
      case flagdnn::testing::PointwiseInputDomain::kScaled:
        result[index] = static_cast<float>(centered) * 0.25F;
        break;
      case flagdnn::testing::PointwiseInputDomain::kTan:
        result[index] = static_cast<float>(centered) / 32.0F;
        break;
      case flagdnn::testing::PointwiseInputDomain::kModuloSigned:
      case flagdnn::testing::PointwiseInputDomain::kComparison:
      case flagdnn::testing::PointwiseInputDomain::kReal:
        result[index] = static_cast<float>(centered) /
                        static_cast<float>(13 + input_index);
        break;
      case flagdnn::testing::PointwiseInputDomain::kLogical:
        throw std::invalid_argument(
            "floating THead input builder cannot materialize logical data");
    }
  }
  return result;
}

void require_supported_type(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.data_type != FLAGDNN_DATA_FLOAT32 &&
      tensor.data_type != FLAGDNN_DATA_FLOAT16 &&
      tensor.data_type != FLAGDNN_DATA_BFLOAT16 &&
      tensor.data_type != FLAGDNN_DATA_BOOLEAN) {
    throw std::invalid_argument(
        "qualified THead validation runner requires a floating or bool type");
  }
}

struct Accuracy {
  double maximum_absolute = 0.0;
  double maximum_relative = 0.0;
};

Accuracy compare(std::span<const float> actual,
                 std::span<const float> expected,
                 const flagdnn::testing::AddTestCase &test_case) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("FlagDNN and acDNN output sizes differ");
  }
  Accuracy result;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = expected[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    result.maximum_absolute = std::max(result.maximum_absolute, absolute);
    result.maximum_relative = std::max(result.maximum_relative, relative);
    if (std::isnan(left) != std::isnan(right) || !std::isfinite(absolute) ||
        (absolute > test_case.absolute_tolerance &&
         relative > test_case.relative_tolerance)) {
      std::ostringstream message;
      message << test_case.name << " differs at output element " << index
              << ": FlagDNN=" << left << " acDNN=" << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << test_case.absolute_tolerance
              << " rtol=" << test_case.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
  return result;
}

void run_case(const flagdnn::testing::AddTestCase &test_case,
              flagdnn::Handle &handle, DeviceStream &stream) {
  require_supported_type(test_case.left);
  require_supported_type(test_case.right);
  require_supported_type(test_case.output);
  auto production = flagdnn::testing::build_flagdnn_add(handle, test_case);
  auto reference = flagdnn::testing::build_add_reference(test_case);

  std::vector<BoundTensor> inputs;
  inputs.push_back(make_input_buffer(test_case.left, 0, stream.get()));
  inputs.push_back(make_input_buffer(test_case.right, 1, stream.get()));
  std::vector<BoundTensor> production_outputs;
  production_outputs.push_back(
      make_output_buffer(test_case.output, stream.get()));
  std::vector<BoundTensor> reference_outputs;
  reference_outputs.push_back(
      make_output_buffer(test_case.output, stream.get()));
  const std::vector<flagdnnBinding_t> production_bindings =
      bindings(inputs, production_outputs);
  const std::vector<flagdnnBinding_t> reference_bindings =
      bindings(inputs, reference_outputs);

  DeviceBuffer production_workspace(production->workspace_size());
  DeviceBuffer reference_workspace(reference->workspace_size());
  check_driver(cuStreamSynchronize(stream.get()),
               "cuStreamSynchronize(before Add)");
  execute(*production, production_bindings, production_workspace, stream);
  execute(*reference, reference_bindings, reference_workspace, stream);
  check_driver(cuStreamSynchronize(stream.get()),
               "cuStreamSynchronize(after Add)");

  const std::vector<float> production_physical =
      read_output(production_outputs.front(), stream.get());
  const std::vector<float> reference_physical =
      read_output(reference_outputs.front(), stream.get());
  require_padding_unchanged("FlagDNN", production_physical,
                            test_case.output);
  require_padding_unchanged("acDNN", reference_physical, test_case.output);
  const Accuracy accuracy =
      compare(gather(production_physical, test_case.output),
              gather(reference_physical, test_case.output), test_case);
  std::cout << test_case.name << ": FlagDNN Graph vs acDNN PASS"
            << " max_abs=" << accuracy.maximum_absolute
            << " max_rel=" << accuracy.maximum_relative << '\n';
}

void emit_skip(const flagdnn::testing::AddTestCase &test_case,
               const CapabilityRecord &record, std::string_view target) {
  if (record.status != CapabilityStatus::kUnsupported ||
      record.reason_code.empty()) {
    throw std::runtime_error("invalid acDNN skip capability");
  }
  std::cout << "[SKIP][acdnn]"
            << " op=add"
            << " case=" << test_case.name
            << " reason=" << record.reason_code
            << " sdk=" << FLAGDNN_THEAD_PPU_SDK_VERSION
            << " acdnn_header=" << ACDNN_VERSION
            << " acdnn_runtime=" << acdnnGetVersion()
            << " target=" << target
            << " dtype=" << data_type_name(test_case.left.data_type)
            << " layout=" << layout_name(test_case.left)
            << " shape=" << shape_name(test_case.output) << '\n';
}

}  // namespace

TemporaryCache::TemporaryCache() {
  const char *configured = std::getenv("FLAGDNN_CACHE_PATH");
  if (configured != nullptr && configured[0] != '\0') {
    path_ = configured;
    std::filesystem::create_directories(path_);
    return;
  }
  std::string pattern =
      (std::filesystem::temp_directory_path() /
       "flagdnn-thead-pointwise-functional-XXXXXX")
          .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    throw std::runtime_error("mkdtemp failed for THead functional cache");
  }
  path_ = created;
  owned_ = true;
}

TemporaryCache::~TemporaryCache() noexcept {
  if (!owned_) {
    return;
  }
  std::error_code ignored;
  std::filesystem::remove_all(path_, ignored);
}

std::size_t element_count(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("functional tensor geometry is invalid");
  }
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("functional tensor extent must be positive");
    }
    result = checked_multiply(result, static_cast<std::size_t>(dimension),
                              "functional tensor element count");
  }
  return result;
}

std::size_t
storage_element_count(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("functional tensor geometry is invalid");
  }
  std::size_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument("functional tensor geometry is invalid");
    }
    const std::size_t term = checked_multiply(
        static_cast<std::size_t>(tensor.dimensions[axis] - 1),
        static_cast<std::size_t>(tensor.strides[axis]),
        "functional tensor storage span");
    if (result > std::numeric_limits<std::size_t>::max() - term) {
      throw std::overflow_error("functional tensor storage span overflows");
    }
    result += term;
  }
  return result;
}

bool is_contiguous(const flagdnn::testing::TestTensor &tensor) {
  std::int64_t expected = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != expected) {
      return false;
    }
    if (tensor.dimensions[axis - 1] >
        std::numeric_limits<std::int64_t>::max() / expected) {
      return false;
    }
    expected *= tensor.dimensions[axis - 1];
  }
  return true;
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return "fp32";
    case FLAGDNN_DATA_FLOAT16:
      return "fp16";
    case FLAGDNN_DATA_BFLOAT16:
      return "bf16";
    case FLAGDNN_DATA_FP8_E4M3:
      return "fp8_e4m3";
    case FLAGDNN_DATA_FP8_E5M2:
      return "fp8_e5m2";
    case FLAGDNN_DATA_BOOLEAN:
      return "bool";
  }
  return "unknown";
}

std::string shape_name(const flagdnn::testing::TestTensor &tensor) {
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result.empty() ? "scalar" : result;
}

std::string layout_name(const flagdnn::testing::TestTensor &tensor) {
  if (is_contiguous(tensor)) {
    return "contiguous";
  }
  if (tensor.dimensions.size() == 4 && tensor.strides.size() == 4 &&
      tensor.strides[1] == 1) {
    return "nhwc";
  }
  return "explicit_strided";
}

BoundTensor make_input_buffer(const flagdnn::testing::TestTensor &tensor,
                              std::size_t input_index, CUstream stream,
                              flagdnn::testing::PointwiseInputDomain domain) {
  require_supported_type(tensor);
  const std::size_t element_alignment = element_size(tensor.data_type);
  if (tensor.binding_byte_offset % element_alignment != 0) {
    throw std::invalid_argument(
        "THead pointwise input binding is misaligned");
  }
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
    if (domain != flagdnn::testing::PointwiseInputDomain::kLogical) {
      throw std::invalid_argument(
          "THead boolean pointwise input requires logical domain");
    }
    std::vector<std::uint8_t> logical(element_count(tensor));
    for (std::size_t index = 0; index < logical.size(); ++index) {
      logical[index] =
          ((index * 17U + input_index * 11U) % 3U) == 0U ? 0U : 1U;
    }
    const std::vector<std::uint8_t> physical =
        scatter_boolean(logical, tensor);
    if (tensor.binding_byte_offset >
        std::numeric_limits<std::size_t>::max() - physical.size()) {
      throw std::overflow_error("boolean input allocation size overflows");
    }
    auto buffer = std::make_unique<DeviceBuffer>(
        tensor.binding_byte_offset + physical.size());
    copy_to_device_async(
        *buffer,
        std::as_bytes(std::span<const std::uint8_t>(physical.data(),
                                                    physical.size())),
        tensor.binding_byte_offset, stream);
    return {tensor, std::move(buffer)};
  }
  const std::vector<float> physical_values =
      scatter(make_input(tensor, input_index, domain), tensor);
  const std::vector<std::byte> physical =
      encode_floating(tensor.data_type, physical_values);
  const std::size_t payload_bytes = physical.size();
  if (tensor.binding_byte_offset >
      std::numeric_limits<std::size_t>::max() - payload_bytes) {
    throw std::overflow_error("input allocation size overflows");
  }
  auto buffer = std::make_unique<DeviceBuffer>(tensor.binding_byte_offset +
                                                payload_bytes);
  copy_to_device_async(*buffer, physical, tensor.binding_byte_offset, stream);
  return {tensor, std::move(buffer)};
}

BoundTensor make_output_buffer(const flagdnn::testing::TestTensor &tensor,
                               CUstream stream) {
  require_supported_type(tensor);
  const std::size_t element_alignment = element_size(tensor.data_type);
  if (tensor.binding_byte_offset % element_alignment != 0) {
    throw std::invalid_argument(
        "THead pointwise output binding is misaligned");
  }
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN) {
    const std::vector<std::uint8_t> initial(storage_element_count(tensor),
                                            0x7fU);
    if (tensor.binding_byte_offset >
        std::numeric_limits<std::size_t>::max() - initial.size()) {
      throw std::overflow_error("boolean output allocation size overflows");
    }
    auto buffer = std::make_unique<DeviceBuffer>(
        tensor.binding_byte_offset + initial.size());
    copy_to_device_async(
        *buffer,
        std::as_bytes(std::span<const std::uint8_t>(initial.data(),
                                                    initial.size())),
        tensor.binding_byte_offset, stream);
    return {tensor, std::move(buffer)};
  }
  const std::vector<float> initial_values(storage_element_count(tensor),
                                          kPaddingSentinel);
  const std::vector<std::byte> initial =
      encode_floating(tensor.data_type, initial_values);
  const std::size_t payload_bytes = initial.size();
  if (tensor.binding_byte_offset >
      std::numeric_limits<std::size_t>::max() - payload_bytes) {
    throw std::overflow_error("output allocation size overflows");
  }
  auto buffer = std::make_unique<DeviceBuffer>(tensor.binding_byte_offset +
                                                payload_bytes);
  copy_to_device_async(*buffer, initial, tensor.binding_byte_offset, stream);
  return {tensor, std::move(buffer)};
}

std::vector<float> read_output(const BoundTensor &tensor, CUstream stream) {
  require_supported_type(tensor.specification);
  if (tensor.specification.data_type == FLAGDNN_DATA_BOOLEAN) {
    std::vector<std::uint8_t> bytes(
        storage_element_count(tensor.specification));
    copy_from_device_async(
        std::as_writable_bytes(std::span<std::uint8_t>(bytes)),
        *tensor.buffer, tensor.specification.binding_byte_offset, stream);
    check_driver(cuStreamSynchronize(stream),
                 "cuStreamSynchronize(read boolean pointwise)");
    std::vector<float> result(bytes.size());
    std::transform(bytes.begin(), bytes.end(), result.begin(),
                   [](std::uint8_t value) { return static_cast<float>(value); });
    return result;
  }
  const std::size_t byte_count = checked_multiply(
      storage_element_count(tensor.specification),
      element_size(tensor.specification.data_type), "output read bytes");
  std::vector<std::byte> bytes(byte_count);
  copy_from_device_async(bytes, *tensor.buffer,
                         tensor.specification.binding_byte_offset, stream);
  check_driver(cuStreamSynchronize(stream),
               "cuStreamSynchronize(read pointwise)");
  return decode_floating(tensor.specification.data_type, bytes);
}

std::vector<float> gather(
    std::span<const float> physical,
    const flagdnn::testing::TestTensor &tensor) {
  if (physical.size() != storage_element_count(tensor)) {
    throw std::invalid_argument("physical pointwise output size mismatch");
  }
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = physical[logical_offset(index, tensor)];
  }
  return result;
}

void require_padding_unchanged(
    std::string_view provider, std::span<const float> physical,
    const flagdnn::testing::TestTensor &tensor) {
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied.at(logical_offset(index, tensor)) = true;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    const float sentinel = tensor.data_type == FLAGDNN_DATA_BOOLEAN
                               ? 127.0F
                               : kPaddingSentinel;
    if (!occupied[index] && physical[index] != sentinel) {
      throw std::runtime_error(std::string(provider) +
                               " modified pointwise output padding");
    }
  }
}

std::vector<flagdnnBinding_t>
bindings(std::span<const BoundTensor> inputs,
         std::span<const BoundTensor> outputs) {
  std::vector<flagdnnBinding_t> result;
  result.reserve(inputs.size() + outputs.size());
  for (const BoundTensor &input : inputs) {
    result.push_back(
        {input.specification.uid,
         input.buffer->at(input.specification.binding_byte_offset)});
  }
  for (const BoundTensor &output : outputs) {
    result.push_back(
        {output.specification.uid,
         output.buffer->at(output.specification.binding_byte_offset)});
  }
  return result;
}

void execute(flagdnn::testing::TestExecutable &executable,
             std::span<const flagdnnBinding_t> tensor_bindings,
             DeviceBuffer &workspace, DeviceStream &stream) {
  executable.prepare(tensor_bindings, stream.opaque());
  executable.execute(tensor_bindings, workspace.data(), workspace.size(),
                     stream.opaque());
}

}  // namespace flagdnn::validation::thead::functional

namespace flagdnn::testing {

int run_add_functional_test(int argc, char **argv,
                            std::span<const AddTestCase> cases) {
  namespace tv = flagdnn::validation::thead;
  namespace functional = tv::functional;
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "functional Add requires COMPILER_EXECUTABLE COMPILER_ENTRY");
    }
    const tv::CapabilityCatalog catalog = tv::CapabilityCatalog::load(
        FLAGDNN_THEAD_ACDNN_CAPABILITY_CATALOG);
    catalog.validate_versions(FLAGDNN_THEAD_PPU_SDK_VERSION, ACDNN_VERSION,
                              static_cast<std::int64_t>(acdnnGetVersion()));

    tv::check_driver(cuInit(0), "cuInit");
    int device_count = 0;
    tv::check_driver(cuDeviceGetCount(&device_count), "cuDeviceGetCount");
    if (device_count <= 0) {
      throw std::runtime_error("PPU driver reported no devices");
    }
    CUdevice device = 0;
    tv::check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    tv::PrimaryContext primary(device);
    tv::ScopedCurrentContext current(primary.get());
    tv::DeviceStream stream;
    functional::TemporaryCache cache;
    flagdnn::Handle handle("thead", 0);
    handle.set_compiler(argv[1], argv[2], cache.path().string());
    const std::string target(handle.target_fingerprint());

    const char *filter = std::getenv("FLAGDNN_ADD_CASE");
    const bool qualify_probes =
        std::getenv("FLAGDNN_THEAD_QUALIFY_PROBES") != nullptr;
    std::size_t selected = 0;
    std::size_t executed = 0;
    std::size_t skipped = 0;
    std::cout << std::setprecision(9);
    for (const AddTestCase &test_case : cases) {
      if (filter != nullptr &&
          test_case.name.find(filter) == std::string::npos) {
        continue;
      }
      ++selected;
      validate_add_case(test_case);
      const tv::CapabilityRecord &record =
          catalog.lookup("add", test_case.name);
      if (record.status == tv::CapabilityStatus::kUnsupported) {
        functional::emit_skip(test_case, record, target);
        ++skipped;
        continue;
      }
      if (record.status == tv::CapabilityStatus::kProbeRequired &&
          !qualify_probes) {
        throw std::runtime_error(
            "unqualified acDNN capability reached functional run: add/" +
            test_case.name);
      }
      functional::run_case(test_case, handle, stream);
      ++executed;
    }
    if (selected == 0) {
      throw std::runtime_error("FLAGDNN_ADD_CASE matched no test cases");
    }
    if (selected != executed + skipped) {
      throw std::runtime_error("THead Add accounting invariant failed");
    }
    std::cout << "FLAGDNN_ADD_FUNCTIONAL: "
              << (executed == 0 ? "SKIP" : "PASS")
              << " cases=" << selected << " executed=" << executed
              << " skipped=" << skipped << '\n';
    return executed == 0 ? 77 : 0;
  } catch (const std::exception &error) {
    std::cerr << "FLAGDNN_ADD_FUNCTIONAL: FAIL reason=" << error.what()
              << '\n';
    return 1;
  }
}

}  // namespace flagdnn::testing
