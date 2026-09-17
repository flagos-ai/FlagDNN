// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "functional/runner_support.hpp"

#include "benchmark/cuda_graph.hpp"
#include "corex_cudnn_status.hpp"
#include <array>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifndef FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG
#define FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG                              \
  "corex_cudnn_capabilities.json"
#endif

namespace flagdnn::iluvatar::validation::functional {
namespace {

constexpr std::uint8_t kBooleanPaddingSentinel = 0xa5U;

float decode_fp8(std::uint8_t bits, int exponent_bits, int mantissa_bits,
                 int exponent_bias, bool finite_nan_encoding) {
  const bool negative = (bits & 0x80U) != 0;
  const unsigned mantissa_mask = (1U << mantissa_bits) - 1U;
  const unsigned exponent_mask = (1U << exponent_bits) - 1U;
  const unsigned mantissa = bits & mantissa_mask;
  const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
  float magnitude = 0.0F;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa),
                           1 - exponent_bias - mantissa_bits);
  } else if (exponent == exponent_mask &&
             (!finite_nan_encoding || mantissa == mantissa_mask)) {
    magnitude = mantissa == 0 && !finite_nan_encoding
                    ? std::numeric_limits<float>::infinity()
                    : std::numeric_limits<float>::quiet_NaN();
  } else {
    magnitude = std::ldexp(1.0F + static_cast<float>(mantissa) /
                                      static_cast<float>(1U << mantissa_bits),
                           static_cast<int>(exponent) - exponent_bias);
  }
  return negative ? -magnitude : magnitude;
}

std::uint8_t encode_fp8(float value, int exponent_bits, int mantissa_bits,
                        int exponent_bias, bool finite_nan_encoding) {
  if (std::isnan(value)) {
    return finite_nan_encoding
               ? static_cast<std::uint8_t>(
                     (((1U << exponent_bits) - 1U) << mantissa_bits) |
                     ((1U << mantissa_bits) - 1U))
               : static_cast<std::uint8_t>(
                     (((1U << exponent_bits) - 1U) << mantissa_bits) | 1U);
  }
  const bool negative = std::signbit(value);
  const float magnitude = std::abs(value);
  std::uint8_t best = 0;
  float best_error = std::numeric_limits<float>::infinity();
  for (unsigned candidate = 0; candidate < 128; ++candidate) {
    const float decoded =
        decode_fp8(static_cast<std::uint8_t>(candidate), exponent_bits,
                   mantissa_bits, exponent_bias, finite_nan_encoding);
    if (!std::isfinite(decoded)) {
      continue;
    }
    const float error = std::abs(decoded - magnitude);
    if (error < best_error ||
        (error == best_error && (candidate & 1U) == 0U && (best & 1U) != 0U)) {
      best_error = error;
      best = static_cast<std::uint8_t>(candidate);
    }
  }
  return static_cast<std::uint8_t>(best | (negative ? 0x80U : 0U));
}

std::uint8_t encode_e4m3(float value) {
  return encode_fp8(value, 4, 3, 7, true);
}

std::uint8_t encode_e5m2(float value) {
  return encode_fp8(value, 5, 2, 15, false);
}

float decode_e4m3(std::uint8_t value) {
  return decode_fp8(value, 4, 3, 7, true);
}

float decode_e5m2(std::uint8_t value) {
  return decode_fp8(value, 5, 2, 15, false);
}

std::filesystem::path temporary_cache() {
  const char *configured = std::getenv("FLAGDNN_CACHE_PATH");
  if (configured != nullptr && configured[0] != '\0') {
    std::filesystem::create_directories(configured);
    return configured;
  }
  std::string pattern = (std::filesystem::temp_directory_path() /
                         "flagdnn-iluvatar-functional-XXXXXX")
                            .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *created = mkdtemp(writable.data());
  if (created == nullptr) {
    throw std::runtime_error("mkdtemp failed for functional cache");
  }
  return created;
}

std::size_t logical_offset(std::size_t logical_index,
                           const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    result += coordinate * static_cast<std::size_t>(tensor.strides[current]);
  }
  return result;
}

std::vector<float> make_input(const PlannedInput &input,
                              std::size_t input_index) {
  const std::size_t count = element_count(input.tensor);
  if (!input.exact_values.empty()) {
    if (input.exact_values.size() == 1) {
      return std::vector<float>(count, input.exact_values.front());
    }
    if (input.exact_values.size() != count) {
      throw std::invalid_argument("exact functional input size mismatch");
    }
    return input.exact_values;
  }
  std::vector<float> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const int centered =
        static_cast<int>((index * 17 + input_index * 11) % 41) - 20;
    const float real =
        static_cast<float>(centered) / static_cast<float>(13 + input_index);
    switch (input.domain) {
    case InputDomain::kReal:
      result[index] = real;
      break;
    case InputDomain::kPositive:
      result[index] = std::abs(real) + 0.5F;
      break;
    case InputDomain::kScaled:
      result[index] = real * 4.0F;
      break;
    case InputDomain::kTan:
      result[index] = static_cast<float>(centered) / 40.0F;
      break;
    case InputDomain::kDivisor:
    case InputDomain::kModulo:
      result[index] = input_index == 1 ? std::abs(real) + 0.5F : real;
      break;
    case InputDomain::kPower:
      result[index] = input_index == 0
                          ? std::abs(real) + 0.5F
                          : std::fmod(std::abs(real), 2.0F) + 0.125F;
      break;
    case InputDomain::kModuloSigned: {
      constexpr float left[] = {-3.0F, -3.0F, 3.0F, 3.0F, -5.5F, 5.5F};
      constexpr float right[] = {2.0F, -2.0F, 2.0F, -2.0F, 2.25F, -2.25F};
      result[index] = input_index == 0 ? left[index % 6] : right[index % 6];
      break;
    }
    case InputDomain::kComparison: {
      const float base =
          static_cast<float>(static_cast<int>((index * 17) % 41) - 20) / 13.0F;
      result[index] = input_index == 0 || index % 3 == 0
                          ? base
                          : (index % 3 == 1 ? base + 0.25F : base - 0.25F);
      break;
    }
    case InputDomain::kLogical:
      result[index] = ((index * 17 + input_index * 11) % 3) == 0 ? 0.0F : 1.0F;
      break;
    case InputDomain::kUnitScale:
      result[index] = 1.0F + real * 0.125F;
      break;
    case InputDomain::kVariance:
      result[index] = std::abs(real) + 1.0F;
      break;
    case InputDomain::kStats:
      result[index] = std::abs(real) + 0.25F;
      break;
    }
  }
  return result;
}

void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const flagdnn::testing::TestTensor &tensor) {
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied[logical_offset(index, tensor)] = true;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    if (!occupied[index] && physical[index] != kPaddingSentinel) {
      throw std::runtime_error(std::string(provider) +
                               " modified output padding for uid " +
                               std::to_string(tensor.uid));
    }
  }
}

std::string data_type_name(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    return "int32";

  case FLAGDNN_DATA_FP8_E8M0:
    return "fp8_e8m0";
  case FLAGDNN_DATA_FLOAT32:
    return "fp32";
  case FLAGDNN_DATA_FLOAT16:
    return "fp16";
  case FLAGDNN_DATA_BFLOAT16:
    return "bf16";
  case FLAGDNN_DATA_BOOLEAN:
    return "bool";
  case FLAGDNN_DATA_FP8_E4M3:
    return "fp8_e4m3";
  case FLAGDNN_DATA_FP8_E5M2:
    return "fp8_e5m2";
  }
  return "unknown";
}

bool is_contiguous(const flagdnn::testing::TestTensor &tensor) {
  std::int64_t stride = 1;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    if (tensor.strides[axis - 1] != stride) {
      return false;
    }
    stride *= tensor.dimensions[axis - 1];
  }
  return true;
}

std::string shape_name(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.empty()) {
    return "scalar";
  }
  std::string result;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (!result.empty()) {
      result.push_back('x');
    }
    result += std::to_string(dimension);
  }
  return result;
}

struct BoundTensor {
  flagdnn::testing::TestTensor specification;
  std::unique_ptr<DeviceBuffer> buffer;
};

BoundTensor make_input_buffer(const PlannedInput &input,
                              std::size_t input_index, Stream &stream) {
  const std::vector<std::uint8_t> bytes =
      encode(scatter(make_input(input, input_index), input.tensor),
             input.tensor.data_type);
  auto buffer = std::make_unique<DeviceBuffer>(
      input.tensor.binding_byte_offset + bytes.size());
  buffer->copy_from_host(bytes.data(), bytes.size(),
                         input.tensor.binding_byte_offset, stream.get());
  return {input.tensor, std::move(buffer)};
}

BoundTensor make_output_buffer(const PlannedOutput &output, Stream &stream) {
  const std::vector<float> initial(storage_element_count(output.tensor),
                                   kPaddingSentinel);
  const std::vector<std::uint8_t> bytes =
      encode(initial, output.tensor.data_type);
  auto buffer = std::make_unique<DeviceBuffer>(
      output.tensor.binding_byte_offset + bytes.size());
  buffer->copy_from_host(bytes.data(), bytes.size(),
                         output.tensor.binding_byte_offset, stream.get());
  return {output.tensor, std::move(buffer)};
}

std::vector<float> read_physical(const BoundTensor &output, Stream &stream) {
  const std::size_t count = storage_element_count(output.specification);
  std::vector<std::uint8_t> bytes(
      count * data_type_size(output.specification.data_type));
  output.buffer->copy_to_host(bytes.data(), bytes.size(),
                              output.specification.binding_byte_offset,
                              stream.get());
  stream.synchronize();
  return decode(bytes, output.specification.data_type, count);
}

std::vector<flagdnnBinding_t>
bindings(const std::vector<BoundTensor> &inputs,
         const std::vector<BoundTensor> &outputs) {
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
             Stream &stream) {
  DeviceBuffer workspace(executable.workspace_size());
  executable.execute(tensor_bindings, workspace.at(),
                     executable.workspace_size(), stream.opaque());
}

void compare(std::span<const float> actual, std::span<const float> reference,
             const PlannedOutput &output, std::string_view case_name) {
  if (actual.size() != reference.size()) {
    throw std::runtime_error("reference output size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double left = actual[index];
    const double right = reference[index];
    if (left == right || (std::isnan(left) && std::isnan(right))) {
      continue;
    }
    if (std::isnan(left) != std::isnan(right)) {
      throw std::runtime_error(std::string(case_name) +
                               " has a NaN mismatch in " + output.label);
    }
    const double absolute = std::abs(left - right);
    const double relative =
        absolute / std::max({std::abs(left), std::abs(right), 1.0e-30});
    if (!std::isfinite(absolute) || (absolute > output.absolute_tolerance &&
                                     relative > output.relative_tolerance)) {
      std::ostringstream message;
      message << case_name << " differs in " << output.label << " at element "
              << index << ": FlagDNN=" << left << " reference=" << right
              << " abs=" << absolute << " rel=" << relative
              << " atol=" << output.absolute_tolerance
              << " rtol=" << output.relative_tolerance;
      throw std::runtime_error(message.str());
    }
  }
}

} // namespace

void check_cuda(CUresult status, const char *operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char *detail = nullptr;
  (void)cuGetErrorString(status, &detail);
  throw std::runtime_error(
      std::string(operation) +
      " failed: " + (detail == nullptr ? "unknown CoreX error" : detail));
}

DriverContext::DriverContext() {
  check_cuda(cuInit(0), "cuInit");
  check_cuda(cuDeviceGet(&device_, 0), "cuDeviceGet");
  check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
             "cuDevicePrimaryCtxRetain");
  check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
}

DriverContext::~DriverContext() noexcept {
  if (context_ != nullptr) {
    (void)cuDevicePrimaryCtxRelease(device_);
  }
}

Stream::Stream() {
  check_cuda(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
             "cuStreamCreate");
}

Stream::~Stream() noexcept {
  if (stream_ != nullptr) {
    (void)cuStreamDestroy(stream_);
  }
}

void Stream::synchronize() const {
  check_cuda(cuStreamSynchronize(stream_), "cuStreamSynchronize");
}

DeviceBuffer::DeviceBuffer(std::size_t bytes)
    : bytes_(std::max<std::size_t>(bytes, 1)) {
  check_cuda(cuMemAlloc(&pointer_, bytes_), "cuMemAlloc");
}

DeviceBuffer::~DeviceBuffer() noexcept {
  if (pointer_ != 0) {
    (void)cuMemFree(pointer_);
  }
}

void *DeviceBuffer::at(std::size_t byte_offset) const {
  if (byte_offset >= bytes_) {
    throw std::invalid_argument("device buffer offset is out of range");
  }
  return reinterpret_cast<void *>(
      static_cast<std::uintptr_t>(pointer_ + byte_offset));
}

void DeviceBuffer::copy_from_host(const void *source, std::size_t bytes,
                                  std::size_t byte_offset,
                                  CUstream stream) const {
  if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
    throw std::invalid_argument("host-to-device copy is out of range");
  }
  check_cuda(cuMemcpyHtoDAsync(pointer_ + byte_offset, source, bytes, stream),
             "cuMemcpyHtoDAsync");
}

void DeviceBuffer::copy_to_host(void *destination, std::size_t bytes,
                                std::size_t byte_offset,
                                CUstream stream) const {
  if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
    throw std::invalid_argument("device-to-host copy is out of range");
  }
  check_cuda(
      cuMemcpyDtoHAsync(destination, pointer_ + byte_offset, bytes, stream),
      "cuMemcpyDtoHAsync");
}

std::size_t data_type_size(flagdnnDataType_t data_type) {
  switch (data_type) {
  case FLAGDNN_DATA_INT32:
    return 4;

  case FLAGDNN_DATA_FLOAT32:
    return 4;
  case FLAGDNN_DATA_FLOAT16:
  case FLAGDNN_DATA_BFLOAT16:
    return 2;
  case FLAGDNN_DATA_BOOLEAN:
  case FLAGDNN_DATA_FP8_E8M0:
  case FLAGDNN_DATA_FP8_E4M3:
  case FLAGDNN_DATA_FP8_E5M2:
    return 1;
  }
  throw std::invalid_argument("unsupported functional data type");
}

std::size_t element_count(const flagdnn::testing::TestTensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0 || result > std::numeric_limits<std::size_t>::max() /
                                       static_cast<std::size_t>(dimension)) {
      throw std::invalid_argument("functional tensor shape is invalid");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

std::size_t storage_element_count(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("functional tensor metadata is invalid");
  }
  if (tensor.dimensions.empty()) {
    return 1;
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument("functional tensor stride is invalid");
    }
    maximum_offset += static_cast<std::size_t>(tensor.dimensions[axis] - 1) *
                      static_cast<std::size_t>(tensor.strides[axis]);
  }
  return maximum_offset + 1;
}

std::vector<float> scatter(std::span<const float> logical,
                           const flagdnn::testing::TestTensor &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("logical tensor size mismatch");
  }
  std::vector<float> result(storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    result[logical_offset(index, tensor)] = logical[index];
  }
  return result;
}

std::vector<float> gather(std::span<const float> physical,
                          const flagdnn::testing::TestTensor &tensor) {
  if (physical.size() < storage_element_count(tensor)) {
    throw std::invalid_argument("physical tensor size mismatch");
  }
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = physical[logical_offset(index, tensor)];
  }
  return result;
}

std::vector<std::uint8_t> encode(std::span<const float> values,
                                 flagdnnDataType_t data_type) {
  const std::size_t element_size = data_type_size(data_type);
  std::vector<std::uint8_t> result(values.size() * element_size);
  if (data_type == FLAGDNN_DATA_FLOAT32) {
    std::memcpy(result.data(), values.data(), result.size());
    return result;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::uint8_t *destination = result.data() + index * element_size;
    switch (data_type) {
    case FLAGDNN_DATA_INT32: {
      const auto value = static_cast<std::int32_t>(values[index]);
      std::memcpy(destination, &value, sizeof(value));
      break;
    }

    case FLAGDNN_DATA_FP8_E8M0:
      throw std::invalid_argument(
          "E8M0 scale storage is not supported by this validation adapter");
    case FLAGDNN_DATA_FLOAT32:
      break;
    case FLAGDNN_DATA_FLOAT16: {
      const __half value = __float2half_rn(values[index]);
      std::memcpy(destination, &value, sizeof(value));
      break;
    }
    case FLAGDNN_DATA_BFLOAT16: {
      const __nv_bfloat16 value = __float2bfloat16_rn(values[index]);
      std::memcpy(destination, &value, sizeof(value));
      break;
    }
    case FLAGDNN_DATA_BOOLEAN:
      *destination = values[index] == kPaddingSentinel
                         ? kBooleanPaddingSentinel
                         : static_cast<std::uint8_t>(values[index] != 0.0F);
      break;
    case FLAGDNN_DATA_FP8_E4M3: {
      *destination = encode_e4m3(values[index]);
      break;
    }
    case FLAGDNN_DATA_FP8_E5M2: {
      *destination = encode_e5m2(values[index]);
      break;
    }
    }
  }
  return result;
}

std::vector<float> decode(std::span<const std::uint8_t> bytes,
                          flagdnnDataType_t data_type, std::size_t count) {
  const std::size_t element_size = data_type_size(data_type);
  if (bytes.size() != count * element_size) {
    throw std::invalid_argument("encoded tensor byte count mismatch");
  }
  std::vector<float> result(count);
  if (data_type == FLAGDNN_DATA_FLOAT32) {
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }
  for (std::size_t index = 0; index < count; ++index) {
    const std::uint8_t *source = bytes.data() + index * element_size;
    switch (data_type) {
    case FLAGDNN_DATA_INT32: {
      std::int32_t value;
      std::memcpy(&value, source, sizeof(value));
      result[index] = static_cast<float>(value);
      break;
    }

    case FLAGDNN_DATA_FP8_E8M0:
      throw std::invalid_argument(
          "E8M0 scale storage is not supported by this validation adapter");
    case FLAGDNN_DATA_FLOAT32:
      break;
    case FLAGDNN_DATA_FLOAT16: {
      __half value;
      std::memcpy(&value, source, sizeof(value));
      result[index] = __half2float(value);
      break;
    }
    case FLAGDNN_DATA_BFLOAT16: {
      __nv_bfloat16 value;
      std::memcpy(&value, source, sizeof(value));
      result[index] = __bfloat162float(value);
      break;
    }
    case FLAGDNN_DATA_BOOLEAN:
      result[index] = *source == kBooleanPaddingSentinel
                          ? kPaddingSentinel
                          : static_cast<float>(*source != 0);
      break;
    case FLAGDNN_DATA_FP8_E4M3: {
      result[index] = decode_e4m3(*source);
      break;
    }
    case FLAGDNN_DATA_FP8_E5M2: {
      result[index] = decode_e5m2(*source);
      break;
    }
    }
  }
  return result;
}

void emit_reference_skip(std::string_view operation, std::string_view case_name,
                         const flagdnn::testing::TestTensor &tensor,
                         std::string_view reason) {
  std::cout << "[SKIP][corex-cudnn] op=" << operation << " case=" << case_name
            << " reason=" << reason << " cudnn_header=7605 cudnn_runtime=7605"
            << " corex=4.4.0 target=corex_71 dtype="
            << data_type_name(tensor.data_type)
            << " layout=" << (is_contiguous(tensor) ? "contiguous" : "strided")
            << " shape=" << shape_name(tensor) << std::endl;
}

FunctionalSuite::FunctionalSuite(int argc, char **argv, std::string operation,
                                 std::string marker)
    : owns_cache_(std::getenv("FLAGDNN_CACHE_PATH") == nullptr ||
                  std::getenv("FLAGDNN_CACHE_PATH")[0] == '\0'),
      cache_path_(temporary_cache()), handle_("iluvatar", 0),
      catalog_(CorexCudnnCapabilityCatalog::load(
          FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG)),
      operation_(std::move(operation)), marker_(std::move(marker)),
      benchmark_(marker_.ends_with("_BENCHMARK")),
      qualify_candidates_(std::getenv("FLAGDNN_ILUVATAR_QUALIFY_CANDIDATES") !=
                          nullptr) {
  if (argc != 3) {
    throw std::invalid_argument(
        "functional runner requires COMPILER_EXECUTABLE COMPILER_ENTRY");
  }
  handle_.set_compiler(argv[1], argv[2], cache_path_.string());
  std::cout << std::setprecision(9);
}

FunctionalSuite::~FunctionalSuite() noexcept {
  std::error_code ignored;
  if (owns_cache_)
    std::filesystem::remove_all(cache_path_, ignored);
}

void FunctionalSuite::run(const CasePlan &plan,
                          const BuildExecutable &build_production,
                          const BuildExecutable &build_reference,
                          const HostReference &host_reference,
                          bool probe_reference) {
  if (finished_ || plan.operation != operation_ || plan.case_name.empty() ||
      plan.outputs.empty()) {
    throw std::invalid_argument("functional case plan is invalid");
  }
  bool qualified = false, candidate = false, run_reference = false;
  std::string skip_reason;
  if (probe_reference) {
    candidate = true;
    run_reference = true;
  } else if (!host_reference) {
    const auto &record = catalog_.lookup(plan.operation, plan.case_name);
    qualified = record.classification ==
                CorexCudnnCatalogClassification::kQualifiedSupported;
    candidate =
        record.classification == CorexCudnnCatalogClassification::kCandidate;
    if (candidate && !qualify_candidates_) {
      throw std::runtime_error(
          "unqualified capability reached functional run: " + plan.operation +
          "/" + plan.case_name);
    }
    run_reference = qualified || candidate;
    skip_reason = record.reason_code;
  }

  std::vector<BoundTensor> inputs;
  inputs.reserve(plan.inputs.size());
  for (std::size_t index = 0; index < plan.inputs.size(); ++index) {
    inputs.push_back(make_input_buffer(plan.inputs[index], index, stream_));
  }
  std::vector<BoundTensor> production_outputs;
  std::vector<BoundTensor> reference_outputs;
  production_outputs.reserve(plan.outputs.size());
  reference_outputs.reserve(plan.outputs.size());
  for (const PlannedOutput &output : plan.outputs) {
    production_outputs.push_back(make_output_buffer(output, stream_));
    if (run_reference) {
      reference_outputs.push_back(make_output_buffer(output, stream_));
    }
  }
  const std::vector<flagdnnBinding_t> production_bindings =
      bindings(inputs, production_outputs);

  std::unique_ptr<flagdnn::testing::TestExecutable> reference;
  if (run_reference) {
    const std::vector<flagdnnBinding_t> reference_bindings =
        bindings(inputs, reference_outputs);
    try {
      reference = build_reference();
      execute(*reference, reference_bindings, stream_);
      stream_.synchronize();

    } catch (const CorexCudnnStatusError &error) {
      if (!cudnn_status_is_runtime_capability(error.status()) || qualified) {
        throw;
      }
      run_reference = false;
      reference_outputs.clear();
      skip_reason = "CUDNN_STATUS_NOT_SUPPORTED";
    } catch (const std::exception &error) {
      const std::string_view detail(error.what());
      const bool sdk_not_supported =
          detail.find("not presently supported") != std::string_view::npos;
      if (qualified || !candidate || !sdk_not_supported) {
        throw;
      }
      run_reference = false;
      reference_outputs.clear();
      skip_reason = "CUDNN_STATUS_NOT_SUPPORTED";
    }
  }

  std::vector<std::vector<float>> host_expected;
  if (host_reference) {
    std::vector<std::vector<float>> host_inputs;
    for (const auto &input : inputs) {
      host_inputs.push_back(
          gather(read_physical(input, stream_), input.specification));
    }
    host_expected = host_reference(host_inputs);
    if (host_expected.size() != plan.outputs.size()) {
      throw std::runtime_error("CPU reference output count mismatch");
    }
  }
  std::unique_ptr<flagdnn::testing::TestExecutable> production =
      build_production();
  execute(*production, production_bindings, stream_);
  stream_.synchronize();
  ++production_executed_;
  ++cases_;

  for (std::size_t index = 0; index < plan.outputs.size(); ++index) {
    const std::vector<float> production_physical =
        read_physical(production_outputs[index], stream_);
    require_padding_unchanged("FlagDNN", production_physical,
                              plan.outputs[index].tensor);
    if (host_reference) {
      const auto &specification = plan.outputs[index];
      const auto bytes =
          encode(host_expected[index], specification.tensor.data_type);
      compare(gather(production_physical, specification.tensor),
              decode(bytes, specification.tensor.data_type,
                     host_expected[index].size()),
              specification, plan.case_name);
    }
    if (run_reference) {
      const std::vector<float> reference_physical =
          read_physical(reference_outputs[index], stream_);
      // Descriptor-internal padding is not part of the logical tensor.
      // Keep the stronger no-padding-write check on FlagDNN production, while
      // comparing only the logical CoreX cuDNN elements selected by strides.
      compare(gather(production_physical, plan.outputs[index].tensor),
              gather(reference_physical, plan.outputs[index].tensor),
              plan.outputs[index], plan.case_name);
    }
  }

  if (benchmark_ && run_reference) {
    namespace timing = flagdnn::iluvatar::validation::benchmark;
    const auto reference_bindings = bindings(inputs, reference_outputs);
    DeviceBuffer production_workspace(production->workspace_size());
    DeviceBuffer reference_workspace(reference->workspace_size());
    const auto submit_production = [&] {
      production->execute(production_bindings, production_workspace.at(),
                          production->workspace_size(), stream_.opaque());
    };
    const auto submit_reference = [&] {
      reference->execute(reference_bindings, reference_workspace.at(),
                         reference->workspace_size(), stream_.opaque());
    };
    // Match the shared benchmark's warmup, sample and replay counts. All
    // allocations, CPU oracles and compilation precede CUDA Graph capture.
    for (int i = 0; i < 10; ++i) {
      submit_production();
      submit_reference();
    }
    stream_.synchronize();
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_.get());
    timing::CapturedExecutionBatch production_batch(cuda_stream, 50,
                                                    submit_production);
    timing::CapturedExecutionBatch reference_batch(cuda_stream, 50,
                                                   submit_reference);
    timing::CudaEventTimer timer;
    std::array<std::vector<double>, 2> samples;
    for (int i = 0; i < 20; ++i) {
      for (int j = 0; j < 2; ++j) {
        const auto provider = (i + j) % 2;
        samples[provider].push_back(timer.measure_microseconds_per_execution(
            cuda_stream, provider == 0 ? production_batch : reference_batch));
      }
    }
    stream_.synchronize();
    for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
      const auto actual = read_physical(production_outputs[i], stream_);
      const auto expected = read_physical(reference_outputs[i], stream_);
      require_padding_unchanged("FlagDNN postcheck", actual,
                                plan.outputs[i].tensor);
      compare(gather(actual, plan.outputs[i].tensor),
              gather(expected, plan.outputs[i].tensor), plan.outputs[i],
              plan.case_name);
      if (host_reference) {
        const auto bytes =
            encode(host_expected[i], plan.outputs[i].tensor.data_type);
        compare(gather(actual, plan.outputs[i].tensor),
                decode(bytes, plan.outputs[i].tensor.data_type,
                       host_expected[i].size()),
                plan.outputs[i], plan.case_name);
      }
    }
    for (std::size_t i = 0; i < samples.size(); ++i) {
      timing::require_positive_finite_samples(samples[i], 20,
                                              i ? "corex_cudnn" : "flagdnn");
      std::cout
          << "{\"schema_version\":1,\"kind\":\"steady_state\",\"provider\":\""
          << (i ? "corex_cudnn" : "flagdnn") << "\",\"case\":\""
          << plan.case_name << "\",\"unit\":\"us\",\"median\":"
          << timing::percentile(samples[i], 0.5)
          << ",\"p90\":" << timing::percentile(samples[i], 0.9)
          << ",\"samples\":[";
      for (std::size_t j = 0; j < samples[i].size(); ++j) {
        if (j)
          std::cout << ',';
        std::cout << samples[i][j];
      }
      std::cout << "]}" << std::endl;
    }
  }
  if (host_reference && !benchmark_) {
    if (run_reference)
      std::cout << plan.case_name << ": FlagDNN Graph vs CoreX cuDNN PASS"
                << std::endl;
    ++reference_executed_;
    std::cout << plan.case_name << ": FlagDNN Graph vs CPU reference PASS"
              << std::endl;
    return;
  }
  if (run_reference) {
    ++reference_executed_;
    std::cout << plan.case_name << ": FlagDNN Graph vs CoreX cuDNN PASS"
              << std::endl;
    return;
  }
  if (skip_reason.empty()) {
    throw std::runtime_error("reference SKIP has no stable reason");
  }
  ++reference_skipped_;
  const auto &representative = plan.inputs.empty() ? plan.outputs.front().tensor
                                                   : plan.inputs.front().tensor;
  std::cout << "[SKIP][corex-cudnn]"
            << " op=" << plan.operation << " case=" << plan.case_name
            << " reason=" << skip_reason << " cudnn_header=7605"
            << " cudnn_runtime=7605"
            << " corex=4.4.0"
            << " target=corex_71"
            << " dtype=" << data_type_name(representative.data_type)
            << " layout="
            << (is_contiguous(representative) ? "contiguous" : "strided")
            << " shape=" << shape_name(representative) << std::endl;
}

void FunctionalSuite::run_raw(
    const CasePlan &plan, const BuildExecutable &build_production,
    const std::vector<std::vector<std::uint8_t>> &input_values,
    const std::vector<std::vector<std::uint8_t>> &expected_values,
    const BuildExecutable &build_reference) {
  if (finished_ || plan.operation != operation_ || plan.case_name.empty() ||
      input_values.size() != plan.inputs.size() ||
      expected_values.size() != plan.outputs.size() || plan.outputs.empty())
    throw std::invalid_argument("raw functional plan is invalid");
  auto scatter_bytes = [](const auto &values, const auto &tensor) {
    const auto width = data_type_size(tensor.data_type);
    if (values.size() != element_count(tensor) * width)
      throw std::invalid_argument("raw tensor size mismatch");
    std::vector<std::uint8_t> bytes(storage_element_count(tensor) * width,
                                    0xA5);
    for (std::size_t i = 0; i < element_count(tensor); ++i)
      std::copy_n(values.data() + i * width, width,
                  bytes.data() + logical_offset(i, tensor) * width);
    return bytes;
  };
  std::vector<BoundTensor> inputs, outputs;
  std::vector<std::vector<std::uint8_t>> expected;
  auto bind = [&](const auto &tensor, const auto &bytes, auto &bound) {
    auto buffer = std::make_unique<DeviceBuffer>(tensor.binding_byte_offset +
                                                 bytes.size());
    buffer->copy_from_host(bytes.data(), bytes.size(),
                           tensor.binding_byte_offset, stream_.get());
    stream_.synchronize();
    bound.push_back({tensor, std::move(buffer)});
  };
  for (std::size_t i = 0; i < plan.inputs.size(); ++i)
    bind(plan.inputs[i].tensor,
         scatter_bytes(input_values[i], plan.inputs[i].tensor), inputs);
  for (std::size_t i = 0; i < plan.outputs.size(); ++i) {
    const auto &tensor = plan.outputs[i].tensor;
    expected.push_back(scatter_bytes(expected_values[i], tensor));
    bind(tensor, std::vector<std::uint8_t>(expected.back().size(), 0xA5),
         outputs);
  }
  auto production = build_production();
  execute(*production, bindings(inputs, outputs), stream_);
  stream_.synchronize();
  const auto verify = [&](const auto &actual_outputs, std::string_view provider) {
    for (std::size_t i = 0; i < actual_outputs.size(); ++i) {
      std::vector<std::uint8_t> actual(expected[i].size());
      actual_outputs[i].buffer->copy_to_host(
          actual.data(), actual.size(),
          actual_outputs[i].specification.binding_byte_offset, stream_.get());
      stream_.synchronize();
      if (actual != expected[i])
        throw std::runtime_error(
            plan.case_name + " " + std::string(provider) +
            " differs from exact CPU reference or changed output padding");
    }
  };
  verify(outputs, "FlagDNN");
  if (build_reference) {
    // Keep the full byte patterns and padding oracle when qualifying storage
    // copies; converting BOOL bytes to float would hide copy corruption.
    std::vector<BoundTensor> reference_outputs;
    for (std::size_t i = 0; i < plan.outputs.size(); ++i)
      bind(plan.outputs[i].tensor,
           std::vector<std::uint8_t>(expected[i].size(), 0xA5), reference_outputs);
    auto reference = build_reference();
    execute(*reference, bindings(inputs, reference_outputs), stream_);
    stream_.synchronize();
    verify(reference_outputs, "CoreX cuDNN");
  }
  ++cases_;
  ++production_executed_;
  ++reference_executed_;
  std::cout << plan.case_name
            << (build_reference ? ": FlagDNN Graph and CoreX cuDNN"
                                : ": FlagDNN Graph")
            << " vs CPU bit-exact reference PASS" << std::endl;
}

void FunctionalSuite::skip_benchmark_case(const CasePlan &plan,
                                          std::string_view reason) {
  if (!benchmark_ || finished_ || plan.operation != operation_)
    throw std::logic_error("invalid explicit benchmark skip");
  ++cases_;
  ++reference_skipped_;
  emit_reference_skip(plan.operation, plan.case_name,
                      plan.inputs.empty() ? plan.outputs.front().tensor
                                          : plan.inputs.front().tensor,
                      reason);
}

int FunctionalSuite::finish() {
  if (finished_ || cases_ == 0 ||
      (!benchmark_ && cases_ != production_executed_) ||
      cases_ != reference_executed_ + reference_skipped_) {
    throw std::runtime_error("functional suite accounting invariant failed");
  }
  finished_ = true;
  const bool all_skipped = reference_executed_ == 0;
  if (benchmark_) {
    std::cout << marker_ << ": " << (all_skipped ? "SKIP" : "PASS")
              << " cases=" << cases_
              << " comparable_executed=" << reference_executed_
              << " reference_skipped=" << reference_skipped_ << std::endl;
    return all_skipped ? 77 : 0;
  }
  std::cout << marker_ << ": " << (all_skipped ? "SKIP" : "PASS")
            << " cases=" << cases_
            << " production_executed=" << production_executed_
            << " reference_executed=" << reference_executed_
            << " reference_skipped=" << reference_skipped_ << std::endl;
  return all_skipped ? 77 : 0;
}

InputDomain pointwise_input_domain(int domain) {
  if (domain < 0 || domain > 9) {
    throw std::invalid_argument("unknown pointwise input domain");
  }
  return static_cast<InputDomain>(domain);
}

} // namespace flagdnn::iluvatar::validation::functional
