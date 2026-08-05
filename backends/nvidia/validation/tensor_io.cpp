/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "validation/tensor_io.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstring>

namespace flagdnn::validation::nvidia::tensor_io {
namespace {

constexpr std::uint8_t kBooleanPaddingSentinel = 0xA5U;

}  // namespace

float padding_sentinel() noexcept { return kPaddingSentinel; }

std::size_t data_type_size(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_FLOAT32:
      return 4;
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16:
      return 2;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      return 1;
  }
  throw std::invalid_argument("unsupported validation tensor data type");
}

std::vector<std::uint8_t> encode(
    std::span<const float> physical,
    flagdnnDataType_t data_type,
    BooleanEncoding boolean_encoding) {
  if (data_type == FLAGDNN_DATA_BOOLEAN &&
      boolean_encoding == BooleanEncoding::kBitPacked) {
    std::vector<std::uint8_t> result((physical.size() + 7U) / 8U, 0U);
    for (std::size_t index = 0; index < physical.size(); ++index) {
      if (physical[index] != 0.0F &&
          physical[index] != kPaddingSentinel) {
        result[index / 8U] |=
            static_cast<std::uint8_t>(1U << (index % 8U));
      }
    }
    return result;
  }

  const std::size_t element_size = data_type_size(data_type);
  std::vector<std::uint8_t> result(physical.size() * element_size);
  if (data_type == FLAGDNN_DATA_FLOAT32) {
    std::memcpy(result.data(), physical.data(), result.size());
    return result;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    std::uint8_t* destination = result.data() + index * element_size;
    switch (data_type) {
      case FLAGDNN_DATA_FLOAT32:
        break;
      case FLAGDNN_DATA_FLOAT16: {
        const __half value = __float2half_rn(physical[index]);
        std::memcpy(destination, &value, sizeof(value));
        break;
      }
      case FLAGDNN_DATA_BFLOAT16: {
        const __nv_bfloat16 value = __float2bfloat16_rn(physical[index]);
        std::memcpy(destination, &value, sizeof(value));
        break;
      }
      case FLAGDNN_DATA_BOOLEAN:
        *destination = physical[index] == kPaddingSentinel
                           ? kBooleanPaddingSentinel
                           : static_cast<std::uint8_t>(physical[index] != 0.0F);
        break;
      case FLAGDNN_DATA_FP8_E4M3: {
        const __nv_fp8_e4m3 value(physical[index]);
        std::memcpy(destination, &value, sizeof(value));
        break;
      }
      case FLAGDNN_DATA_FP8_E5M2: {
        const __nv_fp8_e5m2 value(physical[index]);
        std::memcpy(destination, &value, sizeof(value));
        break;
      }
    }
  }
  return result;
}

std::vector<float> decode(
    std::span<const std::uint8_t> bytes,
    flagdnnDataType_t data_type,
    std::size_t physical_element_count,
    BooleanEncoding boolean_encoding) {
  if (data_type == FLAGDNN_DATA_BOOLEAN &&
      boolean_encoding == BooleanEncoding::kBitPacked) {
    if (bytes.size() < (physical_element_count + 7U) / 8U) {
      throw std::invalid_argument("packed BOOLEAN buffer is too small");
    }
    std::vector<float> result(physical_element_count);
    for (std::size_t index = 0; index < physical_element_count; ++index) {
      result[index] = static_cast<float>(
          (bytes[index / 8U] >> (index % 8U)) & 1U);
    }
    return result;
  }

  const std::size_t element_size = data_type_size(data_type);
  if (bytes.size() != physical_element_count * element_size) {
    throw std::invalid_argument("encoded tensor byte count is invalid");
  }
  std::vector<float> result(physical_element_count);
  if (data_type == FLAGDNN_DATA_FLOAT32) {
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
  }
  for (std::size_t index = 0; index < physical_element_count; ++index) {
    const std::uint8_t* source = bytes.data() + index * element_size;
    switch (data_type) {
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
                            : static_cast<float>(*source != 0U);
        break;
      case FLAGDNN_DATA_FP8_E4M3: {
        __nv_fp8_e4m3 value;
        std::memcpy(&value, source, sizeof(value));
        result[index] = static_cast<float>(value);
        break;
      }
      case FLAGDNN_DATA_FP8_E5M2: {
        __nv_fp8_e5m2 value;
        std::memcpy(&value, source, sizeof(value));
        result[index] = static_cast<float>(value);
        break;
      }
    }
  }
  return result;
}

}  // namespace flagdnn::validation::nvidia::tensor_io
