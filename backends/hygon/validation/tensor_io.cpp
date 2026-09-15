/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "tensor_io.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace flagdnn::validation::hygon::tensor_io {
namespace {

constexpr std::uint8_t kBooleanPaddingSentinel = 0xA5U;

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  const std::uint32_t mantissa = bits & 0x7FFFFFU;
  if (exponent == 0xFFU) {
    return static_cast<std::uint16_t>(sign |
                                      (mantissa == 0 ? 0x7C00U : 0x7E00U));
  }
  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return static_cast<std::uint16_t>(sign);
    }
    std::uint32_t significand = mantissa | 0x800000U;
    const int shift = 14 - half_exponent;
    const std::uint32_t round =
        (1U << (shift - 1)) - 1U + ((significand >> shift) & 1U);
    return static_cast<std::uint16_t>(sign | ((significand + round) >> shift));
  }
  std::uint32_t rounded = mantissa + 0xFFFU + ((mantissa >> 13U) & 1U);
  std::uint32_t output_exponent = static_cast<std::uint32_t>(half_exponent);
  if ((rounded & 0x800000U) != 0U) {
    rounded = 0;
    ++output_exponent;
    if (output_exponent >= 31U) {
      return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
  }
  return static_cast<std::uint16_t>(sign | (output_exponent << 10U) |
                                    (rounded >> 13U));
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(value) & 0x8000U)
                             << 16U;
  const std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x3FFU;
  std::uint32_t output = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      output = sign;
    } else {
      int normalized_exponent = -14;
      while ((mantissa & 0x400U) == 0U) {
        mantissa <<= 1U;
        --normalized_exponent;
      }
      mantissa &= 0x3FFU;
      output = sign |
               (static_cast<std::uint32_t>(normalized_exponent + 127) << 23U) |
               (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    output = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    output = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(output);
}

std::uint16_t float_to_bfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) != 0x7F800000U) {
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
  } else if ((bits & 0x007FFFFFU) != 0U) {
    bits |= 0x00400000U;
  }
  return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

} // namespace

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
  throw std::invalid_argument("unsupported validation tensor data type");
}

std::vector<std::uint8_t> encode(std::span<const float> physical,
                                 flagdnnDataType_t data_type) {
  const std::size_t element_size = data_type_size(data_type);
  std::vector<std::uint8_t> result(physical.size() * element_size);
  for (std::size_t index = 0; index < physical.size(); ++index) {
    std::uint8_t *destination = result.data() + index * element_size;
    if (data_type == FLAGDNN_DATA_FLOAT32) {
      std::memcpy(destination, &physical[index], sizeof(float));
    } else if (data_type == FLAGDNN_DATA_FLOAT16) {
      const std::uint16_t value = float_to_half(physical[index]);
      std::memcpy(destination, &value, sizeof(value));
    } else if (data_type == FLAGDNN_DATA_BFLOAT16) {
      const std::uint16_t value = float_to_bfloat16(physical[index]);
      std::memcpy(destination, &value, sizeof(value));
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      *destination = physical[index] == kPaddingSentinel
                         ? kBooleanPaddingSentinel
                         : static_cast<std::uint8_t>(physical[index] != 0.0F);
    } else {
      throw std::invalid_argument("FP8 validation encoding is not implemented");
    }
  }
  return result;
}

std::vector<float> decode(std::span<const std::uint8_t> bytes,
                          flagdnnDataType_t data_type,
                          std::size_t physical_element_count) {
  const std::size_t element_size = data_type_size(data_type);
  if (bytes.size() != physical_element_count * element_size) {
    throw std::invalid_argument("encoded tensor byte count is invalid");
  }
  std::vector<float> result(physical_element_count);
  for (std::size_t index = 0; index < physical_element_count; ++index) {
    const std::uint8_t *source = bytes.data() + index * element_size;
    if (data_type == FLAGDNN_DATA_FLOAT32) {
      std::memcpy(&result[index], source, sizeof(float));
    } else if (data_type == FLAGDNN_DATA_FLOAT16) {
      std::uint16_t value = 0;
      std::memcpy(&value, source, sizeof(value));
      result[index] = half_to_float(value);
    } else if (data_type == FLAGDNN_DATA_BFLOAT16) {
      std::uint16_t value = 0;
      std::memcpy(&value, source, sizeof(value));
      result[index] = bfloat16_to_float(value);
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      result[index] = *source == kBooleanPaddingSentinel
                          ? kPaddingSentinel
                          : static_cast<float>(*source != 0U);
    } else {
      throw std::invalid_argument("FP8 validation decoding is not implemented");
    }
  }
  return result;
}

} // namespace flagdnn::validation::hygon::tensor_io
