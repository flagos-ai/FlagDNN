/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "tensor_io.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace flagdnn::validation::ascend::tensor_io {
namespace {

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
  std::uint32_t mantissa = bits & 0x7FFFFFU;

  if (exponent == 0xFFU) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    std::uint16_t payload = static_cast<std::uint16_t>(mantissa >> 13U);
    if (payload == 0) {
      payload = 1;
    }
    return static_cast<std::uint16_t>(sign | 0x7C00U | payload);
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7C00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000U;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder_mask = (1U << shift) - 1U;
    const std::uint32_t remainder = mantissa & remainder_mask;
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  std::uint32_t rounded_mantissa = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1FFFU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_mantissa & 1U) != 0U)) {
    ++rounded_mantissa;
    if (rounded_mantissa == 0x400U) {
      rounded_mantissa = 0;
      if (half_exponent + 1 >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7C00U);
      }
      return static_cast<std::uint16_t>(
          sign | (static_cast<std::uint16_t>(half_exponent + 1) << 10U));
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<std::uint16_t>(half_exponent) << 10U) |
      static_cast<std::uint16_t>(rounded_mantissa));
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1FU;
  std::uint32_t mantissa = value & 0x3FFU;
  std::uint32_t result = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      int unbiased = -14;
      while ((mantissa & 0x400U) == 0U) {
        mantissa <<= 1U;
        --unbiased;
      }
      mantissa &= 0x3FFU;
      result = sign |
               (static_cast<std::uint32_t>(unbiased + 127) << 23U) |
               (mantissa << 13U);
    }
  } else if (exponent == 0x1FU) {
    result = sign | 0x7F800000U | (mantissa << 13U);
  } else {
    result = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(result);
}

std::uint16_t float_to_bfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7F800000U) == 0x7F800000U &&
      (bits & 0x007FFFFFU) != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float fp8_decode(std::uint8_t value, flagdnnDataType_t type) {
  if (type == FLAGDNN_DATA_FP8_E8M0)
    return value == 255 ? std::numeric_limits<float>::quiet_NaN()
                        : std::ldexp(1.0F, static_cast<int>(value) - 127);
  const int bits = type == FLAGDNN_DATA_FP8_E4M3 ? 3 : 2;
  const int bias = bits == 3 ? 7 : 15;
  const int exponent = (value & 127) >> bits;
  const int mantissa = value & ((1 << bits) - 1);
  float result;
  if ((bits == 3 && (value & 127) == 127) ||
      (bits == 2 && exponent == 31 && mantissa != 0))
    result = std::numeric_limits<float>::quiet_NaN();
  else if (bits == 2 && exponent == 31)
    result = std::numeric_limits<float>::infinity();
  else
    result = exponent == 0
                 ? std::ldexp(static_cast<float>(mantissa), 1 - bias - bits)
                 : std::ldexp(1.0F + static_cast<float>(mantissa) / (1 << bits),
                              exponent - bias);
  return value & 128 ? -result : result;
}

std::uint8_t fp8_encode(float value, flagdnnDataType_t type) {
  // FP8 is a storage format in this adapter. Padding and copied NaN payloads
  // are checked as bytes, never through floating-point equality.
  if (value == kPaddingSentinel)
    return kBytePaddingSentinel;
  if (type == FLAGDNN_DATA_FP8_E8M0) {
    if (!(value > 0.0F) || !std::isfinite(value))
      return 255;
    int exponent = 0;
    const float mantissa = std::frexp(value, &exponent);
    const int power = exponent - (mantissa == 0.5F ? 1 : 0);
    return static_cast<std::uint8_t>(std::clamp(power + 127, 0, 255));
  }
  if (std::isnan(value))
    return 255;
  const int limit = type == FLAGDNN_DATA_FP8_E4M3 ? 126 : 123;
  const float magnitude = std::abs(value);
  int lo = 0, hi = limit;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (fp8_decode(static_cast<std::uint8_t>(mid), type) < magnitude)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo) {
    const float lower =
        magnitude - fp8_decode(static_cast<std::uint8_t>(lo - 1), type);
    const float upper =
        fp8_decode(static_cast<std::uint8_t>(lo), type) - magnitude;
    if (lower < upper || (lower == upper && (lo & 1)))
      --lo;
  }
  return static_cast<std::uint8_t>(lo | (std::signbit(value) ? 128 : 0));
}

void require_supported_type(flagdnnDataType_t data_type) {
  if (data_type != FLAGDNN_DATA_INT32 && data_type != FLAGDNN_DATA_FLOAT32 &&
      data_type != FLAGDNN_DATA_FLOAT16 &&
      data_type != FLAGDNN_DATA_BFLOAT16 &&
      data_type != FLAGDNN_DATA_BOOLEAN && !is_fp8_storage(data_type)) {
    throw std::invalid_argument(
        "unsupported Ascend validation tensor storage type");
  }
}

}  // namespace

std::size_t data_type_size(flagdnnDataType_t data_type) {
  require_supported_type(data_type);
  if (data_type == FLAGDNN_DATA_FLOAT32 || data_type == FLAGDNN_DATA_INT32) {
    return 4U;
  }
  if (data_type == FLAGDNN_DATA_BOOLEAN || is_fp8_storage(data_type)) {
    return 1U;
  }
  return 2U;
}

std::vector<std::uint8_t> encode(std::span<const float> values,
                                 flagdnnDataType_t data_type) {
  require_supported_type(data_type);
  const std::size_t element_size = data_type_size(data_type);
  std::vector<std::uint8_t> result(
      checked_multiply(values.size(), element_size, "encoded tensor size"));
  for (std::size_t index = 0; index < values.size(); ++index) {
    std::uint8_t* destination = result.data() + index * element_size;
    if (is_fp8_storage(data_type)) {
      *destination = fp8_encode(values[index], data_type);
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      if (values[index] == kPaddingSentinel) {
        *destination = kBooleanPaddingSentinel;
      } else if (values[index] == 0.0F) {
        *destination = 0U;
      } else if (values[index] == 1.0F) {
        *destination = 1U;
      } else {
        throw std::invalid_argument(
            "BOOLEAN encoding requires canonical 0/1 values");
      }
    } else if (data_type == FLAGDNN_DATA_INT32) {
      if (!std::isfinite(values[index]) || double(values[index]) < INT32_MIN ||
          double(values[index]) > INT32_MAX)
        throw std::invalid_argument("INT32 conversion is out of range");
      const auto value = static_cast<std::int32_t>(values[index]);
      std::memcpy(destination, &value, sizeof(value));
    } else if (data_type == FLAGDNN_DATA_FLOAT32) {
      std::memcpy(destination, &values[index], sizeof(float));
    } else {
      const std::uint16_t encoded =
          data_type == FLAGDNN_DATA_FLOAT16
              ? float_to_half(values[index])
              : float_to_bfloat16(values[index]);
      std::memcpy(destination, &encoded, sizeof(encoded));
    }
  }
  return result;
}

std::vector<float> decode(std::span<const std::uint8_t> bytes,
                          flagdnnDataType_t data_type,
                          std::size_t element_count_value) {
  require_supported_type(data_type);
  const std::size_t element_size = data_type_size(data_type);
  if (bytes.size() != checked_multiply(element_count_value,
                                       element_size,
                                       "decoded tensor size")) {
    throw std::invalid_argument("encoded tensor byte count is invalid");
  }
  std::vector<float> result(element_count_value);
  for (std::size_t index = 0; index < result.size(); ++index) {
    const std::uint8_t* source = bytes.data() + index * element_size;
    if (is_fp8_storage(data_type)) {
      result[index] = fp8_decode(*source, data_type);
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      if (*source > 1U) {
        throw std::invalid_argument(
            "BOOLEAN decoding requires canonical 0/1 bytes");
      }
      result[index] = static_cast<float>(*source);
    } else if (data_type == FLAGDNN_DATA_INT32) {
      std::int32_t value = 0;
      std::memcpy(&value, source, sizeof(value));
      result[index] = static_cast<float>(value);
    } else if (data_type == FLAGDNN_DATA_FLOAT32) {
      std::memcpy(&result[index], source, sizeof(float));
    } else {
      std::uint16_t encoded = 0;
      std::memcpy(&encoded, source, sizeof(encoded));
      result[index] = data_type == FLAGDNN_DATA_FLOAT16
                          ? half_to_float(encoded)
                          : bfloat16_to_float(encoded);
    }
  }
  return result;
}

std::vector<float> quantize(std::span<const float> values,
                            flagdnnDataType_t data_type) {
  const std::vector<std::uint8_t> bytes = encode(values, data_type);
  return decode(bytes, data_type, values.size());
}

float quantize_scalar(double value, flagdnnDataType_t data_type) {
  const float narrowed = static_cast<float>(value);
  const std::vector<float> result = quantize(
      std::span<const float>(&narrowed, 1), data_type);
  return result.front();
}

}  // namespace flagdnn::validation::ascend::tensor_io
