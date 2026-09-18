// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "numeric_types.hpp"

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace flagdnn::validation::thead {
namespace {

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent = (bits >> 23U) & 0xffU;
  const std::uint32_t fraction = bits & 0x7fffffU;
  if (exponent == 0xffU) {
    if (fraction == 0) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    std::uint16_t payload = static_cast<std::uint16_t>(fraction >> 13U);
    payload = static_cast<std::uint16_t>(payload | 0x0200U);
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload);
  }

  const int half_exponent = static_cast<int>(exponent) - 127 + 15;
  if (half_exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (half_exponent <= 0) {
    if (half_exponent < -10) {
      return sign;
    }
    const std::uint32_t significand = fraction | 0x800000U;
    const unsigned shift = static_cast<unsigned>(14 - half_exponent);
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder =
        significand & ((std::uint32_t{1} << shift) - 1U);
    const std::uint32_t halfway = std::uint32_t{1} << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  std::uint32_t rounded_fraction = fraction >> 13U;
  const std::uint32_t remainder = fraction & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded_fraction & 1U) != 0U)) {
    ++rounded_fraction;
  }
  const std::uint32_t encoded =
      (static_cast<std::uint32_t>(half_exponent) << 10U) + rounded_fraction;
  if (encoded >= 0x7c00U) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  return static_cast<std::uint16_t>(sign | encoded);
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t fraction = value & 0x03ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (fraction == 0) {
      bits = sign;
    } else {
      int unbiased = -14;
      while ((fraction & 0x0400U) == 0U) {
        fraction <<= 1U;
        --unbiased;
      }
      fraction &= 0x03ffU;
      bits = sign |
             (static_cast<std::uint32_t>(unbiased + 127) << 23U) |
             (fraction << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (fraction << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (fraction << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t float_to_bfloat(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U &&
      (bits & 0x007fffffU) != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float bfloat_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

void append_u16(std::vector<std::byte> &output, std::uint16_t value) {
  output.push_back(static_cast<std::byte>(value & 0xffU));
  output.push_back(static_cast<std::byte>((value >> 8U) & 0xffU));
}

void append_u32(std::vector<std::byte> &output, std::uint32_t value) {
  output.push_back(static_cast<std::byte>(value & 0xffU));
  output.push_back(static_cast<std::byte>((value >> 8U) & 0xffU));
  output.push_back(static_cast<std::byte>((value >> 16U) & 0xffU));
  output.push_back(static_cast<std::byte>((value >> 24U) & 0xffU));
}

std::uint16_t read_u16(std::span<const std::byte> bytes,
                       std::size_t offset) {
  return static_cast<std::uint16_t>(
      std::to_integer<std::uint8_t>(bytes[offset])) |
         static_cast<std::uint16_t>(
             std::to_integer<std::uint8_t>(bytes[offset + 1]))
             << 8U;
}

std::uint32_t read_u32(std::span<const std::byte> bytes,
                       std::size_t offset) {
  return static_cast<std::uint32_t>(
      std::to_integer<std::uint8_t>(bytes[offset])) |
         static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(bytes[offset + 1]))
             << 8U |
         static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(bytes[offset + 2]))
             << 16U |
         static_cast<std::uint32_t>(
             std::to_integer<std::uint8_t>(bytes[offset + 3]))
             << 24U;
}

}  // namespace

std::size_t element_size(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
    case FLAGDNN_DATA_FLOAT32:
      return sizeof(float);
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16:
      return sizeof(std::uint16_t);
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      return sizeof(std::uint8_t);
  }
  throw std::invalid_argument(
      "THead validation storage does not support this data type");
}

std::vector<std::byte>
encode_floating(flagdnnDataType_t data_type, std::span<const float> values) {
  if (values.size() > std::numeric_limits<std::size_t>::max() /
                          element_size(data_type)) {
    throw std::overflow_error("THead validation encoding size overflows");
  }
  std::vector<std::byte> result;
  result.reserve(values.size() * element_size(data_type));
  for (const float value : values) {
    switch (data_type) {
      case FLAGDNN_DATA_INT32:
      case FLAGDNN_DATA_FP8_E8M0:
        throw std::invalid_argument(
            "THead validation does not support INT32 or E8M0 here");
      case FLAGDNN_DATA_FLOAT32:
        append_u32(result, std::bit_cast<std::uint32_t>(value));
        break;
      case FLAGDNN_DATA_FLOAT16:
        append_u16(result, float_to_half(value));
        break;
      case FLAGDNN_DATA_BFLOAT16:
        append_u16(result, float_to_bfloat(value));
        break;
      case FLAGDNN_DATA_BOOLEAN:
      case FLAGDNN_DATA_FP8_E4M3:
      case FLAGDNN_DATA_FP8_E5M2:
        throw std::invalid_argument(
            "THead floating validation encoding requires a floating type");
    }
  }
  return result;
}

std::vector<float>
decode_floating(flagdnnDataType_t data_type,
                std::span<const std::byte> bytes) {
  const std::size_t width = element_size(data_type);
  if ((data_type != FLAGDNN_DATA_FLOAT32 && data_type != FLAGDNN_DATA_FLOAT16 &&
       data_type != FLAGDNN_DATA_BFLOAT16) ||
      bytes.size() % width != 0) {
    throw std::invalid_argument(
        "THead floating validation bytes do not match the data type");
  }
  std::vector<float> result;
  result.reserve(bytes.size() / width);
  for (std::size_t offset = 0; offset < bytes.size(); offset += width) {
    switch (data_type) {
      case FLAGDNN_DATA_INT32:
      case FLAGDNN_DATA_FP8_E8M0:
        throw std::invalid_argument(
            "THead validation does not support INT32 or E8M0 here");
      case FLAGDNN_DATA_FLOAT32:
        result.push_back(std::bit_cast<float>(read_u32(bytes, offset)));
        break;
      case FLAGDNN_DATA_FLOAT16:
        result.push_back(half_to_float(read_u16(bytes, offset)));
        break;
      case FLAGDNN_DATA_BFLOAT16:
        result.push_back(bfloat_to_float(read_u16(bytes, offset)));
        break;
      case FLAGDNN_DATA_BOOLEAN:
      case FLAGDNN_DATA_FP8_E4M3:
      case FLAGDNN_DATA_FP8_E5M2:
        throw std::invalid_argument(
            "THead floating validation decoding requires a floating type");
    }
  }
  return result;
}

}  // namespace flagdnn::validation::thead
