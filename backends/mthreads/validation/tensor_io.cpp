/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/tensor_io.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace flagdnn::validation::mthreads::tensor_io {
namespace {

constexpr std::uint8_t kBooleanPaddingSentinel = 0x5aU;

std::uint16_t float_to_half(float value) {
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  const std::uint16_t sign =
      static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const std::uint32_t exponent_bits = (bits >> 23U) & 0xffU;
  std::uint32_t mantissa = bits & 0x7fffffU;
  if (exponent_bits == 0xffU) {
    if (mantissa == 0) {
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
    return static_cast<std::uint16_t>(
        sign | 0x7c00U | std::max(1U, mantissa >> 13U));
  }

  int exponent = static_cast<int>(exponent_bits) - 127 + 15;
  if (exponent >= 31) {
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  }
  if (exponent <= 0) {
    if (exponent < -10) {
      return sign;
    }
    mantissa |= 0x800000U;
    const unsigned int shift =
        static_cast<unsigned int>(14 - exponent);
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder =
        mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1U);
    if (remainder > halfway ||
        (remainder == halfway && (rounded & 1U) != 0U)) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }

  std::uint32_t rounded = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U ||
      (remainder == 0x1000U && (rounded & 1U) != 0U)) {
    ++rounded;
    if (rounded == 0x400U) {
      rounded = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00U);
      }
    }
  }
  return static_cast<std::uint16_t>(
      sign | (static_cast<unsigned int>(exponent) << 10U) | rounded);
}

float half_to_float(std::uint16_t value) {
  const std::uint32_t sign =
      static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  std::uint32_t exponent = (value >> 10U) & 0x1fU;
  std::uint32_t mantissa = value & 0x3ffU;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int normalized_exponent = -14;
      while ((mantissa & 0x400U) == 0U) {
        mantissa <<= 1U;
        --normalized_exponent;
      }
      mantissa &= 0x3ffU;
      bits = sign |
             (static_cast<std::uint32_t>(normalized_exponent + 127)
              << 23U) |
             (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    exponent = exponent - 15U + 127U;
    bits = sign | (exponent << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

std::uint16_t float_to_bfloat16(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000U) == 0x7f800000U &&
      (bits & 0x007fffffU) != 0U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t least_significant = (bits >> 16U) & 1U;
  bits += 0x7fffU + least_significant;
  return static_cast<std::uint16_t>(bits >> 16U);
}

float bfloat16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

float positive_fp8_to_float(std::uint8_t bits, bool e4m3) {
  const unsigned int mantissa_bits = e4m3 ? 3U : 2U;
  const unsigned int exponent_bias = e4m3 ? 7U : 15U;
  const unsigned int exponent_mask = e4m3 ? 0x0fU : 0x1fU;
  const unsigned int mantissa_mask = (1U << mantissa_bits) - 1U;
  const unsigned int exponent =
      (static_cast<unsigned int>(bits) >> mantissa_bits) & exponent_mask;
  const unsigned int mantissa =
      static_cast<unsigned int>(bits) & mantissa_mask;
  if (exponent == 0U) {
    return std::ldexp(
        static_cast<float>(mantissa),
        1 - static_cast<int>(exponent_bias) -
            static_cast<int>(mantissa_bits));
  }
  return std::ldexp(
      1.0F + static_cast<float>(mantissa) /
                 static_cast<float>(1U << mantissa_bits),
      static_cast<int>(exponent) - static_cast<int>(exponent_bias));
}

const std::vector<float>& positive_fp8_values(bool e4m3) {
  static const std::vector<float> e4m3_values = [] {
    std::vector<float> result(0x7fU);
    for (std::size_t bits = 0; bits < result.size(); ++bits) {
      result[bits] = positive_fp8_to_float(
          static_cast<std::uint8_t>(bits), true);
    }
    return result;
  }();
  static const std::vector<float> e5m2_values = [] {
    std::vector<float> result(0x7cU);
    for (std::size_t bits = 0; bits < result.size(); ++bits) {
      result[bits] = positive_fp8_to_float(
          static_cast<std::uint8_t>(bits), false);
    }
    return result;
  }();
  return e4m3 ? e4m3_values : e5m2_values;
}

std::uint8_t float_to_fp8(float value, bool e4m3) {
  if (std::isnan(value)) {
    return e4m3 ? 0x7fU : 0x7eU;
  }
  const std::uint8_t sign = std::signbit(value) ? 0x80U : 0U;
  const float magnitude = std::abs(value);
  const std::vector<float>& values = positive_fp8_values(e4m3);
  const auto upper = std::lower_bound(
      values.begin(), values.end(), magnitude);
  std::size_t selected = 0;
  if (upper == values.end()) {
    selected = values.size() - 1;
  } else if (upper == values.begin()) {
    selected = 0;
  } else {
    const std::size_t high = static_cast<std::size_t>(
        upper - values.begin());
    const std::size_t low = high - 1;
    const float low_distance = magnitude - values[low];
    const float high_distance = values[high] - magnitude;
    if (high_distance < low_distance) {
      selected = high;
    } else if (low_distance < high_distance) {
      selected = low;
    } else {
      selected = (low & 1U) == 0U ? low : high;
    }
  }
  return static_cast<std::uint8_t>(
      sign | static_cast<std::uint8_t>(selected));
}

std::uint8_t float_to_e8m0(float value) {
  if (value == kPaddingSentinel) return 253;
  if (!(value >= 0.0F) || !std::isfinite(value)) return 255;
  if (value == 0.0F) return 0;
  const double exponent = std::log2(static_cast<double>(value));
  return static_cast<std::uint8_t>(
      std::clamp(std::round(exponent) + 127.0, 0.0, 254.0));
}

float e8m0_to_float(std::uint8_t value) {
  return value == 255 ? std::numeric_limits<float>::quiet_NaN()
                      : std::ldexp(1.0F, static_cast<int>(value) - 127);
}

float fp8_to_float(std::uint8_t bits, bool e4m3) {
  const std::uint8_t magnitude = bits & 0x7fU;
  if ((e4m3 && magnitude == 0x7fU) ||
      (!e4m3 && magnitude > 0x7cU)) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  if (!e4m3 && magnitude == 0x7cU) {
    return (bits & 0x80U) == 0U
               ? std::numeric_limits<float>::infinity()
               : -std::numeric_limits<float>::infinity();
  }
  const float value = positive_fp8_to_float(magnitude, e4m3);
  return (bits & 0x80U) == 0U ? value : -value;
}

void validate_tensor(const TensorDescriptor& tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("tensor dimensions/strides are invalid");
  }
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "tensor dimensions/strides must be positive");
    }
  }
  static_cast<void>(data_type_size(tensor.data_type));
}

std::uint16_t read_u16(
    std::span<const std::uint8_t> bytes, std::size_t offset) {
  return static_cast<std::uint16_t>(bytes[offset]) |
         static_cast<std::uint16_t>(bytes[offset + 1]) << 8U;
}

void append_u16(
    std::vector<std::uint8_t>& output,
    std::size_t offset,
    std::uint16_t value) {
  output[offset] = static_cast<std::uint8_t>(value & 0xffU);
  output[offset + 1] = static_cast<std::uint8_t>(value >> 8U);
}

}  // namespace

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
  throw std::invalid_argument(
      "mthreads validation tensor data type is unsupported");
}

std::size_t element_count(
    const TensorDescriptor& tensor) {
  validate_tensor(tensor);
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (result > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error("tensor element count overflows size_t");
    }
    result *= value;
  }
  return result;
}

std::size_t storage_element_count(
    const TensorDescriptor& tensor) {
  validate_tensor(tensor);
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[axis]);
    const std::size_t stride =
        static_cast<std::size_t>(tensor.strides[axis]);
    if (dimension - 1 >
        (std::numeric_limits<std::size_t>::max() - maximum_offset) /
            stride) {
      throw std::overflow_error("tensor storage span overflows size_t");
    }
    maximum_offset += (dimension - 1) * stride;
  }
  return maximum_offset + 1;
}

std::vector<std::size_t> logical_offsets(
    const TensorDescriptor& tensor) {
  const std::size_t elements = element_count(tensor);
  std::vector<std::size_t> result;
  result.reserve(elements);
  for (std::size_t linear = 0; linear < elements; ++linear) {
    std::size_t remaining = linear;
    std::size_t offset = 0;
    for (std::size_t trailing = 0;
         trailing < tensor.dimensions.size(); ++trailing) {
      const std::size_t axis = tensor.dimensions.size() - 1 - trailing;
      const std::size_t dimension =
          static_cast<std::size_t>(tensor.dimensions[axis]);
      const std::size_t coordinate = remaining % dimension;
      remaining /= dimension;
      offset += coordinate * static_cast<std::size_t>(tensor.strides[axis]);
    }
    result.push_back(offset);
  }
  return result;
}

std::vector<float> make_input(
    const TensorDescriptor& tensor,
    std::size_t input_index) {
  std::vector<float> result(element_count(tensor));
  for (std::size_t index = 0; index < result.size(); ++index) {
    const int centered = static_cast<int>(
        (index * 17U + input_index * 11U) % 41U) - 20;
    result[index] = tensor.data_type == FLAGDNN_DATA_FP8_E8M0
                        ? std::ldexp(1.0F, centered % 7)
                    : tensor.data_type == FLAGDNN_DATA_INT32
                        ? static_cast<float>(centered)
                    : tensor.data_type == FLAGDNN_DATA_BOOLEAN
                        ? static_cast<float>(index % 2)
                        : static_cast<float>(centered) /
                              static_cast<float>(13U + input_index);
  }
  return result;
}

std::vector<float> scatter(
    std::span<const float> logical,
    const TensorDescriptor& tensor,
    float padding) {
  const std::vector<std::size_t> offsets = logical_offsets(tensor);
  if (logical.size() != offsets.size()) {
    throw std::invalid_argument("logical tensor value count differs");
  }
  std::vector<float> result(storage_element_count(tensor), padding);
  for (std::size_t index = 0; index < offsets.size(); ++index) {
    result[offsets[index]] = logical[index];
  }
  return result;
}

std::vector<float> gather(
    std::span<const float> physical,
    const TensorDescriptor& tensor) {
  if (physical.size() != storage_element_count(tensor)) {
    throw std::invalid_argument("physical tensor value count differs");
  }
  const std::vector<std::size_t> offsets = logical_offsets(tensor);
  std::vector<float> result;
  result.reserve(offsets.size());
  for (const std::size_t offset : offsets) {
    result.push_back(physical[offset]);
  }
  return result;
}

std::vector<std::uint8_t> encode(
    std::span<const float> values, flagdnnDataType_t data_type) {
  const std::size_t width = data_type_size(data_type);
  if (values.size() > std::numeric_limits<std::size_t>::max() / width) {
    throw std::overflow_error("encoded tensor size overflows size_t");
  }
  std::vector<std::uint8_t> output(values.size() * width);
  for (std::size_t index = 0; index < values.size(); ++index) {
    const std::size_t offset = index * width;
    if (data_type == FLAGDNN_DATA_FLOAT32 || data_type == FLAGDNN_DATA_INT32) {
      const std::uint32_t bits =
          data_type == FLAGDNN_DATA_INT32
              ? std::bit_cast<std::uint32_t>(
                    static_cast<std::int32_t>(values[index]))
              : std::bit_cast<std::uint32_t>(values[index]);
      for (unsigned int byte = 0; byte < 4; ++byte) {
        output[offset + byte] = static_cast<std::uint8_t>(
            (bits >> (byte * 8U)) & 0xffU);
      }
    } else if (data_type == FLAGDNN_DATA_FLOAT16) {
      append_u16(output, offset, float_to_half(values[index]));
    } else if (data_type == FLAGDNN_DATA_BFLOAT16) {
      append_u16(output, offset, float_to_bfloat16(values[index]));
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      if (values[index] == kPaddingSentinel) {
        output[offset] = kBooleanPaddingSentinel;
      } else if (values[index] == 0.0F || values[index] == 1.0F) {
        output[offset] = static_cast<std::uint8_t>(values[index]);
      } else {
        throw std::invalid_argument(
            "BOOLEAN encoding requires canonical 0/1 values");
      }
    } else if (data_type == FLAGDNN_DATA_FP8_E8M0) {
      output[offset] = float_to_e8m0(values[index]);
    } else if (data_type == FLAGDNN_DATA_FP8_E4M3) {
      output[offset] = float_to_fp8(values[index], true);
    } else if (data_type == FLAGDNN_DATA_FP8_E5M2) {
      output[offset] = float_to_fp8(values[index], false);
    }
  }
  return output;
}

std::vector<float> decode(
    std::span<const std::uint8_t> bytes, flagdnnDataType_t data_type) {
  const std::size_t width = data_type_size(data_type);
  if (bytes.size() % width != 0) {
    throw std::invalid_argument("encoded tensor byte count is misaligned");
  }
  std::vector<float> output(bytes.size() / width);
  for (std::size_t index = 0; index < output.size(); ++index) {
    const std::size_t offset = index * width;
    if (data_type == FLAGDNN_DATA_FLOAT32 || data_type == FLAGDNN_DATA_INT32) {
      std::uint32_t bits = 0;
      for (unsigned int byte = 0; byte < 4; ++byte) {
        bits |= static_cast<std::uint32_t>(bytes[offset + byte])
                << (byte * 8U);
      }
      output[index] =
          data_type == FLAGDNN_DATA_INT32
              ? static_cast<float>(std::bit_cast<std::int32_t>(bits))
              : std::bit_cast<float>(bits);
    } else if (data_type == FLAGDNN_DATA_FLOAT16) {
      output[index] = half_to_float(read_u16(bytes, offset));
    } else if (data_type == FLAGDNN_DATA_BFLOAT16) {
      output[index] = bfloat16_to_float(read_u16(bytes, offset));
    } else if (data_type == FLAGDNN_DATA_BOOLEAN) {
      output[index] = static_cast<float>(bytes[offset]);
    } else if (data_type == FLAGDNN_DATA_FP8_E8M0) {
      output[index] = e8m0_to_float(bytes[offset]);
    } else if (data_type == FLAGDNN_DATA_FP8_E4M3) {
      output[index] = fp8_to_float(bytes[offset], true);
    } else if (data_type == FLAGDNN_DATA_FP8_E5M2) {
      output[index] = fp8_to_float(bytes[offset], false);
    }
  }
  return output;
}

float quantize_scalar(float value, flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return static_cast<float>(static_cast<std::int32_t>(value));

    case FLAGDNN_DATA_FP8_E8M0:
      return e8m0_to_float(float_to_e8m0(value));
    case FLAGDNN_DATA_FLOAT32:
      return value;
    case FLAGDNN_DATA_FLOAT16:
      return half_to_float(float_to_half(value));
    case FLAGDNN_DATA_BFLOAT16:
      return bfloat16_to_float(float_to_bfloat16(value));
    case FLAGDNN_DATA_FP8_E4M3:
      return fp8_to_float(float_to_fp8(value, true), true);
    case FLAGDNN_DATA_FP8_E5M2:
      return fp8_to_float(float_to_fp8(value, false), false);
    case FLAGDNN_DATA_BOOLEAN:
      break;
  }
  throw std::invalid_argument(
      "scalar quantization requires a numeric tensor type");
}

void require_padding_unchanged(
    std::string_view provider,
    std::span<const std::uint8_t> encoded,
    const TensorDescriptor& tensor,
    float padding) {
  const std::size_t width = data_type_size(tensor.data_type);
  const std::size_t storage = storage_element_count(tensor);
  if (encoded.size() != storage * width) {
    throw std::invalid_argument("encoded padding buffer size differs");
  }
  std::vector<bool> logical(storage, false);
  for (const std::size_t offset : logical_offsets(tensor)) {
    logical[offset] = true;
  }
  const std::vector<std::uint8_t> expected =
      encode(std::span<const float>(&padding, 1), tensor.data_type);
  for (std::size_t element = 0; element < storage; ++element) {
    if (logical[element]) {
      continue;
    }
    const auto actual = encoded.subspan(element * width, width);
    if (!std::equal(actual.begin(), actual.end(), expected.begin())) {
      throw std::runtime_error(
          std::string(provider) + " modified tensor padding at element " +
          std::to_string(element));
    }
  }
}

void require_bytes_equal(
    std::string_view description,
    std::span<const std::uint8_t> actual,
    std::span<const std::uint8_t> expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(
        std::string(description) + " byte count differs");
  }
  const auto mismatch = std::mismatch(
      actual.begin(), actual.end(), expected.begin(), expected.end());
  if (mismatch.first != actual.end()) {
    throw std::runtime_error(
        std::string(description) + " changed at byte " +
        std::to_string(
            static_cast<std::size_t>(mismatch.first - actual.begin())));
  }
}

}  // namespace flagdnn::validation::mthreads::tensor_io
