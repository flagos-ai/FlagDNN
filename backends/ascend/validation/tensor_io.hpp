/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_TENSOR_IO_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_TENSOR_IO_HPP_

#include <flagdnn/flagdnn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::ascend::tensor_io {

inline constexpr float kPaddingSentinel = -97.0F;
inline constexpr std::uint8_t kBooleanPaddingSentinel = 0xA5U;
inline constexpr std::uint8_t kBytePaddingSentinel = 0xA5U;

[[nodiscard]] constexpr bool is_fp8_storage(flagdnnDataType_t type) noexcept {
  return type == FLAGDNN_DATA_FP8_E4M3 || type == FLAGDNN_DATA_FP8_E5M2 ||
         type == FLAGDNN_DATA_FP8_E8M0;
}

[[nodiscard]] constexpr bool is_raw_storage(flagdnnDataType_t type) noexcept {
  return type == FLAGDNN_DATA_INT32 || type == FLAGDNN_DATA_BOOLEAN ||
         is_fp8_storage(type);
}

[[nodiscard]] std::size_t data_type_size(flagdnnDataType_t data_type);

[[nodiscard]] inline std::size_t checked_add(std::size_t left,
                                             std::size_t right,
                                             std::string_view description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::overflow_error(std::string(description) + " overflows size_t");
  }
  return left + right;
}

[[nodiscard]] inline std::size_t checked_multiply(
    std::size_t left,
    std::size_t right,
    std::string_view description) {
  if (left != 0 &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error(std::string(description) + " overflows size_t");
  }
  return left * right;
}

template <typename Tensor>
void validate_layout(const Tensor& tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("validation tensor metadata is invalid");
  }

  struct Axis {
    std::size_t stride;
    std::size_t dimension;
  };
  std::vector<Axis> axes;
  axes.reserve(tensor.dimensions.size());
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "validation tensor dimensions and strides must be positive");
    }
    if (tensor.dimensions[axis] > 1) {
      axes.push_back(
          {static_cast<std::size_t>(tensor.strides[axis]),
           static_cast<std::size_t>(tensor.dimensions[axis])});
    }
  }
  std::sort(axes.begin(), axes.end(), [](const Axis& left, const Axis& right) {
    return left.stride < right.stride;
  });
  std::size_t required_span = 1;
  for (const Axis& axis : axes) {
    if (axis.stride < required_span) {
      throw std::invalid_argument("validation tensor layout overlaps");
    }
    required_span = checked_add(
        required_span,
        checked_multiply(axis.dimension - 1,
                         axis.stride,
                         "validation tensor storage span"),
        "validation tensor storage span");
  }
}

template <typename Tensor>
[[nodiscard]] std::size_t element_count(const Tensor& tensor) {
  validate_layout(tensor);
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    result = checked_multiply(result,
                              static_cast<std::size_t>(dimension),
                              "validation tensor element count");
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::size_t storage_element_count(const Tensor& tensor) {
  validate_layout(tensor);
  std::size_t result = 1;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    result = checked_add(
        result,
        checked_multiply(
            static_cast<std::size_t>(tensor.dimensions[axis] - 1),
            static_cast<std::size_t>(tensor.strides[axis]),
            "validation tensor storage span"),
        "validation tensor storage span");
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::size_t encoded_byte_count(const Tensor& tensor) {
  return checked_multiply(storage_element_count(tensor),
                          data_type_size(tensor.data_type),
                          "validation tensor byte count");
}

template <typename Tensor>
[[nodiscard]] std::size_t allocation_byte_count(const Tensor& tensor) {
  return checked_add(tensor.binding_byte_offset,
                     encoded_byte_count(tensor),
                     "validation tensor allocation size");
}

template <typename Tensor>
[[nodiscard]] std::size_t physical_offset(std::size_t logical_index,
                                          const Tensor& tensor) {
  if (logical_index >= element_count(tensor)) {
    throw std::out_of_range("logical tensor index is out of range");
  }
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    const std::size_t coordinate = logical_index % dimension;
    logical_index /= dimension;
    result = checked_add(
        result,
        checked_multiply(coordinate,
                         static_cast<std::size_t>(tensor.strides[current]),
                         "validation tensor physical offset"),
        "validation tensor physical offset");
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::size_t physical_offset_unchecked(
    std::size_t logical_index,
    const Tensor& tensor) noexcept {
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

template <typename Tensor>
[[nodiscard]] std::vector<float> scatter(std::span<const float> logical,
                                         const Tensor& tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("logical input size does not match tensor");
  }
  std::vector<float> physical(storage_element_count(tensor),
                              kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    physical[physical_offset_unchecked(index, tensor)] = logical[index];
  }
  return physical;
}

template <typename Tensor>
[[nodiscard]] std::vector<float> gather(std::span<const float> physical,
                                        const Tensor& tensor) {
  if (physical.size() < storage_element_count(tensor)) {
    throw std::invalid_argument("physical input is smaller than tensor storage");
  }
  std::vector<float> logical(element_count(tensor));
  for (std::size_t index = 0; index < logical.size(); ++index) {
    logical[index] = physical[physical_offset_unchecked(index, tensor)];
  }
  return logical;
}

template <typename Tensor>
void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const Tensor& tensor) {
  if (physical.size() < storage_element_count(tensor)) {
    throw std::invalid_argument("physical output is smaller than tensor storage");
  }
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied[physical_offset_unchecked(index, tensor)] = true;
  }
  for (std::size_t index = 0; index < physical.size(); ++index) {
    if (!occupied[index] && physical[index] != kPaddingSentinel) {
      throw std::runtime_error(
          std::string(provider) +
          " modified output padding at storage element " +
          std::to_string(index));
    }
  }
}

[[nodiscard]] std::vector<std::uint8_t> encode(
    std::span<const float> values,
    flagdnnDataType_t data_type);

[[nodiscard]] std::vector<float> decode(
    std::span<const std::uint8_t> bytes,
    flagdnnDataType_t data_type,
    std::size_t element_count);

template <typename Tensor>
[[nodiscard]] std::vector<float> decode_storage(
    std::string_view provider,
    std::span<const std::uint8_t> bytes,
    const Tensor& tensor) {
  const std::size_t storage_count = storage_element_count(tensor);
  if (bytes.size() != encoded_byte_count(tensor)) {
    throw std::invalid_argument("encoded tensor storage byte count is invalid");
  }
  if (tensor.data_type != FLAGDNN_DATA_BOOLEAN) {
    return decode(bytes, tensor.data_type, storage_count);
  }

  std::vector<bool> occupied(storage_count, false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied[physical_offset_unchecked(index, tensor)] = true;
  }

  // Check padding before interpreting any truth value.  This ordering makes a
  // padding write impossible to hide behind a simultaneously non-canonical
  // result byte.
  for (std::size_t index = 0; index < storage_count; ++index) {
    if (!occupied[index] && bytes[index] != kBooleanPaddingSentinel) {
      throw std::runtime_error(
          std::string(provider) +
          " modified BOOLEAN output padding at storage byte " +
          std::to_string(index));
    }
  }

  std::vector<float> physical(storage_count, kPaddingSentinel);
  for (std::size_t logical_index = 0;
       logical_index < element_count(tensor);
       ++logical_index) {
    const std::size_t storage_index =
        physical_offset_unchecked(logical_index, tensor);
    const std::uint8_t value = bytes[storage_index];
    if (value > 1U) {
      throw std::runtime_error(
          std::string(provider) +
          " produced a non-canonical BOOLEAN value at logical element " +
          std::to_string(logical_index));
    }
    physical[storage_index] = static_cast<float>(value);
  }
  return physical;
}

[[nodiscard]] std::vector<float> quantize(
    std::span<const float> values,
    flagdnnDataType_t data_type);

[[nodiscard]] float quantize_scalar(double value,
                                    flagdnnDataType_t data_type);

}  // namespace flagdnn::validation::ascend::tensor_io

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_TENSOR_IO_HPP_
