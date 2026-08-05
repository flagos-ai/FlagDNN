/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_TENSOR_IO_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_TENSOR_IO_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::nvidia::tensor_io {

inline constexpr float kPaddingSentinel = -97.0F;

enum class BooleanEncoding {
  kByte,
  kBitPacked,
};

[[nodiscard]] float padding_sentinel() noexcept;
[[nodiscard]] std::size_t data_type_size(flagdnnDataType_t data_type);

// Both functional TestTensor and benchmark TensorSpec intentionally satisfy
// this small structural contract: data_type, dimensions, and strides.
template <typename Tensor>
[[nodiscard]] std::size_t element_count(const Tensor& tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument("validation tensor dimension must be positive");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::size_t storage_element_count(const Tensor& tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("validation tensor metadata is invalid");
  }
  if (tensor.dimensions.empty()) {
    return 1;
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "validation tensor dimensions and strides must be positive");
    }
    maximum_offset +=
        static_cast<std::size_t>(tensor.dimensions[axis] - 1) *
        static_cast<std::size_t>(tensor.strides[axis]);
  }
  return maximum_offset + 1;
}

template <typename Tensor>
[[nodiscard]] std::size_t logical_offset(std::size_t logical_index,
                                         const Tensor& tensor) {
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
  std::vector<float> physical(
      storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    physical[logical_offset(index, tensor)] = logical[index];
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
    logical[index] = physical[logical_offset(index, tensor)];
  }
  return logical;
}

template <typename Tensor>
[[nodiscard]] std::size_t encoded_byte_count(
    const Tensor& tensor,
    BooleanEncoding boolean_encoding) {
  const std::size_t storage_count = storage_element_count(tensor);
  if (tensor.data_type == FLAGDNN_DATA_BOOLEAN &&
      boolean_encoding == BooleanEncoding::kBitPacked) {
    return (storage_count + 7U) / 8U;
  }
  return storage_count * data_type_size(tensor.data_type);
}

[[nodiscard]] std::vector<std::uint8_t> encode(
    std::span<const float> physical,
    flagdnnDataType_t data_type,
    BooleanEncoding boolean_encoding);

[[nodiscard]] std::vector<float> decode(
    std::span<const std::uint8_t> bytes,
    flagdnnDataType_t data_type,
    std::size_t physical_element_count,
    BooleanEncoding boolean_encoding);

template <typename Tensor>
void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const Tensor& tensor) {
  if (physical.size() < storage_element_count(tensor)) {
    throw std::invalid_argument("physical output is smaller than tensor storage");
  }
  std::vector<bool> occupied(physical.size(), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied[logical_offset(index, tensor)] = true;
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

}  // namespace flagdnn::validation::nvidia::tensor_io


// Preserve the compact functional-test spelling while allowing the existing
// cuDNN helpers to add their own members to this namespace.
namespace flagdnn::testing::cuda {
using validation::nvidia::tensor_io::BooleanEncoding;
using validation::nvidia::tensor_io::data_type_size;
using validation::nvidia::tensor_io::decode;
using validation::nvidia::tensor_io::element_count;
using validation::nvidia::tensor_io::encode;
using validation::nvidia::tensor_io::encoded_byte_count;
using validation::nvidia::tensor_io::gather;
using validation::nvidia::tensor_io::logical_offset;
using validation::nvidia::tensor_io::padding_sentinel;
using validation::nvidia::tensor_io::require_padding_unchanged;
using validation::nvidia::tensor_io::scatter;
using validation::nvidia::tensor_io::storage_element_count;
}  // namespace flagdnn::testing::cuda

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_TENSOR_IO_HPP_
