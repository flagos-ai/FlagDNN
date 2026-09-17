/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_IO_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_IO_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::hygon::tensor_io {

inline constexpr float kPaddingSentinel = -97.0F;

[[nodiscard]] std::size_t data_type_size(flagdnnDataType_t data_type);

template <typename Tensor>
[[nodiscard]] std::size_t element_count(const Tensor &tensor) {
  std::size_t result = 1;
  for (const std::int64_t dimension : tensor.dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(
          "validation tensor dimension must be positive");
    }
    if (result > std::numeric_limits<std::size_t>::max() /
                     static_cast<std::size_t>(dimension)) {
      throw std::overflow_error("validation tensor element count overflows");
    }
    result *= static_cast<std::size_t>(dimension);
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::size_t storage_element_count(const Tensor &tensor) {
  if (tensor.dimensions.size() != tensor.strides.size()) {
    throw std::invalid_argument("validation tensor metadata is invalid");
  }
  std::size_t maximum_offset = 0;
  for (std::size_t axis = 0; axis < tensor.dimensions.size(); ++axis) {
    if (tensor.dimensions[axis] <= 0 || tensor.strides[axis] <= 0) {
      throw std::invalid_argument(
          "validation tensor dimensions and strides must be positive");
    }
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[axis] - 1);
    const std::size_t stride = static_cast<std::size_t>(tensor.strides[axis]);
    if (dimension != 0 &&
        stride > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("validation tensor storage size overflows");
    }
    const std::size_t contribution = dimension * stride;
    if (maximum_offset >=
        std::numeric_limits<std::size_t>::max() - contribution) {
      throw std::overflow_error("validation tensor storage size overflows");
    }
    maximum_offset += contribution;
  }
  return maximum_offset + 1;
}

template <typename Tensor>
[[nodiscard]] std::size_t logical_offset(std::size_t logical_index,
                                         const Tensor &tensor) {
  std::size_t result = 0;
  for (std::size_t axis = tensor.dimensions.size(); axis != 0; --axis) {
    const std::size_t current = axis - 1;
    const std::size_t dimension =
        static_cast<std::size_t>(tensor.dimensions[current]);
    result += (logical_index % dimension) *
              static_cast<std::size_t>(tensor.strides[current]);
    logical_index /= dimension;
  }
  return result;
}

template <typename Tensor>
[[nodiscard]] std::vector<float> scatter(std::span<const float> logical,
                                         const Tensor &tensor) {
  if (logical.size() != element_count(tensor)) {
    throw std::invalid_argument("logical input size does not match tensor");
  }
  std::vector<float> physical(storage_element_count(tensor), kPaddingSentinel);
  for (std::size_t index = 0; index < logical.size(); ++index) {
    physical[logical_offset(index, tensor)] = logical[index];
  }
  return physical;
}

template <typename Tensor>
[[nodiscard]] std::vector<float> gather(std::span<const float> physical,
                                        const Tensor &tensor) {
  if (physical.size() < storage_element_count(tensor)) {
    throw std::invalid_argument(
        "physical input is smaller than tensor storage");
  }
  std::vector<float> logical(element_count(tensor));
  for (std::size_t index = 0; index < logical.size(); ++index) {
    logical[index] = physical[logical_offset(index, tensor)];
  }
  return logical;
}

[[nodiscard]] std::vector<std::uint8_t> encode(std::span<const float> physical,
                                               flagdnnDataType_t data_type);
[[nodiscard]] std::vector<float> decode(std::span<const std::uint8_t> bytes,
                                        flagdnnDataType_t data_type,
                                        std::size_t physical_element_count);

template <typename Tensor>
void require_padding_unchanged(std::string_view provider,
                               std::span<const float> physical,
                               const Tensor &tensor) {
  std::vector<bool> occupied(storage_element_count(tensor), false);
  for (std::size_t index = 0; index < element_count(tensor); ++index) {
    occupied[logical_offset(index, tensor)] = true;
  }
  for (std::size_t index = 0; index < occupied.size(); ++index) {
    if (!occupied[index] && physical[index] != kPaddingSentinel) {
      throw std::runtime_error(std::string(provider) +
                               " modified output padding at element " +
                               std::to_string(index));
    }
  }
}

} // namespace flagdnn::validation::hygon::tensor_io

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_TENSOR_IO_HPP_
