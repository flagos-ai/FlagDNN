/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_IO_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_IO_HPP_

#include "backends/mthreads/validation/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace flagdnn::validation::mthreads::tensor_io {

inline constexpr float kPaddingSentinel = 137.25F;

[[nodiscard]] std::size_t data_type_size(flagdnnDataType_t data_type);
[[nodiscard]] std::size_t element_count(
    const TensorDescriptor& tensor);
[[nodiscard]] std::size_t storage_element_count(
    const TensorDescriptor& tensor);
[[nodiscard]] std::vector<std::size_t> logical_offsets(
    const TensorDescriptor& tensor);
[[nodiscard]] std::vector<float> make_input(
    const TensorDescriptor& tensor,
    std::size_t input_index);
[[nodiscard]] std::vector<float> scatter(
    std::span<const float> logical,
    const TensorDescriptor& tensor,
    float padding = kPaddingSentinel);
[[nodiscard]] std::vector<float> gather(
    std::span<const float> physical,
    const TensorDescriptor& tensor);
[[nodiscard]] std::vector<std::uint8_t> encode(
    std::span<const float> values, flagdnnDataType_t data_type);
[[nodiscard]] std::vector<float> decode(
    std::span<const std::uint8_t> bytes, flagdnnDataType_t data_type);
[[nodiscard]] float quantize_scalar(
    float value, flagdnnDataType_t data_type);

void require_padding_unchanged(
    std::string_view provider,
    std::span<const std::uint8_t> encoded,
    const TensorDescriptor& tensor,
    float padding = kPaddingSentinel);
void require_bytes_equal(
    std::string_view description,
    std::span<const std::uint8_t> actual,
    std::span<const std::uint8_t> expected);

}  // namespace flagdnn::validation::mthreads::tensor_io

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_IO_HPP_
