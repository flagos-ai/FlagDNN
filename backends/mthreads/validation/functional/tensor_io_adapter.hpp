/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_TENSOR_IO_ADAPTER_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_TENSOR_IO_ADAPTER_HPP_

#include "backends/mthreads/validation/tensor_io.hpp"
#include "common/common.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace flagdnn::validation::mthreads {

[[nodiscard]] inline TensorDescriptor describe_tensor(
    const flagdnn::testing::TestTensor& tensor) {
  return {
      tensor.uid,
      tensor.data_type,
      tensor.dimensions,
      tensor.strides,
      tensor.binding_byte_offset,
  };
}

namespace tensor_io {

[[nodiscard]] inline std::size_t element_count(
    const flagdnn::testing::TestTensor& tensor) {
  return element_count(describe_tensor(tensor));
}

[[nodiscard]] inline std::size_t storage_element_count(
    const flagdnn::testing::TestTensor& tensor) {
  return storage_element_count(describe_tensor(tensor));
}

[[nodiscard]] inline std::vector<std::size_t> logical_offsets(
    const flagdnn::testing::TestTensor& tensor) {
  return logical_offsets(describe_tensor(tensor));
}

[[nodiscard]] inline std::vector<float> make_input(
    const flagdnn::testing::TestTensor& tensor,
    std::size_t input_index) {
  return make_input(describe_tensor(tensor), input_index);
}

[[nodiscard]] inline std::vector<float> scatter(
    std::span<const float> logical,
    const flagdnn::testing::TestTensor& tensor,
    float padding = kPaddingSentinel) {
  return scatter(logical, describe_tensor(tensor), padding);
}

[[nodiscard]] inline std::vector<float> gather(
    std::span<const float> physical,
    const flagdnn::testing::TestTensor& tensor) {
  return gather(physical, describe_tensor(tensor));
}

inline void require_padding_unchanged(
    std::string_view provider,
    std::span<const std::uint8_t> encoded,
    const flagdnn::testing::TestTensor& tensor,
    float padding = kPaddingSentinel) {
  require_padding_unchanged(
      provider, encoded, describe_tensor(tensor), padding);
}

}  // namespace tensor_io
}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_TENSOR_IO_ADAPTER_HPP_
