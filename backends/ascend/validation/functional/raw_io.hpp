/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "validation/functional/aclnn_plan.hpp"
#include <cstring>
#include <type_traits>
namespace flagdnn::testing::ascend {
inline std::vector<std::uint8_t>
logical_bytes(std::span<const std::uint8_t> bytes, const TestTensor &t) {
  if (bytes.size() != io::encoded_byte_count(t))
    throw std::invalid_argument("incorrect raw tensor storage byte count");
  const auto width = io::data_type_size(t.data_type),
             count = io::element_count(t);
  std::vector<std::uint8_t> result(count * width);
  for (std::size_t i = 0; i < count; ++i)
    std::memcpy(result.data() + i * width,
                bytes.data() + io::physical_offset_unchecked(i, t) * width,
                width);
  return result;
}
template <class T>
std::vector<T> logical_values(std::span<const std::uint8_t> bytes,
                              const TestTensor &t) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (sizeof(T) != io::data_type_size(t.data_type))
    throw std::invalid_argument("incorrect raw tensor value width");
  const auto gathered = logical_bytes(bytes, t);
  std::vector<T> result(gathered.size() / sizeof(T));
  std::memcpy(result.data(), gathered.data(), gathered.size());
  return result;
}
template <class T>
std::vector<std::uint8_t> value_bytes(std::span<const T> values) {
  static_assert(std::is_trivially_copyable_v<T>);
  std::vector<std::uint8_t> result(values.size_bytes());
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}
inline void check_raw_padding(std::span<const std::uint8_t> bytes,
                              const TestTensor &t) {
  if (bytes.size() != io::encoded_byte_count(t))
    throw std::invalid_argument("incorrect raw tensor storage byte count");
  const auto width = io::data_type_size(t.data_type),
             count = io::storage_element_count(t);
  std::vector<bool> used(count);
  for (std::size_t i = 0; i < io::element_count(t); ++i)
    used[io::physical_offset_unchecked(i, t)] = true;
  const auto sentinel =
      io::encode(std::vector<float>{io::kPaddingSentinel}, t.data_type);
  for (std::size_t i = 0; i < count; ++i)
    if (!used[i] && !std::equal(sentinel.begin(), sentinel.end(),
                                bytes.begin() + i * width))
      throw std::runtime_error("FlagDNN modified output padding");
}
template <class Generator>
std::vector<std::uint8_t> storage_bytes(const TestTensor &t, Generator value) {
  if (!io::is_fp8_storage(t.data_type))
    throw std::invalid_argument("raw byte generator requires FP8 storage");
  std::vector<std::uint8_t> result(io::encoded_byte_count(t),
                                  io::kBytePaddingSentinel);
  for (std::size_t i = 0; i < io::element_count(t); ++i)
    result[io::physical_offset_unchecked(i, t)] =
        static_cast<std::uint8_t>(value(i));
  return result;
}
template <class Generator>
std::vector<std::uint8_t> integer_bytes(const TestTensor &t, Generator value) {
  auto result = io::encode(
      std::vector<float>(io::storage_element_count(t), io::kPaddingSentinel),
      t.data_type);
  for (std::size_t i = 0; i < io::element_count(t); ++i) {
    const std::int32_t v = value(i);
    std::memcpy(result.data() + 4 * io::physical_offset_unchecked(i, t), &v, 4);
  }
  return result;
}
} // namespace flagdnn::testing::ascend
