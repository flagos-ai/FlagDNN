/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
namespace flagdnn::reference::cpu::detail {
inline std::size_t byte_count(std::span<const std::int64_t> shape,
                              std::size_t element_bytes = 1) {
  if (shape.empty() || element_bytes == 0)
    throw std::invalid_argument("CPU reference shape/element width is empty");
  auto count = element_bytes;
  for (const auto dimension : shape) {
    if (dimension <= 0)
      throw std::invalid_argument("CPU reference dimension must be positive");
    if (static_cast<std::uint64_t>(dimension) >
        std::numeric_limits<std::size_t>::max() / count)
      throw std::overflow_error("CPU reference shape overflows");
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}
}  // namespace flagdnn::reference::cpu::detail
