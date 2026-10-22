/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_REFERENCE_CPU_POINTWISE_INT32_HPP_
#define FLAGDNN_REFERENCE_CPU_POINTWISE_INT32_HPP_

#include <bit>
#include <cstdint>

namespace flagdnn::reference::cpu {

// INT32 arithmetic wraps modulo 2^32. Keep all operand bits; converting via
// float would lose adjacent integers above 2^24 and mishandle overflow.
[[nodiscard]] inline std::int32_t subtract_int32(
    std::int32_t left, std::int32_t right, std::int32_t alpha = 1) noexcept {
  const auto a = std::bit_cast<std::uint32_t>(left);
  const auto b = std::bit_cast<std::uint32_t>(right);
  const auto scale = std::bit_cast<std::uint32_t>(alpha);
  return std::bit_cast<std::int32_t>(a - scale * b);
}

}  // namespace flagdnn::reference::cpu
#endif
