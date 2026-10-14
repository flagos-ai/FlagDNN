/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_

#include "reference/cpu/pointwise.hpp"
#include <cstddef>

namespace flagdnn::validation::ascend::gelu_contract {

// Keep the production semantic check strict.  CANN 9's direct aclnnGelu
// reference can differ from the mathematical erf definition by the same scale
// as the standard tanh approximation, so only ACLNN-host and triangle checks
// receive the wider, measured reference allowance below.
inline constexpr double kFlagdnnAbsoluteTolerance = 2.0e-5;
inline constexpr double kFlagdnnRelativeTolerance = 2.0e-5;
inline constexpr double kAclnnExactAbsoluteTolerance = 5.0e-4;
inline constexpr double kAclnnExactRelativeTolerance = 5.0e-4;
inline constexpr double kAccurateLogitMaximumAbsoluteError = 5.0e-6;

inline constexpr std::size_t kScaledSentinelIndex = 21;
inline constexpr std::size_t kScaledElementCount = 24;

using flagdnn::reference::cpu::gelu_exact;
using flagdnn::reference::cpu::gelu_approximate_tanh;
using flagdnn::reference::cpu::gelu_accurate_logit;
[[nodiscard]] inline float exact(float value) noexcept {
  return gelu_exact(value);
}
[[nodiscard]] inline float approximate_tanh(float value) noexcept {
  return gelu_approximate_tanh(value);
}
[[nodiscard]] inline float accurate_logit(float value) noexcept {
  return gelu_accurate_logit(value);
}

[[nodiscard]] inline float scaled_input(std::size_t index) noexcept {
  const int centered = static_cast<int>((index * 17U) % 41U) - 20;
  return static_cast<float>(centered) / 13.0F * 4.0F;
}

}  // namespace flagdnn::validation::ascend::gelu_contract

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_GELU_CONTRACT_HPP_
