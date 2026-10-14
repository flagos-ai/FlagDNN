/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_

#include "reference/cpu/pointwise.hpp"

namespace flagdnn::validation::ascend::sigmoid_backward_contract {

// FlagDNN's binary pointwise ports are ordered as (gradient, logit).  ACLNN's
// backward primitive instead consumes (gradient, sigmoid(logit)), so both the
// direct reference and the independent host oracle share this explicit
// semantic boundary.
using flagdnn::reference::cpu::sigmoid;
[[nodiscard]] inline float evaluate(float gradient, float logit) noexcept {
  return flagdnn::reference::cpu::sigmoid_backward(gradient, logit);
}

}  // namespace flagdnn::validation::ascend::sigmoid_backward_contract

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_SIGMOID_BACKWARD_CONTRACT_HPP_
