/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_POINTWISE_ATTRIBUTES_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_POINTWISE_ATTRIBUTES_HPP_

#include <flagdnn/flagdnn.h>

namespace flagdnn::validation::mthreads {

// Functional and benchmark providers use frontend setters taking float.
// Match those effective attributes before invoking the generic CPU oracle;
// input/output dtype quantization does not round the double attribute fields.
[[nodiscard]] inline flagdnnPointwiseAttributes_t
normalize_frontend_pointwise_attributes(
    const flagdnnPointwiseAttributes_t& attributes) {
  auto result = attributes;
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    result.relu_lower_clip = static_cast<float>(result.relu_lower_clip);
  }
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U) {
    result.relu_upper_clip = static_cast<float>(result.relu_upper_clip);
  }
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U) {
    result.relu_lower_clip_slope =
        static_cast<float>(result.relu_lower_clip_slope);
  }
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
    result.swish_beta = static_cast<float>(result.swish_beta);
  }
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U) {
    result.elu_alpha = static_cast<float>(result.elu_alpha);
  }
  if ((result.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) != 0U) {
    result.softplus_beta = static_cast<float>(result.softplus_beta);
  }
  return result;
}

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_FUNCTIONAL_POINTWISE_ATTRIBUTES_HPP_
