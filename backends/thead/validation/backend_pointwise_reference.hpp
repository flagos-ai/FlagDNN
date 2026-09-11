// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_BACKEND_POINTWISE_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_BACKEND_POINTWISE_REFERENCE_HPP_

#include "common/common.hpp"

#include <acdnn.h>
#include <acdnn_backend.h>

#include <memory>
#include <string>
#include <vector>

namespace flagdnn::validation::thead {

struct BackendPointwiseReferenceSpecification {
  acdnnPointwiseMode_t mode = ACDNN_POINTWISE_IDENTITY_FWD;
  std::vector<flagdnn::testing::TestTensor> inputs;
  flagdnn::testing::TestTensor output;
  std::string primitive;
  double relu_lower_clip = 0.0;
  double relu_upper_clip = 0.0;
  double relu_lower_clip_slope = 0.0;
  double elu_alpha = 1.0;
  double softplus_beta = 1.0;
  double swish_beta = 1.0;
  float alpha1 = 1.0F;
  float alpha2 = 1.0F;
  bool constant_one_numerator = false;
  // Explicit raw storage access for the acDNN FP8 codec. Reading produces
  // signed byte values; writing accepts integral byte codes in [0, 255].
  // This flag never treats acDNN INT8 arithmetic as native FP8 arithmetic.
  bool fp8_storage_bytes = false;
};

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_backend_pointwise_reference(
    BackendPointwiseReferenceSpecification specification);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_BACKEND_POINTWISE_REFERENCE_HPP_
