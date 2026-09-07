// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_POINTWISE_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_POINTWISE_REFERENCE_HPP_

#include "capability.hpp"
#include "common/common.hpp"

#include <memory>
#include <vector>

namespace flagdnn::validation::thead {

struct PointwiseReferenceSpecification {
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_ADD;
  std::vector<flagdnn::testing::TestTensor> inputs;
  flagdnn::testing::TestTensor output;
  double alpha = 1.0;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
};

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_acdnn_pointwise_reference(
    const PointwiseReferenceSpecification &specification,
    const CapabilityRecord &capability);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_POINTWISE_REFERENCE_HPP_
