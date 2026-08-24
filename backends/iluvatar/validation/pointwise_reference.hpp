// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_POINTWISE_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_POINTWISE_REFERENCE_HPP_

#include "common/common.hpp"

#include <flagdnn/flagdnn.h>

#include <memory>
#include <span>
#include <vector>

namespace flagdnn::iluvatar::validation {

struct ClassicPointwiseReferenceSpec {
  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_NOT_SET;
  std::vector<flagdnn::testing::TestTensor> inputs;
  flagdnn::testing::TestTensor output;
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  double alpha = 1.0;
};

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_pointwise_reference(ClassicPointwiseReferenceSpec specification);

[[nodiscard]] std::unique_ptr<flagdnn::testing::TestExecutable>
make_classic_add_square_reference(const flagdnn::testing::TestTensor &left,
                                  const flagdnn::testing::TestTensor &right,
                                  const flagdnn::testing::TestTensor &output);

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_POINTWISE_REFERENCE_HPP_
