// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "common/matmul.hpp"

#include <stdexcept>

namespace flagdnn::testing {

std::unique_ptr<MatmulExecutable>
build_matmul_reference(const MatmulTestCase &) {
  throw std::logic_error(
      "MatMul has no exact CoreX cuDNN 7.6.5 classic primitive");
}

} // namespace flagdnn::testing
