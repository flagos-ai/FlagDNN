/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_HPP_

#include <flagdnn/flagdnn.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flagdnn::validation::mthreads {

// Validation-owned descriptor shared by the functional and benchmark
// adapters. It deliberately has no dependency on either common test model.
// binding_byte_offset describes the test allocation entrance only; every
// flagdnnBinding_t passed to an executable already points at that entrance.
struct TensorDescriptor {
  std::int64_t uid = 0;
  flagdnnDataType_t data_type = FLAGDNN_DATA_FLOAT32;
  std::vector<std::int64_t> dimensions;
  std::vector<std::int64_t> strides;
  std::size_t binding_byte_offset = 0;
};

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_TENSOR_HPP_
