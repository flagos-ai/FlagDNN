// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_REFERENCE_TENSOR_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_REFERENCE_TENSOR_HPP_

#include "common/common.hpp"

#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace flagdnn::iluvatar::validation {

struct ReferenceTensor {
  std::int64_t uid = 0;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  std::vector<int> dimensions;
  std::vector<int> strides;
  std::size_t byte_offset = 0;
};

[[nodiscard]] cudnnDataType_t
corex_cudnn_data_type(flagdnnDataType_t data_type);
[[nodiscard]] std::size_t flagdnn_data_type_size(flagdnnDataType_t data_type);
[[nodiscard]] ReferenceTensor
make_reference_tensor(const flagdnn::testing::TestTensor &tensor);

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_REFERENCE_TENSOR_HPP_
