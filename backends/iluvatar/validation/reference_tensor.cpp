// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "reference_tensor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace flagdnn::iluvatar::validation {
namespace {

int checked_int(std::int64_t value, const char *name) {
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("cuDNN ") + name +
                                " does not fit the positive int ABI");
  }
  return static_cast<int>(value);
}

} // namespace

cudnnDataType_t corex_cudnn_data_type(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      throw std::invalid_argument(
          "INT32 is not supported by this validation adapter");

    case FLAGDNN_DATA_FLOAT32:
      return CUDNN_DATA_FLOAT;
    case FLAGDNN_DATA_FLOAT16:
      return CUDNN_DATA_HALF;
    case FLAGDNN_DATA_BFLOAT16:
      return CUDNN_DATA_BFLOAT16;
    case FLAGDNN_DATA_BOOLEAN:
      // FlagDNN BOOLEAN bindings use one byte per logical 0/1 value. CoreX
      // cuDNN 7.6.5 exposes NOT over the matching signed one-byte storage.
      return CUDNN_DATA_INT8;
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      throw std::invalid_argument(
          "CoreX cudnn.h 7605 has no exact FP8 data type");
  }
  throw std::invalid_argument("unknown FlagDNN data type");
}

std::size_t flagdnn_data_type_size(flagdnnDataType_t data_type) {
  switch (data_type) {
    case FLAGDNN_DATA_INT32:
      return 4;

    case FLAGDNN_DATA_FLOAT32:
      return 4;
    case FLAGDNN_DATA_FLOAT16:
    case FLAGDNN_DATA_BFLOAT16:
      return 2;
    case FLAGDNN_DATA_BOOLEAN:
    case FLAGDNN_DATA_FP8_E8M0:
    case FLAGDNN_DATA_FP8_E4M3:
    case FLAGDNN_DATA_FP8_E5M2:
      return 1;
  }
  throw std::invalid_argument("unknown FlagDNN data type");
}

ReferenceTensor
make_reference_tensor(const flagdnn::testing::TestTensor &tensor) {
  if (tensor.uid <= 0 || tensor.dimensions.size() != tensor.strides.size() ||
      tensor.dimensions.size() > CUDNN_DIM_MAX) {
    throw std::invalid_argument("reference tensor metadata is invalid");
  }
  ReferenceTensor result;
  result.uid = tensor.uid;
  result.data_type = corex_cudnn_data_type(tensor.data_type);
  result.byte_offset = tensor.binding_byte_offset;
  result.dimensions.reserve(std::max<std::size_t>(4, tensor.dimensions.size()));
  result.strides.reserve(std::max<std::size_t>(4, tensor.strides.size()));

  const std::size_t padding =
      tensor.dimensions.size() < 4 ? 4 - tensor.dimensions.size() : 0;
  std::int64_t leading_stride = 1;
  if (!tensor.dimensions.empty()) {
    const std::int64_t first_dimension = tensor.dimensions.front();
    const std::int64_t first_stride = tensor.strides.front();
    if (first_dimension <= 0 || first_stride <= 0 ||
        first_stride >
            std::numeric_limits<std::int64_t>::max() / first_dimension) {
      throw std::invalid_argument("reference tensor leading span is invalid");
    }
    leading_stride = first_dimension * first_stride;
  }
  for (std::size_t index = 0; index < padding; ++index) {
    result.dimensions.push_back(1);
    result.strides.push_back(checked_int(leading_stride, "leading stride"));
  }
  for (const std::int64_t dimension : tensor.dimensions) {
    result.dimensions.push_back(checked_int(dimension, "dimension"));
  }
  for (const std::int64_t stride : tensor.strides) {
    result.strides.push_back(checked_int(stride, "stride"));
  }
  if (result.dimensions.empty()) {
    result.dimensions.assign(4, 1);
    result.strides.assign(4, 1);
  }
  return result;
}

} // namespace flagdnn::iluvatar::validation
