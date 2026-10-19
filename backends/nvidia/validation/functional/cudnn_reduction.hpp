/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_CUDNN_REDUCTION_HPP_
#define FLAGDNN_NVIDIA_CUDNN_REDUCTION_HPP_
#include "common/reduction.hpp"
namespace flagdnn::testing::cuda {
// Executed on cuDNN 9.24 / SM90: Graph supports BF16 SUM -> FP32;
// legacy ReduceTensor supplies FLOAT/HALF SUM, AVG and MUL. Native graph
// probes also validate the shared INT32/BF16 SUM-to-FP32 cases below.
inline bool cudnn_reduction_case(const ReductionTestCase& value) {
  if (value.input.data_type == FLAGDNN_DATA_BFLOAT16 ||
      value.input.data_type == FLAGDNN_DATA_INT32)
    return (value.input.dimensions.size() == 2 ||
            value.axis == static_cast<std::int32_t>(
                              value.input.dimensions.size() - 1)) &&
           value.output.data_type == FLAGDNN_DATA_FLOAT32 &&
           value.mode == FLAGDNN_REDUCTION_ADD;
  return value.input.data_type == FLAGDNN_DATA_FLOAT32 ||
         value.input.data_type == FLAGDNN_DATA_FLOAT16;
}

}  // namespace flagdnn::testing::cuda
#endif
