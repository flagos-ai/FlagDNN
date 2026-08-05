/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_GRAPH_LOWERING_LOWERING_HPP_
#define FLAGDNN_GRAPH_LOWERING_LOWERING_HPP_

#include "graph/types.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::native {

struct LoweredOperation {
  std::vector<std::pair<std::string, std::int64_t>> parameters;
  std::vector<std::pair<std::string, double>> real_parameters;
  std::vector<std::pair<std::string, std::vector<std::int64_t>>>
      integer_array_parameters;
};

[[nodiscard]] std::string_view operation_name(
    const OperationSpec& operation);
[[nodiscard]] flagdnnDataType_t operation_compute_data_type(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_operation(
    const OperationSpec& operation);

[[nodiscard]] flagdnnPointwiseMode_t pointwise_mode(
    const OperationSpec& operation);
[[nodiscard]] std::string_view pointwise_operation_name(
    flagdnnPointwiseMode_t mode);
[[nodiscard]] LoweredOperation lower_unary_pointwise(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_add(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_pointwise(
    const OperationSpec& operation);

[[nodiscard]] flagdnnReductionMode_t reduction_mode(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_reduction(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_matmul(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_sdpa(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_sdpa_backward(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_sdpa_fp8(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_sdpa_fp8_backward(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_reshape(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_transpose(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_slice(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_convolution_fprop(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_convolution_backward(
    const OperationSpec& operation, bool data_gradient);
[[nodiscard]] LoweredOperation lower_normalization_forward(
    const OperationSpec& operation, bool rmsnorm);
[[nodiscard]] LoweredOperation lower_batchnorm(
    const OperationSpec& operation);
[[nodiscard]] LoweredOperation lower_batchnorm_inference(
    const OperationSpec& operation);

}  // namespace flagdnn::native

#endif  // FLAGDNN_GRAPH_LOWERING_LOWERING_HPP_
