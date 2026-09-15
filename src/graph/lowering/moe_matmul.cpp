/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "graph/lowering/lowering.hpp"
#include "graph/lowering/helpers.hpp"
#include <limits>
namespace flagdnn::native {
LoweredOperation lower_moe_matmul(const OperationSpec& operation) {
  const bool backward =
      operation.custom_operation_name == "moe_grouped_matmul_bwd";
  const auto mode = backward ? 0 : integer_attribute(operation, "mode");
  const auto top_k = backward ? 1 : integer_attribute(operation, "top_k");
  if (mode < 0 || mode > 2 || top_k <= 0 ||
      top_k > std::numeric_limits<std::int32_t>::max())
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "invalid MoE routing attributes");
  require_port_count(operation, 3 + mode, 1);
  const auto& token = require_port(operation.inputs, "token", "input");
  const auto& matrix =
      require_port(operation.inputs, backward ? "doutput" : "weight", "input");
  const auto& offsets =
      require_port(operation.inputs, "first_token_offset", "input");
  const auto& output = require_port(operation.outputs,
                                    backward ? "dweight" : "output", "output");
  for (const auto& port : operation.inputs)
    require_non_overlapping_tensor(port.tensor, "MoE input");
  require_non_overlapping_tensor(output, "MoE output");
  const auto type = token.data_type;
  if (type != FLAGDNN_DATA_FLOAT16 && type != FLAGDNN_DATA_BFLOAT16 &&
      type != FLAGDNN_DATA_FP8_E4M3 && type != FLAGDNN_DATA_FP8_E5M2)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MoE requires FP16, BF16 or FP8 input");
  require_same_data_type(token, matrix, "MoE matrix input types must match");
  require_floating_data_type(output, "MoE output must be FP32, FP16 or BF16");
  if (token.dimensions.size() != 3 || matrix.dimensions.size() != 3 ||
      token.dimensions[0] != 1)
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "MoE requires rank-three tensors and token batch dimension one");
  if (offsets.dimensions.size() != 3)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MoE offsets require rank three");
  const auto experts = backward ? offsets.dimensions[0] : matrix.dimensions[0];
  if (offsets.data_type != FLAGDNN_DATA_INT32 ||
      offsets.dimensions != std::vector<std::int64_t>{experts, 1, 1})
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MoE first token offsets must be INT32 [E,1,1]");
  const auto k = token.dimensions[2], n = matrix.dimensions[2],
             tokens = token.dimensions[1];
  std::int64_t routed = tokens;
  if (backward) {
    if (matrix.dimensions[0] != 1 || matrix.dimensions[1] != tokens)
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "MoE backward token and gradient counts must match");
  } else if (matrix.dimensions[1] != k) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MoE contraction dimensions differ");
  }
  if (mode) {
    const auto& index = require_port(operation.inputs, "token_index", "input");
    if (mode == 1 && index.dimensions.size() == 3) routed = index.dimensions[1];
    if (index.data_type != FLAGDNN_DATA_INT32 ||
        index.dimensions != std::vector<std::int64_t>{1, routed, 1})
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     "MoE token indices must be INT32 [1,routed_tokens,1]");
    if (mode == 2) {
      const auto& ks = require_port(operation.inputs, "token_ks", "input");
      if (ks.data_type != FLAGDNN_DATA_INT32 ||
          ks.dimensions != index.dimensions || routed % top_k)
        throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                       "MoE scatter requires matching INT32 top-k indices and "
                       "divisible routed count");
    }
  }
  if (routed > std::numeric_limits<std::int32_t>::max() ||
      tokens > std::numeric_limits<std::int32_t>::max())
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "MoE routing uses INT32 token indices");
  const auto expected = backward ? std::vector<std::int64_t>{experts, k, n}
                                 : std::vector<std::int64_t>{1, routed, n};
  if (output.dimensions != expected)
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "MoE output shape is inconsistent");
  return {{{"mode", mode},
           {"top_k", top_k},
           {"experts", experts},
           {"tokens", tokens},
           {"routed_tokens", routed},
           {"k", k},
           {"n", n}},
          {},
          {}};
}
}  // namespace flagdnn::native
