/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace flagdnn::native {
namespace {

constexpr std::int64_t kUnboundedDiagonal = INT64_C(1) << 30;

void require_attention_tensor(const TensorSpec& tensor, const char* name) {
  require_non_overlapping_tensor(tensor, name);
  require_floating_data_type(
      tensor, "SDPA tensors must use a floating data type");
  if (tensor.dimensions.size() != 4) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        std::string(name) + " must be a rank-4 BHSD tensor");
  }
}

bool is_fp8_data_type(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FP8_E4M3 ||
         data_type == FLAGDNN_DATA_FP8_E5M2;
}

void require_fp8_attention_tensor(const TensorSpec& tensor,
                                  const char* name) {
  require_non_overlapping_tensor(tensor, name);
  if (!is_fp8_data_type(tensor.data_type)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   std::string(name) + " must use an FP8 data type");
  }
  if (tensor.dimensions.size() != 4) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        std::string(name) + " must be a rank-4 BHSD tensor");
  }
}

void require_fp32_scalar(const TensorSpec& tensor, const char* name) {
  require_non_overlapping_tensor(tensor, name);
  if (tensor.data_type != FLAGDNN_DATA_FLOAT32 ||
      tensor.element_count() != 1) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   std::string(name) +
                       " must be a one-element float32 tensor");
  }
}

struct AttentionShape {
  std::int64_t batch;
  std::int64_t heads;
  std::int64_t key_heads;
  std::int64_t value_heads;
  std::int64_t sequence_q;
  std::int64_t sequence_kv;
  std::int64_t head_dimension;
  std::int64_t value_dimension;
};

AttentionShape validate_qkv(const TensorSpec& q,
                            const TensorSpec& k,
                            const TensorSpec& v,
                            bool fp8 = false) {
  if (fp8) {
    require_fp8_attention_tensor(q, "FP8 SDPA Q");
    require_fp8_attention_tensor(k, "FP8 SDPA K");
    require_fp8_attention_tensor(v, "FP8 SDPA V");
  } else {
    require_attention_tensor(q, "SDPA Q");
    require_attention_tensor(k, "SDPA K");
    require_attention_tensor(v, "SDPA V");
  }
  require_same_data_type(q, k, "SDPA Q/K data types must match");
  require_same_data_type(q, v, "SDPA Q/V data types must match");
  const auto& qd = q.dimensions;
  const auto& kd = k.dimensions;
  const auto& vd = v.dimensions;
  if (qd[0] != kd[0] || qd[0] != vd[0]) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA Q/K/V batch dimensions must match");
  }
  if (qd[3] != kd[3]) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA Q/K head dimensions must match");
  }
  if (kd[2] != vd[2]) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA K/V sequence dimensions must match");
  }
  if (qd[1] % kd[1] != 0 || qd[1] % vd[1] != 0) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA query heads must be divisible by key and value heads");
  }
  return {qd[0], qd[1], kd[1], vd[1], qd[2], kd[2], qd[3], vd[3]};
}

void validate_bias(const TensorSpec& bias,
                   const TensorSpec& q,
                   const AttentionShape& shape) {
  require_attention_tensor(bias, "SDPA bias");
  require_same_data_type(q, bias, "SDPA bias data type must match Q");
  if ((bias.dimensions[0] != 1 &&
       bias.dimensions[0] != shape.batch) ||
      (bias.dimensions[1] != 1 &&
       bias.dimensions[1] != shape.heads) ||
      bias.dimensions[2] != shape.sequence_q ||
      bias.dimensions[3] != shape.sequence_kv) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA bias must broadcast over B/H and match the Q/KV sequences");
  }
}

LoweredOperation attention_parameters(const OperationSpec& operation,
                                      const AttentionShape& shape,
                                      bool backward,
                                      bool has_dbias) {
  const bool scale_set = boolean_attribute(operation, "attn_scale_set");
  const double configured_scale = real_attribute(operation, "attn_scale");
  const double scale =
      scale_set ? configured_scale
                : 1.0 / std::sqrt(static_cast<double>(
                            shape.head_dimension));
  const bool left_set = boolean_attribute(operation, "left_bound_set");
  const bool right_set = boolean_attribute(operation, "right_bound_set");
  const std::int64_t left =
      integer_attribute(operation, "diagonal_band_left_bound");
  const std::int64_t right =
      integer_attribute(operation, "diagonal_band_right_bound");
  if (left_set && left < 1) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA diagonal left bound must be at least one");
  }
  if (right_set && right < 0) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA diagonal right bound must be nonnegative");
  }
  const std::int64_t alignment =
      integer_attribute(operation, "diagonal_alignment");
  if (alignment != FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT &&
      alignment != FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA diagonal alignment is invalid");
  }
  const std::int64_t shift =
      alignment == FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT
          ? shape.sequence_kv - shape.sequence_q
          : 0;
  const std::int64_t min_diag =
      left_set ? 1 - left + shift : -kUnboundedDiagonal;
  const std::int64_t max_diag =
      right_set ? right + shift : kUnboundedDiagonal;
  const bool banded = left_set || right_set;
  const bool causal_top_left =
      alignment == FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT &&
      !left_set && right_set && right == 0 &&
      shape.sequence_q == shape.sequence_kv;
  const bool reverse_causal =
      alignment == FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT &&
      !left_set && right_set && right == 0;
  return {{{"batch", shape.batch},
           {"heads", shape.heads},
           {"key_heads", shape.key_heads},
           {"value_heads", shape.value_heads},
           {"sequence_q", shape.sequence_q},
           {"sequence_kv", shape.sequence_kv},
           {"head_dimension", shape.head_dimension},
           {"value_dimension", shape.value_dimension},
           {"q_per_k", shape.heads / shape.key_heads},
           {"q_per_v", shape.heads / shape.value_heads},
           {"min_diag", min_diag},
           {"max_diag", max_diag},
           {"has_bias", boolean_attribute(operation, "has_bias") ? 1 : 0},
           {"has_dbias", has_dbias ? 1 : 0},
           {"banded", banded ? 1 : 0},
           {"causal_top_left", causal_top_left ? 1 : 0},
           {"reverse_causal", reverse_causal ? 1 : 0},
           {"generate_stats",
            backward
                ? 1
                : (boolean_attribute(operation, "generate_stats") ? 1 : 0)}},
          {{"attn_scale", scale}},
          {}};
}

}  // namespace

LoweredOperation lower_sdpa(const OperationSpec& operation) {
  const bool has_bias = boolean_attribute(operation, "has_bias");
  require_port_count(operation, has_bias ? 4 : 3, 2);
  const TensorSpec& q = require_port(operation.inputs, "q", "input");
  const TensorSpec& k = require_port(operation.inputs, "k", "input");
  const TensorSpec& v = require_port(operation.inputs, "v", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "o", "output");
  const TensorSpec& stats =
      require_port(operation.outputs, "stats", "output");
  const AttentionShape shape = validate_qkv(q, k, v);
  require_attention_tensor(output, "SDPA output");
  require_same_data_type(q, output,
                         "SDPA output data type must match Q");
  if (output.dimensions !=
      std::vector<std::int64_t>{shape.batch,
                                shape.heads,
                                shape.sequence_q,
                                shape.value_dimension}) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA output shape is incorrect");
  }
  require_non_overlapping_tensor(stats, "SDPA stats");
  if (stats.data_type != FLAGDNN_DATA_FLOAT32 ||
      stats.dimensions !=
          std::vector<std::int64_t>{shape.batch,
                                    shape.heads,
                                    shape.sequence_q,
                                    1}) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA stats must be float32 with shape [B,H,SQ,1]");
  }
  if (has_bias) {
    validate_bias(
        require_port(operation.inputs, "bias", "input"), q, shape);
  }
  return attention_parameters(operation, shape, false, false);
}

LoweredOperation lower_sdpa_backward(const OperationSpec& operation) {
  const bool has_bias = boolean_attribute(operation, "has_bias");
  const bool has_dbias = boolean_attribute(operation, "has_dbias");
  require_port_count(
      operation, has_bias ? 7 : 6, has_dbias ? 4 : 3);
  const TensorSpec& q = require_port(operation.inputs, "q", "input");
  const TensorSpec& k = require_port(operation.inputs, "k", "input");
  const TensorSpec& v = require_port(operation.inputs, "v", "input");
  const TensorSpec& output =
      require_port(operation.inputs, "o", "input");
  const TensorSpec& doutput =
      require_port(operation.inputs, "do", "input");
  const TensorSpec& stats =
      require_port(operation.inputs, "stats", "input");
  const TensorSpec& dq =
      require_port(operation.outputs, "dq", "output");
  const TensorSpec& dk =
      require_port(operation.outputs, "dk", "output");
  const TensorSpec& dv =
      require_port(operation.outputs, "dv", "output");
  const AttentionShape shape = validate_qkv(q, k, v);
  for (const auto& [tensor, expected, name] :
       std::vector<std::tuple<const TensorSpec*,
                              const TensorSpec*,
                              const char*>>{{&output, &doutput, "O/dO"},
                                           {&q, &dq, "Q/dQ"},
                                           {&k, &dk, "K/dK"},
                                           {&v, &dv, "V/dV"}}) {
    require_attention_tensor(*tensor, name);
    require_attention_tensor(*expected, name);
    require_same_shape(*tensor, *expected,
                       "SDPA backward primal/gradient shapes must match");
    require_same_data_type(
        *tensor,
        *expected,
        "SDPA backward primal/gradient data types must match");
  }
  const std::vector<std::int64_t> expected_output{
      shape.batch, shape.heads, shape.sequence_q, shape.value_dimension};
  if (output.dimensions != expected_output) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "SDPA backward O/dO shape is incorrect");
  }
  require_non_overlapping_tensor(stats, "SDPA backward stats");
  if (stats.data_type != FLAGDNN_DATA_FLOAT32 ||
      stats.dimensions !=
          std::vector<std::int64_t>{shape.batch,
                                    shape.heads,
                                    shape.sequence_q,
                                    1}) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA backward stats must be float32 with shape [B,H,SQ,1]");
  }
  if (has_bias) {
    validate_bias(
        require_port(operation.inputs, "bias", "input"), q, shape);
  }
  if (has_dbias) {
    validate_bias(
        require_port(operation.outputs, "dbias", "output"), q, shape);
  }
  return attention_parameters(operation, shape, true, has_dbias);
}

LoweredOperation lower_sdpa_fp8(const OperationSpec& operation) {
  const bool has_bias = boolean_attribute(operation, "has_bias");
  if (has_bias) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "FP8 SDPA does not support bias");
  }
  require_port_count(operation, 9, 4);
  const TensorSpec& q = require_port(operation.inputs, "q", "input");
  const TensorSpec& k = require_port(operation.inputs, "k", "input");
  const TensorSpec& v = require_port(operation.inputs, "v", "input");
  const AttentionShape shape = validate_qkv(q, k, v, true);
  const TensorSpec& output =
      require_port(operation.outputs, "o", "output");
  require_fp8_attention_tensor(output, "FP8 SDPA output");
  require_same_data_type(
      q, output, "FP8 SDPA output data type must match Q");
  if (output.dimensions !=
      std::vector<std::int64_t>{shape.batch,
                                shape.heads,
                                shape.sequence_q,
                                shape.value_dimension}) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "FP8 SDPA output shape is incorrect");
  }
  const TensorSpec& stats =
      require_port(operation.outputs, "stats", "output");
  require_non_overlapping_tensor(stats, "FP8 SDPA stats");
  if (stats.data_type != FLAGDNN_DATA_FLOAT32 ||
      stats.dimensions !=
          std::vector<std::int64_t>{shape.batch,
                                    shape.heads,
                                    shape.sequence_q,
                                    1}) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FP8 SDPA stats must be float32 with shape [B,H,SQ,1]");
  }
  for (const std::string_view name :
       {"descale_q", "descale_k", "descale_v", "descale_s",
        "scale_s", "scale_o"}) {
    require_fp32_scalar(
        require_port(operation.inputs, name, "input"),
        ("FP8 SDPA " + std::string(name)).c_str());
  }
  require_fp32_scalar(
      require_port(operation.outputs, "amax_s", "output"),
      "FP8 SDPA amax S");
  require_fp32_scalar(
      require_port(operation.outputs, "amax_o", "output"),
      "FP8 SDPA amax O");
  return attention_parameters(operation, shape, false, false);
}

LoweredOperation lower_sdpa_fp8_backward(
    const OperationSpec& operation) {
  require_port_count(operation, 18, 7);
  const TensorSpec& q = require_port(operation.inputs, "q", "input");
  const TensorSpec& k = require_port(operation.inputs, "k", "input");
  const TensorSpec& v = require_port(operation.inputs, "v", "input");
  const AttentionShape shape = validate_qkv(q, k, v, true);
  if (shape.key_heads != shape.value_heads ||
      shape.head_dimension != shape.value_dimension ||
      shape.head_dimension > 128) {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "FP8 SDPA backward requires matching K/V heads, D == V <= 128");
  }
  const TensorSpec& output =
      require_port(operation.inputs, "o", "input");
  const TensorSpec& doutput =
      require_port(operation.inputs, "do", "input");
  const TensorSpec& stats =
      require_port(operation.inputs, "stats", "input");
  const TensorSpec& dq =
      require_port(operation.outputs, "dq", "output");
  const TensorSpec& dk =
      require_port(operation.outputs, "dk", "output");
  const TensorSpec& dv =
      require_port(operation.outputs, "dv", "output");
  for (const auto& [tensor, expected, name] :
       std::vector<std::tuple<const TensorSpec*,
                              const TensorSpec*,
                              const char*>>{{&q, &dq, "Q/dQ"},
                                           {&k, &dk, "K/dK"},
                                           {&v, &dv, "V/dV"}}) {
    require_fp8_attention_tensor(*expected, name);
    require_same_shape(
        *tensor, *expected,
        "FP8 SDPA backward primal/gradient shapes must match");
    require_same_data_type(
        *tensor, *expected,
        "FP8 SDPA backward primal/gradient data types must match");
  }
  require_fp8_attention_tensor(output, "FP8 SDPA backward O");
  require_fp8_attention_tensor(doutput, "FP8 SDPA backward dO");
  const std::vector<std::int64_t> expected_output{
      shape.batch, shape.heads, shape.sequence_q, shape.value_dimension};
  if (output.dimensions != expected_output ||
      doutput.dimensions != expected_output) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FP8 SDPA backward O/dO shape is incorrect");
  }
  require_same_data_type(
      q, output, "FP8 SDPA backward O data type must match Q");
  require_same_data_type(
      q, doutput, "FP8 SDPA backward dO data type must match Q");
  require_non_overlapping_tensor(stats, "FP8 SDPA backward stats");
  if (stats.data_type != FLAGDNN_DATA_FLOAT32 ||
      stats.dimensions !=
          std::vector<std::int64_t>{shape.batch,
                                    shape.heads,
                                    shape.sequence_q,
                                    1}) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FP8 SDPA backward stats must be float32 with shape [B,H,SQ,1]");
  }
  for (const std::string_view name :
       {"descale_q", "descale_k", "descale_v", "descale_o",
        "descale_do", "descale_s", "descale_dp", "scale_s",
        "scale_dq", "scale_dk", "scale_dv", "scale_dp"}) {
    require_fp32_scalar(
        require_port(operation.inputs, name, "input"),
        ("FP8 SDPA backward " + std::string(name)).c_str());
  }
  for (const std::string_view name :
       {"amax_dq", "amax_dk", "amax_dv", "amax_dp"}) {
    require_fp32_scalar(
        require_port(operation.outputs, name, "output"),
        ("FP8 SDPA backward " + std::string(name)).c_str());
  }
  return attention_parameters(operation, shape, true, false);
}

}  // namespace flagdnn::native
