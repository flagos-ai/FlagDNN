/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "graph/lowering/lowering.hpp"

#include "graph/lowering/helpers.hpp"

#include <cstdint>
#include <string_view>
#include <vector>

namespace flagdnn::native {
namespace {

bool is_comparison_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE;
}

bool is_logical_binary_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR;
}

LoweredOperation lower_binary_pointwise(
    const OperationSpec& operation) {
  require_port_count(operation, 2, 1);
  const TensorSpec& left = require_port(operation.inputs, "left", "input");
  const TensorSpec& right = require_port(operation.inputs, "right", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(left, "left");
  require_non_overlapping_tensor(right, "right");
  require_non_overlapping_tensor(output, "output");
  require_same_data_type(
      left, right, "binary pointwise input data types must match");
  const flagdnnPointwiseMode_t mode =
      operation.operation == FLAGDNN_OPERATION_POINTWISE
          ? pointwise_mode(operation)
          : FLAGDNN_POINTWISE_ADD;
  if (mode == FLAGDNN_POINTWISE_SIGMOID_BWD &&
      (left.dimensions != right.dimensions ||
       output.dimensions != left.dimensions)) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "sigmoid backward tensors must have equal shapes");
  }
  if (is_comparison_mode(mode)) {
    require_floating_data_type(
        left, "comparison pointwise inputs must use a floating data type");
    require_boolean_data_type(
        output, "comparison pointwise output data type must be BOOLEAN");
  } else if (is_logical_binary_mode(mode)) {
    require_boolean_data_type(
        left, "logical pointwise inputs must use BOOLEAN data type");
    require_boolean_data_type(
        output, "logical pointwise output must use BOOLEAN data type");
  } else {
    require_floating_data_type(
        left, "numeric binary pointwise tensors must use a floating data type");
    require_same_data_type(
        left, output, "binary pointwise input/output data types must match");
  }
  const std::vector<std::int64_t> expected =
      broadcast_dimensions(left, right);
  if (output.dimensions != expected) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "binary pointwise output shape does not match "
                   "broadcast result");
  }
  return {{{"n_elements", output.element_count()},
           {"pointwise_mode", static_cast<std::int64_t>(mode)}},
          {{"alpha", real_attribute(operation, "alpha")}},
          {}};
}

LoweredOperation lower_ternary_pointwise(
    const OperationSpec& operation) {
  require_port_count(operation, 3, 1);
  const TensorSpec& a = require_port(operation.inputs, "a", "input");
  const TensorSpec& b = require_port(operation.inputs, "b", "input");
  const TensorSpec& t = require_port(operation.inputs, "t", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(a, "A");
  require_non_overlapping_tensor(b, "B");
  require_non_overlapping_tensor(t, "T");
  require_non_overlapping_tensor(output, "output");
  require_floating_data_type(
      a, "ternary pointwise A/B/output tensors must be floating");
  require_same_data_type(
      a, b, "ternary pointwise A/B data types must match");
  require_same_data_type(
      a, output, "ternary pointwise A/output data types must match");
  require_boolean_data_type(
      t, "ternary pointwise T predicate must use BOOLEAN data type");
  TensorSpec partial;
  partial.dimensions = broadcast_dimensions(a, b);
  const std::vector<std::int64_t> expected =
      broadcast_dimensions(partial, t);
  if (output.dimensions != expected) {
    throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                   "ternary pointwise output shape does not match "
                   "broadcast result");
  }
  return {{{"n_elements", output.element_count()}}, {}, {}};
}

}  // namespace

flagdnnPointwiseMode_t pointwise_mode(const OperationSpec& operation) {
  return static_cast<flagdnnPointwiseMode_t>(
      integer_attribute(operation, "mode"));
}

std::string_view pointwise_operation_name(flagdnnPointwiseMode_t mode) {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
      return "add";
    case FLAGDNN_POINTWISE_SUB:
      return "sub";
    case FLAGDNN_POINTWISE_MUL:
      return "mul";
    case FLAGDNN_POINTWISE_DIV:
      return "div";
    case FLAGDNN_POINTWISE_MIN:
      return "min";
    case FLAGDNN_POINTWISE_MAX:
      return "max";
    case FLAGDNN_POINTWISE_MOD:
      return "mod";
    case FLAGDNN_POINTWISE_POW:
      return "pow";
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
      return "logical_not";
    case FLAGDNN_POINTWISE_CMP_EQ:
      return "cmp_eq";
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return "cmp_neq";
    case FLAGDNN_POINTWISE_CMP_GT:
      return "cmp_gt";
    case FLAGDNN_POINTWISE_CMP_GE:
      return "cmp_ge";
    case FLAGDNN_POINTWISE_CMP_LT:
      return "cmp_lt";
    case FLAGDNN_POINTWISE_CMP_LE:
      return "cmp_le";
    case FLAGDNN_POINTWISE_LOGICAL_AND:
      return "logical_and";
    case FLAGDNN_POINTWISE_LOGICAL_OR:
      return "logical_or";
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return "sigmoid_backward";
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return "binary_select";
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
      return "sigmoid";
    case FLAGDNN_POINTWISE_TANH_FWD:
      return "tanh";
    case FLAGDNN_POINTWISE_ELU_FWD:
      return "elu";
    case FLAGDNN_POINTWISE_GELU_FWD:
      return "gelu";
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      return "softplus";
    case FLAGDNN_POINTWISE_SWISH_FWD:
      return "swish";
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return "gelu_approx_tanh";
    case FLAGDNN_POINTWISE_RELU_FWD:
      return "relu";
    case FLAGDNN_POINTWISE_SQRT:
      return "sqrt";
    case FLAGDNN_POINTWISE_ERF:
      return "erf";
    case FLAGDNN_POINTWISE_IDENTITY:
      return "identity";
    case FLAGDNN_POINTWISE_EXP:
      return "exp";
    case FLAGDNN_POINTWISE_LOG:
      return "log";
    case FLAGDNN_POINTWISE_NEG:
      return "neg";
    case FLAGDNN_POINTWISE_ABS:
      return "abs";
    case FLAGDNN_POINTWISE_CEIL:
      return "ceil";
    case FLAGDNN_POINTWISE_COS:
      return "cos";
    case FLAGDNN_POINTWISE_FLOOR:
      return "floor";
    case FLAGDNN_POINTWISE_RSQRT:
      return "rsqrt";
    case FLAGDNN_POINTWISE_SIN:
      return "sin";
    case FLAGDNN_POINTWISE_TAN:
      return "tan";
    case FLAGDNN_POINTWISE_RECIPROCAL:
      return "reciprocal";
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                 "configured pointwise operation has an invalid mode");
}

LoweredOperation lower_unary_pointwise(const OperationSpec& operation) {
  require_port_count(operation, 1, 1);
  const TensorSpec& input = require_port(operation.inputs, "input", "input");
  const TensorSpec& output =
      require_port(operation.outputs, "output", "output");
  require_non_overlapping_tensor(input, "input");
  require_non_overlapping_tensor(output, "output");
  require_same_shape(input, output, "pointwise input/output shapes must match");
  require_same_data_type(
      input, output, "pointwise input/output data types must match");

  flagdnnPointwiseMode_t mode = FLAGDNN_POINTWISE_RELU_FWD;
  double relu_lower_clip = 0.0;
  double relu_upper_clip = 0.0;
  double relu_lower_clip_slope = 0.0;
  bool relu_upper_clip_set = false;
  double swish_beta = 1.0;
  double elu_alpha = 1.0;
  double softplus_beta = 1.0;
  if (operation.operation == FLAGDNN_OPERATION_POINTWISE) {
    mode = pointwise_mode(operation);
    relu_lower_clip = real_attribute(operation, "relu_lower_clip");
    relu_upper_clip = real_attribute(operation, "relu_upper_clip");
    relu_lower_clip_slope =
        real_attribute(operation, "relu_lower_clip_slope");
    relu_upper_clip_set =
        boolean_attribute(operation, "relu_upper_clip_set");
    swish_beta = real_attribute(operation, "swish_beta");
    elu_alpha = real_attribute(operation, "elu_alpha");
    softplus_beta = real_attribute(operation, "softplus_beta");
  }

  if (mode == FLAGDNN_POINTWISE_LOGICAL_NOT) {
    require_boolean_data_type(
        input, "logical NOT input/output data types must be BOOLEAN");
  } else {
    require_floating_data_type(
        input, "numeric unary pointwise tensors must use a floating data type");
  }
  return {{{"n_elements", input.element_count()},
           {"has_upper_clip", relu_upper_clip_set ? 1 : 0}},
          {{"negative_slope", relu_lower_clip_slope},
           {"lower_clip", relu_lower_clip},
           {"upper_clip", relu_upper_clip},
           {"swish_beta", swish_beta},
           {"elu_alpha", elu_alpha},
           {"softplus_beta", softplus_beta}},
          {}};
}

LoweredOperation lower_add(const OperationSpec& operation) {
  return lower_binary_pointwise(operation);
}

LoweredOperation lower_pointwise(const OperationSpec& operation) {
  switch (pointwise_mode(operation)) {
    case FLAGDNN_POINTWISE_ADD:
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
    case FLAGDNN_POINTWISE_MOD:
    case FLAGDNN_POINTWISE_POW:
    case FLAGDNN_POINTWISE_CMP_EQ:
    case FLAGDNN_POINTWISE_CMP_NEQ:
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LT:
    case FLAGDNN_POINTWISE_CMP_LE:
    case FLAGDNN_POINTWISE_LOGICAL_AND:
    case FLAGDNN_POINTWISE_LOGICAL_OR:
    case FLAGDNN_POINTWISE_SIGMOID_BWD:
      return lower_binary_pointwise(operation);
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return lower_ternary_pointwise(operation);
    case FLAGDNN_POINTWISE_RELU_FWD:
    case FLAGDNN_POINTWISE_SQRT:
    case FLAGDNN_POINTWISE_ERF:
    case FLAGDNN_POINTWISE_IDENTITY:
    case FLAGDNN_POINTWISE_EXP:
    case FLAGDNN_POINTWISE_LOG:
    case FLAGDNN_POINTWISE_NEG:
    case FLAGDNN_POINTWISE_ABS:
    case FLAGDNN_POINTWISE_CEIL:
    case FLAGDNN_POINTWISE_COS:
    case FLAGDNN_POINTWISE_FLOOR:
    case FLAGDNN_POINTWISE_RSQRT:
    case FLAGDNN_POINTWISE_SIN:
    case FLAGDNN_POINTWISE_TAN:
    case FLAGDNN_POINTWISE_RECIPROCAL:
    case FLAGDNN_POINTWISE_LOGICAL_NOT:
    case FLAGDNN_POINTWISE_SIGMOID_FWD:
    case FLAGDNN_POINTWISE_TANH_FWD:
    case FLAGDNN_POINTWISE_ELU_FWD:
    case FLAGDNN_POINTWISE_GELU_FWD:
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    case FLAGDNN_POINTWISE_SWISH_FWD:
    case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
      return lower_unary_pointwise(operation);
    case FLAGDNN_POINTWISE_NOT_SET:
      break;
  }
  throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                 "configured pointwise operation has an invalid mode");
}

}  // namespace flagdnn::native
