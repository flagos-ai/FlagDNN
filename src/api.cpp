/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include <flagdnn/flagdnn.h>

#include "internal.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

template <typename Function>
flagdnnStatus_t api_call(Function&& function) {
  flagdnn::native::clear_last_error();
  try {
    std::forward<Function>(function)();
    return FLAGDNN_STATUS_SUCCESS;
  } catch (const flagdnn::native::ApiError& error) {
    flagdnn::native::set_last_error(error.what());
    return error.status();
  } catch (const std::bad_alloc&) {
    flagdnn::native::set_last_error("host memory allocation failed");
    return FLAGDNN_STATUS_ALLOC_FAILED;
  } catch (const std::exception& error) {
    flagdnn::native::set_last_error(
        "unexpected FlagDNN error: " + std::string(error.what()));
    return FLAGDNN_STATUS_INTERNAL_ERROR;
  } catch (...) {
    flagdnn::native::set_last_error("unknown FlagDNN error");
    return FLAGDNN_STATUS_INTERNAL_ERROR;
  }
}

void require_pointer(const void* pointer, const char* message) {
  if (pointer == nullptr) {
    throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE, message);
  }
}

bool valid_data_type(flagdnnDataType_t data_type) {
  return data_type == FLAGDNN_DATA_FLOAT32 ||
         data_type == FLAGDNN_DATA_FLOAT16 ||
         data_type == FLAGDNN_DATA_BFLOAT16 ||
         data_type == FLAGDNN_DATA_BOOLEAN ||
         data_type == FLAGDNN_DATA_FP8_E4M3 ||
         data_type == FLAGDNN_DATA_FP8_E5M2;
}

bool valid_operation(flagdnnOperation_t operation) {
  return operation == FLAGDNN_OPERATION_RELU ||
         operation == FLAGDNN_OPERATION_ADD ||
         operation == FLAGDNN_OPERATION_REDUCTION ||
         operation == FLAGDNN_OPERATION_CONVOLUTION_FPROP ||
         operation == FLAGDNN_OPERATION_POINTWISE ||
         operation == FLAGDNN_OPERATION_MATMUL ||
         operation == FLAGDNN_OPERATION_SDPA ||
         operation == FLAGDNN_OPERATION_SDPA_BACKWARD ||
         operation == FLAGDNN_OPERATION_SDPA_FP8 ||
         operation == FLAGDNN_OPERATION_SDPA_FP8_BACKWARD;
}

void require_configured_tensor(flagdnnTensorDescriptor_t descriptor,
                               const char* message);

bool valid_descriptor_token(std::string_view value) {
  if (value.empty() || value.size() > 128 ||
      value.front() < static_cast<char>(97) ||
      value.front() > static_cast<char>(122)) {
    return false;
  }
  return std::all_of(value.begin(), value.end(), [](unsigned char character) {
    return (character >= static_cast<unsigned char>(97) &&
            character <= static_cast<unsigned char>(122)) ||
           (character >= static_cast<unsigned char>(48) &&
            character <= static_cast<unsigned char>(57)) ||
           character == static_cast<unsigned char>(95);
  });
}

void require_mutable_custom_operation(
    flagdnnOperationDescriptor_t descriptor) {
  require_pointer(descriptor, "operation descriptor is null");
  if (descriptor->specification.operation != FLAGDNN_OPERATION_CUSTOM) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "generic descriptor setter requires a named operation descriptor");
  }
  if (descriptor->specification.configured) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "cannot modify a finalized operation descriptor");
  }
}

std::string require_descriptor_token(const char* value,
                                     const char* null_message,
                                     const char* invalid_message) {
  require_pointer(value, null_message);
  std::string result(value);
  if (!valid_descriptor_token(result)) {
    throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                    invalid_message);
  }
  return result;
}

void append_custom_port(flagdnnOperationDescriptor_t descriptor,
                        const char* port_name,
                        flagdnnTensorDescriptor_t tensor,
                        int32_t is_optional,
                        bool input) {
  require_mutable_custom_operation(descriptor);
  require_configured_tensor(tensor, "generic operation tensor is null");
  if (is_optional != 0 && is_optional != 1) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "generic operation optional flag must be zero or one");
  }
  std::string name = require_descriptor_token(
      port_name,
      "generic operation port name is null",
      "generic operation port name must use lower_snake_case");
  auto& ports = input ? descriptor->specification.inputs
                      : descriptor->specification.outputs;
  const bool duplicate = std::any_of(
      ports.begin(), ports.end(), [&](const flagdnn::native::OperationPort& port) {
        return port.name == name;
      });
  if (duplicate) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "generic operation port name is duplicated");
  }
  ports.push_back({std::move(name), tensor->specification, is_optional != 0});
}

std::string custom_attribute_name(flagdnnOperationDescriptor_t descriptor,
                                  const char* attribute_name) {
  require_mutable_custom_operation(descriptor);
  return require_descriptor_token(
      attribute_name,
      "generic operation attribute name is null",
      "generic operation attribute name must use lower_snake_case");
}

bool valid_unary_pointwise_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_RELU_FWD ||
         mode == FLAGDNN_POINTWISE_SQRT ||
         mode == FLAGDNN_POINTWISE_ERF ||
         mode == FLAGDNN_POINTWISE_IDENTITY ||
         mode == FLAGDNN_POINTWISE_EXP ||
         mode == FLAGDNN_POINTWISE_LOG ||
         mode == FLAGDNN_POINTWISE_NEG ||
         mode == FLAGDNN_POINTWISE_ABS ||
         mode == FLAGDNN_POINTWISE_CEIL ||
         mode == FLAGDNN_POINTWISE_COS ||
         mode == FLAGDNN_POINTWISE_FLOOR ||
         mode == FLAGDNN_POINTWISE_RSQRT ||
         mode == FLAGDNN_POINTWISE_SIN ||
         mode == FLAGDNN_POINTWISE_TAN ||
         mode == FLAGDNN_POINTWISE_RECIPROCAL ||
         mode == FLAGDNN_POINTWISE_LOGICAL_NOT ||
         mode == FLAGDNN_POINTWISE_SIGMOID_FWD ||
         mode == FLAGDNN_POINTWISE_TANH_FWD ||
         mode == FLAGDNN_POINTWISE_ELU_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_FWD ||
         mode == FLAGDNN_POINTWISE_SOFTPLUS_FWD ||
         mode == FLAGDNN_POINTWISE_SWISH_FWD ||
         mode == FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD;
}

bool valid_binary_pointwise_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_ADD ||
         mode == FLAGDNN_POINTWISE_SUB ||
         mode == FLAGDNN_POINTWISE_MUL ||
         mode == FLAGDNN_POINTWISE_DIV ||
         mode == FLAGDNN_POINTWISE_MIN ||
         mode == FLAGDNN_POINTWISE_MAX ||
         mode == FLAGDNN_POINTWISE_MOD ||
         mode == FLAGDNN_POINTWISE_POW ||
         mode == FLAGDNN_POINTWISE_CMP_EQ ||
         mode == FLAGDNN_POINTWISE_CMP_NEQ ||
         mode == FLAGDNN_POINTWISE_CMP_GT ||
         mode == FLAGDNN_POINTWISE_CMP_GE ||
         mode == FLAGDNN_POINTWISE_CMP_LT ||
         mode == FLAGDNN_POINTWISE_CMP_LE ||
         mode == FLAGDNN_POINTWISE_LOGICAL_AND ||
         mode == FLAGDNN_POINTWISE_LOGICAL_OR ||
         mode == FLAGDNN_POINTWISE_SIGMOID_BWD;
}

bool valid_ternary_pointwise_mode(flagdnnPointwiseMode_t mode) {
  return mode == FLAGDNN_POINTWISE_BINARY_SELECT;
}

bool valid_reduction_mode(flagdnnReductionMode_t mode) {
  return mode == FLAGDNN_REDUCTION_ADD ||
         mode == FLAGDNN_REDUCTION_AVG ||
         mode == FLAGDNN_REDUCTION_MUL;
}

void require_operation_type(flagdnnOperationDescriptor_t descriptor,
                            flagdnnOperation_t expected) {
  require_pointer(descriptor, "operation descriptor is null");
  if (descriptor->specification.operation != expected) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "operation descriptor type does not match setter");
  }
}

void require_configured_tensor(flagdnnTensorDescriptor_t descriptor,
                               const char* message) {
  require_pointer(descriptor, message);
  if (!descriptor->specification.configured) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_NOT_INITIALIZED,
        "tensor descriptor must be configured before use in an operation");
  }
}

void configure_operation_ports(
    flagdnnOperationDescriptor_t descriptor,
    std::initializer_list<
        std::pair<const char*, flagdnnTensorDescriptor_t>> inputs,
    std::initializer_list<
        std::pair<const char*, flagdnnTensorDescriptor_t>> outputs) {
  auto& specification = descriptor->specification;
  specification.inputs.clear();
  specification.outputs.clear();
  specification.attributes.clear();
  specification.configured = false;
  specification.inputs.reserve(inputs.size());
  specification.outputs.reserve(outputs.size());
  for (const auto& [name, tensor] : inputs) {
    specification.inputs.push_back({name, tensor->specification, false});
  }
  for (const auto& [name, tensor] : outputs) {
    specification.outputs.push_back({name, tensor->specification, false});
  }
}

void set_integer_attribute(flagdnnOperationDescriptor_t descriptor,
                           const char* name,
                           std::int64_t value) {
  descriptor->specification.attributes.insert_or_assign(name, value);
}

void set_real_attribute(flagdnnOperationDescriptor_t descriptor,
                        const char* name,
                        double value) {
  descriptor->specification.attributes.insert_or_assign(name, value);
}

void set_boolean_attribute(flagdnnOperationDescriptor_t descriptor,
                           const char* name,
                           bool value) {
  descriptor->specification.attributes.insert_or_assign(name, value);
}

void set_integer_array_attribute(
    flagdnnOperationDescriptor_t descriptor,
    const char* name,
    const std::int64_t values[],
    std::size_t count) {
  descriptor->specification.attributes.insert_or_assign(
      name, std::vector<std::int64_t>(values, values + count));
}

void configure_add_operation(flagdnnOperationDescriptor_t descriptor,
                             flagdnnTensorDescriptor_t left,
                             flagdnnTensorDescriptor_t right,
                             flagdnnTensorDescriptor_t output,
                             double alpha) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_ADD);
  require_configured_tensor(left, "left descriptor is null");
  require_configured_tensor(right, "right descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  const double maximum_alpha =
      static_cast<double>(std::numeric_limits<float>::max());
  if (!std::isfinite(alpha) || alpha < -maximum_alpha ||
      alpha > maximum_alpha) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "Add alpha must be finite and representable as float32");
  }
  configure_operation_ports(
      descriptor,
      {{"left", left}, {"right", right}},
      {{"output", output}});
  set_real_attribute(descriptor, "alpha", alpha);
  descriptor->specification.configured = true;
}

void configure_matmul_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t output) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_MATMUL);
  require_configured_tensor(a, "A descriptor is null");
  require_configured_tensor(b, "B descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  configure_operation_ports(
      descriptor, {{"a", a}, {"b", b}}, {{"output", output}});
  descriptor->specification.configured = true;
}

flagdnnSdpaAttributes_t normalized_sdpa_attributes(
    const flagdnnSdpaAttributes_t* attributes) {
  flagdnnSdpaAttributes_t result = FLAGDNN_SDPA_ATTRIBUTES_INITIALIZER;
  if (attributes != nullptr) {
    if (attributes->struct_size < sizeof(flagdnnSdpaAttributes_t) ||
        attributes->version != FLAGDNN_SDPA_ATTRIBUTES_VERSION) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "SDPA attributes have an incompatible size or version");
    }
    result = *attributes;
  }
  if ((result.flags & ~FLAGDNN_SDPA_ATTRIBUTE_FLAGS_ALL) != 0U) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA attributes contain unknown flags");
  }
  if (result.generate_stats != 0 && result.generate_stats != 1) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA generate_stats must be either zero or one");
  }
  if (result.diagonal_alignment != FLAGDNN_DIAGONAL_ALIGNMENT_TOP_LEFT &&
      result.diagonal_alignment !=
          FLAGDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "SDPA diagonal alignment is invalid");
  }
  if ((result.flags & FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE) != 0U) {
    const double maximum =
        static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(result.attn_scale) || result.attn_scale <= 0.0 ||
        result.attn_scale > maximum) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "SDPA attention scale must be positive and representable as float32");
    }
  }
  constexpr std::int64_t kMaximumDiagonalBound = INT64_C(1) << 29;
  for (const auto& [flag, value] :
       std::array<std::pair<std::uint64_t, std::int64_t>, 2>{
           {{FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND,
             result.diagonal_band_left_bound},
            {FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND,
             result.diagonal_band_right_bound}}}) {
    if ((result.flags & flag) != 0U &&
        (value < -kMaximumDiagonalBound ||
         value > kMaximumDiagonalBound)) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "SDPA diagonal bound is outside the supported range");
    }
  }
  return result;
}

void set_sdpa_attributes(flagdnnOperationDescriptor_t descriptor,
                         const flagdnnSdpaAttributes_t& attributes) {
  set_boolean_attribute(
      descriptor,
      "attn_scale_set",
      (attributes.flags & FLAGDNN_SDPA_ATTRIBUTE_ATTN_SCALE) != 0U);
  set_real_attribute(descriptor, "attn_scale", attributes.attn_scale);
  set_boolean_attribute(
      descriptor,
      "left_bound_set",
      (attributes.flags & FLAGDNN_SDPA_ATTRIBUTE_LEFT_BOUND) != 0U);
  set_integer_attribute(descriptor,
                        "diagonal_band_left_bound",
                        attributes.diagonal_band_left_bound);
  set_boolean_attribute(
      descriptor,
      "right_bound_set",
      (attributes.flags & FLAGDNN_SDPA_ATTRIBUTE_RIGHT_BOUND) != 0U);
  set_integer_attribute(descriptor,
                        "diagonal_band_right_bound",
                        attributes.diagonal_band_right_bound);
  set_integer_attribute(descriptor,
                        "diagonal_alignment",
                        static_cast<std::int64_t>(
                            attributes.diagonal_alignment));
  set_boolean_attribute(
      descriptor, "generate_stats", attributes.generate_stats != 0);
}

void configure_sdpa_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    const flagdnnSdpaAttributes_t* attributes) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_SDPA);
  require_configured_tensor(q, "SDPA Q descriptor is null");
  require_configured_tensor(k, "SDPA K descriptor is null");
  require_configured_tensor(v, "SDPA V descriptor is null");
  require_configured_tensor(output, "SDPA output descriptor is null");
  require_configured_tensor(stats, "SDPA stats descriptor is null");
  if (bias != nullptr) {
    require_configured_tensor(bias, "SDPA bias descriptor is invalid");
  }
  const flagdnnSdpaAttributes_t normalized =
      normalized_sdpa_attributes(attributes);
  auto& specification = descriptor->specification;
  specification.inputs = {
      {"q", q->specification, false},
      {"k", k->specification, false},
      {"v", v->specification, false}};
  if (bias != nullptr) {
    specification.inputs.push_back({"bias", bias->specification, false});
  }
  specification.outputs = {
      {"o", output->specification, false},
      {"stats", stats->specification, false}};
  specification.attributes.clear();
  set_boolean_attribute(descriptor, "has_bias", bias != nullptr);
  set_sdpa_attributes(descriptor, normalized);
  specification.configured = true;
}

void configure_sdpa_backward_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t dbias,
    const flagdnnSdpaAttributes_t* attributes) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_SDPA_BACKWARD);
  for (const auto& [tensor, message] :
       std::array<std::pair<flagdnnTensorDescriptor_t, const char*>, 9>{
           {{q, "SDPA backward Q descriptor is null"},
            {k, "SDPA backward K descriptor is null"},
            {v, "SDPA backward V descriptor is null"},
            {output, "SDPA backward O descriptor is null"},
            {doutput, "SDPA backward dO descriptor is null"},
            {stats, "SDPA backward stats descriptor is null"},
            {dq, "SDPA backward dQ descriptor is null"},
            {dk, "SDPA backward dK descriptor is null"},
            {dv, "SDPA backward dV descriptor is null"}}}) {
    require_configured_tensor(tensor, message);
  }
  if (bias != nullptr) {
    require_configured_tensor(
        bias, "SDPA backward bias descriptor is invalid");
  }
  if (dbias != nullptr) {
    require_configured_tensor(
        dbias, "SDPA backward dBias descriptor is invalid");
  }
  const flagdnnSdpaAttributes_t normalized =
      normalized_sdpa_attributes(attributes);
  auto& specification = descriptor->specification;
  specification.inputs = {
      {"q", q->specification, false},
      {"k", k->specification, false},
      {"v", v->specification, false},
      {"o", output->specification, false},
      {"do", doutput->specification, false},
      {"stats", stats->specification, false}};
  if (bias != nullptr) {
    specification.inputs.push_back({"bias", bias->specification, false});
  }
  specification.outputs = {
      {"dq", dq->specification, false},
      {"dk", dk->specification, false},
      {"dv", dv->specification, false}};
  if (dbias != nullptr) {
    specification.outputs.push_back({"dbias", dbias->specification, false});
  }
  specification.attributes.clear();
  set_boolean_attribute(descriptor, "has_bias", bias != nullptr);
  set_boolean_attribute(descriptor, "has_dbias", dbias != nullptr);
  set_sdpa_attributes(descriptor, normalized);
  specification.configured = true;
}

void configure_sdpa_fp8_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_o,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t amax_s,
    flagdnnTensorDescriptor_t amax_o,
    const flagdnnSdpaAttributes_t* attributes) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_SDPA_FP8);
  for (const auto& [tensor, message] :
       std::array<std::pair<flagdnnTensorDescriptor_t, const char*>, 13>{
           {{q, "FP8 SDPA Q descriptor is null"},
            {k, "FP8 SDPA K descriptor is null"},
            {v, "FP8 SDPA V descriptor is null"},
            {descale_q, "FP8 SDPA descale-Q descriptor is null"},
            {descale_k, "FP8 SDPA descale-K descriptor is null"},
            {descale_v, "FP8 SDPA descale-V descriptor is null"},
            {descale_s, "FP8 SDPA descale-S descriptor is null"},
            {scale_s, "FP8 SDPA scale-S descriptor is null"},
            {scale_o, "FP8 SDPA scale-O descriptor is null"},
            {output, "FP8 SDPA output descriptor is null"},
            {stats, "FP8 SDPA stats descriptor is null"},
            {amax_s, "FP8 SDPA amax-S descriptor is null"},
            {amax_o, "FP8 SDPA amax-O descriptor is null"}}}) {
    require_configured_tensor(tensor, message);
  }
  if (bias != nullptr) {
    require_configured_tensor(bias, "FP8 SDPA bias descriptor is invalid");
  }
  const flagdnnSdpaAttributes_t normalized =
      normalized_sdpa_attributes(attributes);
  auto& specification = descriptor->specification;
  specification.inputs = {
      {"q", q->specification, false},
      {"k", k->specification, false},
      {"v", v->specification, false},
      {"descale_q", descale_q->specification, false},
      {"descale_k", descale_k->specification, false},
      {"descale_v", descale_v->specification, false},
      {"descale_s", descale_s->specification, false},
      {"scale_s", scale_s->specification, false},
      {"scale_o", scale_o->specification, false}};
  if (bias != nullptr) {
    specification.inputs.push_back({"bias", bias->specification, false});
  }
  specification.outputs = {
      {"o", output->specification, false},
      {"stats", stats->specification, false},
      {"amax_s", amax_s->specification, false},
      {"amax_o", amax_o->specification, false}};
  specification.attributes.clear();
  set_boolean_attribute(descriptor, "has_bias", bias != nullptr);
  set_sdpa_attributes(descriptor, normalized);
  specification.configured = true;
}

void configure_sdpa_fp8_backward_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_o,
    flagdnnTensorDescriptor_t descale_doutput,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t descale_dp,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_dq,
    flagdnnTensorDescriptor_t scale_dk,
    flagdnnTensorDescriptor_t scale_dv,
    flagdnnTensorDescriptor_t scale_dp,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t amax_dq,
    flagdnnTensorDescriptor_t amax_dk,
    flagdnnTensorDescriptor_t amax_dv,
    flagdnnTensorDescriptor_t amax_dp,
    const flagdnnSdpaAttributes_t* attributes) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_SDPA_FP8_BACKWARD);
  for (const auto& [tensor, message] :
       std::array<std::pair<flagdnnTensorDescriptor_t, const char*>, 25>{
           {{q, "FP8 SDPA backward Q descriptor is null"},
            {k, "FP8 SDPA backward K descriptor is null"},
            {v, "FP8 SDPA backward V descriptor is null"},
            {output, "FP8 SDPA backward O descriptor is null"},
            {doutput, "FP8 SDPA backward dO descriptor is null"},
            {stats, "FP8 SDPA backward stats descriptor is null"},
            {descale_q, "FP8 SDPA backward descale-Q descriptor is null"},
            {descale_k, "FP8 SDPA backward descale-K descriptor is null"},
            {descale_v, "FP8 SDPA backward descale-V descriptor is null"},
            {descale_o, "FP8 SDPA backward descale-O descriptor is null"},
            {descale_doutput,
             "FP8 SDPA backward descale-dO descriptor is null"},
            {descale_s, "FP8 SDPA backward descale-S descriptor is null"},
            {descale_dp, "FP8 SDPA backward descale-dP descriptor is null"},
            {scale_s, "FP8 SDPA backward scale-S descriptor is null"},
            {scale_dq, "FP8 SDPA backward scale-dQ descriptor is null"},
            {scale_dk, "FP8 SDPA backward scale-dK descriptor is null"},
            {scale_dv, "FP8 SDPA backward scale-dV descriptor is null"},
            {scale_dp, "FP8 SDPA backward scale-dP descriptor is null"},
            {dq, "FP8 SDPA backward dQ descriptor is null"},
            {dk, "FP8 SDPA backward dK descriptor is null"},
            {dv, "FP8 SDPA backward dV descriptor is null"},
            {amax_dq, "FP8 SDPA backward amax-dQ descriptor is null"},
            {amax_dk, "FP8 SDPA backward amax-dK descriptor is null"},
            {amax_dv, "FP8 SDPA backward amax-dV descriptor is null"},
            {amax_dp, "FP8 SDPA backward amax-dP descriptor is null"}}}) {
    require_configured_tensor(tensor, message);
  }
  const flagdnnSdpaAttributes_t normalized =
      normalized_sdpa_attributes(attributes);
  auto& specification = descriptor->specification;
  specification.inputs = {
      {"q", q->specification, false},
      {"k", k->specification, false},
      {"v", v->specification, false},
      {"o", output->specification, false},
      {"do", doutput->specification, false},
      {"stats", stats->specification, false},
      {"descale_q", descale_q->specification, false},
      {"descale_k", descale_k->specification, false},
      {"descale_v", descale_v->specification, false},
      {"descale_o", descale_o->specification, false},
      {"descale_do", descale_doutput->specification, false},
      {"descale_s", descale_s->specification, false},
      {"descale_dp", descale_dp->specification, false},
      {"scale_s", scale_s->specification, false},
      {"scale_dq", scale_dq->specification, false},
      {"scale_dk", scale_dk->specification, false},
      {"scale_dv", scale_dv->specification, false},
      {"scale_dp", scale_dp->specification, false}};
  specification.outputs = {
      {"dq", dq->specification, false},
      {"dk", dk->specification, false},
      {"dv", dv->specification, false},
      {"amax_dq", amax_dq->specification, false},
      {"amax_dk", amax_dk->specification, false},
      {"amax_dv", amax_dv->specification, false},
      {"amax_dp", amax_dp->specification, false}};
  specification.attributes.clear();
  set_boolean_attribute(descriptor, "has_bias", false);
  set_boolean_attribute(descriptor, "has_dbias", false);
  set_sdpa_attributes(descriptor, normalized);
  specification.configured = true;
}

void configure_pointwise_binary_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    double alpha) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_POINTWISE);
  require_configured_tensor(left, "left descriptor is null");
  require_configured_tensor(right, "right descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  if (!valid_binary_pointwise_mode(mode)) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "pointwise mode is not a supported binary operation");
  }
  const double maximum_alpha =
      static_cast<double>(std::numeric_limits<float>::max());
  if (!std::isfinite(alpha) || alpha < -maximum_alpha ||
      alpha > maximum_alpha) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "pointwise alpha must be finite and representable as float32");
  }
  if (mode != FLAGDNN_POINTWISE_ADD &&
      mode != FLAGDNN_POINTWISE_SUB && alpha != 1.0) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "pointwise alpha is only supported by ADD and SUB modes");
  }
  configure_operation_ports(
      descriptor,
      {{"left", left}, {"right", right}},
      {{"output", output}});
  set_integer_attribute(descriptor, "mode", static_cast<std::int64_t>(mode));
  set_real_attribute(descriptor, "alpha", alpha);
  descriptor->specification.configured = true;
}

void configure_pointwise_ternary_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t t,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_POINTWISE);
  require_configured_tensor(a, "A descriptor is null");
  require_configured_tensor(b, "B descriptor is null");
  require_configured_tensor(t, "T descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  if (!valid_ternary_pointwise_mode(mode)) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "pointwise mode is not a supported ternary operation");
  }
  configure_operation_ports(
      descriptor,
      {{"a", a}, {"b", b}, {"t", t}},
      {{"output", output}});
  set_integer_attribute(descriptor, "mode", static_cast<std::int64_t>(mode));
  descriptor->specification.configured = true;
}

float checked_pointwise_attribute(double value, const char* name) {
  const double maximum =
      static_cast<double>(std::numeric_limits<float>::max());
  if (!std::isfinite(value) || value < -maximum || value > maximum) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        std::string(name) +
            " must be finite and representable as float32");
  }
  return static_cast<float>(value);
}

void configure_pointwise_unary_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    const flagdnnPointwiseAttributes_t* attributes) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_POINTWISE);
  require_configured_tensor(input, "input descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  if (!valid_unary_pointwise_mode(mode)) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "pointwise mode is not a supported unary operation");
  }

  flagdnnPointwiseAttributes_t normalized =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  if (attributes != nullptr) {
    if (attributes->struct_size < sizeof(flagdnnPointwiseAttributes_t) ||
        attributes->version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "pointwise attributes have an incompatible size or version");
    }
    normalized = *attributes;
  }
  if ((normalized.flags & ~FLAGDNN_POINTWISE_ATTRIBUTE_FLAGS_ALL) != 0U) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "pointwise attributes contain unknown flags");
  }

  std::uint64_t allowed_flags = 0U;
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_FWD:
      allowed_flags =
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
          FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
      break;
    case FLAGDNN_POINTWISE_SWISH_FWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
      break;
    case FLAGDNN_POINTWISE_ELU_FWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
      break;
    case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
      break;
    default:
      break;
  }
  if ((normalized.flags & ~allowed_flags) != 0U) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "pointwise attributes are not valid for the selected mode");
  }

  float relu_lower_clip = 0.0F;
  float relu_upper_clip = 0.0F;
  float relu_lower_clip_slope = 0.0F;
  float swish_beta = 1.0F;
  float elu_alpha = 1.0F;
  float softplus_beta = 1.0F;
  const bool relu_upper_clip_set =
      (normalized.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0U;
  if ((normalized.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP) != 0U) {
    relu_lower_clip = checked_pointwise_attribute(
        normalized.relu_lower_clip, "ReLU lower clip");
  }
  if (relu_upper_clip_set) {
    relu_upper_clip = checked_pointwise_attribute(
        normalized.relu_upper_clip, "ReLU upper clip");
  }
  if ((normalized.flags &
       FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE) != 0U) {
    relu_lower_clip_slope = checked_pointwise_attribute(
        normalized.relu_lower_clip_slope, "ReLU lower clip slope");
  }
  if ((normalized.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA) != 0U) {
    swish_beta = checked_pointwise_attribute(
        normalized.swish_beta, "SWISH beta");
  }
  if ((normalized.flags & FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA) != 0U) {
    elu_alpha = checked_pointwise_attribute(
        normalized.elu_alpha, "ELU alpha");
  }
  if ((normalized.flags & FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA) !=
      0U) {
    softplus_beta = checked_pointwise_attribute(
        normalized.softplus_beta, "SOFTPLUS beta");
  }
  if (softplus_beta <= 0.0F) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE, "SOFTPLUS beta must be positive");
  }
  if (relu_upper_clip_set && relu_upper_clip < relu_lower_clip) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "ReLU upper clip must not be less than its lower clip");
  }

  configure_operation_ports(
      descriptor, {{"input", input}}, {{"output", output}});
  set_integer_attribute(descriptor, "mode", static_cast<std::int64_t>(mode));
  set_real_attribute(descriptor, "relu_lower_clip", relu_lower_clip);
  set_real_attribute(descriptor, "relu_upper_clip", relu_upper_clip);
  set_real_attribute(
      descriptor, "relu_lower_clip_slope", relu_lower_clip_slope);
  set_boolean_attribute(
      descriptor, "relu_upper_clip_set", relu_upper_clip_set);
  set_real_attribute(descriptor, "swish_beta", swish_beta);
  set_real_attribute(descriptor, "elu_alpha", elu_alpha);
  set_real_attribute(descriptor, "softplus_beta", softplus_beta);
  descriptor->specification.configured = true;
}

void configure_reduction_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnReductionMode_t mode,
    int32_t axis,
    int32_t keep_dimensions,
    flagdnnTensorDescriptor_t output) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_REDUCTION);
  require_configured_tensor(input, "input descriptor is null");
  require_configured_tensor(output, "output descriptor is null");
  if (!valid_reduction_mode(mode)) {
    throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                    "reduction mode is invalid");
  }
  if (keep_dimensions != 0 && keep_dimensions != 1) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "keep_dimensions must be either zero or one");
  }
  configure_operation_ports(
      descriptor, {{"input", input}}, {{"output", output}});
  set_integer_attribute(descriptor, "mode", static_cast<std::int64_t>(mode));
  set_integer_attribute(descriptor, "axis", axis);
  set_boolean_attribute(descriptor, "keep_dimensions", keep_dimensions != 0);
  descriptor->specification.configured = true;
}

void configure_convolution_fprop_operation(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    int32_t spatial_rank,
    const int64_t pre_padding[],
    const int64_t post_padding[],
    const int64_t stride[],
    const int64_t dilation[],
    int64_t groups,
    flagdnnTensorDescriptor_t output) {
  require_operation_type(descriptor, FLAGDNN_OPERATION_CONVOLUTION_FPROP);
  require_configured_tensor(input, "input descriptor is null");
  require_configured_tensor(filter, "filter descriptor is null");
  if (spatial_rank < 1 || spatial_rank > 3) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "convolution spatial rank must be in the inclusive range [1, 3]");
  }
  require_pointer(pre_padding, "pre_padding is null");
  require_pointer(post_padding, "post_padding is null");
  require_pointer(stride, "stride is null");
  require_pointer(dilation, "dilation is null");
  require_configured_tensor(output, "output descriptor is null");
  configure_operation_ports(
      descriptor,
      {{"input", input}, {"filter", filter}},
      {{"output", output}});
  set_integer_attribute(descriptor, "spatial_rank", spatial_rank);
  const std::size_t rank = static_cast<std::size_t>(spatial_rank);
  set_integer_array_attribute(descriptor, "pre_padding", pre_padding, rank);
  set_integer_array_attribute(descriptor, "post_padding", post_padding, rank);
  set_integer_array_attribute(descriptor, "stride", stride, rank);
  set_integer_array_attribute(descriptor, "dilation", dilation, rank);
  set_integer_attribute(descriptor, "groups", groups);
  descriptor->specification.configured = true;
}

flagdnnBuildOptions_t normalized_build_options(
    const flagdnnBuildOptions_t* options) {
  flagdnnBuildOptions_t result{
      sizeof(flagdnnBuildOptions_t), FLAGDNN_BUILD_OPTIONS_VERSION, 0};
  if (options == nullptr) {
    return result;
  }
  if (options->struct_size < sizeof(flagdnnBuildOptions_t) ||
      options->version != FLAGDNN_BUILD_OPTIONS_VERSION) {
    throw flagdnn::native::ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "build options have an incompatible size or version");
  }
  return *options;
}

}  // namespace

extern "C" {

size_t flagdnnGetVersion(void) {
  return FLAGDNN_VERSION_NUMBER;
}

const char* flagdnnGetVersionString(void) { return FLAGDNN_VERSION_STRING; }

const char* flagdnnGetErrorString(flagdnnStatus_t status) {
  switch (status) {
    case FLAGDNN_STATUS_SUCCESS:
      return "success";
    case FLAGDNN_STATUS_INVALID_VALUE:
      return "invalid value";
    case FLAGDNN_STATUS_NOT_INITIALIZED:
      return "not initialized";
    case FLAGDNN_STATUS_ALLOC_FAILED:
      return "allocation failed";
    case FLAGDNN_STATUS_NOT_SUPPORTED:
      return "not supported";
    case FLAGDNN_STATUS_COMPILATION_FAILED:
      return "compilation failed";
    case FLAGDNN_STATUS_BACKEND_ERROR:
      return "backend error";
    case FLAGDNN_STATUS_INTERNAL_ERROR:
      return "internal error";
  }
  return "unknown status";
}

const char* flagdnnGetLastErrorString(void) {
  return flagdnn::native::last_error();
}

flagdnnStatus_t flagdnnCreate(flagdnnHandle_t* handle) {
  return flagdnnCreateWithBackend(FLAGDNN_BACKEND_AUTO, 0, handle);
}

flagdnnStatus_t flagdnnCreateWithBackend(flagdnnBackend_t backend,
                                         int32_t device_ordinal,
                                         flagdnnHandle_t* handle) {
  return api_call([&] {
    require_pointer(handle, "handle output pointer is null");
    *handle = nullptr;
    std::unique_ptr<flagdnnContext> result =
        std::make_unique<flagdnnContext>(backend, device_ordinal);
    *handle = result.release();
  });
}

flagdnnStatus_t flagdnnCreateWithBackendName(const char* backend_name,
                                             int32_t device_ordinal,
                                             flagdnnHandle_t* handle) {
  return api_call([&] {
    require_pointer(backend_name, "backend name is null");
    require_pointer(handle, "handle output pointer is null");
    *handle = nullptr;
    std::unique_ptr<flagdnnContext> result =
        std::make_unique<flagdnnContext>(backend_name, device_ordinal);
    *handle = result.release();
  });
}

flagdnnStatus_t flagdnnDestroy(flagdnnHandle_t handle) {
  return api_call([&] {
    require_pointer(handle, "handle is null");
    delete handle;
  });
}

flagdnnStatus_t flagdnnGetBackendName(flagdnnHandle_t handle,
                                      const char** backend_name) {
  return api_call([&] {
    require_pointer(handle, "handle is null");
    require_pointer(backend_name, "backend name output pointer is null");
    *backend_name = handle->implementation.backend_name().c_str();
  });
}

flagdnnStatus_t flagdnnGetTargetFingerprint(
    flagdnnHandle_t handle,
    const char** target_fingerprint) {
  return api_call([&] {
    require_pointer(handle, "handle is null");
    require_pointer(target_fingerprint,
                    "target fingerprint output pointer is null");
    *target_fingerprint =
        handle->implementation.target_fingerprint().c_str();
  });
}


flagdnnStatus_t flagdnnSetCompilerConfig(flagdnnHandle_t handle,
                                         const char* compiler_executable,
                                         const char* compiler_path,
                                         const char* cache_directory) {
  return api_call([&] {
    require_pointer(handle, "handle is null");
    require_pointer(compiler_executable, "compiler executable is null");
    require_pointer(compiler_path, "compiler path is null");
    require_pointer(cache_directory, "cache directory is null");
    handle->implementation.set_compiler(
        compiler_executable, compiler_path, cache_directory);
  });
}

flagdnnStatus_t flagdnnCreateTensorDescriptor(
    flagdnnTensorDescriptor_t* descriptor) {
  return api_call([&] {
    require_pointer(descriptor, "tensor descriptor output pointer is null");
    *descriptor = nullptr;
    std::unique_ptr<flagdnnTensorDescriptor> result =
        std::make_unique<flagdnnTensorDescriptor>();
    *descriptor = result.release();
  });
}

flagdnnStatus_t flagdnnDestroyTensorDescriptor(
    flagdnnTensorDescriptor_t descriptor) {
  return api_call([&] {
    require_pointer(descriptor, "tensor descriptor is null");
    delete descriptor;
  });
}

flagdnnStatus_t flagdnnSetTensorNdDescriptor(
    flagdnnTensorDescriptor_t descriptor,
    int64_t uid,
    flagdnnDataType_t data_type,
    int32_t rank,
    const int64_t dimensions[],
    const int64_t strides[]) {
  return api_call([&] {
    require_pointer(descriptor, "tensor descriptor is null");
    if (uid <= 0) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "tensor UID must be greater than zero");
    }
    if (!valid_data_type(data_type)) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "tensor data type is invalid");
    }
    if (rank < 0 || rank > 8) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "tensor rank must be in the inclusive range [0, 8]");
    }
    if (rank != 0) {
      require_pointer(dimensions, "tensor dimensions are null");
      require_pointer(strides, "tensor strides are null");
    }

    flagdnn::native::TensorSpec value;
    value.configured = true;
    value.uid = uid;
    value.data_type = data_type;
    if (rank != 0) {
      value.dimensions.assign(dimensions, dimensions + rank);
      value.strides.assign(strides, strides + rank);
    }
    for (int32_t index = 0; index < rank; ++index) {
      if (value.dimensions[static_cast<std::size_t>(index)] <= 0 ||
          value.strides[static_cast<std::size_t>(index)] <= 0) {
        throw flagdnn::native::ApiError(
            FLAGDNN_STATUS_INVALID_VALUE,
            "tensor dimensions and element strides must be positive");
      }
    }
    (void)value.element_count();
    (void)value.storage_size_in_bytes();
    descriptor->specification = std::move(value);
  });
}

flagdnnStatus_t flagdnnGetTensorNdDescriptor(
    flagdnnTensorDescriptor_t descriptor,
    int32_t requested_rank,
    int64_t* uid,
    flagdnnDataType_t* data_type,
    int32_t* actual_rank,
    int64_t dimensions[],
    int64_t strides[]) {
  return api_call([&] {
    require_pointer(descriptor, "tensor descriptor is null");
    require_pointer(uid, "tensor UID output pointer is null");
    require_pointer(data_type, "tensor data type output pointer is null");
    require_pointer(actual_rank, "tensor rank output pointer is null");
    if (!descriptor->specification.configured) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_NOT_INITIALIZED,
          "tensor descriptor has not been configured");
    }
    const std::size_t rank = descriptor->specification.dimensions.size();
    if (requested_rank < 0 ||
        static_cast<std::size_t>(requested_rank) < rank) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "requested tensor rank is smaller than descriptor rank");
    }
    if (rank != 0) {
      require_pointer(dimensions, "tensor dimensions output is null");
      require_pointer(strides, "tensor strides output is null");
    }
    *uid = descriptor->specification.uid;
    *data_type = descriptor->specification.data_type;
    *actual_rank = static_cast<int32_t>(rank);
    if (rank != 0) {
      std::copy(descriptor->specification.dimensions.begin(),
                descriptor->specification.dimensions.end(),
                dimensions);
      std::copy(descriptor->specification.strides.begin(),
                descriptor->specification.strides.end(),
                strides);
    }
  });
}

flagdnnStatus_t flagdnnGetTensorSizeInBytes(
    flagdnnTensorDescriptor_t descriptor,
    size_t* size_in_bytes) {
  return api_call([&] {
    require_pointer(descriptor, "tensor descriptor is null");
    require_pointer(size_in_bytes, "tensor size output pointer is null");
    *size_in_bytes = descriptor->specification.storage_size_in_bytes();
  });
}

flagdnnStatus_t flagdnnSetTensorDescriptorVirtual(
    flagdnnTensorDescriptor_t descriptor,
    int32_t is_virtual) {
  return api_call([&] {
    require_configured_tensor(descriptor, "tensor descriptor is null");
    if (is_virtual != 0 && is_virtual != 1) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "is_virtual must be either zero or one");
    }
    descriptor->specification.is_virtual = is_virtual != 0;
  });
}

flagdnnStatus_t flagdnnGetTensorDescriptorVirtual(
    flagdnnTensorDescriptor_t descriptor,
    int32_t* is_virtual) {
  return api_call([&] {
    require_configured_tensor(descriptor, "tensor descriptor is null");
    require_pointer(is_virtual, "virtual tensor output pointer is null");
    *is_virtual = descriptor->specification.is_virtual ? 1 : 0;
  });
}

flagdnnStatus_t flagdnnSetTensorDescriptorAlignment(
    flagdnnTensorDescriptor_t descriptor,
    int64_t alignment) {
  return api_call([&] {
    require_configured_tensor(descriptor, "tensor descriptor is null");
    if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "tensor alignment must be a positive power of two");
    }
    descriptor->specification.alignment = alignment;
  });
}

flagdnnStatus_t flagdnnGetTensorDescriptorAlignment(
    flagdnnTensorDescriptor_t descriptor,
    int64_t* alignment) {
  return api_call([&] {
    require_configured_tensor(descriptor, "tensor descriptor is null");
    require_pointer(alignment, "tensor alignment output pointer is null");
    *alignment = descriptor->specification.alignment;
  });
}

flagdnnStatus_t flagdnnCreateOperationDescriptor(
    flagdnnOperation_t operation,
    flagdnnOperationDescriptor_t* descriptor) {
  return api_call([&] {
    if (!valid_operation(operation)) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "operation type is invalid");
    }
    require_pointer(descriptor,
                    "operation descriptor output pointer is null");
    *descriptor = nullptr;
    std::unique_ptr<flagdnnOperationDescriptor> result =
        std::make_unique<flagdnnOperationDescriptor>(operation);
    *descriptor = result.release();
  });
}

flagdnnStatus_t flagdnnCreateOperationDescriptorByName(
    const char* operation_kind,
    flagdnnOperationDescriptor_t* descriptor) {
  return api_call([&] {
    require_pointer(descriptor,
                    "operation descriptor output pointer is null");
    *descriptor = nullptr;
    std::string kind = require_descriptor_token(
        operation_kind,
        "operation kind is null",
        "operation kind must use lower_snake_case");
    std::unique_ptr<flagdnnOperationDescriptor> result =
        std::make_unique<flagdnnOperationDescriptor>(
            FLAGDNN_OPERATION_CUSTOM);
    result->specification.custom_operation_name = std::move(kind);
    *descriptor = result.release();
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorInput(
    flagdnnOperationDescriptor_t descriptor,
    const char* port_name,
    flagdnnTensorDescriptor_t tensor,
    int32_t is_optional) {
  return api_call([&] {
    append_custom_port(descriptor, port_name, tensor, is_optional, true);
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorOutput(
    flagdnnOperationDescriptor_t descriptor,
    const char* port_name,
    flagdnnTensorDescriptor_t tensor,
    int32_t is_optional) {
  return api_call([&] {
    append_custom_port(descriptor, port_name, tensor, is_optional, false);
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorAttributeInt64(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    int64_t value) {
  return api_call([&] {
    std::string name = custom_attribute_name(descriptor, attribute_name);
    descriptor->specification.attributes.insert_or_assign(
        std::move(name), static_cast<std::int64_t>(value));
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorAttributeDouble(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    double value) {
  return api_call([&] {
    if (!std::isfinite(value)) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "generic operation double attribute must be finite");
    }
    std::string name = custom_attribute_name(descriptor, attribute_name);
    descriptor->specification.attributes.insert_or_assign(
        std::move(name), value);
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorAttributeBoolean(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    int32_t value) {
  return api_call([&] {
    if (value != 0 && value != 1) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "generic operation boolean attribute must be zero or one");
    }
    std::string name = custom_attribute_name(descriptor, attribute_name);
    descriptor->specification.attributes.insert_or_assign(
        std::move(name), value != 0);
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorAttributeString(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    const char* value) {
  return api_call([&] {
    require_pointer(value, "generic operation string attribute is null");
    std::string owned_value(value);
    if (owned_value.size() > 4096) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "generic operation string attribute is too long");
    }
    std::string name = custom_attribute_name(descriptor, attribute_name);
    descriptor->specification.attributes.insert_or_assign(
        std::move(name), std::move(owned_value));
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorAttributeInt64Array(
    flagdnnOperationDescriptor_t descriptor,
    const char* attribute_name,
    const int64_t values[],
    size_t value_count) {
  return api_call([&] {
    if (value_count > 1024) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "generic operation integer array attribute is too long");
    }
    if (value_count != 0) {
      require_pointer(values,
                      "generic operation integer array attribute is null");
    }
    std::string name = custom_attribute_name(descriptor, attribute_name);
    std::vector<std::int64_t> owned_values;
    if (value_count != 0) {
      owned_values.assign(values, values + value_count);
    }
    descriptor->specification.attributes.insert_or_assign(
        std::move(name), std::move(owned_values));
  });
}

flagdnnStatus_t flagdnnFinalizeOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor) {
  return api_call([&] {
    require_mutable_custom_operation(descriptor);
    if (descriptor->specification.outputs.empty()) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "generic operation descriptor requires at least one output");
    }
    descriptor->specification.configured = true;
  });
}

flagdnnStatus_t flagdnnDestroyOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor) {
  return api_call([&] {
    require_pointer(descriptor, "operation descriptor is null");
    delete descriptor;
  });
}

flagdnnStatus_t flagdnnGetOperationDescriptorType(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnOperation_t* operation) {
  return api_call([&] {
    require_pointer(descriptor, "operation descriptor is null");
    require_pointer(operation, "operation type output pointer is null");
    *operation = descriptor->specification.operation;
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorName(
    flagdnnOperationDescriptor_t descriptor,
    const char* name) {
  return api_call([&] {
    require_pointer(descriptor, "operation descriptor is null");
    require_pointer(name, "operation name is null");
    std::string owned_name(name);
    if (owned_name.size() > 1024) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "operation name is too long");
    }
    descriptor->specification.name = std::move(owned_name);
  });
}

flagdnnStatus_t flagdnnSetOperationDescriptorComputeDataType(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnDataType_t data_type) {
  return api_call([&] {
    require_pointer(descriptor, "operation descriptor is null");
    if (!valid_data_type(data_type)) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "operation compute data type is invalid");
    }
    descriptor->specification.compute_data_type = data_type;
    descriptor->specification.has_compute_data_type = true;
  });
}

flagdnnStatus_t flagdnnSetPointwiseUnaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_pointwise_unary_operation(
        descriptor, input, mode, output, nullptr);
  });
}

flagdnnStatus_t
flagdnnSetPointwiseUnaryOperationDescriptorWithAttributes(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    const flagdnnPointwiseAttributes_t* attributes) {
  return api_call([&] {
    configure_pointwise_unary_operation(
        descriptor, input, mode, output, attributes);
  });
}

flagdnnStatus_t flagdnnSetPointwiseBinaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_pointwise_binary_operation(
        descriptor, left, right, mode, output, 1.0);
  });
}

flagdnnStatus_t
flagdnnSetPointwiseBinaryOperationDescriptorWithAlpha(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output,
    double alpha) {
  return api_call([&] {
    configure_pointwise_binary_operation(
        descriptor, left, right, mode, output, alpha);
  });
}

flagdnnStatus_t flagdnnSetPointwiseTernaryOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t t,
    flagdnnPointwiseMode_t mode,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_pointwise_ternary_operation(
        descriptor, a, b, t, mode, output);
  });
}

flagdnnStatus_t flagdnnSetReluOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    require_operation_type(descriptor, FLAGDNN_OPERATION_RELU);
    require_configured_tensor(input, "input descriptor is null");
    require_configured_tensor(output, "output descriptor is null");
    configure_operation_ports(
        descriptor, {{"input", input}}, {{"output", output}});
    descriptor->specification.configured = true;
  });
}

flagdnnStatus_t flagdnnSetAddOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_add_operation(descriptor, left, right, output, 1.0);
  });
}

flagdnnStatus_t flagdnnSetAddOperationDescriptorWithAlpha(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t left,
    flagdnnTensorDescriptor_t right,
    flagdnnTensorDescriptor_t output,
    double alpha) {
  return api_call([&] {
    configure_add_operation(descriptor, left, right, output, alpha);
  });
}

flagdnnStatus_t flagdnnSetMatmulOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t a,
    flagdnnTensorDescriptor_t b,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_matmul_operation(descriptor, a, b, output);
  });
}

flagdnnStatus_t flagdnnSetSdpaOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    const flagdnnSdpaAttributes_t* attributes) {
  return api_call([&] {
    configure_sdpa_operation(
        descriptor, q, k, v, bias, output, stats, attributes);
  });
}

flagdnnStatus_t flagdnnSetSdpaBackwardOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t dbias,
    const flagdnnSdpaAttributes_t* attributes) {
  return api_call([&] {
    configure_sdpa_backward_operation(descriptor,
                                      q,
                                      k,
                                      v,
                                      output,
                                      doutput,
                                      stats,
                                      bias,
                                      dq,
                                      dk,
                                      dv,
                                      dbias,
                                      attributes);
  });
}

flagdnnStatus_t flagdnnSetSdpaFp8OperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_o,
    flagdnnTensorDescriptor_t bias,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t amax_s,
    flagdnnTensorDescriptor_t amax_o,
    const flagdnnSdpaAttributes_t* attributes) {
  return api_call([&] {
    configure_sdpa_fp8_operation(descriptor,
                                 q,
                                 k,
                                 v,
                                 descale_q,
                                 descale_k,
                                 descale_v,
                                 descale_s,
                                 scale_s,
                                 scale_o,
                                 bias,
                                 output,
                                 stats,
                                 amax_s,
                                 amax_o,
                                 attributes);
  });
}

flagdnnStatus_t flagdnnSetSdpaFp8BackwardOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t q,
    flagdnnTensorDescriptor_t k,
    flagdnnTensorDescriptor_t v,
    flagdnnTensorDescriptor_t output,
    flagdnnTensorDescriptor_t doutput,
    flagdnnTensorDescriptor_t stats,
    flagdnnTensorDescriptor_t descale_q,
    flagdnnTensorDescriptor_t descale_k,
    flagdnnTensorDescriptor_t descale_v,
    flagdnnTensorDescriptor_t descale_o,
    flagdnnTensorDescriptor_t descale_doutput,
    flagdnnTensorDescriptor_t descale_s,
    flagdnnTensorDescriptor_t descale_dp,
    flagdnnTensorDescriptor_t scale_s,
    flagdnnTensorDescriptor_t scale_dq,
    flagdnnTensorDescriptor_t scale_dk,
    flagdnnTensorDescriptor_t scale_dv,
    flagdnnTensorDescriptor_t scale_dp,
    flagdnnTensorDescriptor_t dq,
    flagdnnTensorDescriptor_t dk,
    flagdnnTensorDescriptor_t dv,
    flagdnnTensorDescriptor_t amax_dq,
    flagdnnTensorDescriptor_t amax_dk,
    flagdnnTensorDescriptor_t amax_dv,
    flagdnnTensorDescriptor_t amax_dp,
    const flagdnnSdpaAttributes_t* attributes) {
  return api_call([&] {
    configure_sdpa_fp8_backward_operation(descriptor,
                                          q,
                                          k,
                                          v,
                                          output,
                                          doutput,
                                          stats,
                                          descale_q,
                                          descale_k,
                                          descale_v,
                                          descale_o,
                                          descale_doutput,
                                          descale_s,
                                          descale_dp,
                                          scale_s,
                                          scale_dq,
                                          scale_dk,
                                          scale_dv,
                                          scale_dp,
                                          dq,
                                          dk,
                                          dv,
                                          amax_dq,
                                          amax_dk,
                                          amax_dv,
                                          amax_dp,
                                          attributes);
  });
}

flagdnnStatus_t flagdnnSetReductionOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnReductionMode_t mode,
    int32_t axis,
    int32_t keep_dimensions,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_reduction_operation(
        descriptor, input, mode, axis, keep_dimensions, output);
  });
}

flagdnnStatus_t flagdnnSetReductionSumOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    int32_t axis,
    int32_t keep_dimensions,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_reduction_operation(descriptor,
                                  input,
                                  FLAGDNN_REDUCTION_ADD,
                                  axis,
                                  keep_dimensions,
                                  output);
  });
}

flagdnnStatus_t flagdnnSetConvolutionFpropOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    int32_t spatial_rank,
    const int64_t pre_padding[],
    const int64_t post_padding[],
    const int64_t stride[],
    const int64_t dilation[],
    int64_t groups,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_convolution_fprop_operation(descriptor,
                                          input,
                                          filter,
                                          spatial_rank,
                                          pre_padding,
                                          post_padding,
                                          stride,
                                          dilation,
                                          groups,
                                          output);
  });
}

flagdnnStatus_t flagdnnSetConv2dFpropOperationDescriptor(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    const int64_t padding[2],
    const int64_t stride[2],
    const int64_t dilation[2],
    int64_t groups,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    require_pointer(padding, "padding is null");
    configure_convolution_fprop_operation(descriptor,
                                          input,
                                          filter,
                                          2,
                                          padding,
                                          padding,
                                          stride,
                                          dilation,
                                          groups,
                                          output);
  });
}

flagdnnStatus_t
flagdnnSetConv2dFpropOperationDescriptorWithAsymmetricPadding(
    flagdnnOperationDescriptor_t descriptor,
    flagdnnTensorDescriptor_t input,
    flagdnnTensorDescriptor_t filter,
    const int64_t pre_padding[2],
    const int64_t post_padding[2],
    const int64_t stride[2],
    const int64_t dilation[2],
    int64_t groups,
    flagdnnTensorDescriptor_t output) {
  return api_call([&] {
    configure_convolution_fprop_operation(descriptor,
                                          input,
                                          filter,
                                          2,
                                          pre_padding,
                                          post_padding,
                                          stride,
                                          dilation,
                                          groups,
                                          output);
  });
}

flagdnnStatus_t flagdnnCreateGraph(flagdnnGraph_t* graph) {
  return api_call([&] {
    require_pointer(graph, "graph output pointer is null");
    *graph = nullptr;
    std::unique_ptr<flagdnnGraph> result = std::make_unique<flagdnnGraph>();
    *graph = result.release();
  });
}

flagdnnStatus_t flagdnnDestroyGraph(flagdnnGraph_t graph) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    delete graph;
  });
}

flagdnnStatus_t flagdnnSetGraphName(flagdnnGraph_t graph,
                                    const char* name) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    require_pointer(name, "graph name is null");
    if (graph->specification.finalized) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "cannot modify a finalized graph");
    }
    std::string owned_name(name);
    if (owned_name.size() > 1024) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "graph name is too long");
    }
    graph->specification.name = std::move(owned_name);
  });
}

flagdnnStatus_t flagdnnGraphAddOperation(
    flagdnnGraph_t graph,
    flagdnnOperationDescriptor_t operation) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    require_pointer(operation, "operation descriptor is null");
    if (graph->specification.finalized) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "cannot modify a finalized graph");
    }
    if (!operation->specification.configured) {
      throw flagdnn::native::ApiError(
          FLAGDNN_STATUS_NOT_INITIALIZED,
          "operation descriptor is not configured");
    }
    graph->specification.operations.push_back(operation->specification);
  });
}

flagdnnStatus_t flagdnnValidateGraph(flagdnnGraph_t graph) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    flagdnn::native::validate_graph_structure(graph->specification);
  });
}

flagdnnStatus_t flagdnnFinalizeGraph(flagdnnGraph_t graph) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    if (graph->specification.finalized) {
      throw flagdnn::native::ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                                      "graph is already finalized");
    }
    flagdnn::native::validate_graph_structure(graph->specification);
    graph->specification.finalized = true;
  });
}

flagdnnStatus_t flagdnnGetGraphOperationCount(
    flagdnnGraph_t graph,
    size_t* operation_count) {
  return api_call([&] {
    require_pointer(graph, "graph is null");
    require_pointer(operation_count, "operation count output pointer is null");
    *operation_count = graph->specification.operations.size();
  });
}

flagdnnStatus_t flagdnnBuildExecutable(
    flagdnnHandle_t handle,
    flagdnnGraph_t graph,
    const flagdnnBuildOptions_t* options,
    flagdnnExecutable_t* executable) {
  return api_call([&] {
    require_pointer(handle, "handle is null");
    require_pointer(graph, "graph is null");
    require_pointer(executable, "executable output pointer is null");
    *executable = nullptr;
    const flagdnnBuildOptions_t build_options =
        normalized_build_options(options);
    std::unique_ptr<flagdnnExecutable> result =
        std::make_unique<flagdnnExecutable>(
            flagdnn::native::build_graph_executable(
                handle->implementation,
                graph->specification,
                build_options));
    *executable = result.release();
  });
}

flagdnnStatus_t flagdnnDestroyExecutable(flagdnnExecutable_t executable) {
  return api_call([&] {
    require_pointer(executable, "executable is null");
    delete executable;
  });
}

flagdnnStatus_t flagdnnGetExecutableOperationCount(
    flagdnnExecutable_t executable,
    size_t* operation_count) {
  return api_call([&] {
    require_pointer(executable, "executable is null");
    require_pointer(operation_count, "operation count output pointer is null");
    *operation_count = executable->implementation->operation_count();
  });
}

flagdnnStatus_t flagdnnGetExecutableWorkspaceSize(
    flagdnnExecutable_t executable,
    size_t* workspace_size) {
  return api_call([&] {
    require_pointer(executable, "executable is null");
    require_pointer(workspace_size, "workspace size output pointer is null");
    *workspace_size = executable->implementation->workspace_size();
  });
}

flagdnnStatus_t flagdnnExecuteAsync(flagdnnExecutable_t executable,
                                    const flagdnnBinding_t bindings[],
                                    size_t binding_count,
                                    void* workspace,
                                    size_t workspace_size,
                                    flagdnnStream_t caller_stream) {
  return api_call([&] {
    require_pointer(executable, "executable is null");
    executable->implementation->execute(bindings,
                                        binding_count,
                                        workspace,
                                        workspace_size,
                                        caller_stream);
  });
}

}  // extern "C"
