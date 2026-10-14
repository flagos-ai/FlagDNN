/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "reference/cpu/pointwise.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::reference::cpu {

std::vector<std::uint8_t> evaluate_identity_bytes(
    std::span<const std::uint8_t> input) {
  return {input.begin(), input.end()};
}

float gelu_exact(float value) noexcept {
  constexpr float inverse_sqrt_two = 0.70710678118654752440F;
  return 0.5F * value * (1.0F + std::erf(value * inverse_sqrt_two));
}

float gelu_approximate_tanh(float value) noexcept {
  constexpr float sqrt_two_over_pi = 0.79788456080286535588F;
  constexpr float cubic_scale = 0.044715F;
  const float cubic = value * value * value;
  return 0.5F * value *
         (1.0F + std::tanh(sqrt_two_over_pi * (value + cubic_scale * cubic)));
}

float gelu_accurate_logit(float value) noexcept {
  const float squared = value * value;
  float polynomial = 2.454685183586082e-6F;
  polynomial = std::fma(polynomial, squared, -6.585686530333067e-5F);
  polynomial = std::fma(polynomial, squared, -2.1926247170679162e-4F);
  polynomial = std::fma(polynomial, squared, 7.286941707962942e-2F);
  polynomial = std::fma(polynomial, squared, 1.5957015276323492F);
  const float argument = value * polynomial;
  return value * (argument >= 0.0F
      ? 1.0F / (1.0F + std::exp(-argument))
      : std::exp(argument) / (1.0F + std::exp(argument)));
}

float sigmoid(float logit) noexcept {
  const float exponential = std::exp(-std::abs(logit));
  return logit >= 0.0F ? 1.0F / (1.0F + exponential)
                       : exponential / (1.0F + exponential);
}

float sigmoid_backward(float gradient, float logit) noexcept {
  const float exponential = std::exp(-std::abs(logit));
  const float inverse = 1.0F / (1.0F + exponential);
  return gradient * exponential * inverse * inverse;
}
namespace {

std::size_t checked_element_count(std::span<const std::int64_t> dimensions,
                                  std::string_view role) {
  if (dimensions.empty()) {
    throw std::invalid_argument(std::string(role) +
                                " dimensions must not be empty");
  }
  std::size_t count = 1;
  for (const std::int64_t dimension : dimensions) {
    if (dimension <= 0) {
      throw std::invalid_argument(std::string(role) +
                                  " dimensions must be positive");
    }
    const std::size_t value = static_cast<std::size_t>(dimension);
    if (count > std::numeric_limits<std::size_t>::max() / value) {
      throw std::overflow_error(std::string(role) + " element count overflows");
    }
    count *= value;
  }
  return count;
}

template <typename T>
void validate_input(std::span<const T> values,
                    std::span<const std::int64_t> dimensions,
                    std::span<const std::int64_t> output_dimensions,
                    std::string_view role) {
  if (values.size() != checked_element_count(dimensions, role)) {
    throw std::invalid_argument(std::string(role) +
                                " value count does not match dimensions");
  }
  if (dimensions.size() > output_dimensions.size()) {
    throw std::invalid_argument(std::string(role) +
                                " rank exceeds output rank");
  }
  const std::size_t leading = output_dimensions.size() - dimensions.size();
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    const std::int64_t input_dimension = dimensions[axis];
    const std::int64_t output_dimension = output_dimensions[leading + axis];
    if (input_dimension != 1 && input_dimension != output_dimension) {
      throw std::invalid_argument(std::string(role) +
                                  " dimensions do not broadcast to output");
    }
  }
}

std::size_t broadcast_index(std::size_t output_index,
                            std::span<const std::int64_t> input_dimensions,
                            std::span<const std::int64_t> output_dimensions) {
  const std::size_t leading =
      output_dimensions.size() - input_dimensions.size();
  std::size_t input_index = 0;
  std::size_t input_stride = 1;
  for (std::size_t axis = output_dimensions.size(); axis != 0; --axis) {
    const std::size_t output_axis = axis - 1;
    const std::size_t output_dimension =
        static_cast<std::size_t>(output_dimensions[output_axis]);
    const std::size_t coordinate = output_index % output_dimension;
    output_index /= output_dimension;
    if (output_axis < leading) {
      continue;
    }
    const std::size_t input_axis = output_axis - leading;
    if (input_dimensions[input_axis] != 1) {
      input_index += coordinate * input_stride;
    }
    input_stride *= static_cast<std::size_t>(input_dimensions[input_axis]);
  }
  return input_index;
}

}  // namespace

bool supports_binary_pointwise(flagdnnPointwiseMode_t mode) noexcept {
  switch (mode) {
    case FLAGDNN_POINTWISE_ADD:
    case FLAGDNN_POINTWISE_SUB:
    case FLAGDNN_POINTWISE_MUL:
    case FLAGDNN_POINTWISE_MIN:
    case FLAGDNN_POINTWISE_MAX:
    case FLAGDNN_POINTWISE_DIV:
    case FLAGDNN_POINTWISE_POW:
    case FLAGDNN_POINTWISE_MOD:
    case FLAGDNN_POINTWISE_CMP_EQ:
    case FLAGDNN_POINTWISE_CMP_NEQ:
    case FLAGDNN_POINTWISE_CMP_GT:
    case FLAGDNN_POINTWISE_CMP_GE:
    case FLAGDNN_POINTWISE_CMP_LT:
    case FLAGDNN_POINTWISE_CMP_LE:
      return true;
    default:
      return false;
  }
}

std::vector<float> evaluate_binary_pointwise(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions, std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions) {
  return evaluate_binary_pointwise_with_alpha(
      mode, left, left_dimensions, right, right_dimensions, output_dimensions,
      1.0);
}

std::vector<float> evaluate_binary_pointwise_with_alpha(
    flagdnnPointwiseMode_t mode, std::span<const float> left,
    std::span<const std::int64_t> left_dimensions, std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, double alpha) {
  if (!supports_binary_pointwise(mode)) {
    throw std::invalid_argument(
        "CPU reference does not support this pointwise mode");
  }
  const std::size_t output_count =
      checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");

  std::vector<float> output(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    const float left_value =
        left[broadcast_index(index, left_dimensions, output_dimensions)];
    const float right_value =
        right[broadcast_index(index, right_dimensions, output_dimensions)];
    switch (mode) {
      case FLAGDNN_POINTWISE_ADD:
        output[index] = left_value + static_cast<float>(alpha) * right_value;
        break;
      case FLAGDNN_POINTWISE_SUB:
        output[index] = left_value - static_cast<float>(alpha) * right_value;
        break;
      case FLAGDNN_POINTWISE_MUL:
        output[index] = left_value * right_value;
        break;
      case FLAGDNN_POINTWISE_MIN:
        // Preserve a numeric operand when only its counterpart is NaN.
        output[index] = std::fmin(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_MAX:
        output[index] = std::fmax(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_DIV:
        output[index] = left_value / right_value;
        break;
      case FLAGDNN_POINTWISE_POW:
        output[index] = static_cast<float>(std::pow(left_value, right_value));
        break;
      case FLAGDNN_POINTWISE_MOD:
        output[index] = std::fmod(left_value, right_value);
        break;
      case FLAGDNN_POINTWISE_CMP_EQ:
        output[index] = left_value == right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_NEQ:
        output[index] = left_value != right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_GT:
        output[index] = left_value > right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_GE:
        output[index] = left_value >= right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_LT:
        output[index] = left_value < right_value ? 1.0F : 0.0F;
        break;
      case FLAGDNN_POINTWISE_CMP_LE:
        output[index] = left_value <= right_value ? 1.0F : 0.0F;
        break;
      default:
        throw std::logic_error("validated CPU pointwise mode became invalid");
    }
  }
  return output;
}

std::vector<std::int32_t> evaluate_binary_pointwise_int32(
    flagdnnPointwiseMode_t mode, std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions, std::int32_t alpha) {
  if (!supports_binary_pointwise(mode)) {
    throw std::invalid_argument(
        "CPU reference does not support this pointwise mode");
  }
  const std::size_t output_count =
      checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");

  std::vector<std::int32_t> output(output_count);
  for (std::size_t index = 0; index < output_count; ++index) {
    const std::int32_t left_value =
        left[broadcast_index(index, left_dimensions, output_dimensions)];
    const std::int32_t right_value =
        right[broadcast_index(index, right_dimensions, output_dimensions)];
    output[index] = pointwise_integer_reference(
        mode, left_value, right_value, false, alpha);
  }
  return output;
}

bool supports_activation_backward(flagdnnPointwiseMode_t mode) noexcept {
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_BWD:
    case FLAGDNN_POINTWISE_ELU_BWD:
    case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
    case FLAGDNN_POINTWISE_SWISH_BWD:
      return true;
    default:
      return false;
  }
}

std::vector<float> evaluate_activation_backward(
    flagdnnPointwiseMode_t mode, std::span<const float> dy,
    std::span<const float> x, const flagdnnPointwiseAttributes_t& attributes) {
  if (!supports_activation_backward(mode)) {
    throw std::invalid_argument(
        "CPU reference does not support this activation backward mode");
  }
  if (dy.empty() || dy.size() != x.size()) {
    throw std::invalid_argument("activation backward DY and X sizes must match");
  }
  std::uint64_t allowed_flags = 0;
  switch (mode) {
    case FLAGDNN_POINTWISE_RELU_BWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
                      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
                      FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
      break;
    case FLAGDNN_POINTWISE_ELU_BWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
      break;
    case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
      break;
    case FLAGDNN_POINTWISE_SWISH_BWD:
      allowed_flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
      break;
    default:
      break;
  }
  if (attributes.struct_size != sizeof(attributes) ||
      attributes.version != FLAGDNN_POINTWISE_ATTRIBUTES_VERSION ||
      (attributes.flags & ~allowed_flags) != 0) {
    throw std::invalid_argument("activation backward attribute ABI is invalid");
  }
  // The public API applies fields only when their presence bit is set.
  const auto attribute = [&](std::uint64_t flag, double value,
                             double fallback) {
    if ((attributes.flags & flag) == 0) return fallback;
    if (!std::isfinite(value)) {
      throw std::invalid_argument("activation backward attribute is not finite");
    }
    return value;
  };
  const double lower = attribute(FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP,
                                 attributes.relu_lower_clip, 0.0);
  const double slope =
      attribute(FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE,
                attributes.relu_lower_clip_slope, 0.0);
  const bool has_upper =
      (attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP) != 0;
  const double upper = attribute(FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP,
                                 attributes.relu_upper_clip, 0.0);
  const double elu_alpha = attribute(FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA,
                                     attributes.elu_alpha, 1.0);
  const double swish_beta = attribute(FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA,
                                      attributes.swish_beta, 1.0);
  const double softplus_beta =
      attribute(FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA,
                attributes.softplus_beta, 1.0);
  if ((has_upper && upper < lower) || softplus_beta <= 0.0) {
    throw std::invalid_argument("activation backward attributes are invalid");
  }
  const auto sigmoid = [](double value) {
    // Both branches avoid exponential overflow, including at the tails.
    if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
  };
  std::vector<float> result(x.size());
  for (std::size_t index = 0; index < x.size(); ++index) {
    const double value = x[index];
    double derivative = 0.0;
    switch (mode) {
      case FLAGDNN_POINTWISE_RELU_BWD: {
        // At the lower boundary use the lower-branch slope. Clipping at
        // the upper boundary suppresses the derivative there as well.
        const double forward =
            value < lower ? lower + slope * (value - lower) : value;
        derivative = has_upper && forward >= upper
                         ? 0.0
                         : (value > lower ? 1.0 : slope);
        break;
      }
      case FLAGDNN_POINTWISE_ELU_BWD:
        derivative = value > 0.0 ? 1.0 : elu_alpha * std::exp(value);
        break;
      case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
        derivative = sigmoid(softplus_beta * value);
        break;
      case FLAGDNN_POINTWISE_SWISH_BWD: {
        const double scaled = swish_beta * value;
        const double probability = sigmoid(scaled);
        derivative = probability;
        if (probability != 0.0 && probability != 1.0) {
          derivative += scaled * probability * (1.0 - probability);
        }
        break;
      }
      default:
        throw std::logic_error("validated activation backward mode is invalid");
    }
    result[index] = static_cast<float>(dy[index] * derivative);
  }
  return result;
}

std::vector<std::int32_t> evaluate_binary_select_int32(
    std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::uint8_t> predicate,
    std::span<const std::int64_t> predicate_dimensions,
    std::span<const std::int64_t> output_dimensions) {
  const auto count = checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");
  validate_input(predicate, predicate_dimensions, output_dimensions, "predicate");
  std::vector<std::int32_t> output(count);
  for (std::size_t index = 0; index < count; ++index) {
    output[index] = pointwise_integer_reference(
        FLAGDNN_POINTWISE_BINARY_SELECT,
        left[broadcast_index(index, left_dimensions, output_dimensions)],
        right[broadcast_index(index, right_dimensions, output_dimensions)],
        predicate[broadcast_index(index, predicate_dimensions, output_dimensions)]
            != 0,
        1);
  }
  return output;
}

std::vector<float> evaluate_add_square(
    std::span<const float> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const float> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions,
    const PointwiseQuantizer& quantize_virtual) {
  if (!quantize_virtual)
    throw std::invalid_argument("AddSquare virtual quantizer is required");
  (void)checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");
  auto square = evaluate_binary_pointwise(FLAGDNN_POINTWISE_MUL,
      right, right_dimensions, right, right_dimensions, right_dimensions);
  square = quantize_virtual(square);
  if (square.size() != right.size())
    throw std::invalid_argument("AddSquare virtual quantizer changed value count");
  return evaluate_binary_pointwise(FLAGDNN_POINTWISE_ADD,
      left, left_dimensions, square, right_dimensions, output_dimensions);
}

std::vector<std::int32_t> evaluate_add_square_int32(
    std::span<const std::int32_t> left,
    std::span<const std::int64_t> left_dimensions,
    std::span<const std::int32_t> right,
    std::span<const std::int64_t> right_dimensions,
    std::span<const std::int64_t> output_dimensions) {
  (void)checked_element_count(output_dimensions, "output");
  validate_input(left, left_dimensions, output_dimensions, "left input");
  validate_input(right, right_dimensions, output_dimensions, "right input");
  const auto square = evaluate_binary_pointwise_int32(FLAGDNN_POINTWISE_MUL,
      right, right_dimensions, right, right_dimensions, right_dimensions);
  return evaluate_binary_pointwise_int32(FLAGDNN_POINTWISE_ADD,
      left, left_dimensions, square, right_dimensions, output_dimensions);
}

std::int32_t pointwise_integer_reference(flagdnnPointwiseMode_t mode,
                                         std::int32_t left, std::int32_t right,
                                         bool predicate, std::int32_t alpha) {
  const auto a = static_cast<std::uint32_t>(left);
  const auto b = static_cast<std::uint32_t>(right);
  const auto wrap = [](std::uint32_t value) {
    return std::bit_cast<std::int32_t>(value);
  };
  switch (mode) {
    case FLAGDNN_POINTWISE_IDENTITY:
      return left;
    case FLAGDNN_POINTWISE_ADD:
      return wrap(a + static_cast<std::uint32_t>(alpha) * b);
    case FLAGDNN_POINTWISE_SUB:
      return wrap(a - static_cast<std::uint32_t>(alpha) * b);
    case FLAGDNN_POINTWISE_MUL:
      return wrap(a * b);
    case FLAGDNN_POINTWISE_DIV:
      return right == 0 ? 0
                        : wrap(static_cast<std::uint32_t>(
                              static_cast<std::int64_t>(left) / right));
    case FLAGDNN_POINTWISE_MOD:
      return right == 0 ? 0
                        : static_cast<std::int32_t>(
                              static_cast<std::int64_t>(left) % right);
    case FLAGDNN_POINTWISE_POW: {
      if (right < 0)
        return left == 1 ? 1 : left == -1 ? ((right % 2) ? -1 : 1) : 0;
      // Small exponents allow a straightforward repeated-multiplication oracle.
      std::uint32_t result = 1;
      for (std::int32_t exponent = 0; exponent < right; ++exponent) result *= a;
      return wrap(result);
    }
    case FLAGDNN_POINTWISE_MIN:
      return std::min(left, right);
    case FLAGDNN_POINTWISE_MAX:
      return std::max(left, right);
    case FLAGDNN_POINTWISE_CMP_EQ:
      return left == right;
    case FLAGDNN_POINTWISE_CMP_NEQ:
      return left != right;
    case FLAGDNN_POINTWISE_CMP_GT:
      return left > right;
    case FLAGDNN_POINTWISE_CMP_GE:
      return left >= right;
    case FLAGDNN_POINTWISE_CMP_LT:
      return left < right;
    case FLAGDNN_POINTWISE_CMP_LE:
      return left <= right;
    case FLAGDNN_POINTWISE_BINARY_SELECT:
      return predicate ? left : right;
    default:
      throw std::invalid_argument(
          "integer oracle does not implement this mode");
  }
}

}  // namespace flagdnn::reference::cpu
