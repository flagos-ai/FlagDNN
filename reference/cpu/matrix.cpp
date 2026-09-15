/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/matrix.hpp"

#include <bit>
#include <functional>
#include <numeric>
namespace flagdnn::reference::cpu {
std::vector<float> evaluate_matmul(const MatmulParameters& parameters,
                                   std::span<const float> a,
                                   std::span<const float> b) {
  const std::size_t output_rank = parameters.output_shape.size();
  const std::size_t batch_rank = output_rank - 2;
  const std::size_t m =
      static_cast<std::size_t>(parameters.output_shape[output_rank - 2]);
  const std::size_t n =
      static_cast<std::size_t>(parameters.output_shape[output_rank - 1]);
  const std::size_t k = static_cast<std::size_t>(parameters.a_shape.back());
  const std::size_t batch_count =
      static_cast<std::size_t>(std::accumulate(
          parameters.output_shape.begin(), parameters.output_shape.end(),
          std::int64_t{1}, std::multiplies<>())) /
      (m * n);
  const bool tf32 = parameters.input_precision == 2 ||
                    (parameters.input_precision == 0 &&
                     parameters.input_type == FLAGDNN_DATA_FLOAT32 &&
                     output_rank <= 3 && k % 4 == 0 && n % 4 == 0);
  const auto rounded_tf32 = [](float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return std::bit_cast<float>((bits + 0xFFFU + ((bits >> 13) & 1U)) &
                                0xFFFFE000U);
  };
  const auto batch_offset = [batch_rank](
                                std::span<const std::int64_t> input,
                                std::span<const std::size_t> coordinates) {
    const std::size_t input_batch_rank = input.size() - 2;
    const std::size_t leading = batch_rank - input_batch_rank;
    std::size_t result = 0;
    for (std::size_t axis = 0; axis < input_batch_rank; ++axis) {
      const std::size_t dimension = static_cast<std::size_t>(input[axis]);
      result *= dimension;
      if (dimension != 1) {
        result += coordinates[leading + axis];
      }
    }
    return result;
  };

  std::vector<float> result(static_cast<std::size_t>(std::accumulate(
      parameters.output_shape.begin(), parameters.output_shape.end(),
      std::int64_t{1}, std::multiplies<>())));
  std::vector<std::size_t> batch_coordinates(batch_rank);
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    std::size_t remaining = batch;
    for (std::size_t axis = batch_rank; axis != 0; --axis) {
      const std::size_t current = axis - 1;
      const std::size_t dimension =
          static_cast<std::size_t>(parameters.output_shape[current]);
      batch_coordinates[current] = remaining % dimension;
      remaining /= dimension;
    }
    const std::size_t a_base =
        batch_offset(parameters.a_shape, batch_coordinates) * m * k;
    const std::size_t b_base =
        batch_offset(parameters.b_shape, batch_coordinates) * k * n;
    const std::size_t output_base = batch * m * n;
    for (std::size_t row = 0; row < m; ++row) {
      for (std::size_t column = 0; column < n; ++column) {
        double accumulator = 0.0;
        for (std::size_t reduction = 0; reduction < k; ++reduction) {
          const float left = a[a_base + row * k + reduction];
          const float right = b[b_base + reduction * n + column];
          accumulator += tf32 ? static_cast<double>(rounded_tf32(left)) *
                                    rounded_tf32(right)
                              : static_cast<double>(left) * right;
        }
        result[output_base + row * n + column] =
            static_cast<float>(accumulator);
      }
    }
  }
  return result;
}

std::vector<float> evaluate_convolution(
    const ConvolutionParameters& parameters,
    const std::vector<std::vector<float>>& inputs) {
  // TF32 contracts round operands before multiplication. Comparing against
  // full-precision inputs would measure input quantization as a kernel error.
  auto rounded_inputs = parameters.input_precision == 2
                            ? inputs
                            : std::vector<std::vector<float>>{};
  for (auto& tensor : rounded_inputs) {
    for (float& value : tensor) {
      const auto bits = std::bit_cast<std::uint32_t>(value);
      value = std::bit_cast<float>((bits + 0xFFFU + ((bits >> 13) & 1U)) &
                                   0xFFFFE000U);
    }
  }
  const auto& reference_inputs =
      parameters.input_precision == 2 ? rounded_inputs : inputs;
  const auto& xd = parameters.x_shape;
  const auto& wd = parameters.w_shape;
  const auto& yd = parameters.y_shape;
  const auto product = [](auto begin, auto end) {
    return std::accumulate(begin, end, std::int64_t{1}, std::multiplies<>());
  };
  const auto input_area = product(xd.begin() + 2, xd.end()),
             output_area = product(yd.begin() + 2, yd.end());
  const auto volume = product(wd.begin() + 2, wd.end());
  const auto ci = xd[1] / parameters.groups, co = wd[0] / parameters.groups;
  const auto& output =
      parameters.direction == ConvolutionDirection::kForward
          ? parameters.y_shape
      : parameters.direction == ConvolutionDirection::kInputGradient
          ? parameters.x_shape
          : parameters.w_shape;
  std::vector<double> result(product(output.begin(), output.end()));
  std::vector<std::int64_t> spatial(yd.size() - 2), kernel(wd.size() - 2);
  for (std::int64_t n = 0; n < xd[0]; ++n) {
    for (std::int64_t channel = 0; channel < wd[0]; ++channel) {
      for (std::int64_t position = 0; position < output_area; ++position) {
        auto remaining = position;
        for (std::size_t axis = spatial.size(); axis > 0; --axis) {
          spatial[axis - 1] = remaining % yd[axis + 1];
          remaining /= yd[axis + 1];
        }
        const auto y_index = (n * wd[0] + channel) * output_area + position;
        for (std::int64_t c = 0; c < ci; ++c) {
          for (std::int64_t window = 0; window < volume; ++window) {
            auto window_index = window;
            for (std::size_t axis = kernel.size(); axis > 0; --axis) {
              kernel[axis - 1] = window_index % wd[axis + 1];
              window_index /= wd[axis + 1];
            }
            bool valid = true;
            std::int64_t source = 0, weight_position = 0;
            for (std::size_t axis = 0; axis < spatial.size(); ++axis) {
              const auto coordinate = spatial[axis] * parameters.stride[axis] -
                                      parameters.pre_padding[axis] +
                                      kernel[axis] * parameters.dilation[axis];
              valid &= coordinate >= 0 && coordinate < xd[axis + 2];
              source = source * xd[axis + 2] + coordinate;
              const auto weight_coordinate =
                  parameters.flip_kernel ? wd[axis + 2] - 1 - kernel[axis]
                                         : kernel[axis];
              weight_position =
                  weight_position * wd[axis + 2] + weight_coordinate;
            }
            if (!valid) continue;
            const auto x_index =
                (n * xd[1] + channel / co * ci + c) * input_area + source;
            const auto w_index = (channel * ci + c) * volume + weight_position;
            if (parameters.direction == ConvolutionDirection::kForward)
              result[y_index] +=
                  static_cast<double>(reference_inputs[0][x_index]) *
                  reference_inputs[1][w_index];
            else if (parameters.direction ==
                     ConvolutionDirection::kInputGradient)
              result[x_index] +=
                  static_cast<double>(reference_inputs[2][y_index]) *
                  reference_inputs[1][w_index];
            else
              result[w_index] +=
                  static_cast<double>(reference_inputs[2][y_index]) *
                  reference_inputs[0][x_index];
          }
        }
      }
    }
  }
  return std::vector<float>(result.begin(), result.end());
}

std::vector<float> evaluate_reduction(const ReductionParameters& parameters,
                                      std::span<const float> input) {
  const auto axis = parameters.axis < 0
                        ? parameters.axis + static_cast<std::int32_t>(
                                                parameters.input_shape.size())
                        : parameters.axis;
  const auto extent = static_cast<std::size_t>(parameters.input_shape[axis]);
  std::size_t inner = 1;
  for (std::size_t i = axis + 1; i < parameters.input_shape.size(); ++i)
    inner *= parameters.input_shape[i];
  std::vector<float> output(input.size() / extent);
  for (std::size_t index = 0; index < output.size(); ++index) {
    double value = parameters.mode == FLAGDNN_REDUCTION_MUL ? 1.0 : 0.0;
    for (std::size_t r = 0; r < extent; ++r) {
      const double x =
          input[(index / inner * extent + r) * inner + index % inner];
      if (parameters.mode == FLAGDNN_REDUCTION_MUL)
        value *= x;
      else
        value += x;
    }
    output[index] = static_cast<float>(
        parameters.mode == FLAGDNN_REDUCTION_AVG ? value / extent : value);
  }
  return output;
}

}  // namespace flagdnn::reference::cpu
