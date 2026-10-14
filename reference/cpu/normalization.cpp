/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/normalization.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
namespace flagdnn::reference::cpu {
namespace {
using Shape = std::vector<std::int64_t>;
std::size_t count(const Shape& shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                         std::multiplies<>());
}
struct Geometry {
  Shape shape, parameter_shape, statistic_shape;
  std::size_t elements, groups, parameters, reduction;
  explicit Geometry(const NormalizationParameters& test_case) {
    shape = test_case.dimensions;
    parameter_shape = test_case.parameter_dimensions;
    parameter_shape.insert(parameter_shape.begin(),
                           shape.size() - parameter_shape.size(), 1);
    statistic_shape = shape;
    for (auto axis : test_case.axes) statistic_shape[axis] = 1;
    elements = count(shape);
    groups = count(statistic_shape);
    parameters = count(parameter_shape);
    reduction = elements / groups;
  }
  std::size_t project(std::size_t index, const Shape& target) const {
    std::size_t result = 0, stride = 1;
    for (std::size_t axis = shape.size(); axis > 0; --axis) {
      const auto coordinate = index % shape[axis - 1];
      index /= shape[axis - 1];
      if (target[axis - 1] != 1) result += coordinate * stride;
      stride *= target[axis - 1];
    }
    return result;
  }
};
std::pair<std::vector<double>, std::vector<double>> statistics(
    const Geometry& geometry, const std::vector<float>& x, bool rms,
    double epsilon) {
  std::vector<double> means(geometry.groups), variance(geometry.groups);
  if (!rms) {
    for (std::size_t i = 0; i < x.size(); ++i)
      means[geometry.project(i, geometry.statistic_shape)] += x[i];
    for (auto& mean : means) mean /= geometry.reduction;
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto group = geometry.project(i, geometry.statistic_shape);
    const auto centered = x[i] - means[group];
    variance[group] += centered * centered;
  }
  for (auto& var : variance)
    var = 1.0 / std::sqrt(var / geometry.reduction + epsilon);
  return {means, variance};
}
}  // namespace
void populate_normalization_statistics(
    const NormalizationParameters& test_case,
    std::vector<std::vector<float>>& inputs) {
  if (!test_case.operation.ends_with("_backward")) return;
  const bool rms = test_case.operation == "rmsnorm_backward";
  auto [mean, inverse] =
      statistics(Geometry(test_case), inputs[1], rms, test_case.epsilon);
  if (!rms) inputs[3].assign(mean.begin(), mean.end());
  inputs[rms ? 3 : 4].assign(inverse.begin(), inverse.end());
}
namespace {
std::vector<std::vector<float>> evaluate_normalization_impl(
    const NormalizationParameters& test_case,
    const std::vector<std::vector<float>>& inputs, bool rms) {
  const Geometry geometry(test_case);
  const bool backward = test_case.operation.ends_with("_backward");
  const auto& x = inputs[backward ? 1 : 0];
  const auto& scale = inputs[backward ? 2 : 1];
  auto [mean, inverse] = statistics(geometry, x, rms, test_case.epsilon);
  if (backward) {
    if (!rms) mean.assign(inputs[3].begin(), inputs[3].end());
    inverse.assign(inputs[rms ? 3 : 4].begin(), inputs[rms ? 3 : 4].end());
  }
  std::vector<float> output(geometry.elements);
  if (!backward) {
    for (std::size_t i = 0; i < x.size(); ++i) {
      const auto g = geometry.project(i, geometry.statistic_shape),
                 p = geometry.project(i, geometry.parameter_shape);
      output[i] = static_cast<float>((x[i] - mean[g]) * inverse[g] * scale[p] +
                                     inputs[2][p]);
    }
    return {output, std::vector<float>(mean.begin(), mean.end()),
            std::vector<float>(inverse.begin(), inverse.end())};
  }
  std::vector<double> sum(geometry.groups), sum_normalized(geometry.groups),
      ds(geometry.parameters), db(geometry.parameters);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto g = geometry.project(i, geometry.statistic_shape),
               p = geometry.project(i, geometry.parameter_shape);
    const auto normalized = (x[i] - mean[g]) * inverse[g],
               dy = static_cast<double>(inputs[0][i]);
    sum[g] += dy * scale[p];
    sum_normalized[g] += dy * scale[p] * normalized;
    ds[p] += dy * normalized;
    db[p] += dy;
  }
  for (std::size_t i = 0; i < x.size(); ++i) {
    const auto g = geometry.project(i, geometry.statistic_shape),
               p = geometry.project(i, geometry.parameter_shape);
    const auto normalized = (x[i] - mean[g]) * inverse[g];
    output[i] = static_cast<float>(
        inverse[g] *
        (inputs[0][i] * scale[p] - (rms ? 0.0 : sum[g] / geometry.reduction) -
         normalized * sum_normalized[g] / geometry.reduction));
  }
  return {output, std::vector<float>(ds.begin(), ds.end()),
          std::vector<float>(db.begin(), db.end())};
}
}  // namespace

std::vector<std::vector<float>> evaluate_normalization(
    const NormalizationParameters& test_case,
    const std::vector<std::vector<float>>& inputs) {
  return evaluate_normalization_impl(test_case, inputs,
                                     test_case.operation == "rmsnorm_backward");
}

namespace {
void require_epsilon(double epsilon) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0)
    throw std::invalid_argument("CPU normalization epsilon is invalid");
}
void require_finite(std::span<const float> values) {
  if (std::any_of(values.begin(), values.end(),
                  [](float value) { return !std::isfinite(value); }))
    throw std::invalid_argument("CPU normalization values are non-finite");
}
void require_batchnorm_shapes(std::span<const float> input, std::size_t spatial,
                              std::size_t channels,
                              std::span<const float> a,
                              std::span<const float> b,
                              std::span<const float> c) {
  if (input.empty() || channels == 0 || spatial == 0 ||
      input.size() % spatial != 0 || (input.size() / spatial) % channels != 0 ||
      a.size() != channels || b.size() != channels || c.size() != channels)
    throw std::invalid_argument("CPU BatchNorm input shape is invalid");
}
}  // namespace

std::vector<std::vector<float>> evaluate_row_normalization(
    bool rms, std::span<const float> input, std::span<const float> scale,
    std::span<const float> bias, double epsilon) {
  require_epsilon(epsilon);
  if (input.empty() || scale.empty() || bias.size() != scale.size() ||
      input.size() % scale.size() != 0 ||
      input.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::invalid_argument("CPU row normalization input shape is invalid");
  require_finite(input);
  require_finite(scale);
  require_finite(bias);
  NormalizationParameters p;
  p.operation = rms ? "rmsnorm" : "layernorm";
  p.dimensions = {static_cast<std::int64_t>(input.size() / scale.size()),
                  static_cast<std::int64_t>(scale.size())};
  p.parameter_dimensions = {p.dimensions[1]};
  p.axes = {1};
  p.epsilon = epsilon;
  auto result = evaluate_normalization_impl(p, {{input.begin(), input.end()},
      {scale.begin(), scale.end()}, {bias.begin(), bias.end()}}, rms);
  for (const auto& values : result) require_finite(values);
  return result;
}

void variance_to_inverse_standard_deviation(std::span<float> variance,
                                            double epsilon) {
  require_epsilon(epsilon);
  for (float& value : variance) {
    if (!std::isfinite(value) || value < 0.0F)
      throw std::invalid_argument("CPU normalization variance is invalid");
    value = static_cast<float>(1.0 / std::sqrt(static_cast<double>(value) + epsilon));
  }
}

std::vector<float> evaluate_batchnorm_inference(
    std::span<const float> input, std::size_t spatial,
    std::span<const float> mean, std::span<const float> inverse_std,
    std::span<const float> scale, std::span<const float> bias) {
  const auto channels = scale.size();
  require_batchnorm_shapes(input, spatial, channels, mean, inverse_std, bias);
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto channel = (i / spatial) % channels;
    output[i] = (input[i] - mean[channel]) * inverse_std[channel] *
                    scale[channel] + bias[channel];
  }
  return output;
}

std::vector<std::vector<float>> evaluate_batchnorm_training(
    std::span<const float> input, std::size_t spatial,
    std::span<const float> scale, std::span<const float> bias,
    std::span<const float> previous_mean, std::span<const float> previous_variance,
    double epsilon, double momentum) {
  require_epsilon(epsilon);
  if (!std::isfinite(momentum) || momentum < 0 || momentum > 1)
    throw std::invalid_argument("CPU BatchNorm momentum is invalid");
  const auto channels = scale.size();
  require_batchnorm_shapes(input, spatial, channels, bias, previous_mean,
                           previous_variance);
  const auto reduction = input.size() / channels;
  std::vector<double> sums(channels), deviations(channels);
  std::vector<float> mean(channels), inverse(channels), next_mean(channels),
      next_variance(channels);
  for (std::size_t i = 0; i < input.size(); ++i)
    sums[(i / spatial) % channels] += input[i];
  for (std::size_t c = 0; c < channels; ++c)
    mean[c] = static_cast<float>(sums[c] / static_cast<double>(reduction));
  for (std::size_t i = 0; i < input.size(); ++i) {
    const auto c = (i / spatial) % channels;
    const double delta = static_cast<double>(input[i]) - mean[c];
    deviations[c] += delta * delta;
  }
  for (std::size_t c = 0; c < channels; ++c) {
    inverse[c] = static_cast<float>(
        1.0 / std::sqrt(deviations[c] / static_cast<double>(reduction) + epsilon));
    const double unbiased = reduction > 1
        ? deviations[c] / static_cast<double>(reduction - 1) : 0.0;
    next_mean[c] = static_cast<float>((1.0 - momentum) * previous_mean[c] +
                                    momentum * mean[c]);
    next_variance[c] = static_cast<float>((1.0 - momentum) * previous_variance[c] +
                                        momentum * unbiased);
  }
  return {evaluate_batchnorm_inference(input, spatial, mean, inverse, scale, bias),
          std::move(mean), std::move(inverse), std::move(next_mean),
          std::move(next_variance)};
}
}  // namespace flagdnn::reference::cpu
