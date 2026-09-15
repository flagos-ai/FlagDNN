/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "reference/cpu/normalization.hpp"

#include <cmath>
#include <numeric>
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
std::vector<std::vector<float>> evaluate_normalization(
    const NormalizationParameters& test_case,
    const std::vector<std::vector<float>>& inputs) {
  const Geometry geometry(test_case);
  const bool backward = test_case.operation.ends_with("_backward"),
             rms = test_case.operation == "rmsnorm_backward";
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
}  // namespace flagdnn::reference::cpu
