/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "reference/cpu/pointwise.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cpu = flagdnn::reference::cpu;
using Shape = std::vector<std::int64_t>;

using LegacyEvaluator = std::vector<float> (*)(
    flagdnnPointwiseMode_t, std::span<const float>,
    std::span<const std::int64_t>, std::span<const float>,
    std::span<const std::int64_t>, std::span<const std::int64_t>);
static_assert(std::is_same_v<decltype(&cpu::evaluate_binary_pointwise),
                             LegacyEvaluator>);

template <typename T>
void expect_values(std::string_view name, const std::vector<T>& actual,
                   const std::vector<T>& expected) {
  if (actual != expected) {
    throw std::runtime_error(std::string(name) + " differs");
  }
}

template <typename Exception = std::invalid_argument, typename Operation>
void expect_error(std::string_view name, Operation operation) {
  try {
    operation();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error(std::string(name) + " was accepted");
}

void verify_float() {
  const std::vector<float> left = {1.0F, -2.0F};
  const std::vector<float> right = {3.0F, 4.0F, -5.0F};
  const Shape left_shape = {2, 1};
  const Shape right_shape = {3};
  const Shape output_shape = {2, 3};
  const auto evaluate = [&](flagdnnPointwiseMode_t mode, double alpha = 1.0) {
    return cpu::evaluate_binary_pointwise_with_alpha(
        mode, left, left_shape, right, right_shape, output_shape, alpha);
  };
  expect_values<float>("ADD broadcast", evaluate(FLAGDNN_POINTWISE_ADD),
                       {4, 5, -4, 1, 2, -7});
  expect_values<float>("ADD alpha", evaluate(FLAGDNN_POINTWISE_ADD, 0.5),
                       {2.5, 3, -1.5, -0.5, 0, -4.5});
  expect_values<float>("SUB alpha", evaluate(FLAGDNN_POINTWISE_SUB, 2),
                       {-5, -7, 11, -8, -10, 8});
  expect_values<float>("MUL broadcast", evaluate(FLAGDNN_POINTWISE_MUL),
                       {3, 4, -5, -6, -8, 10});
  expect_values<float>("MAX broadcast", evaluate(FLAGDNN_POINTWISE_MAX),
                       {3, 4, 1, 3, 4, -2});
  expect_values<float>("MIN broadcast", evaluate(FLAGDNN_POINTWISE_MIN),
                       {1, 1, -5, -2, -2, -5});
  expect_values<float>("comparison broadcast", evaluate(FLAGDNN_POINTWISE_CMP_EQ),
                       {0, 0, 0, 0, 0, 0});

  const auto paired = [](flagdnnPointwiseMode_t mode,
                         const std::vector<float>& a,
                         const std::vector<float>& b) {
    const Shape shape = {static_cast<std::int64_t>(a.size())};
    return cpu::evaluate_binary_pointwise(mode, a, shape, b, shape, shape);
  };
  expect_values<float>("DIV signed", paired(FLAGDNN_POINTWISE_DIV,
                       {-7.5F, 6.0F}, {2.5F, -4.0F}), {-3.0F, -1.5F});
  expect_values<float>("MOD signed", paired(FLAGDNN_POINTWISE_MOD,
                       {-7.5F, 7.5F}, {2.0F, -2.0F}), {-1.5F, 1.5F});
  expect_values<float>("POW", paired(FLAGDNN_POINTWISE_POW,
                       {-2, 4, 2}, {3, 0.5, -2}), {-8, 2, 0.25});
  expect_values<float>("CMP_EQ", paired(FLAGDNN_POINTWISE_CMP_EQ,
                       {1, -2, 3}, {1, 2, 3}), {1, 0, 1});

  const float comparison_nan = std::numeric_limits<float>::quiet_NaN();
  expect_values<float>("CMP_NEQ IEEE NaN", paired(FLAGDNN_POINTWISE_CMP_NEQ,
                       {comparison_nan, 1, -1}, {comparison_nan, 1, 0}), {1, 0, 1});
  expect_values<float>("CMP_GT ordered", paired(FLAGDNN_POINTWISE_CMP_GT,
                       {comparison_nan, 1, -1}, {0, 1, -2}), {0, 0, 1});
  expect_values<float>("CMP_GE ordered", paired(FLAGDNN_POINTWISE_CMP_GE,
                       {comparison_nan, 1, -1}, {0, 1, -2}), {0, 1, 1});
  expect_values<float>("CMP_LT ordered", paired(FLAGDNN_POINTWISE_CMP_LT,
                       {comparison_nan, 1, -1}, {0, 1, 0}), {0, 0, 1});
  expect_values<float>("CMP_LE ordered", paired(FLAGDNN_POINTWISE_CMP_LE,
                       {comparison_nan, 1, -1}, {0, 1, 0}), {0, 1, 1});

  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const auto mode : {FLAGDNN_POINTWISE_MIN, FLAGDNN_POINTWISE_MAX}) {
    const auto values = paired(mode, {nan, 2.0F, nan}, {2.0F, nan, nan});
    if (values[0] != 2.0F || values[1] != 2.0F || !std::isnan(values[2])) {
      throw std::runtime_error(
          "MIN/MAX must preserve the numeric operand for one-sided NaN");
    }
  }
}

void verify_legacy_float_special_values() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const auto verify = [](flagdnnPointwiseMode_t mode,
                         const std::vector<float>& left,
                         const std::vector<float>& right,
                         const std::vector<float>& expected) {
    const Shape shape = {static_cast<std::int64_t>(left.size())};
    const auto actual = cpu::evaluate_binary_pointwise(
        mode, left, shape, right, shape, shape);
    if (actual.size() != expected.size()) {
      throw std::runtime_error("legacy float output size changed");
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (std::isnan(expected[index]) ? !std::isnan(actual[index])
          : actual[index] != expected[index] ||
            std::signbit(actual[index]) != std::signbit(expected[index])) {
        throw std::runtime_error("legacy float special value changed");
      }
    }
  };
  verify(FLAGDNN_POINTWISE_DIV,
         {0.0F, -0.0F, 1, -1, 1, -1, inf, nan},
         {2, 2, 0.0F, 0.0F, inf, inf, inf, 1},
         {0.0F, -0.0F, inf, -inf, 0.0F, -0.0F, nan, nan});
  verify(FLAGDNN_POINTWISE_MOD,
         {-4, 4, -0.0F, 0.0F, inf, 3, nan},
         {2, -2, 2, 2, 2, inf, 2},
         {-0.0F, 0.0F, -0.0F, 0.0F, nan, 3, nan});
  verify(FLAGDNN_POINTWISE_POW,
         {-0.0F, -0.0F, nan, nan, 1, 2, inf, -inf},
         {3, 2, 0, 2, nan, inf, -1, 3},
         {-0.0F, 0.0F, 1, nan, 1, inf, 0.0F, -inf});
  verify(FLAGDNN_POINTWISE_CMP_EQ,
         {nan, inf, -0.0F, 0.0F, -inf},
         {nan, inf, 0.0F, -0.0F, inf},
         {0, 1, 1, 1, 0});
}

void expect_near(std::string_view name, const std::vector<float>& actual,
                 const std::vector<float>& expected, double tolerance = 1e-6) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error(std::string(name) + " size differs");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) >
            tolerance * std::max(1.0F, std::abs(expected[index]))) {
      throw std::runtime_error(std::string(name) + " differs at " +
                               std::to_string(index));
    }
  }
}

void verify_activation_backward() {
  const auto evaluate = [](auto mode, const std::vector<float>& dy,
                            const std::vector<float>& x,
                            flagdnnPointwiseAttributes_t attributes =
                                FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER) {
    return cpu::evaluate_activation_backward(mode, dy, x, attributes);
  };
  flagdnnPointwiseAttributes_t attributes =
      FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  expect_values<float>("ReLU lower boundary",
      evaluate(FLAGDNN_POINTWISE_RELU_BWD, {2, 3, -4}, {-1, 0, 1}),
      {0, 0, -4});

  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP |
                     FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP |
                     FLAGDNN_POINTWISE_ATTRIBUTE_RELU_LOWER_CLIP_SLOPE;
  attributes.relu_lower_clip = -1;
  attributes.relu_upper_clip = 2;
  attributes.relu_lower_clip_slope = 0.25;
  expect_values<float>("clipped leaky ReLU boundaries",
      evaluate(FLAGDNN_POINTWISE_RELU_BWD, {2, 2, -3, 4, 5},
               {-2, -1, 0, 2, 3}, attributes), {0.5F, 0.5F, -3, 0, 0});
  attributes.relu_lower_clip_slope = -2;
  expect_values<float>("clipping applies to the lower forward branch",
      evaluate(FLAGDNN_POINTWISE_RELU_BWD, {2, 3}, {-3, -2}, attributes),
      {0, -6});

  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
  attributes.elu_alpha = 0.25;
  expect_near("ELU alpha and lower boundary",
      evaluate(FLAGDNN_POINTWISE_ELU_BWD, {2, 3, -4}, {-1, 0, 1}, attributes),
      {static_cast<float>(0.5 / std::exp(1.0)), 0.75F, -4});
  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  attributes.softplus_beta = 2;
  expect_near("softplus beta and stable tails",
      evaluate(FLAGDNN_POINTWISE_SOFTPLUS_BWD, {2, -4, 3, 5},
               {-1000, 0, 1000, -1}, attributes),
      {0, -2, 3, static_cast<float>(5.0 / (1.0 + std::exp(2.0)))});
  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SWISH_BETA;
  attributes.swish_beta = 0;
  expect_values<float>("swish beta zero",
      evaluate(FLAGDNN_POINTWISE_SWISH_BWD, {2, -4, 3}, {-1, 0, 1},
               attributes), {1, -2, 1.5F});
  attributes.swish_beta = 2;
  expect_values<float>("swish saturated tails",
      evaluate(FLAGDNN_POINTWISE_SWISH_BWD, {2, -4}, {-1000, 1000},
               attributes), {0, -4});

  // A separate central difference of the forward functions checks smooth
  // gradients without copying the implementation's derivative expression.
  constexpr double step = 1e-4;
  const std::vector<float> x = {-3.0F, -0.75F, 0.25F, 2.0F};
  for (const auto mode : {FLAGDNN_POINTWISE_ELU_BWD,
                           FLAGDNN_POINTWISE_SOFTPLUS_BWD,
                           FLAGDNN_POINTWISE_SWISH_BWD}) {
    attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
    const auto forward = [&](double value) {
      if (mode == FLAGDNN_POINTWISE_ELU_BWD)
        return value > 0 ? value : std::expm1(value);
      if (mode == FLAGDNN_POINTWISE_SOFTPLUS_BWD)
        return std::log1p(std::exp(value));
      return value / (1.0 + std::exp(-value));
    };
    std::vector<float> derivative;
    for (const float value : x) {
      derivative.push_back(static_cast<float>(
          (forward(value + step) - forward(value - step)) / (2 * step)));
    }
    expect_near("forward finite difference",
        evaluate(mode, {1, 1, 1, 1}, x), derivative);
  }

  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.elu_alpha = 5;  // Unflagged values must not change the public API.
  expect_values<float>("unflagged alpha keeps the default",
      evaluate(FLAGDNN_POINTWISE_ELU_BWD, {2}, {0}, attributes), {2});
  expect_error("unsupported gradient", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_ADD, {1}, {2});
  });
  expect_error("gradient input size", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_ELU_BWD, {1, 2}, {2});
  });
  expect_error("empty gradient input", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_ELU_BWD, {}, {});
  });
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_SOFTPLUS_BETA;
  attributes.softplus_beta = 0;
  expect_error("non-positive softplus beta", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_SOFTPLUS_BWD, {1}, {2}, attributes);
  });
  expect_error("gradient attributes for another mode", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_ELU_BWD, {1}, {2}, attributes);
  });
  attributes = FLAGDNN_POINTWISE_ATTRIBUTES_INITIALIZER;
  attributes.flags = FLAGDNN_POINTWISE_ATTRIBUTE_ELU_ALPHA;
  attributes.elu_alpha = std::numeric_limits<double>::infinity();
  expect_error("non-finite activation attribute", [&] {
    (void)evaluate(FLAGDNN_POINTWISE_ELU_BWD, {1}, {2}, attributes);
  });
  if (cpu::supports_activation_backward(FLAGDNN_POINTWISE_NOT_SET)) {
    throw std::runtime_error("unset activation gradient reported supported");
  }
}

void verify_add_square() {
  const std::vector<float> left = {1.0F, -2.0F};
  const std::vector<float> right = {1.5F, -2.5F, 3.5F};
  std::size_t calls = 0;
  const auto round_virtual = [&](std::span<const float> values) {
    ++calls;
    std::vector<float> rounded;
    for (const auto value : values) rounded.push_back(std::floor(value));
    return rounded;
  };
  expect_values<float>("AddSquare virtual rounding before broadcast/add",
      cpu::evaluate_add_square(left, Shape{2, 1}, right, Shape{3},
                               Shape{2, 3}, round_virtual),
      {3, 7, 13, 0, 4, 10});
  if (calls != 1) throw std::runtime_error("AddSquare virtual quantization count differs");
  expect_error("AddSquare missing quantizer", [&] {
    (void)cpu::evaluate_add_square(left, Shape{2, 1}, right, Shape{3},
                                   Shape{2, 3}, {});
  });
  expect_error("AddSquare quantizer count mismatch", [&] {
    (void)cpu::evaluate_add_square(left, Shape{2, 1}, right, Shape{3},
        Shape{2, 3}, [](std::span<const float>) { return std::vector<float>{1}; });
  });

}

template <typename T, typename Evaluate>
void verify_invalid_inputs(Evaluate evaluate) {
  const std::vector<T> one = {1};
  const std::vector<T> two = {1, 2};
  expect_error("unsupported mode", [&] {
    evaluate(FLAGDNN_POINTWISE_RELU_FWD, one, Shape{1}, one, Shape{1}, Shape{1});
  });
  expect_error("empty output shape", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, one, Shape{1}, one, Shape{1}, Shape{});
  });
  expect_error("empty input shape", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, one, Shape{}, one, Shape{1}, Shape{1});
  });
  expect_error("value-count mismatch", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, two, Shape{1}, one, Shape{1}, Shape{1});
  });
  expect_error("input rank exceeds output", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, one, Shape{1, 1}, one, Shape{1}, Shape{1});
  });
  expect_error("incompatible broadcast", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, two, Shape{2}, one, Shape{1}, Shape{3});
  });
  expect_error("non-positive dimension", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, one, Shape{-1}, one, Shape{1}, Shape{1});
  });
  expect_error<std::overflow_error>("shape overflow", [&] {
    evaluate(FLAGDNN_POINTWISE_ADD, one, Shape{1}, one, Shape{1},
             Shape{std::numeric_limits<std::int64_t>::max(), 3});
  });
}

int main() {
  try {
    verify_float();
    verify_legacy_float_special_values();
    verify_activation_backward();
    verify_add_square();
    verify_invalid_inputs<float>([](auto mode, const auto& a, const auto& a_shape,
                                    const auto& b, const auto& b_shape,
                                    const auto& shape) {
      (void)cpu::evaluate_binary_pointwise(mode, a, a_shape, b, b_shape, shape);
    });

    std::cout << "PASS CPU pointwise contracts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
