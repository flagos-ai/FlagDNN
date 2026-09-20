/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "reference/cpu/pointwise.hpp"

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

void verify_integer() {
  constexpr auto minimum = std::numeric_limits<std::int32_t>::min();
  constexpr auto maximum = std::numeric_limits<std::int32_t>::max();
  const auto paired = [](flagdnnPointwiseMode_t mode,
                         const std::vector<std::int32_t>& a,
                         const std::vector<std::int32_t>& b,
                         std::int32_t alpha = 1) {
    const Shape shape = {static_cast<std::int64_t>(a.size())};
    return cpu::evaluate_binary_pointwise_int32(
        mode, a, shape, b, shape, shape, alpha);
  };
  expect_values<std::int32_t>("INT32 ADD wrap",
      paired(FLAGDNN_POINTWISE_ADD, {maximum, minimum}, {1, -1}),
      {minimum, maximum});
  expect_values<std::int32_t>("INT32 SUB wrap and alpha",
      paired(FLAGDNN_POINTWISE_SUB, {minimum, 3}, {1, 4}, 2),
      {maximum - 1, -5});
  expect_values<std::int32_t>("INT32 MUL wrap",
      paired(FLAGDNN_POINTWISE_MUL, {maximum, minimum}, {2, -1}),
      {-2, minimum});
  expect_values<std::int32_t>("INT32 DIV signed overflow and zero",
      paired(FLAGDNN_POINTWISE_DIV, {-7, 7, minimum, 3}, {3, -3, -1, 0}),
      {-2, -2, minimum, 0});
  expect_values<std::int32_t>("INT32 MOD signed overflow and zero",
      paired(FLAGDNN_POINTWISE_MOD, {-7, 7, minimum, 3}, {3, -3, -1, 0}),
      {-1, 1, 0, 0});
  expect_values<std::int32_t>("INT32 POW negative and wrap",
      paired(FLAGDNN_POINTWISE_POW, {-1, -1, 1, 2, 0, 2},
             {-3, -2, -9, -3, 0, 31}),
      {-1, 1, 1, 0, 1, minimum});
  expect_values<std::int32_t>("INT32 MAX exact",
      paired(FLAGDNN_POINTWISE_MAX, {maximum, minimum, 16777217},
             {maximum - 1, minimum + 1, 16777216}),
      {maximum, minimum + 1, 16777217});
  expect_values<std::int32_t>("INT32 MIN exact",
      paired(FLAGDNN_POINTWISE_MIN, {maximum, minimum},
             {maximum - 1, minimum + 1}),
      {maximum - 1, minimum});
  expect_values<std::int32_t>("INT32 CMP_EQ exact",
      paired(FLAGDNN_POINTWISE_CMP_EQ, {16777217, minimum, maximum},
             {16777216, minimum, maximum}), {0, 1, 1});

  const std::vector<std::int32_t> left = {16777217, -2};
  const std::vector<std::int32_t> right = {3, 4, -5};
  expect_values<std::int32_t>("INT32 ADD broadcast and alpha",
      cpu::evaluate_binary_pointwise_int32(FLAGDNN_POINTWISE_ADD,
          left, Shape{2, 1}, right, Shape{3}, Shape{2, 3}, 2),
      {16777223, 16777225, 16777207, 4, 6, -12});
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
    verify_integer();
    verify_invalid_inputs<float>([](auto mode, const auto& a, const auto& a_shape,
                                    const auto& b, const auto& b_shape,
                                    const auto& shape) {
      (void)cpu::evaluate_binary_pointwise(mode, a, a_shape, b, b_shape, shape);
    });
    verify_invalid_inputs<std::int32_t>(
        [](auto mode, const auto& a, const auto& a_shape, const auto& b,
           const auto& b_shape, const auto& shape) {
          (void)cpu::evaluate_binary_pointwise_int32(
              mode, a, a_shape, b, b_shape, shape);
        });
    std::cout << "PASS CPU pointwise contracts\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
