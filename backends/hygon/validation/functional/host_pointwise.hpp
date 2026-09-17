/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "pointwise_runner_support.hpp"

namespace flagdnn::testing::hygon_functional::pointwise {

// Independent host mathematics. This code neither calls nor imports the
// device kernel and is used only for functional accuracy, never timing.
inline std::size_t broadcast_index(std::size_t index, const TestTensor &input,
                                   const TestTensor &output) {
  std::size_t result = 0, stride = 1;
  for (std::size_t d = output.dimensions.size(); d-- > 0;) {
    const auto coordinate = index % output.dimensions[d];
    index /= output.dimensions[d];
    if (d + input.dimensions.size() >= output.dimensions.size()) {
      const auto extent = input.dimensions[d + input.dimensions.size() -
                                           output.dimensions.size()];
      if (extent != 1)
        result += coordinate * stride;
      stride *= extent;
    }
  }
  return result;
}

inline double
host_pointwise_value(flagdnnPointwiseMode_t mode, double a, double b,
                     bool predicate, double alpha,
                     const flagdnnPointwiseAttributes_t &attributes) {
  const double slope = attributes.relu_lower_clip_slope;
  const double lower = attributes.relu_lower_clip;
  const bool clipped =
      attributes.flags & FLAGDNN_POINTWISE_ATTRIBUTE_RELU_UPPER_CLIP;
  const double upper = attributes.relu_upper_clip;
  const double beta = attributes.swish_beta;
  const double elu = attributes.elu_alpha;
  const double softplus = attributes.softplus_beta;
  const auto sigmoid = [](double x) {
    return x >= 0 ? 1 / (1 + std::exp(-x)) : std::exp(x) / (1 + std::exp(x));
  };
  const auto relu = [&](double x) {
    const auto y = x < lower ? lower + slope * (x - lower) : x;
    return clipped ? std::min(y, upper) : y;
  };
  switch (mode) {
  case FLAGDNN_POINTWISE_ADD:
    return a + alpha * b;
  case FLAGDNN_POINTWISE_SUB:
    return a - alpha * b;
  case FLAGDNN_POINTWISE_MUL:
    return a * b;
  case FLAGDNN_POINTWISE_DIV:
    return a / b;
  case FLAGDNN_POINTWISE_POW:
    return std::pow(a, b);
  case FLAGDNN_POINTWISE_MOD:
    return std::fmod(a, b);
  case FLAGDNN_POINTWISE_MIN:
    return std::min(a, b);
  case FLAGDNN_POINTWISE_MAX:
    return std::max(a, b);
  case FLAGDNN_POINTWISE_CMP_EQ:
    return a == b;
  case FLAGDNN_POINTWISE_CMP_NEQ:
    return a != b;
  case FLAGDNN_POINTWISE_CMP_GT:
    return a > b;
  case FLAGDNN_POINTWISE_CMP_GE:
    return a >= b;
  case FLAGDNN_POINTWISE_CMP_LT:
    return a < b;
  case FLAGDNN_POINTWISE_CMP_LE:
    return a <= b;
  case FLAGDNN_POINTWISE_LOGICAL_AND:
    return a != 0 && b != 0;
  case FLAGDNN_POINTWISE_LOGICAL_OR:
    return a != 0 || b != 0;
  case FLAGDNN_POINTWISE_LOGICAL_NOT:
    return a == 0;
  case FLAGDNN_POINTWISE_BINARY_SELECT:
    return predicate ? a : b;
  case FLAGDNN_POINTWISE_ABS:
    return std::abs(a);
  case FLAGDNN_POINTWISE_NEG:
    return -a;
  case FLAGDNN_POINTWISE_IDENTITY:
    return a;
  case FLAGDNN_POINTWISE_CEIL:
    return std::ceil(a);
  case FLAGDNN_POINTWISE_FLOOR:
    return std::floor(a);
  case FLAGDNN_POINTWISE_SQRT:
    return std::sqrt(a);
  case FLAGDNN_POINTWISE_RSQRT:
    return 1 / std::sqrt(a);
  case FLAGDNN_POINTWISE_RECIPROCAL:
    return 1 / a;
  case FLAGDNN_POINTWISE_EXP:
    return std::exp(a);
  case FLAGDNN_POINTWISE_LOG:
    return std::log(a);
  case FLAGDNN_POINTWISE_ERF:
    return std::erf(a);
  case FLAGDNN_POINTWISE_SIN:
    return std::sin(a);
  case FLAGDNN_POINTWISE_COS:
    return std::cos(a);
  case FLAGDNN_POINTWISE_TAN:
    return std::tan(a);
  case FLAGDNN_POINTWISE_RELU_FWD:
    return relu(a);
  case FLAGDNN_POINTWISE_SIGMOID_FWD:
    return sigmoid(a);
  case FLAGDNN_POINTWISE_TANH_FWD:
    return std::tanh(a);
  case FLAGDNN_POINTWISE_ELU_FWD:
    return a > 0 ? a : elu * std::expm1(a);
  case FLAGDNN_POINTWISE_SOFTPLUS_FWD:
    return (std::max(softplus * a, 0.0) +
            std::log1p(std::exp(-std::abs(softplus * a)))) /
           softplus;
  case FLAGDNN_POINTWISE_SWISH_FWD:
    return a * sigmoid(beta * a);
  case FLAGDNN_POINTWISE_GELU_FWD:
    return 0.5 * a * std::erfc(-a / std::sqrt(2.0));
  case FLAGDNN_POINTWISE_GELU_APPROX_TANH_FWD:
    return 0.5 * a *
           (1 + std::tanh(std::sqrt(2 / std::acos(-1.0)) *
                          (a + 0.044715 * a * a * a)));
  case FLAGDNN_POINTWISE_RELU_BWD:
    return a * (clipped && relu(b) >= upper ? 0 : b > lower ? 1 : slope);
  case FLAGDNN_POINTWISE_SIGMOID_BWD: {
    const auto s = sigmoid(b);
    return a * s * (1 - s);
  }
  case FLAGDNN_POINTWISE_TANH_BWD: {
    const auto c = std::cosh(b);
    return a / (c * c);
  }
  case FLAGDNN_POINTWISE_ELU_BWD:
    return a * (b > 0 ? 1 : elu * std::exp(b));
  case FLAGDNN_POINTWISE_SOFTPLUS_BWD:
    return a * sigmoid(softplus * b);
  case FLAGDNN_POINTWISE_SWISH_BWD: {
    const auto s = sigmoid(beta * b);
    return a * (s + beta * b * s * (1 - s));
  }
  case FLAGDNN_POINTWISE_GELU_BWD:
    return a * (0.5 * std::erfc(-b / std::sqrt(2.0)) +
                b * std::exp(-b * b / 2) / std::sqrt(2 * std::acos(-1.0)));
  case FLAGDNN_POINTWISE_GELU_APPROX_TANH_BWD: {
    const double k = std::sqrt(2 / std::acos(-1.0));
    const auto t = std::tanh(k * (b + 0.044715 * b * b * b));
    return a *
           (0.5 * (1 + t) + 0.5 * b * (1 - t * t) * k * (1 + 0.134145 * b * b));
  }
  default:
    throw std::invalid_argument("host pointwise reference mode is missing");
  }
}

inline CaseResult
run_host_pointwise_case(const PointwiseTestCase &test_case,
                        const std::function<flagdnn::Handle &()> &get_handle,
                        hv::Stream &stream) {
  auto executable = build_flagdnn_pointwise(get_handle(), test_case);
  auto buffers = prepare_buffers(
      test_case.inputs, test_case.output, test_case.input_domains,
      BindingAddress::kTensorEntrance, LogicalInputCapture::kEnabled, stream);
  std::vector<float> expected(io::element_count(test_case.output));
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto input = [&](std::size_t j) -> double {
      return j < buffers.logical_inputs.size()
                 ? buffers.logical_inputs[j][broadcast_index(
                       i, test_case.inputs[j], test_case.output)]
                 : 0;
    };
    expected[i] = static_cast<float>(
        host_pointwise_value(test_case.mode, input(0), input(1), input(2) != 0,
                             test_case.alpha, test_case.attributes));
  }
  expected = quantize_logical_output(expected, test_case.output);
  hv::DeviceBuffer workspace(executable->workspace_size());
  execute(*executable, buffers.bindings, workspace, stream);
  stream.synchronize();
  compare_outputs(read_output(buffers, stream, "FlagDNN"), expected,
                  test_case.absolute_tolerance, test_case.relative_tolerance,
                  test_case.name, "independent host reference");
  std::cout << test_case.name
            << ": FlagDNN Graph vs independent host reference PASS\n";
  return CaseResult::kExecuted;
}
} // namespace flagdnn::testing::hygon_functional::pointwise
