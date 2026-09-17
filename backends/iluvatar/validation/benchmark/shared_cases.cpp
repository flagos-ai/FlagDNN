// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "benchmark/shared_cases.hpp"
#include "common/convolution.hpp"
#include "common/layout.hpp"
#include "common/matmul.hpp"
#include "common/pointwise.hpp"
#include "common/reduction.hpp"
#include <algorithm>
#include <cctype>
namespace flagdnn::iluvatar::validation::benchmark {
namespace {
namespace b = flagdnn::benchmarking;
namespace t = flagdnn::testing;
b::TensorSpec tensor(const t::TestTensor &value) {
  return {value.uid, value.data_type, value.dimensions, value.strides,
          value.binding_byte_offset};
}
b::BenchmarkCase base(std::string name, b::Operation op,
                      std::initializer_list<t::TestTensor> tensors) {
  b::BenchmarkCase result;
  result.name = std::move(name) + "_shared";
  result.operation = op;
  for (const auto &value : tensors)
    result.tensors.push_back(tensor(value));
  return result;
}
} // namespace
std::vector<b::BenchmarkCase> shared_benchmark_cases(std::string_view marker) {
  std::string op(marker.substr(8, marker.size() - 8 - 10));
  std::transform(op.begin(), op.end(), op.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  std::vector<b::BenchmarkCase> result;
  if (op == "identity" || op == "logical_not" || op == "logical_and" ||
      op == "logical_or") {
    t::PointwiseCaseDefinition definition;
    definition.operation_name = op;
    definition.mode = op == "identity"      ? FLAGDNN_POINTWISE_IDENTITY
                      : op == "logical_not" ? FLAGDNN_POINTWISE_LOGICAL_NOT
                      : op == "logical_and" ? FLAGDNN_POINTWISE_LOGICAL_AND
                                            : FLAGDNN_POINTWISE_LOGICAL_OR;
    const bool logical = op != "identity";
    definition.input_domain = logical ? t::PointwiseInputDomain::kLogical
                                      : t::PointwiseInputDomain::kReal;
    const auto cases = (op == "logical_and" || op == "logical_or")
                           ? t::make_binary_pointwise_cases(definition)
                           : t::make_unary_pointwise_cases(definition);
    for (const auto &c : cases) {
      auto v = base(c.name, b::Operation::kPointwise, {});
      for (const auto &input : c.inputs)
        v.tensors.push_back(tensor(input));
      v.tensors.push_back(tensor(c.output));
      v.pointwise_mode = c.mode;
      v.pointwise_attributes = c.attributes;
      v.input_domain =
          logical ? b::InputDomain::kLogical : b::InputDomain::kReal;
      v.absolute_tolerance = c.absolute_tolerance;
      v.relative_tolerance = c.relative_tolerance;
      result.push_back(std::move(v));
    }
  } else if (op == "reshape" || op == "transpose" || op == "slice") {
    const auto mode = op == "reshape"     ? t::LayoutOperation::kReshape
                      : op == "transpose" ? t::LayoutOperation::kTranspose
                                          : t::LayoutOperation::kSlice;
    const auto operation = op == "reshape"     ? b::Operation::kReshape
                           : op == "transpose" ? b::Operation::kTranspose
                                               : b::Operation::kSlice;
    for (const auto &c : t::make_layout_cases(mode)) {
      auto v = base(c.name, operation, {c.input, c.output});
      v.reshape.dimensions = c.output.dimensions;
      v.reshape.strides = c.output.strides;
      v.transpose.permutation = c.permutation;
      v.slice.slices = c.slices;
      v.slice.strides = c.slice_strides;
      result.push_back(std::move(v));
    }
  } else if (op == "matmul") {
    for (const auto &c : t::make_matmul_cases()) {
      if (!c.input_precision || c.output.dimensions.size() > 3)
        continue;
      auto v = base(c.name, b::Operation::kMatmul, {c.a, c.b, c.output});
      v.absolute_tolerance = c.absolute_tolerance;
      v.relative_tolerance = c.relative_tolerance;
      result.push_back(std::move(v));
    }
  } else if (op == "conv_fprop" || op == "conv_dgrad" || op == "conv_wgrad") {
    const auto direction = op == "conv_fprop" ? t::ConvolutionDirection::kFprop
                           : op == "conv_dgrad"
                               ? t::ConvolutionDirection::kDgrad
                               : t::ConvolutionDirection::kWgrad;
    const auto operation = op == "conv_fprop" ? b::Operation::kConvolutionFprop
                           : op == "conv_dgrad"
                               ? b::Operation::kConvolutionDgrad
                               : b::Operation::kConvolutionWgrad;
    for (const auto &c : t::make_convolution_cases(direction)) {
      if (!c.input_precision)
        continue;
      auto v = base(c.name, operation, {});
      const auto inputs = direction == t::ConvolutionDirection::kFprop
                              ? std::vector{c.x, c.w}
                          : direction == t::ConvolutionDirection::kDgrad
                              ? std::vector{c.y, c.w}
                              : std::vector{c.y, c.x};
      for (const auto &input : inputs)
        v.tensors.push_back(tensor(input));
      v.tensors.push_back(tensor(t::convolution_output_tensor(c)));
      v.convolution = {static_cast<std::int32_t>(c.stride.size()),
                       c.pre_padding,
                       c.post_padding,
                       c.stride,
                       c.dilation,
                       c.groups,
                       static_cast<b::ConvolutionMode>(c.mode)};
      v.absolute_tolerance = c.absolute_tolerance;
      v.relative_tolerance = c.relative_tolerance;
      result.push_back(std::move(v));
    }
  } else if (op == "reduction") {
    for (const auto &c : t::make_reduction_cases()) {
      if (c.name.find("_to_fp32_") == std::string::npos)
        continue;
      // Includes the NVIDIA cuDNN group and its integer/BF16 capability skips.
      auto v = base(c.name, b::Operation::kReduction, {c.input, c.output});
      v.reduction_mode = c.mode;
      v.reduction_axis = c.axis;
      v.keep_dimensions = c.keep_dimensions;
      v.absolute_tolerance = c.absolute_tolerance;
      v.relative_tolerance = c.relative_tolerance;
      result.push_back(std::move(v));
    }
  }
  return result;
}
} // namespace flagdnn::iluvatar::validation::benchmark
