/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "common/case.hpp"
#include "parallel_matmul_reference.hpp"
#include "reference/cpu/benchmark.hpp"
#include "reference/cpu/matrix.hpp"
#include "reference/cpu/pointwise.hpp"
#include "tensor_io.hpp"

namespace flagdnn::benchmarking {
inline bool matrix_operation(Operation op) {
  return op == Operation::kMatmul || op == Operation::kConvolutionFprop ||
         op == Operation::kConvolutionDgrad ||
         op == Operation::kConvolutionWgrad;
}
// Used only for validation outside device-event timing. Default Hygon FP32
// kernels use IEEE inputs; explicit TF32 cases supply precision 2.
inline std::vector<std::vector<float>> matrix_reference(
    const BenchmarkCase& spec, const std::vector<std::vector<float>>& values,
    int precision = 1) {
  namespace cpu = reference::cpu;
  namespace io = validation::hygon::tensor_io;
  std::vector<float> result;
  if (spec.operation == Operation::kMatmul) {
    result = validation::hygon::parallel_matmul_reference(
        {spec.tensors[0].dimensions, spec.tensors[1].dimensions,
         spec.tensors[2].dimensions, spec.tensors[0].data_type, precision},
        values.at(0), values.at(1));
  } else {
    if (!matrix_operation(spec.operation))
      throw std::invalid_argument(
          "matrix oracle requires matmul or convolution");
    const bool f = spec.operation == Operation::kConvolutionFprop;
    const bool d = spec.operation == Operation::kConvolutionDgrad;
    cpu::ConvolutionParameters p;
    p.x_shape = spec.tensors[f ? 0 : d ? 2 : 1].dimensions;
    p.w_shape = spec.tensors[f || d ? 1 : 2].dimensions;
    p.y_shape = spec.tensors[f ? 2 : 0].dimensions;
    p.stride = spec.convolution.stride;
    p.pre_padding = spec.convolution.pre_padding;
    p.dilation = spec.convolution.dilation;
    p.groups = spec.convolution.groups;
    p.input_precision = precision;
    p.flip_kernel = spec.convolution.mode == ConvolutionMode::kConvolution;
    p.direction = f   ? cpu::ConvolutionDirection::kForward
                  : d ? cpu::ConvolutionDirection::kInputGradient
                      : cpu::ConvolutionDirection::kWeightGradient;
    std::vector<std::vector<float>> xyz(3);
    if (f) {
      xyz[0] = values.at(0);
      xyz[1] = values.at(1);
    } else if (d) {
      xyz[2] = values.at(0);
      xyz[1] = values.at(1);
    } else {
      xyz[2] = values.at(0);
      xyz[0] = values.at(1);
    }
    result = cpu::evaluate_convolution(p, xyz);
  }
  const auto dtype = spec.tensors.back().data_type;
  return {io::decode(io::encode(result, dtype), dtype, result.size())};
}

inline std::vector<std::vector<float>> cbr_reference(
    const BenchmarkCase& spec, const std::vector<std::vector<float>>& inputs) {
  namespace io = validation::hygon::tensor_io;
  BenchmarkCase conv;
  conv.operation = Operation::kConvolutionFprop;
  conv.convolution = spec.graph.nodes.at(0).convolution;
  conv.tensors = {spec.tensors.at(0), spec.tensors.at(1),
                  spec.graph.intermediates.at(0)};
  auto values = matrix_reference(conv, {inputs.at(0), inputs.at(1)})[0];
  const auto& shape = conv.tensors.back().dimensions;
  values = reference::cpu::evaluate_binary_pointwise_with_alpha(
      FLAGDNN_POINTWISE_ADD, values, shape, inputs.at(2),
      spec.tensors.at(2).dimensions, shape, 1.0);
  const auto dtype = spec.graph.intermediates.at(1).data_type;
  values = io::decode(io::encode(values, dtype), dtype, values.size());
  BenchmarkCase activation;
  activation.operation = Operation::kRelu;
  activation.tensors = {spec.graph.intermediates.at(1), spec.tensors.back()};
  return reference::cpu::evaluate_benchmark(
      activation, {values},
      [](std::span<const float> logical, flagdnnDataType_t type) {
        return io::decode(io::encode(logical, type), type, logical.size());
      });
}
}  // namespace flagdnn::benchmarking
