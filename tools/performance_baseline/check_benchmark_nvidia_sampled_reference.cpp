/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
/* CPU-only check for the sampled oracle used by benchmark_nvidia_cudnn.cpp.
   c++ -O2 -std=c++20 -I. -Iinclude -Ibuild/nvidia/generated/include \
     tools/performance_baseline/check_benchmark_nvidia_sampled_reference.cpp \
     build/nvidia/benchmark/libflagdnn_benchmark_cases.a \
     build/nvidia/reference/cpu/libflagdnn_validation_cpu_reference.a \
     -o /tmp/check_benchmark_nvidia_sampled_reference
   /tmp/check_benchmark_nvidia_sampled_reference
*/
#include "benchmark/common/cases.hpp"
#include "reference/cpu/benchmark.hpp"
#include "tools/performance_baseline/benchmark_nvidia_sampled_reference.hpp"

#include <cmath>
#include <iostream>

int main() {
  using namespace flagdnn::benchmarking;
  using namespace standalone_cudnn;
  const auto quantize = [](std::span<const float> values, flagdnnDataType_t) {
    return std::vector<float>(values.begin(), values.end());
  };
  std::size_t checked = 0;
  for (auto cases : {matmul_cases(), conv_dgrad_cases(), conv_wgrad_cases()}) {
    for (auto& spec : cases) {
      if (spec.tensors[0].data_type != FLAGDNN_DATA_FLOAT32) continue;
      if (product(spec.tensors[0].dimensions) > 100000 ||
          product(spec.tensors[1].dimensions) > 100000 ||
          product(spec.tensors.back().dimensions) > 100000) continue;
      std::vector<std::vector<float>> inputs;
      for (std::size_t t = 0; t < input_tensor_count(spec); ++t) {
        std::vector<float> values(product(spec.tensors[t].dimensions));
        for (std::size_t i = 0; i < values.size(); ++i) {
          values[i] = float(int((i * 17 + t * 11) % 41) - 20) / float(13 + t);
        }
        inputs.push_back(values);
      }
      const auto indices = sample_indices(product(spec.tensors.back().dimensions), 64);
      const auto actual = sampled_reference(spec, inputs, indices);
      const auto expected =
          flagdnn::reference::cpu::evaluate_benchmark(spec, inputs, quantize).at(0);
      for (std::size_t i = 0; i < indices.size(); ++i) {
        if (std::abs(actual[i] - expected[indices[i]]) > 0.01F) {
          std::cerr << spec.name << ' ' << indices[i] << ' ' << actual[i] << ' '
                    << expected[indices[i]] << '\n';
          return 1;
        }
      }
      ++checked;
    }
  }

  // Forward/backward adjoint identity, including groups, stride, dilation and
  // flipped kernels. This checks the new FProp formula using the existing
  // public CPU Dgrad oracle, which uses a different loop organization.
  for (auto mode : {ConvolutionMode::kCrossCorrelation, ConvolutionMode::kConvolution}) {
    BenchmarkCase forward;
    forward.operation = Operation::kConvolutionFprop;
    forward.tensors = {tensor(1, {2, 4, 7, 8}), tensor(2, {6, 2, 2, 3}),
                       tensor(3, {2, 6, 4, 4})};
    forward.convolution = {2, {1, 1}, {1, 0}, {2, 2}, {2, 1}, 2, mode};
    std::vector<std::vector<float>> inputs;
    for (int t = 0; t < 3; ++t) {
      std::vector<float> values(product(forward.tensors[t].dimensions));
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = float(int((i * 7 + t) % 17) - 8) / 9;
      }
      inputs.push_back(values);
    }
    const auto indices = sample_indices(inputs[2].size(), inputs[2].size());
    const auto y = sampled_reference(forward, {inputs[0], inputs[1]}, indices);
    auto backward = forward;
    backward.operation = Operation::kConvolutionDgrad;
    backward.tensors = {forward.tensors[2], forward.tensors[1], forward.tensors[0]};
    const auto dx = flagdnn::reference::cpu::evaluate_benchmark(
        backward, {inputs[2], inputs[1]}, quantize).at(0);
    double lhs = 0, rhs = 0;
    for (std::size_t i = 0; i < y.size(); ++i) lhs += double(y[i]) * inputs[2][i];
    for (std::size_t i = 0; i < dx.size(); ++i) rhs += double(dx[i]) * inputs[0][i];
    if (std::abs(lhs - rhs) > 0.0001) {
      std::cerr << "adjoint mismatch " << lhs << ' ' << rhs << '\n';
      return 1;
    }
    ++checked;
  }
  std::cout << "Sampled CPU reference validated against full CPU oracle and "
               "adjoint identities: " << checked << " cases\n";
}
