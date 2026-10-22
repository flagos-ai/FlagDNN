/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "pointwise_runner_support.hpp"
#include "benchmark/common/case.hpp"
#include "reference/cpu/benchmark.hpp"

namespace flagdnn::testing::hygon_functional::pointwise {

// Use the public CPU oracle, independently checked against NVIDIA cuDNN.
// Host evaluation is only for functional accuracy, never GPU timing.
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

inline CaseResult
run_host_pointwise_case(const PointwiseTestCase &test_case,
                        const std::function<flagdnn::Handle &()> &get_handle,
                        hv::Stream &stream) {
  auto executable = build_flagdnn_pointwise(get_handle(), test_case);
  auto buffers = prepare_buffers(
      test_case.inputs, test_case.output, test_case.input_domains,
      BindingAddress::kTensorEntrance, LogicalInputCapture::kEnabled, stream);
  benchmarking::BenchmarkCase specification;
  specification.name = test_case.name;
  specification.operation = benchmarking::Operation::kPointwise;
  specification.pointwise_mode = test_case.mode;
  specification.pointwise_attributes = test_case.attributes;
  specification.add_alpha = test_case.alpha;
  const auto append_tensor = [&](const TestTensor &tensor) {
    specification.tensors.push_back(
        {tensor.uid, tensor.data_type, tensor.dimensions, tensor.strides,
         tensor.binding_byte_offset});
  };
  for (const auto &input : test_case.inputs)
    append_tensor(input);
  append_tensor(test_case.output);
  const auto reference = reference::cpu::evaluate_benchmark(
      specification, buffers.logical_inputs,
      [](std::span<const float> values, flagdnnDataType_t type) {
        return io::decode(io::encode(values, type), type, values.size());
      });
  if (reference.size() != 1)
    throw std::logic_error("pointwise CPU reference output count mismatch");
  const auto &expected = reference.front();
  hv::DeviceBuffer workspace(executable->workspace_size());
  execute(*executable, buffers.bindings, workspace, stream);
  stream.synchronize();
  compare_outputs(read_output(buffers, stream, "FlagDNN"), expected,
                  test_case.absolute_tolerance, test_case.relative_tolerance,
                  test_case.name, "public CPU reference");
  std::cout << test_case.name
            << ": FlagDNN Graph vs public CPU reference PASS\n";
  return CaseResult::kExecuted;
}
} // namespace flagdnn::testing::hygon_functional::pointwise
