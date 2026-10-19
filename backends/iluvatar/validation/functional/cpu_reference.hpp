// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "benchmark/common/case.hpp"
#include "functional/runner_support.hpp"
#include "reference/cpu/benchmark.hpp"

namespace flagdnn::iluvatar::validation::functional {
// Only descriptor adaptation lives here. All CPU arithmetic is implemented
// in reference/cpu, shared with the independent benchmark oracle.
inline benchmarking::BenchmarkCase cpu_case(const CasePlan &plan) {
  benchmarking::BenchmarkCase result;
  result.name = plan.case_name;
  result.output_count = plan.outputs.size();
  const auto append = [&](const testing::TestTensor &tensor) {
    result.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                              tensor.strides, tensor.binding_byte_offset});
  };
  for (const auto &input : plan.inputs) append(input.tensor);
  for (const auto &output : plan.outputs) append(output.tensor);
  return result;
}

inline HostReference cpu_reference(benchmarking::BenchmarkCase specification) {
  return [specification = std::move(specification)](const auto &inputs) {
    return reference::cpu::evaluate_benchmark(
        specification, inputs,
        [](std::span<const float> values, flagdnnDataType_t dtype) {
          return decode(encode(values, dtype), dtype, values.size());
        });
  };
}

inline HostReference pointwise_cpu_reference(
    const CasePlan &plan, flagdnnPointwiseMode_t mode, double alpha = 1,
    flagdnnPointwiseAttributes_t attributes =
        benchmarking::default_pointwise_attributes()) {
  auto specification = cpu_case(plan);
  specification.operation = benchmarking::Operation::kPointwise;
  specification.pointwise_mode = mode;
  specification.pointwise_attributes = attributes;
  specification.add_alpha = alpha;
  return cpu_reference(std::move(specification));
}
} // namespace flagdnn::iluvatar::validation::functional
