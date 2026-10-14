/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_MTHREADS_NATIVE_POINTWISE_CASE_HPP_
#define FLAGDNN_MTHREADS_NATIVE_POINTWISE_CASE_HPP_

#include "benchmark/common/case.hpp"
#include "tests/common/pointwise.hpp"

namespace flagdnn::validation::mthreads::equivalent {

// Shared by the runtime adapter and its CPU baseline contract. Preserve every
// per-input domain: native attribute probes use Real upstream gradients and
// Scaled activations, unlike ordinary common performance cases.
inline benchmarking::InputDomain pointwise_input_domain(testing::PointwiseInputDomain value) {
  using D = testing::PointwiseInputDomain;
  switch (value) {
    case D::kReal: return benchmarking::InputDomain::kReal;
    case D::kPositive: return benchmarking::InputDomain::kPositive;
    case D::kScaled: return benchmarking::InputDomain::kScaled;
    case D::kTan: return benchmarking::InputDomain::kTan;
    case D::kDivisor: return benchmarking::InputDomain::kDivisor;
    case D::kModulo: return benchmarking::InputDomain::kModulo;
    case D::kPower: return benchmarking::InputDomain::kPower;
    case D::kModuloSigned: return benchmarking::InputDomain::kModuloSigned;
    case D::kComparison: return benchmarking::InputDomain::kComparison;
    case D::kLogical: return benchmarking::InputDomain::kLogical;
  }
  throw std::invalid_argument("unknown native pointwise input domain");
}

inline benchmarking::BenchmarkCase as_benchmark(const testing::PointwiseTestCase& native) {
  benchmarking::BenchmarkCase result;
  result.name = native.name;
  result.operation = benchmarking::Operation::kPointwise;
  result.pointwise_mode = native.mode;
  result.pointwise_attributes = native.attributes;
  result.add_alpha = native.alpha;
  result.absolute_tolerance = native.absolute_tolerance;
  result.relative_tolerance = native.relative_tolerance;
  if (native.inputs.empty() ||
      native.input_domains.size() != native.inputs.size())
    throw std::invalid_argument("invalid native equivalent input domains");
  result.input_domain = pointwise_input_domain(native.input_domains.front());
  for (auto value : native.input_domains)
    result.input_domains.push_back(pointwise_input_domain(value));
  const auto append = [&](const testing::TestTensor& tensor) {
    result.tensors.push_back({tensor.uid, tensor.data_type, tensor.dimensions,
                              tensor.strides, tensor.binding_byte_offset});
  };
  for (const auto& tensor : native.inputs) append(tensor);
  append(native.output);
  return result;
}

}  // namespace flagdnn::validation::mthreads::equivalent
#endif
