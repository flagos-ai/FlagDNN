/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "tensor_reference.hpp"

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::hipdnn_detail {
namespace {

namespace hv = validation::hygon;

class TensorExecutable final : public HipdnnExecutable {
public:
  TensorExecutable(hv::HipdnnTensorOperation operation,
                   std::vector<hv::ReferenceTensor> tensors)
      : plan_(std::move(operation), std::move(tensors)) {}

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return plan_.workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    plan_.execute(bindings, workspace, workspace_size, stream);
  }

private:
  hv::HipdnnTensorPlan plan_;
};

hv::HipdnnTensorOperation operation(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kReduction:
    return hv::make_hipdnn_reduction_operation(specification.reduction_mode,
                                               specification.reduction_axis,
                                               specification.keep_dimensions);
  case Operation::kSlice:
    return hv::make_hipdnn_slice_operation(specification.slice.slices,
                                           specification.slice.strides);
  case Operation::kReshape:
    return hv::make_hipdnn_tensor_unavailable(
        "hipdnnTransformTensor reshape is not an exact validated primitive");
  case Operation::kTranspose:
    return hv::make_hipdnn_tensor_unavailable(
        "hipdnnTransformTensor does not provide a validated exact transpose");
  case Operation::kMatmul:
    return hv::make_hipdnn_tensor_unavailable(
        "hipDNN exposes no exact primitive MatMul reference on this stack");
  default:
    return hv::make_hipdnn_tensor_unavailable(
        "benchmark family is outside the hipDNN tensor reference adapter");
  }
}

std::vector<hv::ReferenceTensor>
as_reference_tensors(std::span<const TensorSpec> tensors) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(tensors.size());
  for (const TensorSpec &tensor : tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

} // namespace

std::vector<TensorSpec>
tensor_reference_specs(const BenchmarkCase &specification) {
  // Preserve the public benchmark tensor layout exactly.  In particular, a
  // slice output intentionally carries the source-derived gapped strides.
  // Repacking only the hipDNN side to dense would time a different physical
  // workload than FlagDNN and would make the speedup incomparable.  If the
  // installed hipDNN primitive cannot represent those strides, capability
  // validation must fail closed and report a structured SKIP.
  return specification.tensors;
}

std::vector<hv::ReferenceTensor>
tensor_diagnostic_tensors(const BenchmarkCase &specification) {
  return as_reference_tensors(tensor_reference_specs(specification));
}

ProviderCapability tensor_capability(const BenchmarkCase &specification) {
  if (specification.output_count != 1 || specification.tensors.empty()) {
    throw std::invalid_argument(
        "Hygon tensor benchmark adapter requires exactly one output");
  }
  const hv::HipdnnCapability result = hv::hipdnn_tensor_capability(
      operation(specification), tensor_diagnostic_tensors(specification));
  hv::require_valid_hipdnn_adapter_contract(
      result, tensor_operation_name(specification));
  return {result.supported, result.reason};
}

std::unique_ptr<HipdnnExecutable>
build_tensor(const BenchmarkCase &specification) {
  const ProviderCapability support = tensor_capability(specification);
  if (!support.supported) {
    throw BenchmarkUnsupportedError(support.reason);
  }
  return std::make_unique<TensorExecutable>(
      operation(specification), tensor_diagnostic_tensors(specification));
}

std::string tensor_operation_name(const BenchmarkCase &specification) {
  switch (specification.operation) {
  case Operation::kReduction:
    return "reduction";
  case Operation::kMatmul:
    return "matmul";
  case Operation::kReshape:
    return "reshape";
  case Operation::kTranspose:
    return "transpose";
  case Operation::kSlice:
    return "slice";
  default:
    return "tensor";
  }
}

} // namespace flagdnn::benchmarking::hipdnn_detail
