/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "ops.hpp"

#include "benchmark/pointwise_matcher.hpp"
#include "pointwise_reference.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::benchmarking::hipdnn_detail {
namespace {

namespace hv = validation::hygon;

class PointwiseExecutable final : public HipdnnExecutable {
public:
  PointwiseExecutable(hv::HipdnnPointwiseOperation operation,
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
  hv::HipdnnPointwisePlan plan_;
};

hv::HipdnnPointwiseOperation fixed_operation(hv::HipdnnPointwiseKind kind,
                                             double alpha = 1.0) {
  hv::HipdnnPointwiseOperation result;
  result.kind = kind;
  result.alpha = alpha;
  result.unavailable_reason.clear();
  return result;
}

hv::HipdnnPointwiseOperation operation(const BenchmarkCase &specification) {
  if (specification.operation == Operation::kRelu) {
    return fixed_operation(hv::HipdnnPointwiseKind::kRelu);
  }
  if (specification.operation == Operation::kAdd) {
    return fixed_operation(hv::HipdnnPointwiseKind::kAdd,
                           specification.add_alpha);
  }
  if (specification.operation == Operation::kPointwise) {
    return hv::make_hipdnn_pointwise_operation(
        specification.pointwise_mode, specification.pointwise_attributes,
        specification.add_alpha);
  }
  if (matches_add_square_graph(specification)) {
    return fixed_operation(hv::HipdnnPointwiseKind::kAddSquare);
  }
  hv::HipdnnPointwiseOperation result;
  result.unavailable_reason =
      "benchmark graph has no exact hipDNN pointwise primitive sequence";
  return result;
}

std::vector<hv::ReferenceTensor>
reference_tensors(const BenchmarkCase &specification) {
  std::vector<hv::ReferenceTensor> result;
  result.reserve(specification.tensors.size());
  for (const TensorSpec &tensor : specification.tensors) {
    result.push_back(hv::as_reference_tensor(tensor));
  }
  return result;
}

} // namespace

ProviderCapability pointwise_capability(const BenchmarkCase &specification) {
  const hv::HipdnnCapability result = hv::hipdnn_pointwise_capability(
      operation(specification), reference_tensors(specification), true);
  hv::require_valid_hipdnn_adapter_contract(
      result, pointwise_operation_name(specification));
  return {result.supported, result.reason};
}

ProviderCapability
pointwise_capture_capability(const BenchmarkCase &specification) {
  const hv::HipdnnCapability result =
      hv::hipdnn_pointwise_capture_capability(operation(specification));
  hv::require_valid_hipdnn_adapter_contract(
      result, pointwise_operation_name(specification));
  return {result.supported, result.reason};
}

std::unique_ptr<HipdnnExecutable>
build_pointwise(const BenchmarkCase &specification) {
  const ProviderCapability support = pointwise_capability(specification);
  if (!support.supported) {
    throw BenchmarkUnsupportedError(support.reason);
  }
  return std::make_unique<PointwiseExecutable>(
      operation(specification), reference_tensors(specification));
}

std::vector<TensorSpec>
pointwise_reference_specs(const BenchmarkCase &specification) {
  return specification.tensors;
}

std::vector<hv::ReferenceTensor>
pointwise_diagnostic_tensors(const BenchmarkCase &specification) {
  return reference_tensors(specification);
}

std::string pointwise_operation_name(const BenchmarkCase &specification) {
  constexpr std::array<std::string_view, 9> kMarkers = {
      "_strided_", "_perf_",     "_fp32_", "_fp16_",
      "_bf16_",    "_bfloat16_", "_bool_", "_int32_", "_fp8_"};
  const std::string_view case_name = specification.name;
  std::size_t end = std::string_view::npos;
  for (const std::string_view marker : kMarkers) {
    const std::size_t position = case_name.find(marker);
    if (position != std::string_view::npos) {
      end = std::min(end, position);
    }
  }
  return std::string(end == std::string_view::npos ? case_name
                                                   : case_name.substr(0, end));
}

bool pointwise_uses_sequence(const BenchmarkCase &specification) noexcept {
  return hv::hipdnn_pointwise_uses_sequence(operation(specification).kind);
}

} // namespace flagdnn::benchmarking::hipdnn_detail
