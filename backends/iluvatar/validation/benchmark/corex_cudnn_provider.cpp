// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/corex_cudnn_provider.hpp"

#include "benchmark/corex_cudnn_capability_policy.hpp"

#include "benchmark/ops.hpp"

#include <algorithm>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG
#define FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG                              \
  "corex_cudnn_capabilities.json"
#endif

namespace flagdnn::iluvatar::validation::benchmark {
namespace {

class CorexCudnnExecutable final
    : public flagdnn::benchmarking::BenchmarkExecutable {
public:
  explicit CorexCudnnExecutable(
      std::unique_ptr<flagdnn::testing::TestExecutable> executable)
      : executable_(std::move(executable)) {
    if (executable_ == nullptr) {
      throw std::invalid_argument("CoreX cuDNN executable is null");
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return executable_->workspace_size();
  }

  void execute(std::span<const flagdnnBinding_t> bindings, void *workspace,
               std::size_t workspace_size, flagdnnStream_t stream) override {
    executable_->execute(bindings, workspace, workspace_size, stream);
  }

private:
  std::unique_ptr<flagdnn::testing::TestExecutable> executable_;
};

std::string performance_prefix(std::string_view name) {
  const std::size_t separator = name.find("_perf_");
  if (separator == std::string_view::npos || separator == 0) {
    throw std::invalid_argument("benchmark case name has no operation prefix");
  }
  return std::string(name.substr(0, separator));
}

} // namespace

CorexCudnnProvider::CorexCudnnProvider()
    : catalog_(CorexCudnnCapabilityCatalog::load(
          FLAGDNN_ILUVATAR_CUDNN_CAPABILITY_CATALOG)) {}

std::string CorexCudnnProvider::operation_name(
    const flagdnn::benchmarking::BenchmarkCase &specification) const {
  using flagdnn::benchmarking::Operation;
  switch (specification.operation) {
  case Operation::kRelu:
    return "relu";
  case Operation::kPointwise:
    return performance_prefix(specification.name);
  case Operation::kAdd:
    return "add";
  case Operation::kReduction:
    return "reduction";
  case Operation::kConvolutionFprop:
    return "conv_fprop";
  case Operation::kConvolutionDgrad:
    return "conv_dgrad";
  case Operation::kConvolutionWgrad:
    return "conv_wgrad";
  case Operation::kMatmul:
    return "matmul";
  case Operation::kReshape:
    return "reshape";
  case Operation::kTranspose:
    return "transpose";
  case Operation::kSlice:
    return "slice";
  case Operation::kLayernorm:
    return "layernorm";
  case Operation::kRmsnorm:
    return "rmsnorm";
  case Operation::kBatchnorm:
    return "batchnorm";
  case Operation::kBatchnormInference:
    return "batchnorm_inference";
  case Operation::kGraph:
    return performance_prefix(specification.name);
  }
  throw std::invalid_argument("unknown benchmark operation");
}

flagdnn::benchmarking::ProviderCapability CorexCudnnProvider::capability(
    const flagdnn::benchmarking::BenchmarkCase &specification) const {
  const std::string operation = operation_name(specification);
  return benchmark_capability_from_catalog(catalog_, specification, operation);
}

std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
CorexCudnnProvider::build(
    const flagdnn::benchmarking::BenchmarkCase &specification) {
  using flagdnn::benchmarking::Operation;
  std::unique_ptr<flagdnn::testing::TestExecutable> executable;
  switch (specification.operation) {
  case Operation::kRelu:
  case Operation::kPointwise:
  case Operation::kAdd:
    executable = build_pointwise_reference(specification);
    break;
  case Operation::kReduction:
    executable = build_reduction_reference(specification);
    break;
  case Operation::kReshape:
  case Operation::kTranspose:
  case Operation::kSlice:
    executable = build_layout_reference(specification);
    break;
  case Operation::kMatmul:
    executable = build_matmul_reference(specification);
    break;
  case Operation::kConvolutionFprop:
  case Operation::kConvolutionDgrad:
  case Operation::kConvolutionWgrad:
    executable = build_convolution_reference(specification);
    break;
  case Operation::kLayernorm:
  case Operation::kRmsnorm:
  case Operation::kBatchnorm:
  case Operation::kBatchnormInference:
    executable = build_normalization_reference(specification);
    break;
  case Operation::kGraph:
    executable = build_graph_reference(specification);
    break;
  }
  if (executable == nullptr) {
    throw std::logic_error("CoreX cuDNN provider returned no executable");
  }
  return std::make_unique<CorexCudnnExecutable>(std::move(executable));
}

} // namespace flagdnn::iluvatar::validation::benchmark
