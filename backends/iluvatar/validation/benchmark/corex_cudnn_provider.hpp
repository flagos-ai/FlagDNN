// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"
#include "corex_cudnn_reference.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace flagdnn::iluvatar::validation::benchmark {

class CorexCudnnProvider final
    : public flagdnn::benchmarking::BenchmarkProvider {
public:
  CorexCudnnProvider();

  [[nodiscard]] std::string_view name() const noexcept override {
    return "corex_cudnn";
  }
  [[nodiscard]] flagdnn::benchmarking::ProviderCapability capability(
      const flagdnn::benchmarking::BenchmarkCase &specification) const override;
  [[nodiscard]] std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
  build(const flagdnn::benchmarking::BenchmarkCase &specification) override;

  [[nodiscard]] std::string operation_name(
      const flagdnn::benchmarking::BenchmarkCase &specification) const;

private:
  CorexCudnnCapabilityCatalog catalog_;
};

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_COREX_CUDNN_PROVIDER_HPP_
