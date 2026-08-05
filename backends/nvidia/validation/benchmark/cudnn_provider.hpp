/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

namespace flagdnn::benchmarking {

class CudnnProvider final : public BenchmarkProvider {
 public:
  [[nodiscard]] std::string_view name() const noexcept override {
    return "cudnn";
  }

  [[nodiscard]] ProviderCapability capability(
      const BenchmarkCase& specification) const override;
  [[nodiscard]] std::unique_ptr<BenchmarkExecutable> build(
      const BenchmarkCase& specification) override;
};

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_BENCHMARK_CUDNN_PROVIDER_HPP_
