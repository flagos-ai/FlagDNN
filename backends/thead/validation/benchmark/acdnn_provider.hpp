// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_BENCHMARK_ACDNN_PROVIDER_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_BENCHMARK_ACDNN_PROVIDER_HPP_

#include "common/benchmark_provider.hpp"

#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>

namespace flagdnn::validation::thead::benchmark {

enum class ComparableStatus {
  kComparable,
  kUnsupported,
  kProbeRequired,
};

struct ComparableRecord {
  ComparableStatus status = ComparableStatus::kUnsupported;
  std::string reason_code;
  std::string detail;
};

class AcdnnProvider final : public flagdnn::benchmarking::BenchmarkProvider {
 public:
  AcdnnProvider(const std::string &catalog_path, std::string operation,
                bool qualify_probes);

  [[nodiscard]] std::string_view name() const noexcept override {
    return "acdnn";
  }
  [[nodiscard]] flagdnn::benchmarking::ProviderCapability capability(
      const flagdnn::benchmarking::BenchmarkCase &specification) const override;
  [[nodiscard]] std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
  build(const flagdnn::benchmarking::BenchmarkCase &specification) override;

  void require_exact_cases(
      std::span<const flagdnn::benchmarking::BenchmarkCase> cases) const;
  [[nodiscard]] const ComparableRecord &
  lookup(std::string_view case_name) const;

 private:
  std::map<std::string, ComparableRecord, std::less<>> records_;
  std::string operation_;
  bool qualify_probes_ = false;
};

[[nodiscard]] std::unique_ptr<flagdnn::benchmarking::BenchmarkExecutable>
build_acdnn_pointwise_benchmark(
    const flagdnn::benchmarking::BenchmarkCase &specification,
    ComparableStatus qualification);

}  // namespace flagdnn::validation::thead::benchmark

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_BENCHMARK_ACDNN_PROVIDER_HPP_
