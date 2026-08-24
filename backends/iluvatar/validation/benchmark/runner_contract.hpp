// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_RUNNER_CONTRACT_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_RUNNER_CONTRACT_HPP_

#include "benchmark/benchmark_phase.hpp"
#include "benchmark/cuda_graph.hpp"

#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace flagdnn::iluvatar::validation::benchmark {

enum class PairDisposition { kComparable, kSkip };

[[nodiscard]] inline PairDisposition
validate_capture_pair(bool correctness_validated, CaptureCapability flagdnn,
                      CaptureCapability corex_cudnn,
                      int flagdnn_execution_count,
                      int corex_cudnn_execution_count) {
  if (!correctness_validated) {
    throw std::runtime_error(
        "benchmark capture was attempted before correctness validation");
  }
  if (flagdnn == CaptureCapability::kCudnnNotSupported) {
    throw std::runtime_error(
        "FlagDNN capture cannot report a cuDNN capability status");
  }
  if (corex_cudnn == CaptureCapability::kCudnnNotSupported ||
      flagdnn == CaptureCapability::kStreamCaptureUnsupported ||
      corex_cudnn == CaptureCapability::kStreamCaptureUnsupported) {
    return PairDisposition::kSkip;
  }
  if (flagdnn_execution_count <= 0 ||
      flagdnn_execution_count != corex_cudnn_execution_count) {
    throw std::runtime_error("paired CUDA Graph execution counts are unequal");
  }
  return PairDisposition::kComparable;
}

class PairSampleCollector final {
public:
  void add(BenchmarkProviderKind provider, std::vector<double> samples) {
    std::optional<std::vector<double>> &destination =
        provider == BenchmarkProviderKind::kFlagdnn ? flagdnn_ : corex_cudnn_;
    if (destination.has_value()) {
      throw std::runtime_error("duplicate benchmark provider samples");
    }
    destination = std::move(samples);
  }

  void validate(std::size_t expected_count) const {
    if (!flagdnn_.has_value() || !corex_cudnn_.has_value()) {
      throw std::runtime_error("benchmark provider pair is incomplete");
    }
    require_positive_finite_samples(*flagdnn_, expected_count, "flagdnn");
    require_positive_finite_samples(*corex_cudnn_, expected_count,
                                    "corex_cudnn");
    if (flagdnn_->size() != corex_cudnn_->size()) {
      throw std::runtime_error("benchmark provider sample counts differ");
    }
  }

  [[nodiscard]] std::span<const double> flagdnn() const {
    if (!flagdnn_.has_value()) {
      throw std::logic_error("FlagDNN benchmark samples are unavailable");
    }
    return *flagdnn_;
  }

  [[nodiscard]] std::span<const double> corex_cudnn() const {
    if (!corex_cudnn_.has_value()) {
      throw std::logic_error("CoreX cuDNN benchmark samples are unavailable");
    }
    return *corex_cudnn_;
  }

private:
  std::optional<std::vector<double>> flagdnn_;
  std::optional<std::vector<double>> corex_cudnn_;
};

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_RUNNER_CONTRACT_HPP_
