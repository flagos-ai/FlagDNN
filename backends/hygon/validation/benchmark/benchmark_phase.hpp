/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_

#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::benchmarking {

enum class BenchmarkProviderKind {
  kFlagdnn,
  kHipdnn,
};

enum class BenchmarkPhase {
  kProbe,
  kWarmup,
  kCaptureBuild,
  kCaptureReplay,
  kBatchBuild,
  kBatchReplay,
  kFreshValidation,
  kTiming,
};

[[nodiscard]] constexpr std::string_view
benchmark_provider_name(BenchmarkProviderKind provider) {
  switch (provider) {
  case BenchmarkProviderKind::kFlagdnn:
    return "flagdnn";
  case BenchmarkProviderKind::kHipdnn:
    return "hipdnn";
  }
  throw std::invalid_argument("unknown benchmark provider");
}

[[nodiscard]] constexpr std::string_view
benchmark_phase_name(BenchmarkPhase phase) {
  switch (phase) {
  case BenchmarkPhase::kProbe:
    return "probe";
  case BenchmarkPhase::kWarmup:
    return "warmup";
  case BenchmarkPhase::kCaptureBuild:
    return "capture-build";
  case BenchmarkPhase::kCaptureReplay:
    return "capture-replay";
  case BenchmarkPhase::kBatchBuild:
    return "batch-build";
  case BenchmarkPhase::kBatchReplay:
    return "batch-replay";
  case BenchmarkPhase::kFreshValidation:
    return "fresh-validation";
  case BenchmarkPhase::kTiming:
    return "timing";
  }
  throw std::invalid_argument("unknown benchmark phase");
}

[[nodiscard]] inline std::string
benchmark_phase_context(BenchmarkProviderKind provider, BenchmarkPhase phase) {
  return "provider=" + std::string(benchmark_provider_name(provider)) +
         " phase=" + std::string(benchmark_phase_name(phase));
}

} // namespace flagdnn::benchmarking

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_
