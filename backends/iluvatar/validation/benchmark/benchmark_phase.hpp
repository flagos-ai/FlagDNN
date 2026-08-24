// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_

#include <string>
#include <string_view>

namespace flagdnn::iluvatar::validation::benchmark {

enum class BenchmarkProviderKind { kFlagdnn, kCorexCudnn };

enum class BenchmarkPhase {
  kProductionSafety,
  kReferenceCorrectness,
  kWarmup,
  kCaptureBuild,
  kCaptureReplay,
  kTiming,
  kPostcheck,
};

enum class CaptureCapability {
  kSupported,
  kCudnnNotSupported,
  kStreamCaptureUnsupported,
};

[[nodiscard]] inline std::string_view
provider_name(BenchmarkProviderKind provider) noexcept {
  switch (provider) {
  case BenchmarkProviderKind::kFlagdnn:
    return "flagdnn";
  case BenchmarkProviderKind::kCorexCudnn:
    return "corex_cudnn";
  }
  return "unknown";
}

[[nodiscard]] inline std::string_view
phase_name(BenchmarkPhase phase) noexcept {
  switch (phase) {
  case BenchmarkPhase::kProductionSafety:
    return "production_safety";
  case BenchmarkPhase::kReferenceCorrectness:
    return "reference_correctness";
  case BenchmarkPhase::kWarmup:
    return "warmup";
  case BenchmarkPhase::kCaptureBuild:
    return "capture_build";
  case BenchmarkPhase::kCaptureReplay:
    return "capture_replay";
  case BenchmarkPhase::kTiming:
    return "timing";
  case BenchmarkPhase::kPostcheck:
    return "postcheck";
  }
  return "unknown";
}

[[nodiscard]] inline std::string
benchmark_phase_context(BenchmarkProviderKind provider, BenchmarkPhase phase) {
  return "provider=" + std::string(provider_name(provider)) +
         " phase=" + std::string(phase_name(phase));
}

} // namespace flagdnn::iluvatar::validation::benchmark

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_BENCHMARK_BENCHMARK_PHASE_HPP_
