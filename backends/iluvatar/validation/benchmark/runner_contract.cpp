// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/runner_contract.hpp"

#include "benchmark/capability_contract.hpp"

#include <cuda_runtime_api.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace ivb = flagdnn::iluvatar::validation::benchmark;

namespace {

template <typename Callback>
void require_throws(std::string_view name, Callback &&callback) {
  bool threw = false;
  try {
    std::forward<Callback>(callback)();
  } catch (const std::exception &) {
    threw = true;
  }
  if (!threw) {
    throw std::runtime_error(std::string(name) + " did not fail");
  }
}

} // namespace

int main() {
  try {
    ivb::run_capability_contract();
    require_throws("correctness-before-timing", [] {
      (void)ivb::validate_capture_pair(
          false, ivb::CaptureCapability::kSupported,
          ivb::CaptureCapability::kSupported, 4, 4);
    });
    if (ivb::validate_capture_pair(
            true, ivb::CaptureCapability::kSupported,
            ivb::CaptureCapability::kStreamCaptureUnsupported, 4,
            4) != ivb::PairDisposition::kSkip) {
      throw std::runtime_error("one-sided capture failure was timed");
    }
    if (ivb::validate_capture_pair(true, ivb::CaptureCapability::kSupported,
                                   ivb::CaptureCapability::kCudnnNotSupported,
                                   4, 4) != ivb::PairDisposition::kSkip) {
      throw std::runtime_error("cuDNN NOT_SUPPORTED was timed");
    }
    require_throws("unequal-capture-count", [] {
      (void)ivb::validate_capture_pair(true, ivb::CaptureCapability::kSupported,
                                       ivb::CaptureCapability::kSupported, 4,
                                       3);
    });
    if (!ivb::cuda_capture_status_is_capability(
            cudaErrorStreamCaptureUnsupported) ||
        ivb::cuda_capture_status_is_capability(
            cudaErrorStreamCaptureInvalidated) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureInvalidated) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureMerge) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureUnmatched) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureUnjoined) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureIsolation) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureImplicit) ||
        !ivb::cuda_capture_status_is_fatal(cudaErrorStreamCaptureWrongThread)) {
      throw std::runtime_error("CUDA capture error classification is unsafe");
    }

    require_throws("missing-provider", [] {
      ivb::PairSampleCollector pair;
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn, {1.0, 2.0});
      pair.validate(2);
    });
    require_throws("duplicate-provider", [] {
      ivb::PairSampleCollector pair;
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn, {1.0});
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn, {2.0});
    });
    require_throws("unequal-samples", [] {
      ivb::PairSampleCollector pair;
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn, {1.0, 2.0});
      pair.add(ivb::BenchmarkProviderKind::kCorexCudnn, {1.0});
      pair.validate(2);
    });
    require_throws("nonpositive-sample", [] {
      ivb::PairSampleCollector pair;
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn, {1.0, 0.0});
      pair.add(ivb::BenchmarkProviderKind::kCorexCudnn, {1.0, 2.0});
      pair.validate(2);
    });
    require_throws("nonfinite-sample", [] {
      ivb::PairSampleCollector pair;
      pair.add(ivb::BenchmarkProviderKind::kFlagdnn,
               {1.0, std::numeric_limits<double>::infinity()});
      pair.add(ivb::BenchmarkProviderKind::kCorexCudnn, {1.0, 2.0});
      pair.validate(2);
    });

    ivb::PairSampleCollector valid;
    valid.add(ivb::BenchmarkProviderKind::kFlagdnn, {1.0, 3.0, 2.0});
    valid.add(ivb::BenchmarkProviderKind::kCorexCudnn, {2.0, 4.0, 3.0});
    valid.validate(3);
    if (ivb::percentile(valid.flagdnn(), 0.5) != 2.0 ||
        ivb::validate_capture_pair(true, ivb::CaptureCapability::kSupported,
                                   ivb::CaptureCapability::kSupported, 8,
                                   8) != ivb::PairDisposition::kComparable) {
      throw std::runtime_error("valid benchmark pair contract failed");
    }
    std::cout << "PASS: Iluvatar paired benchmark runner contract" << std::endl;
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return 1;
  }
}
