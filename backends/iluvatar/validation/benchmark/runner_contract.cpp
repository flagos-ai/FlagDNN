// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "benchmark/runner_contract.hpp"

#include "benchmark/capability_contract.hpp"
#include "benchmark/corex_cudnn_provider.hpp"
#include "benchmark/shared_cases.hpp"
#include "common/layout.hpp"
#include "common/pointwise.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
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
    // The supplemental copy group must retain non-floating storage and
    // subviews from the shared catalog, including FP8 E8M0 and INT32.
    const auto copy = ivb::shared_benchmark_cases("FLAGDNN_IDENTITY_BENCHMARK");
    const auto expected = flagdnn::testing::make_unary_pointwise_cases(
        {.operation_name = "identity", .mode = FLAGDNN_POINTWISE_IDENTITY});
    if (copy.size() != expected.size())
      throw std::runtime_error(
          "supplemental identity silently filtered shared cases");
    ivb::CorexCudnnProvider naming;
    for (std::size_t i = 0; i < copy.size(); ++i) {
      if (copy[i].tensors[0].data_type != expected[i].inputs[0].data_type ||
          copy[i].tensors[0].dimensions != expected[i].inputs[0].dimensions ||
          copy[i].tensors[0].strides != expected[i].inputs[0].strides ||
          naming.operation_name(copy[i]) != "identity")
        throw std::runtime_error(
            "supplemental copy changed shared tensor metadata");
    }
    for (const auto operation : {"logical_and", "logical_or", "logical_not"}) {
      std::string marker = "FLAGDNN_" + std::string(operation) + "_BENCHMARK";
      std::transform(marker.begin(), marker.end(), marker.begin(),
                     [](unsigned char c) { return std::toupper(c); });
      const auto cases = ivb::shared_benchmark_cases(marker);
      if (cases.empty())
        throw std::runtime_error("missing shared boolean benchmark cases");
      for (const auto &c : cases) {
        if (naming.operation_name(c) != operation ||
            c.tensors.front().data_type != FLAGDNN_DATA_BOOLEAN)
          throw std::runtime_error("invalid shared boolean benchmark metadata");
        if (!naming.capability(c).supported)
          throw std::runtime_error(
              "qualified shared boolean benchmark was skipped: " + c.name);
      }
    }
    for (const auto marker : {"FLAGDNN_IDENTITY_BENCHMARK",
                              "FLAGDNN_RESHAPE_BENCHMARK",
                              "FLAGDNN_TRANSPOSE_BENCHMARK",
                              "FLAGDNN_SLICE_BENCHMARK"}) {
      std::size_t boolean_cases = 0;
      for (const auto &c : ivb::shared_benchmark_cases(marker)) {
        if (c.tensors.front().data_type != FLAGDNN_DATA_BOOLEAN)
          continue;
        ++boolean_cases;
        if (!naming.capability(c).supported)
          throw std::runtime_error("qualified BOOL copy was skipped: " + c.name);
      }
      if (boolean_cases == 0)
        throw std::runtime_error("missing shared BOOL copy benchmark cases");
    }
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
