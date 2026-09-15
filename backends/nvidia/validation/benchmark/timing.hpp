/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_VALIDATION_BENCHMARK_TIMING_HPP_
#define FLAGDNN_NVIDIA_VALIDATION_BENCHMARK_TIMING_HPP_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "validation/cuda_driver.hpp"

namespace flagdnn::validation::nvidia::timing {

class CapturedExecutionBatch {
 public:
  template <typename Function>
  CapturedExecutionBatch(CUstream stream, int execution_count,
                         Function&& execute)
      : execution_count_(execution_count) {
    if (execution_count_ <= 0) {
      throw std::invalid_argument("captured execution count must be positive");
    }

    check_cuda(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED),
               "cuStreamBeginCapture");
    try {
      for (int i = 0; i < execution_count_; ++i) {
        execute();
      }
    } catch (...) {
      CUgraph abandoned_graph = nullptr;
      if (cuStreamEndCapture(stream, &abandoned_graph) == CUDA_SUCCESS &&
          abandoned_graph != nullptr) {
        cuGraphDestroy(abandoned_graph);
      }
      throw;
    }

    check_cuda(cuStreamEndCapture(stream, &graph_), "cuStreamEndCapture");
    try {
      check_cuda(cuGraphInstantiate(&executable_, graph_, 0),
                 "cuGraphInstantiate");
    } catch (...) {
      cuGraphDestroy(graph_);
      graph_ = nullptr;
      throw;
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch&) = delete;
  CapturedExecutionBatch& operator=(const CapturedExecutionBatch&) = delete;

  ~CapturedExecutionBatch() {
    if (executable_ != nullptr) {
      cuGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      cuGraphDestroy(graph_);
    }
  }

  void launch(CUstream stream) const {
    check_cuda(cuGraphLaunch(executable_, stream), "cuGraphLaunch");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

 private:
  CUgraph graph_ = nullptr;
  CUgraphExec executable_ = nullptr;
  int execution_count_ = 0;
};

inline double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty benchmark samples");
  }
  std::sort(values.begin(), values.end());
  const std::size_t index =
      static_cast<std::size_t>(
          std::ceil(fraction * static_cast<double>(values.size()))) -
      1;
  return values[std::min(index, values.size() - 1)];
}

using HostClock = std::chrono::steady_clock;

inline double host_microseconds_since(HostClock::time_point begin) {
  return std::chrono::duration<double, std::micro>(HostClock::now() - begin)
      .count();
}

struct RuntimeMeasurements {
  double build_us = 0;
  double warm_build_us = 0;
  std::size_t workspace_bytes = 0;
  const char* build_cache = "provider_managed";
  std::vector<double> host_submit_us;
};

inline void emit_samples(const char* provider, std::string_view case_name,
                         const std::vector<double>& samples,
                         const RuntimeMeasurements& runtime) {
  std::cout << "{\"schema_version\":3,\"kind\":\"steady_state\","
            << "\"provider\":\"" << provider << "\",\"case\":\"" << case_name
            << "\",\"unit\":\"us\","
            << "\"median\":" << percentile(samples, 0.5)
            << ",\"p90\":" << percentile(samples, 0.9) << ",\"samples\":[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << samples[index];
  }
  std::cout << "],\"host_submit_us\":{\"median\":"
            << percentile(runtime.host_submit_us, 0.5)
            << ",\"p90\":" << percentile(runtime.host_submit_us, 0.9)
            << ",\"samples\":[";
  for (std::size_t index = 0; index < runtime.host_submit_us.size(); ++index) {
    if (index != 0) {
      std::cout << ',';
    }
    std::cout << runtime.host_submit_us[index];
  }
  std::cout << "]},\"build_us\":" << runtime.build_us
            << ",\"warm_build_us\":" << runtime.warm_build_us
            << ",\"workspace_bytes\":" << runtime.workspace_bytes
            << ",\"build_cache\":\"" << runtime.build_cache << "\"";
  std::cout << "}\n";
}

// Profile only graph construction; the second build exercises the warm cache.
template <typename Factory>
auto profile_build(Factory&& factory, RuntimeMeasurements* measurements) {
  if (measurements == nullptr) return factory();
  auto begin = HostClock::now();
  auto executable = factory();
  measurements->build_us = host_microseconds_since(begin);
  measurements->workspace_bytes = executable->workspace_size();
  begin = HostClock::now();
  auto warm = factory();
  measurements->warm_build_us = host_microseconds_since(begin);
  return executable;
}

}  // namespace flagdnn::validation::nvidia::timing
#endif  // FLAGDNN_NVIDIA_VALIDATION_BENCHMARK_TIMING_HPP_
