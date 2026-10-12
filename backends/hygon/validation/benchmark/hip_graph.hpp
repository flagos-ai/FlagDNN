/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_

#include "hip_driver.hpp"
#include "timing_policy.hpp"

#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>

namespace flagdnn::benchmarking {

class CapturedExecutionBatch final {
public:
  template <typename Function>
  CapturedExecutionBatch(hipStream_t stream, int execution_count,
                         Function &&execute)
      : execution_count_(execution_count) {
    if (execution_count_ <= 0) {
      throw std::invalid_argument("captured execution count must be positive");
    }

    validation::hygon::check_hip(
        hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
        "hipStreamBeginCapture");
    try {
      for (int index = 0; index < execution_count_; ++index) {
        execute();
      }
    } catch (...) {
      hipGraph_t abandoned_graph = nullptr;
      if (hipStreamEndCapture(stream, &abandoned_graph) == hipSuccess &&
          abandoned_graph != nullptr) {
        (void)hipGraphDestroy(abandoned_graph);
      }
      throw;
    }

    validation::hygon::check_hip(hipStreamEndCapture(stream, &graph_),
                                 "hipStreamEndCapture");
    try {
      validation::hygon::check_hip(
          hipGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0),
          "hipGraphInstantiate");
    } catch (...) {
      (void)hipGraphDestroy(graph_);
      graph_ = nullptr;
      throw;
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch &) = delete;
  CapturedExecutionBatch &operator=(const CapturedExecutionBatch &) = delete;

  ~CapturedExecutionBatch() {
    if (executable_ != nullptr) {
      (void)hipGraphExecDestroy(executable_);
    }
    if (graph_ != nullptr) {
      (void)hipGraphDestroy(graph_);
    }
  }

  void launch(hipStream_t stream) const {
    validation::hygon::check_hip(hipGraphLaunch(executable_, stream),
                                 "hipGraphLaunch");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

private:
  hipGraph_t graph_ = nullptr;
  hipGraphExec_t executable_ = nullptr;
  int execution_count_ = 0;
};

// A native pair always shares its mode. Direct event batches include gaps
// while the CPU submits work, so records distinguish them from HIP graphs.
class NativeExecutionBatch final {
public:
  template <typename Function>
  NativeExecutionBatch(hipStream_t stream, int count, NativeTimingMode mode,
                       Function &&execute)
      : count_(count), mode_(mode), execute_(std::forward<Function>(execute)) {
    if (count_ <= 0)
      throw std::invalid_argument("batch count must be positive");
    if (mode_ == NativeTimingMode::kGraphBatch)
      graph_ =
          std::make_unique<CapturedExecutionBatch>(stream, count_, execute_);
  }

  void launch(hipStream_t stream) const {
    if (graph_) {
      graph_->launch(stream);
    } else {
      for (int index = 0; index < count_; ++index)
        execute_();
    }
  }

  double measure(validation::hygon::EventTimer &timer,
                 hipStream_t stream) const {
    return measure_native_batch(timer, stream, mode_, count_, execute_,
                                [&] { graph_->launch(stream); });
  }

private:
  int count_;
  NativeTimingMode mode_;
  std::function<void()> execute_;
  std::unique_ptr<CapturedExecutionBatch> graph_;
};

} // namespace flagdnn::benchmarking

#endif // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_HIP_GRAPH_HPP_
