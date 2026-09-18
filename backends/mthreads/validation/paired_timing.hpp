/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_MTHREADS_PAIRED_TIMING_HPP_
#define FLAGDNN_MTHREADS_PAIRED_TIMING_HPP_
#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>

#include "backends/mthreads/validation/musa_driver.hpp"

namespace flagdnn::validation::mthreads::timing {
namespace mv = flagdnn::validation::mthreads;
class CapturedExecutionBatch final {
 public:
  CapturedExecutionBatch(musaStream_t stream, int execution_count,
                         const std::function<void()>& enqueue)
      : execution_count_(execution_count) {
    if (stream == nullptr || execution_count_ <= 0) {
      throw std::invalid_argument("MUSA Graph capture arguments are invalid");
    }
    mv::check_musa(
        musaStreamBeginCapture(stream, musaStreamCaptureModeThreadLocal),
        "musaStreamBeginCapture(benchmark)");
    try {
      for (int index = 0; index < execution_count_; ++index) {
        enqueue();
      }
    } catch (...) {
      musaGraph_t abandoned = nullptr;
      static_cast<void>(musaStreamEndCapture(stream, &abandoned));
      if (abandoned != nullptr) {
        static_cast<void>(musaGraphDestroy(abandoned));
      }
      throw;
    }

    musaGraph_t graph = nullptr;
    mv::check_musa(musaStreamEndCapture(stream, &graph),
                   "musaStreamEndCapture(benchmark)");
    if (graph == nullptr) {
      throw std::runtime_error("MUSA Graph capture returned a null graph");
    }
    try {
      mv::check_musa(musaGraphGetNodes(graph, nullptr, &node_count_),
                     "musaGraphGetNodes(benchmark)");
      if (node_count_ == 0) {
        throw std::runtime_error("MUSA Graph capture produced no nodes");
      }
      mv::check_musa(musaGraphInstantiate(&graph_exec_, graph, 0),
                     "musaGraphInstantiate(benchmark)");
    } catch (...) {
      static_cast<void>(musaGraphDestroy(graph));
      throw;
    }
    mv::check_musa(musaGraphDestroy(graph), "musaGraphDestroy(benchmark)");
  }

  ~CapturedExecutionBatch() {
    if (graph_exec_ != nullptr) {
      static_cast<void>(musaGraphExecDestroy(graph_exec_));
    }
  }

  CapturedExecutionBatch(const CapturedExecutionBatch&) = delete;
  CapturedExecutionBatch& operator=(const CapturedExecutionBatch&) = delete;

  void launch(musaStream_t stream) const {
    mv::check_musa(musaGraphLaunch(graph_exec_, stream),
                   "musaGraphLaunch(benchmark)");
  }

  [[nodiscard]] int execution_count() const noexcept {
    return execution_count_;
  }

  [[nodiscard]] std::size_t node_count() const noexcept { return node_count_; }

 private:
  musaGraphExec_t graph_exec_ = nullptr;
  int execution_count_ = 0;
  std::size_t node_count_ = 0;
};

// Native GroupedMatMul graph replay can hang or produce incorrect outputs.
// Use identical direct submission for both providers in that case.
class EnqueuedExecutionBatch final {
 public:
  EnqueuedExecutionBatch(int execution_count, std::function<void()> enqueue)
      : execution_count_(execution_count), enqueue_(std::move(enqueue)) {}
  void launch(musaStream_t) const {
    for (int i = 0; i < execution_count_; ++i) enqueue_();
  }
  int execution_count() const noexcept { return execution_count_; }

 private:
  int execution_count_;
  std::function<void()> enqueue_;
};

class EventTimer final {
 public:
  EventTimer() {
    mv::check_musa(musaEventCreate(&start_), "musaEventCreate(start)");
    try {
      mv::check_musa(musaEventCreate(&finish_), "musaEventCreate(finish)");
    } catch (...) {
      static_cast<void>(musaEventDestroy(start_));
      start_ = nullptr;
      throw;
    }
  }

  ~EventTimer() {
    if (finish_ != nullptr) {
      static_cast<void>(musaEventDestroy(finish_));
    }
    if (start_ != nullptr) {
      static_cast<void>(musaEventDestroy(start_));
    }
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  template <typename Batch>
  double measure_microseconds(musaStream_t stream, const Batch& batch) {
    mv::check_musa(musaEventRecord(start_, stream), "musaEventRecord(start)");
    batch.launch(stream);
    mv::check_musa(musaEventRecord(finish_, stream),
                   "musaEventRecord(finish)");
    mv::check_musa(musaEventSynchronize(finish_),
                   "musaEventSynchronize(finish)");
    float milliseconds = 0.0F;
    mv::check_musa(musaEventElapsedTime(&milliseconds, start_, finish_),
                   "musaEventElapsedTime");
    const double per_execution = static_cast<double>(milliseconds) * 1000.0 /
                                 static_cast<double>(batch.execution_count());
    if (!std::isfinite(per_execution) || per_execution <= 0.0) {
      throw std::runtime_error(
          "MUSA Event returned a non-positive benchmark duration");
    }
    return per_execution;
  }

 private:
  musaEvent_t start_ = nullptr;
  musaEvent_t finish_ = nullptr;
};

inline void emit(std::string_view provider, std::string_view name,
                 const std::vector<double>& samples) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::cout
      << std::setprecision(12)
      << "{\"schema_version\":1,\"kind\":\"steady_state\",\"provider\":\""
      << provider << "\",\"case\":\"" << name
      << "\",\"unit\":\"us\",\"median\":" << sorted[sorted.size() / 2]
      << ",\"p90\":" << sorted.back() << ",\"samples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples[i];
  }
  std::cout << "]}\n";
}

// Call after both providers' first execution; the caller verifies outputs and
// inputs again after timing. Direct Event timing includes host submission gaps;
// MUSA Graph timing measures captured device work.
template <typename Production, typename Reference>
void paired(std::string_view name, Stream& stream, Production&& production,
            Reference&& reference, bool capture_graph = true) {
  if (!benchmark_enabled()) return;
  stream.synchronize();
  for (int i = 0; i < 5; ++i) {
    production();
    reference();
  }
  stream.synchronize();
  EventTimer timer;
  std::vector<double> flagdnn_samples, mudnn_samples;
  const auto measure = [&](const auto& flagdnn_batch,
                           const auto& mudnn_batch) {
    for (int sample = 0; sample < 7; ++sample) {
      if (sample % 2 == 0) {
        flagdnn_samples.push_back(
            timer.measure_microseconds(stream.get(), flagdnn_batch));
        mudnn_samples.push_back(
            timer.measure_microseconds(stream.get(), mudnn_batch));
      } else {
        mudnn_samples.push_back(
            timer.measure_microseconds(stream.get(), mudnn_batch));
        flagdnn_samples.push_back(
            timer.measure_microseconds(stream.get(), flagdnn_batch));
      }
    }
  };
  std::cout << "[timing] case=" << name << " method="
            << (capture_graph ? "musa_graph" : "musa_event_batch")
            << " execution_count=20\n";
  if (capture_graph) {
    CapturedExecutionBatch flagdnn_batch(stream.get(), 20, production);
    CapturedExecutionBatch mudnn_batch(stream.get(), 20, reference);
    measure(flagdnn_batch, mudnn_batch);
  } else {
    EnqueuedExecutionBatch flagdnn_batch(20, production);
    EnqueuedExecutionBatch mudnn_batch(20, reference);
    measure(flagdnn_batch, mudnn_batch);
  }
  emit("flagdnn", name, flagdnn_samples);
  emit("mudnn", name, mudnn_samples);
}
}  // namespace flagdnn::validation::mthreads::timing
#endif
