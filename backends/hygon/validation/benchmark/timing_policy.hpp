/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_TIMING_POLICY_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_TIMING_POLICY_HPP_

#include <array>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flagdnn::benchmarking {

enum class NativeTimingMode { kGraphBatch, kDirectBatch };

// The installed DTK/hipDNN activation capture path truncates the launch grid
// for non-block-aligned element counts. Direct execution writes the full
// tensor. Select a paired policy before measuring, never as a response to an
// arbitrary numerical failure. Apply it to all shapes/dtypes of these ops so
// a suite does not silently mix timing methods. Other backends are unaffected.
constexpr NativeTimingMode native_timing_mode(std::string_view operation) {
  constexpr std::array<std::string_view, 14> affected = {
      "identity", "abs",
      "relu",     "relu_backward",
      "elu",      "elu_backward",
      "sigmoid",  "sigmoid_backward",
      "swish",    "swish_backward",
      "tanh",     "tanh_backward",
      "softplus", "softplus_backward"};
  for (const auto name : affected)
    if (operation == name)
      return NativeTimingMode::kDirectBatch;
  return NativeTimingMode::kGraphBatch;
}

constexpr std::string_view timing_mode_name(NativeTimingMode mode) {
  return mode == NativeTimingMode::kDirectBatch ? "hip_event_direct_batch"
                                                : "hip_graph_batch";
}

constexpr std::string_view timing_mode_reason(NativeTimingMode mode) {
  return mode == NativeTimingMode::kDirectBatch ? "hipdnn_graph_tail_workaround"
                                                : "standard";
}

// EventTimer already divides by the number of callbacks. A graph callback
// executes N operations, whereas a direct callback executes just one. Keeping
// this distinction here makes the normalization testable without a GPU.
template <class Timer, class Stream, class Direct, class Graph>
double measure_native_batch(Timer &timer, Stream stream, NativeTimingMode mode,
                            int count, Direct &&direct, Graph &&graph) {
  if (count <= 0)
    throw std::invalid_argument("batch count must be positive");
  if (mode == NativeTimingMode::kDirectBatch)
    return timer.measure_microseconds(stream, count,
                                      std::forward<Direct>(direct));
  return timer.measure_microseconds(stream, 1, std::forward<Graph>(graph)) /
         static_cast<double>(count);
}

} // namespace flagdnn::benchmarking

#endif
