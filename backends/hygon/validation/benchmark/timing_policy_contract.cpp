/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "timing_policy.hpp"

#include <iostream>
#include <stdexcept>

namespace {
using namespace flagdnn::benchmarking;
void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
struct Timer {
  int iterations = 0;
  template <class Callback>
  double measure_microseconds(int stream, int count, Callback callback) {
    require(stream == 7, "stream changed");
    iterations = count;
    for (int i = 0; i < count; ++i)
      callback();
    return 250.0 / count;
  }
};
} // namespace

int main() {
  try {
    for (const auto operation :
         {"identity", "abs", "relu", "relu_backward", "elu", "elu_backward",
          "sigmoid", "sigmoid_backward", "swish", "swish_backward", "tanh",
          "tanh_backward", "softplus", "softplus_backward"})
      require(native_timing_mode(operation) == NativeTimingMode::kDirectBatch,
              "known capture-tail operation must use paired direct events");
    for (const auto operation :
         {"add", "sub", "mul", "max", "min", "scale", "add_square", "neg",
          "div", "gelu", "conv_fprop", "matmul", "layernorm",
          "identity_unknown", ""})
      require(native_timing_mode(operation) == NativeTimingMode::kGraphBatch,
              "unaffected operation changed timing policy");
    for (const auto mode :
         {NativeTimingMode::kGraphBatch, NativeTimingMode::kDirectBatch}) {
      Timer timer;
      int direct = 0, graph = 0;
      const double elapsed = measure_native_batch(
          timer, 7, mode, 50, [&] { ++direct; }, [&] { ++graph; });
      require(elapsed == 5, "per-operation time normalized incorrectly");
      require(mode == NativeTimingMode::kGraphBatch
                  ? (direct == 0 && graph == 1 && timer.iterations == 1)
                  : (direct == 50 && graph == 0 && timer.iterations == 50),
              "wrong submission path or execution count");
      bool failed = false;
      try {
        measure_native_batch(
            timer, 7, mode, 0, [] {}, [] {});
      } catch (const std::invalid_argument &) {
        failed = true;
      }
      require(failed, "invalid batch size accepted");
      failed = false;
      try {
        measure_native_batch(
            timer, 7, mode, 50, [] { throw std::runtime_error("direct"); },
            [] { throw std::runtime_error("graph"); });
      } catch (const std::runtime_error &) {
        failed = true;
      }
      require(failed, "execution error silently fell back");
    }
    require(timing_mode_name(NativeTimingMode::kDirectBatch) ==
                    "hip_event_direct_batch" &&
                timing_mode_reason(NativeTimingMode::kDirectBatch) ==
                    "hipdnn_graph_tail_workaround",
            "direct provenance missing");
    require(timing_mode_name(NativeTimingMode::kGraphBatch) ==
                    "hip_graph_batch" &&
                timing_mode_reason(NativeTimingMode::kGraphBatch) == "standard",
            "graph provenance missing");
    std::cout << "PASS: paired timing policy, normalization, execution counts, "
                 "provenance and error propagation\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
