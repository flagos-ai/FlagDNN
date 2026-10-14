/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/equivalent_pointwise.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "backends/mthreads/validation/native_pointwise_case.hpp"
#include "backends/mthreads/validation/benchmark/equivalent_baseline.hpp"

namespace flagdnn::validation::mthreads::equivalent {
namespace {
namespace bench = benchmarking;
namespace baseline_api = benchmarking::mthreads;

double percentile(std::vector<double> values, double fraction) {
  if (values.empty()) throw std::invalid_argument("empty equivalent samples");
  std::sort(values.begin(), values.end());
  return values.at(static_cast<std::size_t>(
                       std::ceil(fraction * values.size())) - 1);
}

const baseline_api::PerformanceBaseline& performance_baseline() {
  static const baseline_api::PerformanceBaseline baseline([] {
    const char* path = std::getenv("FLAGDNN_PERFORMANCE_BASELINE");
    return std::filesystem::path(path && path[0]
                                   ? path
                                   : FLAGDNN_PERFORMANCE_BASELINE_FILE);
  }(), "mthreads");
  return baseline;
}

void emit(const bench::BenchmarkCase& specification,
          const baseline_api::EquivalentBaseline& baseline,
          const std::vector<double>& samples) {
  const double measured = percentile(samples, 0.5);
  const auto previous = std::cout.precision();
  std::cout << std::setprecision(17)
            << "{\"schema_version\":4,\"kind\":\"steady_state\","
               "\"provider\":\"flagdnn\",\"case\":\""
            << specification.name << "\",\"unit\":\"us\",\"median\":"
            << measured << ",\"p90\":" << percentile(samples, 0.9)
            << ",\"samples\":[";
  for (std::size_t i = 0; i < samples.size(); ++i) {
    if (i) std::cout << ',';
    std::cout << samples[i];
  }
  std::cout << "],\"equivalent_baseline\":{"
               "\"provider\":\"nvidia_equivalent\","
               "\"platform\":\"nvidia\",\"target_platform\":\"mthreads\","
               "\"scale_metric\":\""
            << baseline.scale_metric << "\",\"timing_profile\":\""
            << baseline.timing_profile << "\",\"reference_time_us\":"
            << baseline.reference_time_us << ",\"reference_peak\":"
            << baseline.reference_peak << ",\"target_peak\":"
            << baseline.target_peak << ",\"traffic_ratio\":"
            << baseline.traffic_ratio << ",\"reference_validation\":\""
            << baseline.reference_validation
            << "\",\"min_relative_efficiency\":"
            << baseline.min_relative_efficiency << ",\"equivalent_time_us\":"
            << baseline.equivalent_time_us() << ",\"limit_us\":"
            << baseline.limit_us() << ",\"relative_efficiency\":"
            << baseline.equivalent_time_us() / measured << ",\"passed\":"
            << (measured <= baseline.limit_us() ? "true" : "false")
            << "}}\n";
  std::cout << specification.name << ": nvidia_equivalent "
            << (measured <= baseline.limit_us() ? "MEETS_BASELINE"
                                                : "BELOW_BASELINE")
            << " measured_us=" << measured
            << " limit_us=" << baseline.limit_us()
            << " relative_efficiency="
            << baseline.equivalent_time_us() / measured
            << " required=" << baseline.min_relative_efficiency << '\n'
            << std::setprecision(previous);
}

}  // namespace

void run(const bench::BenchmarkCase& specification, Stream& stream,
         const std::function<void()>& enqueue,
         const std::function<void()>& validate,
         const std::function<void()>& reset_outputs) {
  if (!enqueue || !validate || !reset_outputs)
    throw std::invalid_argument("equivalent benchmark requires all callbacks");
  int device = 0;
  check_musa(musaGetDevice(&device), "musaGetDevice(equivalent baseline)");
  musaDeviceProp properties{};
  check_musa(musaGetDeviceProperties(&properties, device),
             "musaGetDeviceProperties(equivalent baseline)");
  const std::string architecture =
      "mp_" + std::to_string(properties.major) +
      std::to_string(properties.minor);
  const auto baseline = performance_baseline().lookup(
      specification, properties.name, architecture);

  stream.synchronize();
  validate();
  const auto& config = specification.benchmark;
  for (int i = 0; i < config.warmup_iterations; ++i) enqueue();
  stream.synchronize();
  timing::CapturedExecutionBatch batch(
      stream.get(), config.iterations_per_sample, enqueue);
  std::cout << "[capture] case=" << specification.name
            << " provider=flagdnn nodes=" << batch.node_count()
            << " executions=" << batch.execution_count()
            << " timing_profile=" << baseline.timing_profile << '\n';
  batch.launch(stream.get());  // Exactly one graph warmup replay.
  stream.synchronize();

  timing::EventTimer timer;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(config.sample_count));
  for (int i = 0; i < config.sample_count; ++i)
    samples.push_back(timer.measure_microseconds(stream.get(), batch));
  validate();  // Check last timed output before overwriting it.
  reset_outputs();
  stream.synchronize();
  batch.launch(stream.get());
  stream.synchronize();
  validate();  // Reject empty/partial graph replay and corrupt inputs/padding.
  std::cout << "[postcheck] case=" << specification.name
            << " cpu_full=true fresh_graph_replay=true\n";
  emit(specification, baseline, samples);
}

void run(const testing::PointwiseTestCase& specification, Stream& stream,
         const std::function<void()>& enqueue,
         const std::function<void()>& validate,
         const std::function<void()>& reset_outputs) {
  run(as_benchmark(specification), stream, enqueue, validate, reset_outputs);
}
}  // namespace flagdnn::validation::mthreads::equivalent
