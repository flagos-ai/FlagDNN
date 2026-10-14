// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#include "benchmark/equivalent.hpp"
#include "benchmark_mode.hpp"
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace flagdnn::validation::thead {
void run_equivalent_benchmark(
    const benchmarking::BenchmarkCase& spec,
    benchmarking::BenchmarkExecutable& executable, DeviceStream& stream,
    const benchmarking::thead::NativeBaselineMetadata* metadata) {
  static const benchmarking::thead::PerformanceBaseline document([] {
    const char* path = std::getenv("FLAGDNN_PERFORMANCE_BASELINE");
    return path && path[0] ? path : FLAGDNN_PERFORMANCE_BASELINE_FILE;
  }());
  CUdevice device;
  check_driver(cuCtxGetDevice(&device), "equivalent current device");
  char name[256]{};
  check_driver(cuDeviceGetName(name, sizeof(name), device), "equivalent device name");
  int major = 0, minor = 0;
  check_driver(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device), "equivalent device major");
  check_driver(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device), "equivalent device minor");
  const auto baseline = document.lookup(spec, name,
      "sm_" + std::to_string(major) + std::to_string(minor), metadata);
  const bool timing = !benchmark_validation_only(spec.name);
  const auto samples = run_cpu_validated_case(spec, executable, stream, metadata, timing);
  if (!timing) return;
  auto ordered = samples;
  std::sort(ordered.begin(), ordered.end());
  const auto measured = ordered.at((ordered.size() + 1) / 2 - 1);
  const auto precision = std::cout.precision();
  std::cout << std::setprecision(17)
            << R"({"schema_version":4,"kind":"steady_state","provider":"flagdnn","case":")"
            << spec.name << R"(","unit":"us","median":)" << measured
            << R"(,"p90":)" << ordered.at((ordered.size() * 9 + 9) / 10 - 1)
            << R"(,"samples":[)";
  for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
  std::cout << R"(],"equivalent_baseline":{"provider":"nvidia_equivalent",)"
            << R"("platform":"nvidia","target_platform":"thead",)"
            << R"("scale_metric":")" << baseline.scale_metric << R"(",)"
            << R"("timing_profile":")" << baseline.timing_profile << R"(",)"
            << R"("reference_validation":")" << baseline.reference_validation << R"(",)"
            << R"("reference_time_us":)" << baseline.reference_time_us
            << R"(,"reference_peak":)" << baseline.reference_peak
            << R"(,"target_peak":)" << baseline.target_peak
            << R"(,"traffic_ratio":)" << baseline.traffic_ratio
            << R"(,"min_relative_efficiency":)" << baseline.min_relative_efficiency
            << R"(,"equivalent_time_us":)" << baseline.equivalent_time_us()
            << R"(,"limit_us":)" << baseline.limit_us()
            << R"(,"relative_efficiency":)" << baseline.equivalent_time_us() / measured
            << R"(,"passed":)" << (measured <= baseline.limit_us() ? "true" : "false") << "}}\n"
            << spec.name << ": nvidia_equivalent "
            << (measured <= baseline.limit_us() ? "MEETS_BASELINE" : "BELOW_BASELINE")
            << " measured_us=" << measured << " limit_us=" << baseline.limit_us() << '\n'
            << std::setprecision(precision);
}
}  // namespace flagdnn::validation::thead
