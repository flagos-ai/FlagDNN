/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_EQUIVALENT_BASELINE_HPP_
#define FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_EQUIVALENT_BASELINE_HPP_

#include "common/case.hpp"
#include "runtime/json.hpp"

#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace flagdnn::benchmarking {

struct EquivalentBaseline {
  std::string scale_metric;
  std::string timing_profile;
  double reference_time_us = 0.0;
  double reference_peak = 0.0;
  double target_peak = 0.0;
  double min_relative_efficiency = 0.0;
  double traffic_ratio = 1.0;
  std::string reference_validation;

  [[nodiscard]] double equivalent_time_us() const;
  [[nodiscard]] double limit_us() const;
};

// Invalid or unmatched evidence is an error, never an unsupported-case skip.
struct NativeBaselineMetadata {
  std::string operation;
  std::int64_t axis = 0;
  int input_precision = 0;
  std::vector<std::pair<double,double>> output_tolerances = {};
  // Extended native catalogs retain their own attributes and deterministic
  // input generators. Both are checked before any timing is accepted.
  std::size_t input_count = 0;
  std::size_t output_count = 1;
  std::function<void(const native::json::Value&)> validate_attributes = {};
  std::function<std::vector<std::vector<float>>()> generate_inputs = {};
};

// Scope only: missing, malformed, or mismatched reference evidence must fail
// lookup, not silently turn an in-scope benchmark into an unsupported skip.
[[nodiscard]] bool supports_equivalent_baseline(
    const BenchmarkCase& specification);

class PerformanceBaseline {
 public:
  explicit PerformanceBaseline(const std::filesystem::path& path);

  [[nodiscard]] EquivalentBaseline lookup(
      const BenchmarkCase& specification, std::string_view device_name,
      std::string_view architecture,
      const NativeBaselineMetadata* metadata = nullptr) const;

 private:
  std::filesystem::path path_;
  native::json::Value document_;
};

using DivPerformanceBaseline = PerformanceBaseline;

}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_HYGON_VALIDATION_BENCHMARK_EQUIVALENT_BASELINE_HPP_
