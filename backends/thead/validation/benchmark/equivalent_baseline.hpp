/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_THEAD_EQUIVALENT_BASELINE_HPP_
#define FLAGDNN_THEAD_EQUIVALENT_BASELINE_HPP_

#include "common/case.hpp"
#include "runtime/json.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace flagdnn::benchmarking::thead {

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

struct NativeBaselineMetadata {
  std::string operation;
  std::int64_t axis = 0;
};

// Scope only. Missing or mismatched evidence is an error during lookup.
[[nodiscard]] bool supports_equivalent_baseline(const BenchmarkCase& specification);

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

}  // namespace flagdnn::benchmarking::thead
#endif
