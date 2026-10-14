/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once

#include "common/pointwise.hpp"
#include "runtime/json.hpp"

#include <filesystem>
#include <string>
#include <string_view>

namespace flagdnn::testing::ascend {

struct EquivalentBaseline {
  std::string reference_case;
  std::string scale_metric = "hbm_copy";
  std::string timing_profile = "native";
  std::string reference_validation = "bit_exact";
  double reference_time_us = 0;
  double reference_peak = 0;
  double target_peak = 0;
  double min_relative_efficiency = 0;
  double traffic_ratio = 1;

  [[nodiscard]] double equivalent_time_us() const;
  [[nodiscard]] double limit_us() const;
  [[nodiscard]] std::string json(double measured_us) const;
};

// Scope is deliberately restricted to the native FP8 identity catalog.
// Missing evidence for an in-scope case is an error, never a capability skip.
[[nodiscard]] bool supports_equivalent_baseline(const PointwiseTestCase &);
[[nodiscard]] std::filesystem::path performance_baseline_path();
[[nodiscard]] std::vector<std::uint8_t>
identity_fp8_logical_bytes(const TestTensor &tensor);

class PerformanceBaseline {
 public:
  explicit PerformanceBaseline(
      const std::filesystem::path &path = performance_baseline_path());
  [[nodiscard]] EquivalentBaseline lookup(
      const PointwiseTestCase &specification, std::string_view device_name,
      std::string_view target_fingerprint) const;

 private:
  std::filesystem::path path_;
  native::json::Value document_;
};

}  // namespace flagdnn::testing::ascend
