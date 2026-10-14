/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_EXACT_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_EXACT_REFERENCE_HPP_

#include "validation/tensor_io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::ascend {

// Only physical buffer placement belongs to the backend. CPU operator math
// produces contiguous logical values in reference/cpu.
template <class Tensor>
inline void store_logical_reference(std::span<const float> logical,
                                    std::span<float> storage,
                                    const Tensor& tensor) {
  if (logical.size() != tensor_io::element_count(tensor) ||
      storage.size() < tensor_io::storage_element_count(tensor))
    throw std::invalid_argument("CPU reference output storage size is invalid");
  for (std::size_t index = 0; index < logical.size(); ++index)
    storage[tensor_io::physical_offset_unchecked(index, tensor)] = logical[index];
}

class ExactReferenceCoverage {
 public:
  explicit ExactReferenceCoverage(std::size_t required)
      : required_(required) {
    if (required == 0U) {
      throw std::invalid_argument(
          "exact-reference coverage requires a nonzero catalog");
    }
  }

  void record_pass(std::string_view case_id) {
    validate_new_case(case_id);
    require_available();
    completed_cases_.emplace(case_id);
    ++passed_;
  }

  void record_skip(std::string_view case_id, std::string_view reason) {
    validate_new_case(case_id);
    if (reason.empty() ||
        std::all_of(reason.begin(), reason.end(), is_space) ||
        reason.find_first_of("\r\n") != std::string_view::npos) {
      throw std::invalid_argument(
          "exact-reference SKIP reason is invalid");
    }
    require_available();
    completed_cases_.emplace(case_id);
    skip_lines_.emplace_back(
        "SKIP case=" + std::string(case_id) + " reason=" +
        std::string(reason));
    ++unsupported_;
  }

  [[nodiscard]] std::size_t required() const noexcept { return required_; }
  [[nodiscard]] std::size_t passed() const noexcept { return passed_; }
  [[nodiscard]] std::size_t unsupported() const noexcept {
    return unsupported_;
  }
  [[nodiscard]] const std::vector<std::string>& skip_lines() const noexcept {
    return skip_lines_;
  }

  void require_complete() const {
    if (passed_ + unsupported_ != required_) {
      throw std::logic_error(
          "exact-reference coverage is incomplete: required=" +
          std::to_string(required_) + " pass=" + std::to_string(passed_) +
          " unsupported=" + std::to_string(unsupported_));
    }
  }

  [[nodiscard]] std::string summary() const {
    require_complete();
    return "ACLNN_REQUIRED_SUMMARY required=" + std::to_string(required_) +
           " pass=" + std::to_string(passed_) + " unsupported=" +
           std::to_string(unsupported_) + " fail=0";
  }

 private:
  static bool is_space(char value) {
    return std::isspace(static_cast<unsigned char>(value)) != 0;
  }

  void validate_new_case(std::string_view case_id) const {
    if (case_id.empty() ||
        std::any_of(case_id.begin(), case_id.end(), is_space)) {
      throw std::invalid_argument(
          "exact-reference case identifier is invalid");
    }
    if (completed_cases_.contains(std::string(case_id))) {
      throw std::logic_error(
          "exact-reference case completion is duplicate: " +
          std::string(case_id));
    }
  }

  void require_available() const {
    if (passed_ + unsupported_ >= required_) {
      throw std::logic_error(
          "exact-reference coverage exceeds the required catalog");
    }
  }

  std::size_t required_ = 0U;
  std::size_t passed_ = 0U;
  std::size_t unsupported_ = 0U;
  std::set<std::string, std::less<>> completed_cases_;
  std::vector<std::string> skip_lines_;
};

inline void compare_exact_reference(std::span<const float> actual,
                                    std::span<const float> reference,
                                    double absolute_tolerance,
                                    double relative_tolerance,
                                    std::string_view case_name) {
  if (case_name.empty()) {
    throw std::invalid_argument(
        "exact-reference comparison requires a case name");
  }
  if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0 ||
      !std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
    throw std::invalid_argument(
        "exact-reference comparison tolerance is invalid");
  }
  if (actual.size() != reference.size()) {
    throw std::invalid_argument(
        std::string(case_name) + " exact-reference size mismatch");
  }
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double actual_value = static_cast<double>(actual[index]);
    const double reference_value = static_cast<double>(reference[index]);
    if (!std::isfinite(actual_value) || !std::isfinite(reference_value)) {
      throw std::runtime_error(
          std::string(case_name) +
          " exact-reference comparison requires finite values at index " +
          std::to_string(index));
    }
    const double absolute_error =
        std::abs(actual_value - reference_value);
    const double scale =
        std::max({std::abs(actual_value), std::abs(reference_value), 1.0e-30});
    const double relative_error = absolute_error / scale;
    if (absolute_error > absolute_tolerance &&
        relative_error > relative_tolerance) {
      throw std::runtime_error(
          std::string(case_name) +
          " exact-reference mismatch at index " + std::to_string(index) +
          ": actual=" + std::to_string(actual_value) +
          " reference=" + std::to_string(reference_value));
    }
  }
}

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_EXACT_REFERENCE_HPP_
