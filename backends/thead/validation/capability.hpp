// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_CAPABILITY_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_CAPABILITY_HPP_

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::validation::thead {

enum class CapabilityStatus {
  kSupported,
  kUnsupported,
  kProbeRequired,
};

enum class ReferencePath {
  kNone,
  kStablePrimitive,
  kBackendDescriptor,
};

struct CapabilityBaseline {
  std::string ppu_sdk_min;
  std::string ppu_sdk_max;
  std::int64_t acdnn_header_min = 0;
  std::int64_t acdnn_header_max = 0;
  std::int64_t acdnn_runtime_min = 0;
  std::int64_t acdnn_runtime_max = 0;
};

struct RankConstraints {
  std::int64_t minimum = 0;
  std::int64_t maximum = 0;
};

struct CapabilityConstraints {
  std::vector<std::string> dtypes;
  std::string compute_type;
  RankConstraints rank;
  std::vector<std::string> shape;
  std::vector<std::string> layouts;
  std::string stride_policy;
  std::string broadcast;
  std::map<std::string, std::vector<std::string>, std::less<>> attributes;
};

struct CapabilityRecord {
  CapabilityStatus status = CapabilityStatus::kUnsupported;
  ReferencePath path = ReferencePath::kNone;
  std::vector<std::string> reference_plan;
  std::optional<CapabilityConstraints> constraints;
  std::string reason_code;
  std::string detail;
};

class CapabilityCatalog final {
 public:
  using Records =
      std::map<std::string,
               std::map<std::string, CapabilityRecord, std::less<>>,
               std::less<>>;

  [[nodiscard]] static CapabilityCatalog parse(std::string_view document);
  [[nodiscard]] static CapabilityCatalog load(const std::string &path);

  void require_exact_cases(
      const std::map<std::string, std::set<std::string>> &expected) const;
  void validate_versions(std::string_view ppu_sdk_version,
                         std::int64_t acdnn_header_version,
                         std::int64_t acdnn_runtime_version) const;

  [[nodiscard]] const CapabilityRecord &
  lookup(std::string_view operation, std::string_view case_name) const;
  [[nodiscard]] const CapabilityBaseline &baseline() const noexcept {
    return baseline_;
  }
  [[nodiscard]] const Records &records() const noexcept { return records_; }

 private:
  CapabilityBaseline baseline_;
  Records records_;
};

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_CAPABILITY_HPP_
