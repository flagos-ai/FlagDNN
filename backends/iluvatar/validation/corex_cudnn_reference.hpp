// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_REFERENCE_HPP_

#include "corex_cudnn_raii.hpp"

#include <map>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace flagdnn::iluvatar::validation {

enum class CorexCudnnCapabilityClass {
  kSupported,
  kSemanticUnsupported,
  kVendorUnsupported,
  kInvalidAdapterContract,
};

struct CorexCudnnCapability {
  CorexCudnnCapabilityClass classification =
      CorexCudnnCapabilityClass::kInvalidAdapterContract;
  std::string reason_code;
  std::string detail;
  std::vector<std::string> primitive_sequence;
};

enum class CorexCudnnCatalogClassification {
  kQualifiedSupported,
  kCandidate,
  kSemanticUnsupported,
  kVendorUnsupported,
};

struct CorexCudnnCatalogRecord {
  CorexCudnnCatalogClassification classification =
      CorexCudnnCatalogClassification::kCandidate;
  std::string reason_code;
  std::string detail;
  std::vector<std::string> primitive_sequence;
};

class CorexCudnnCapabilityCatalog final {
public:
  [[nodiscard]] static CorexCudnnCapabilityCatalog
  parse(std::string_view document);
  [[nodiscard]] static CorexCudnnCapabilityCatalog
  load(const std::string &path);

  void require_exact_cases(
      const std::map<std::string, std::set<std::string>> &expected) const;
  [[nodiscard]] const CorexCudnnCatalogRecord &
  lookup(std::string_view operation, std::string_view case_name) const;
  [[nodiscard]] const std::map<std::string,
                               std::map<std::string, CorexCudnnCatalogRecord>> &
  records() const noexcept {
    return records_;
  }

private:
  std::map<std::string, std::map<std::string, CorexCudnnCatalogRecord>>
      records_;
};

[[nodiscard]] CorexCudnnCapability
corex_cudnn_capability(std::string_view operation, std::string_view case_name);
void require_valid_capability(const CorexCudnnCapability &capability);

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_REFERENCE_HPP_
