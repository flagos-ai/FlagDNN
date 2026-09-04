/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_TARGET_POLICY_HPP_
#define FLAGDNN_BACKENDS_ASCEND_TARGET_POLICY_HPP_

#include <flagdnn/ascend_target_policy_data.hpp>

#include <string_view>

namespace flagdnn::ascend::target_policy {

enum class ResolutionStatus {
  kSupported,
  kInvalidCompatibilitySetting,
  kUnsupportedSoc,
};

struct Resolution {
  ResolutionStatus status = ResolutionStatus::kUnsupportedSoc;
  std::string_view codegen_arch;
  bool uses_compatibility_alias = false;
  bool detects_codegen_arch_from_runtime = false;
};

[[nodiscard]] inline constexpr bool supported_codegen_arch(
    std::string_view value) noexcept {
  for (const std::string_view supported : kSupportedCodegenArches) {
    if (supported == value) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline constexpr bool supported_target_soc(
    std::string_view value) noexcept {
  return supported_codegen_arch(value);
}

[[nodiscard]] inline constexpr bool runtime_detected_codegen_arch(
    std::string_view value) noexcept {
  for (const std::string_view detected : kRuntimeDetectedCodegenArches) {
    if (detected == value) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] inline constexpr Resolution resolve_codegen_arch(
    std::string_view runtime_soc,
    std::string_view compatibility_setting) noexcept {
  const bool compatibility_disabled = compatibility_setting.empty() ||
                                      compatibility_setting == "0";
  const bool compatibility_enabled = compatibility_setting == "1";
  if (!compatibility_disabled && !compatibility_enabled) {
    return {ResolutionStatus::kInvalidCompatibilitySetting, {}, false, false};
  }
  if (compatibility_enabled) {
    if (runtime_soc != kCompatibilityRuntimeSoc) {
      return {ResolutionStatus::kInvalidCompatibilitySetting, {}, false,
              false};
    }
    return {ResolutionStatus::kSupported, kCompatibilityCodegenArch, true,
            false};
  }
  if (supported_codegen_arch(runtime_soc)) {
    /* Runtime-detected targets must not be forced through an environment
     * spelling rejected by the pinned backend. Its driver obtains the exact
     * SoC from rtGetSocVersion and forwards that value to BishengIR. */
    return {ResolutionStatus::kSupported,
            runtime_soc,
            false,
            runtime_detected_codegen_arch(runtime_soc)};
  }
  return {ResolutionStatus::kUnsupportedSoc, {}, false, false};
}

static_assert(resolve_codegen_arch(kCompatibilityRuntimeSoc, "1")
                  .codegen_arch == kCompatibilityCodegenArch);
static_assert(resolve_codegen_arch(kCompatibilityCodegenArch, {})
                  .codegen_arch == kCompatibilityCodegenArch);
static_assert(resolve_codegen_arch(kCompatibilityRuntimeSoc, {})
                  .codegen_arch == kCompatibilityRuntimeSoc);
static_assert(resolve_codegen_arch(kCompatibilityRuntimeSoc, {})
                  .detects_codegen_arch_from_runtime);
static_assert(!resolve_codegen_arch(kCompatibilityRuntimeSoc, "1")
                   .detects_codegen_arch_from_runtime);

}  // namespace flagdnn::ascend::target_policy

#endif  // FLAGDNN_BACKENDS_ASCEND_TARGET_POLICY_HPP_
