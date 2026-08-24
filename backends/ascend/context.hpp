/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_CONTEXT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_CONTEXT_HPP_

#include "backends/backend_api.h"

#include <acl/acl_rt.h>

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace flagdnn::ascend {

namespace detail {
struct ProcessConfigurationSnapshot;
}  // namespace detail

struct EngineBuildContext {
  std::int32_t device_ordinal = -1;
  aclrtContext context = nullptr;
  std::uint32_t ai_core_count = 0;
  std::string target_fingerprint;
  std::string codegen_arch;
  std::string runtime_identity;
  std::string cache_root;
  std::string configuration_identity;
  std::shared_ptr<const detail::ProcessConfigurationSnapshot>
      configuration_snapshot;
};

namespace detail {

enum class DomainPhase {
  kUnbound,
  kInitializing,
  kBound,
  kFailed,
};

struct DomainBinding {
  std::int32_t device_ordinal = -1;
  aclrtContext default_context = nullptr;
  std::uint32_t ai_core_count = 0;
  std::string target_fingerprint;
  std::string codegen_arch;
  std::string runtime_identity;
  std::string cache_root;
  std::string configuration_identity;
  std::shared_ptr<const ProcessConfigurationSnapshot> configuration_snapshot;
};

enum class InitializationDisposition {
  kBound,
  kRetryableFailure,
  kTerminalFailure,
};

struct DomainInitialization {
  InitializationDisposition disposition =
      InitializationDisposition::kTerminalFailure;
  DomainBinding binding;
  flagdnnBackendResult_t result = FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  std::string message;

  [[nodiscard]] static DomainInitialization bound(DomainBinding binding);
  [[nodiscard]] static DomainInitialization retryable(
      flagdnnBackendResult_t result,
      std::string message);
  [[nodiscard]] static DomainInitialization terminal(
      flagdnnBackendResult_t result,
      std::string message);
};

struct DomainSnapshot {
  DomainPhase phase = DomainPhase::kUnbound;
  std::int32_t requested_device = -1;
  std::uint64_t initializer_id = 0;
  bool terminal_failure_latched = false;
};

class DomainCoordinator {
 public:
  using Initializer =
      std::function<DomainInitialization(std::int32_t device_ordinal)>;

  [[nodiscard]] std::shared_ptr<const DomainBinding> acquire(
      std::int32_t device_ordinal,
      const Initializer& initializer);

  void ensure_healthy() const;
  void latch_terminal() noexcept;
  void mark_terminal(flagdnnBackendResult_t result,
                     std::string message) noexcept;

  [[nodiscard]] bool terminal_failure_latched() const noexcept;
  [[nodiscard]] DomainSnapshot snapshot() const;
  [[nodiscard]] std::int32_t bound_device_ordinal(
      aclrtContext expected_context) const;

 private:
  [[noreturn]] void throw_failure_locked() const;

  mutable std::mutex mutex_;
  std::condition_variable condition_;
  DomainPhase phase_ = DomainPhase::kUnbound;
  std::int32_t requested_device_ = -1;
  std::uint64_t initializer_id_ = 0;
  std::shared_ptr<const DomainBinding> binding_;
  flagdnnBackendResult_t failure_result_ =
      FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR;
  std::string failure_message_ = "Ascend process domain is terminal";
  std::atomic<bool> terminal_failure_latch_{false};
};

struct AclRuntimeApi {
  aclError (*get_current_context)(aclrtContext*) = nullptr;
  aclError (*set_current_context)(aclrtContext) = nullptr;
  aclError (*get_device)(std::int32_t*) = nullptr;
  std::int32_t (*get_ai_core_count)(std::uint32_t*) = nullptr;
  const char* (*get_soc_name)() = nullptr;
  aclError (*get_version_string)(char*, char*) = nullptr;
  aclError (*get_version_number)(char*, std::int32_t*) = nullptr;
};

[[nodiscard]] const AclRuntimeApi& production_acl_runtime_api() noexcept;
[[nodiscard]] DomainCoordinator& process_domain() noexcept;

/* Internal test seam. It follows the production ordering and never bypasses
 * the containment decision; tests inject counters in place of AscendCL. */
[[nodiscard]] DomainInitialization initialize_domain_with_api(
    std::int32_t device_ordinal,
    const AclRuntimeApi& api);
[[nodiscard]] std::size_t loaded_python_object_count_for_test();
void verify_pinned_python_files_for_test();

}  // namespace detail

class ContextGuard {
 public:
  explicit ContextGuard(aclrtContext expected_context);
  ContextGuard(aclrtContext expected_context,
               std::int32_t expected_device,
               detail::DomainCoordinator& domain,
               const detail::AclRuntimeApi& api);
  ~ContextGuard() = default;

  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
};

class AscendContext {
 public:
  explicit AscendContext(std::int32_t device_ordinal);
  ~AscendContext() = default;

  AscendContext(const AscendContext&) = delete;
  AscendContext& operator=(const AscendContext&) = delete;

  [[nodiscard]] const std::string& target_fingerprint() const noexcept;
  [[nodiscard]] EngineBuildContext engine_build_context() const;

 private:
  std::shared_ptr<const detail::DomainBinding> binding_;
};

/*
 * Every libtriton_jit registry/cache/hook/module operation uses this process
 * mutex. Callers must never hold the domain mutex while acquiring it.
 */
[[nodiscard]] std::mutex& process_ltj_mutex() noexcept;
void ensure_process_healthy();
void ensure_process_configuration(const EngineBuildContext& context);
/* Allocation-free and domain-mutex-free configuration readback. This may be
 * called while process_ltj_mutex() is held; callers must latch under that lock
 * and publish the terminal diagnostic only after releasing it. */
[[nodiscard]] bool process_configuration_matches(
    const EngineBuildContext& context) noexcept;
/* Safe while process_ltj_mutex() is held. Commit the diagnostic only after
 * releasing that mutex by calling mark_process_terminal(). */
void latch_process_terminal() noexcept;
/* Must not be called while process_ltj_mutex() is held. */
void mark_process_terminal(flagdnnBackendResult_t result,
                           std::string message) noexcept;

}  // namespace flagdnn::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_CONTEXT_HPP_
