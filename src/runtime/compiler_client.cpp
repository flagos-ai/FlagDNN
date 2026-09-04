/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "runtime/compiler_client.hpp"

#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
#include "backends/ascend/target_policy.hpp"
#endif
#include "error.hpp"
#include "runtime/context.hpp"
#include "runtime/json.hpp"
#include "runtime/sha256.hpp"

#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char **environ;

#ifndef FLAGDNN_ASCEND_DEFAULT_CANN_ROOT
#define FLAGDNN_ASCEND_DEFAULT_CANN_ROOT ""
#endif
#ifndef FLAGDNN_ASCEND_DEFAULT_LIBTRITON_JIT_LIBRARY
#define FLAGDNN_ASCEND_DEFAULT_LIBTRITON_JIT_LIBRARY ""
#endif
#ifndef FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_STANDALONE_COMPILER
#define FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_STANDALONE_COMPILER ""
#endif
#ifndef FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_CONFIG
#define FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_CONFIG ""
#endif
#ifndef FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_INCLUDE_DIRECTORY
#define FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_INCLUDE_DIRECTORY ""
#endif
#ifndef FLAGDNN_ASCEND_DEFAULT_PYTHON_MODULE_ROOT
#define FLAGDNN_ASCEND_DEFAULT_PYTHON_MODULE_ROOT ""
#endif
#ifndef FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE
#define FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE ""
#endif
#ifndef FLAGDNN_ASCEND_COMPATIBILITY_ENVIRONMENT
/* Core-only SDKs may load an external Ascend v3 plugin. Keep the current
 * public compatibility switch available to that fallback without making
 * unrelated builds parse or generate Ascend capability data. */
#define FLAGDNN_ASCEND_COMPATIBILITY_ENVIRONMENT \
  "FLAGDNN_ASCEND_910B4_1_COMPAT"
#endif

namespace flagdnn::native {
namespace {

std::atomic<std::uint64_t> identity_temporary_counter{0};
#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
const int compiler_client_anchor = 0;

bool path_is_within(const std::filesystem::path& child,
                    const std::filesystem::path& parent) {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin();
       parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() ||
        *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}
#endif

constexpr int kIdentityTemporaryFailure = 75;
constexpr std::size_t kIdentityQueryAttempts = 3;

class TemporaryIdentityFailure final : public std::exception {
public:
  [[nodiscard]] const char *what() const noexcept override {
    return "compiler identity dependencies changed during discovery";
  }
};

std::chrono::seconds compiler_timeout() {
  constexpr std::uint64_t default_seconds = 1800;
  constexpr std::uint64_t maximum_seconds = 86400;
  const char *configured = std::getenv("FLAGDNN_COMPILER_TIMEOUT_SECONDS");
  if (configured == nullptr || configured[0] == '\0') {
    return std::chrono::seconds(default_seconds);
  }
  const std::string_view value(configured);
  std::uint64_t seconds = 0;
  const auto [end, error] =
      std::from_chars(value.data(), value.data() + value.size(), seconds);
  if (error != std::errc{} || end != value.data() + value.size() ||
      seconds == 0 || seconds > maximum_seconds) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        "FLAGDNN_COMPILER_TIMEOUT_SECONDS must be an integer in [1, 86400]");
  }
  return std::chrono::seconds(seconds);
}

bool is_sha256(std::string_view value) {
  if (value.size() != 64) {
    return false;
  }
  for (const char character : value) {
    if (!((character >= '0' && character <= '9') ||
          (character >= 'a' && character <= 'f'))) {
      return false;
    }
  }
  return true;
}

class TemporaryFile {
public:
  explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {}

  ~TemporaryFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
  }

private:
  std::filesystem::path path_;
};

bool is_private_owned_directory(const std::filesystem::path& path) noexcept {
  struct stat status {};
  struct stat link_status {};
  return ::lstat(path.c_str(), &link_status) == 0 &&
         ::stat(path.c_str(), &status) == 0 &&
         S_ISDIR(link_status.st_mode) && S_ISDIR(status.st_mode) &&
         link_status.st_dev == status.st_dev &&
         link_status.st_ino == status.st_ino && status.st_uid == ::geteuid() &&
         (status.st_mode & 07777) == S_IRWXU &&
         ::access(path.c_str(), R_OK | W_OK | X_OK) == 0;
}

bool is_safe_scratch_parent(const std::filesystem::path& path) noexcept {
  struct stat status {};
  struct stat link_status {};
  if (::lstat(path.c_str(), &link_status) != 0 ||
      ::stat(path.c_str(), &status) != 0 ||
      !S_ISDIR(link_status.st_mode) || !S_ISDIR(status.st_mode) ||
      link_status.st_dev != status.st_dev ||
      link_status.st_ino != status.st_ino ||
      ::access(path.c_str(), R_OK | W_OK | X_OK) != 0) {
    return false;
  }
  if (status.st_uid == ::geteuid() &&
      (status.st_mode & 07777) == S_IRWXU) {
    return true;
  }
  return (status.st_uid == 0 || status.st_uid == ::geteuid()) &&
         (status.st_mode & S_ISVTX) != 0 &&
         (status.st_mode & S_IWOTH) != 0;
}

class AscendProviderScratch {
 public:
  explicit AscendProviderScratch(bool enabled) {
    if (!enabled) {
      return;
    }
    std::error_code path_error;
    const std::filesystem::path selected_base =
        std::filesystem::temp_directory_path(path_error);
    const std::filesystem::path base =
        path_error ? std::filesystem::path{}
                   : std::filesystem::canonical(selected_base, path_error);
    if (path_error || base.empty() || base != selected_base ||
        !is_safe_scratch_parent(base)) {
      throw ApiError(
          FLAGDNN_STATUS_NOT_SUPPORTED,
          "Ascend compiler TMPDIR must be canonical and either private 0700 "
          "or an owner-trusted sticky temporary directory");
    }
    std::string pattern =
        (base / "flagdnn-ascend-provider-XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "cannot create request-private Ascend compiler root: " +
                         std::string(std::strerror(errno)));
    }
    root_ = created;
    cache_ = root_ / "cache";
    temporary_ = root_ / "tmp";
    if (::mkdir(cache_.c_str(), 0700) != 0 ||
        ::mkdir(temporary_.c_str(), 0700) != 0 ||
        ::chmod(cache_.c_str(), 0700) != 0 ||
        ::chmod(temporary_.c_str(), 0700) != 0 ||
        !is_private_owned_directory(root_) ||
        !is_private_owned_directory(cache_) ||
        !is_private_owned_directory(temporary_)) {
      const std::string detail = std::strerror(errno);
      std::error_code ignored;
      std::filesystem::remove_all(root_, ignored);
      root_.clear();
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "cannot prepare request-private Ascend compiler root: " +
                         detail);
    }
  }

  ~AscendProviderScratch() {
    if (!root_.empty()) {
      std::error_code ignored;
      std::filesystem::remove_all(root_, ignored);
    }
  }

  AscendProviderScratch(const AscendProviderScratch&) = delete;
  AscendProviderScratch& operator=(const AscendProviderScratch&) = delete;

  [[nodiscard]] const std::filesystem::path& cache() const noexcept {
    return cache_;
  }
  [[nodiscard]] const std::filesystem::path& temporary() const noexcept {
    return temporary_;
  }

 private:
  std::filesystem::path root_;
  std::filesystem::path cache_;
  std::filesystem::path temporary_;
};

std::string ascend_local_containment_mode() {
  const char* configured =
      std::getenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  const std::string_view mode = configured == nullptr
                                    ? std::string_view{}
                                    : std::string_view(configured);
  if (mode.empty() || mode == "trusted-local") {
    return "trusted-local";
  }
  if (mode == "development") {
    return "development";
  }
  if (mode == "production" || mode == "hardened") {
    throw ApiError(
        FLAGDNN_STATUS_NOT_SUPPORTED,
        "Ascend hardened compiler launch requires an explicitly deployed "
        "supervisor and resource manager; use trusted-local or development "
        "for the in-process launcher");
  }
  throw ApiError(
      FLAGDNN_STATUS_INVALID_VALUE,
      "FLAGDNN_ASCEND_RESOURCE_CONTAINMENT must be trusted-local, "
      "development, production, or hardened");
}

void append_environment(std::vector<std::string>& values,
                        std::string name,
                        std::string value) {
  values.push_back(std::move(name) + "=" + std::move(value));
}

void copy_environment(std::vector<std::string>& values, const char* name) {
  const char* value = std::getenv(name);
  if (value != nullptr && value[0] != '\0') {
    append_environment(values, name, value);
  }
}

std::string canonical_ascend_home() {
  const char* primary = std::getenv("ASCEND_HOME_PATH");
  const char* alias = std::getenv("ASCEND_TOOLKIT_HOME");
  const std::string primary_value =
      primary == nullptr ? std::string{} : std::string(primary);
  const std::string alias_value =
      alias == nullptr ? std::string{} : std::string(alias);
  const std::string configured =
      !primary_value.empty()
          ? primary_value
          : !alias_value.empty() ? alias_value
                                 : FLAGDNN_ASCEND_DEFAULT_CANN_ROOT;
  if (configured.empty()) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend compiler has no canonical CANN root");
  }
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::canonical(configured, error);
  if (error || !std::filesystem::is_directory(resolved)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend compiler CANN root is unavailable");
  }
  if (FLAGDNN_ASCEND_DEFAULT_CANN_ROOT[0] != '\0') {
    error.clear();
    const std::filesystem::path expected = std::filesystem::canonical(
        FLAGDNN_ASCEND_DEFAULT_CANN_ROOT, error);
    if (error || expected != resolved) {
      throw ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "Ascend compiler CANN root differs from the configured plugin");
    }
  }
  if (!primary_value.empty() && !alias_value.empty()) {
    error.clear();
    const std::filesystem::path resolved_alias =
        std::filesystem::canonical(alias_value, error);
    if (error || resolved_alias != resolved) {
      throw ApiError(
          FLAGDNN_STATUS_INVALID_VALUE,
          "ASCEND_HOME_PATH and ASCEND_TOOLKIT_HOME identify different "
          "CANN installations");
    }
  }
  return resolved.string();
}

#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
std::optional<std::filesystem::path> ascend_private_runtime_path(
    std::string_view relative, bool directory) {
  if (FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE[0] == '\0' ||
      relative.empty()) {
    return std::nullopt;
  }
  Dl_info information{};
  if (::dladdr(&compiler_client_anchor, &information) == 0 ||
      information.dli_fname == nullptr) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "cannot locate the loaded FlagDNN runtime");
  }
  const std::filesystem::path relative_path(relative);
  if (relative_path.is_absolute() || relative_path.has_root_path()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "Ascend private runtime path is not relative");
  }
  std::error_code error;
  const std::filesystem::path owner =
      std::filesystem::canonical(information.dli_fname, error);
  if (error) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "cannot canonicalize the loaded FlagDNN runtime");
  }
  const std::filesystem::path lexical_root =
      (owner.parent_path() / FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE)
          .lexically_normal();
  const bool root_exists = std::filesystem::exists(lexical_root, error);
  if (error) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "cannot inspect the Ascend private runtime root");
  }
  if (!root_exists) {
    return std::nullopt;
  }
  if (!std::filesystem::is_directory(lexical_root, error) || error) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime root is not a directory");
  }
  const std::filesystem::file_status root_status =
      std::filesystem::symlink_status(lexical_root, error);
  if (error || std::filesystem::is_symlink(root_status)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime root is symlinked");
  }
  const std::filesystem::path root =
      std::filesystem::canonical(lexical_root, error);
  if (error || root != lexical_root ||
      !path_is_within(root, owner.parent_path())) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime root escapes the SDK library tree");
  }
  const std::filesystem::path candidate =
      (root / relative_path).lexically_normal();
  if (!path_is_within(candidate, root)) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "Ascend private runtime path escapes its root");
  }
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(candidate, error);
  if (error || !std::filesystem::exists(status)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime entry is missing");
  }
  const bool correct_type = directory
                                ? std::filesystem::is_directory(status)
                                : std::filesystem::is_regular_file(status);
  if (!correct_type || std::filesystem::is_symlink(status)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime entry is not a regular path");
  }
  const std::filesystem::path canonical =
      std::filesystem::canonical(candidate, error);
  if (error || !path_is_within(canonical, root)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend private runtime entry escapes its root");
  }
  return canonical;
}
#else
std::optional<std::filesystem::path> ascend_private_runtime_path(
    std::string_view, bool) {
  return std::nullopt;
}
#endif

std::string canonical_identity_path(const char* environment_name,
                                    const char* configured_default,
                                    std::string_view private_relative,
                                    bool directory) {
  const char* environment_value = std::getenv(environment_name);
  const std::optional<std::filesystem::path> private_path =
      ascend_private_runtime_path(private_relative, directory);
  const std::string selected =
      environment_value != nullptr && environment_value[0] != '\0'
          ? environment_value
          : private_path.has_value()
          ? private_path->string()
          : configured_default == nullptr ? std::string{}
                                          : std::string(configured_default);
  if (selected.empty()) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   std::string(environment_name) + " is not configured");
  }
  std::error_code error;
  const std::filesystem::path resolved =
      std::filesystem::canonical(selected, error);
  if (error || (directory && !std::filesystem::is_directory(resolved)) ||
      (!directory && !std::filesystem::is_regular_file(resolved))) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   std::string(environment_name) + " is unavailable");
  }
  if (environment_value != nullptr && environment_value[0] != '\0' &&
      configured_default != nullptr && configured_default[0] != '\0') {
    std::vector<std::filesystem::path> expected_paths;
    if (private_path.has_value()) {
      expected_paths.push_back(*private_path);
    } else {
      error.clear();
      const std::filesystem::path configured =
          std::filesystem::canonical(configured_default, error);
      if (!error) {
        expected_paths.push_back(configured);
      }
    }
    if (std::find(expected_paths.begin(), expected_paths.end(), resolved) ==
        expected_paths.end()) {
      throw ApiError(FLAGDNN_STATUS_INVALID_VALUE,
                     std::string(environment_name) +
                         " differs from the configured Ascend toolchain");
    }
  }
  return resolved.string();
}

#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
flagdnn::ascend::target_policy::Resolution ascend_codegen_target(
    std::string_view target_fingerprint) {
  constexpr std::string_view kPrefix = "ascend_";
  constexpr std::string_view kVersionMarker = "_cann_";
  if (!target_fingerprint.starts_with(kPrefix)) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend target fingerprint has no code-generation arch");
  }
  const std::size_t marker = target_fingerprint.rfind(kVersionMarker);
  if (marker == std::string_view::npos || marker <= kPrefix.size() ||
      marker + kVersionMarker.size() == target_fingerprint.size()) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend target fingerprint is not versioned");
  }
  const std::string_view encoded = target_fingerprint.substr(
      kPrefix.size(), marker - kPrefix.size());
  if (encoded.find_first_not_of(
          "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") !=
      std::string_view::npos) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend target contains an invalid code-generation arch");
  }
  const char* compatibility = std::getenv(
      flagdnn::ascend::target_policy::kCompatibilityEnvironment);
  const flagdnn::ascend::target_policy::Resolution resolution =
      flagdnn::ascend::target_policy::resolve_codegen_arch(
          encoded, compatibility == nullptr ? std::string_view{}
                                            : std::string_view(compatibility));
  if (resolution.status ==
      flagdnn::ascend::target_policy::ResolutionStatus::
          kInvalidCompatibilitySetting) {
    throw ApiError(
        FLAGDNN_STATUS_INVALID_VALUE,
        std::string(
            flagdnn::ascend::target_policy::kCompatibilityEnvironment) +
            " must be unset, '0', or '1', and '1' is valid only on " +
            std::string(
                flagdnn::ascend::target_policy::kCompatibilityRuntimeSoc));
  }
  if (resolution.status !=
      flagdnn::ascend::target_policy::ResolutionStatus::kSupported) {
    throw ApiError(FLAGDNN_STATUS_NOT_SUPPORTED,
                   "Ascend target contains an unsupported SoC identity");
  }
  return resolution;
}
#endif

std::vector<std::string> ascend_compiler_environment(
    const RuntimeContext& context,
    const AscendProviderScratch& scratch,
    std::string_view containment_mode) {
  std::vector<std::string> values;
  values.reserve(48);
  append_environment(values, "TRITON_JIT_BACKEND", "NPU");
#if defined(FLAGDNN_HAS_ASCEND_TARGET_POLICY)
  const flagdnn::ascend::target_policy::Resolution target =
      ascend_codegen_target(context.target_fingerprint());
  if (!target.detects_codegen_arch_from_runtime) {
    append_environment(values, "TRITON_ASCEND_ARCH",
                       std::string(target.codegen_arch));
  }
  if (target.uses_compatibility_alias) {
    append_environment(
        values,
        flagdnn::ascend::target_policy::kCompatibilityEnvironment,
        "1");
  }
#else
  /* An external Ascend plugin has already validated and frozen these values
   * while creating its context. Mirror them into the sanitized compiler child
   * so a core-only SDK remains extensible without embedding backend policy. */
  (void)context;
  copy_environment(values, "TRITON_ASCEND_ARCH");
  copy_environment(values, FLAGDNN_ASCEND_COMPATIBILITY_ENVIRONMENT);
#endif
  append_environment(values, "TRITON_BACKEND", "torch_npu");
  append_environment(values, "TORCH_DEVICE_BACKEND_AUTOLOAD", "0");
  append_environment(values, "FLAGDNN_ASCEND_REQUIRE_COMPILER_MODULES", "1");
  append_environment(values,
                     "FLAGDNN_ASCEND_RESOURCE_CONTAINMENT",
                     std::string(containment_mode));
  const std::string ascend_home = canonical_ascend_home();
  append_environment(values, "ASCEND_HOME_PATH", ascend_home);
  append_environment(values, "ASCEND_TOOLKIT_HOME", ascend_home);
  append_environment(
      values,
      "FLAGDNN_LIBTRITON_JIT_LIBRARY",
      canonical_identity_path(
          "FLAGDNN_LIBTRITON_JIT_LIBRARY",
          FLAGDNN_ASCEND_DEFAULT_LIBTRITON_JIT_LIBRARY,
          "libtriton_jit.so",
          false));
  append_environment(
      values,
      "FLAGDNN_TRITON_JIT_STANDALONE_COMPILER",
      canonical_identity_path(
          "FLAGDNN_TRITON_JIT_STANDALONE_COMPILER",
          FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_STANDALONE_COMPILER,
          "standalone_compile.py",
          false));
  append_environment(
      values,
      "FLAGDNN_TRITON_JIT_CONFIG",
      canonical_identity_path("FLAGDNN_TRITON_JIT_CONFIG",
                              FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_CONFIG,
                              "TritonJITConfig.cmake",
                              false));
  append_environment(
      values,
      "FLAGDNN_TRITON_JIT_INCLUDE_DIRECTORY",
      canonical_identity_path(
          "FLAGDNN_TRITON_JIT_INCLUDE_DIRECTORY",
          FLAGDNN_ASCEND_DEFAULT_TRITON_JIT_INCLUDE_DIRECTORY,
          "include",
          true));
  append_environment(
      values,
      "PYTHONPATH",
      canonical_identity_path("FLAGDNN_ASCEND_PYTHON_MODULE_ROOT",
                              FLAGDNN_ASCEND_DEFAULT_PYTHON_MODULE_ROOT,
                              {},
                              true));
  append_environment(values, "TRITON_CACHE_DIR", scratch.cache().string());
  append_environment(values, "TMPDIR", scratch.temporary().string());
  append_environment(values, "PYTHONDONTWRITEBYTECODE", "1");
  append_environment(values, "PYTHONNOUSERSITE", "1");
  append_environment(values, "PYTHONHASHSEED", "0");
  append_environment(values, "PYTHONSAFEPATH", "1");
  append_environment(values, "LC_ALL", "C");

  static constexpr std::array<const char*, 24> kAllowedInherited = {
      "TRITON_NPU_COMPILER_PATH",
      "MLIR_ROOT",
      "LLVM_ROOT",
      "CC",
      "TRITON_ENABLE_VF_FUSION",
      "TRITON_DISABLE_FFTS",
      "TRITON_ENABLE_LIBDEVICE_SIMT",
      "TRITON_ALL_BLOCKS_PARALLEL",
      "TRITON_DISABLE_LINE_INFO",
      "TRITON_ENABLE_SANITIZER",
      "ENABLE_UNPUBLISHED_FEATURE",
      "ENABLE_PRINT_UB_BITS",
      "TRITON_MEMORY_DISPLAY",
      "LLVM_EXTRACT_DI_LOCAL_VARIABLES",
      "TRITON_ENABLE_TASKQUEUE",
      "TRITON_DEVICE_PRINT",
      "TRITON_GRID_WARN_PRINT",
      "TRITON_DISABLE_PRECOMPILE",
      "TRITON_ALLOW_NON_CONSTEXPR_GLOBALS",
      "PATH",
      "LD_LIBRARY_PATH",
      "FLAGDNN_BACKEND_ROOT",
      "FLAGDNN_KERNEL_SOURCE_ROOT",
      "FLAGDNN_TUNING_ROOT",
  };
  for (const char* name : kAllowedInherited) {
    copy_environment(values, name);
  }
  return values;
}

void run_compiler_process(const RuntimeContext& context,
                          const std::vector<std::string>& owned_arguments,
                          std::string_view action,
                          bool retry_temporary_identity_failure = false) {
  if (context.compiler_executable().empty() || context.compiler().empty()) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler is not configured; call flagdnnSetCompilerConfig");
  }
  if (owned_arguments.size() < 2 ||
      owned_arguments[0] != context.compiler_executable() ||
      owned_arguments[1] != context.compiler()) {
    throw ApiError(FLAGDNN_STATUS_INTERNAL_ERROR,
                   "external compiler argument construction is invalid");
  }

  std::vector<char *> arguments;
  arguments.reserve(owned_arguments.size() + 1);
  for (const std::string &argument : owned_arguments) {
    arguments.push_back(const_cast<char *>(argument.c_str()));
  }
  arguments.push_back(nullptr);

  const bool ascend = context.backend_name() == "ascend";
  std::vector<std::string> owned_environment;
  std::vector<char *> environment;
  pid_t child = -1;
  int spawn_result = 0;
  std::chrono::seconds timeout;

  // Environment mutation is serialized by RuntimeContext. Keep the shared
  // lock through posix_spawnp so both inherited and Ascend-sanitized child
  // environments are coherent snapshots.
  std::shared_lock environment_lock(process_environment_mutex());
  timeout = compiler_timeout();
  const std::string ascend_containment =
      ascend ? ascend_local_containment_mode() : std::string{};
  AscendProviderScratch provider_scratch(ascend);
  if (ascend) {
    owned_environment = ascend_compiler_environment(
        context, provider_scratch, ascend_containment);
  } else if (environ != nullptr) {
    for (char **entry = environ; *entry != nullptr; ++entry) {
      owned_environment.emplace_back(*entry);
    }
  }
  environment.reserve(owned_environment.size() + 1);
  for (std::string &entry : owned_environment) {
    environment.push_back(entry.data());
  }
  environment.push_back(nullptr);

  posix_spawnattr_t spawn_attributes;
  int attribute_result = posix_spawnattr_init(&spawn_attributes);
  if (attribute_result != 0) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot initialize external compiler process attributes: " +
                       std::string(std::strerror(attribute_result)));
  }
  attribute_result =
      posix_spawnattr_setflags(&spawn_attributes, POSIX_SPAWN_SETPGROUP);
  if (attribute_result == 0) {
    // A process-group value of zero creates a group whose ID is the child PID.
    // This lets timeout handling terminate Python plus ptxas/other descendants.
    attribute_result = posix_spawnattr_setpgroup(&spawn_attributes, 0);
  }
  if (attribute_result != 0) {
    (void)posix_spawnattr_destroy(&spawn_attributes);
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "cannot configure external compiler process group: " +
                       std::string(std::strerror(attribute_result)));
  }
  spawn_result = posix_spawnp(&child, arguments[0], nullptr,
                              &spawn_attributes, arguments.data(),
                              environment.data());
  environment_lock.unlock();
  (void)posix_spawnattr_destroy(&spawn_attributes);
  if (spawn_result != 0) {
    const std::string message = "cannot start external compiler for " +
                                std::string(action) + ": " +
                                std::string(std::strerror(spawn_result));
    if (spawn_result == ENOENT) {
      throw CompilerExecutableUnavailable(message);
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, message);
  }

  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int status = 0;
  pid_t wait_result = -1;
  for (;;) {
    wait_result = waitpid(child, &status, WNOHANG);
    if (wait_result == child) {
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      (void)kill(-child, SIGKILL);
      do {
        wait_result = waitpid(child, &status, 0);
      } while (wait_result < 0 && errno == EINTR);
      throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                     "external compiler " + std::string(action) +
                         " timed out after " + std::to_string(timeout.count()) +
                         " seconds");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  if (wait_result != child) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "waitpid failed for external compiler: " +
                       std::string(std::strerror(errno)));
  }
  if (retry_temporary_identity_failure && WIFEXITED(status) &&
      WEXITSTATUS(status) == kIdentityTemporaryFailure) {
    throw TemporaryIdentityFailure();
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    std::ostringstream message;
    message << "external compiler " << action << " failed";
    if (WIFEXITED(status)) {
      message << " with exit code " << WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
      message << " with signal " << WTERMSIG(status);
    }
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED, message.str());
  }
}

struct CompilerIdentityResponse {
  struct DependencySnapshot {
    std::filesystem::path path;
    std::string fingerprint;
    std::string content_sha256;
  };

  std::string digest;
  std::vector<std::filesystem::path> dependencies;
  std::vector<DependencySnapshot> snapshots;
  bool has_snapshots = false;
  bool dependencies_complete = false;
};

CompilerIdentityResponse
read_compiler_identity(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler did not produce its identity");
  }
  std::string value((std::istreambuf_iterator<char>(input)),
                    std::istreambuf_iterator<char>());
  constexpr std::size_t maximum_identity_bytes = 8U * 1024U * 1024U;
  if (input.bad() || value.size() > maximum_identity_bytes) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "compiler identity file is invalid");
  }
  const std::size_t newline = value.find('\n');
  std::string digest = value.substr(0, newline);
  if (!digest.empty() && digest.back() == '\r') {
    digest.pop_back();
  }
  if (!is_sha256(digest)) {
    throw ApiError(FLAGDNN_STATUS_COMPILATION_FAILED,
                   "external compiler returned an invalid identity");
  }

  CompilerIdentityResponse result;
  result.digest = std::move(digest);
  if (newline == std::string::npos) {
    return result;
  }
  const std::string_view metadata(value.data() + newline + 1,
                                  value.size() - newline - 1);
  if (std::all_of(metadata.begin(), metadata.end(), [](char character) {
        return std::isspace(static_cast<unsigned char>(character)) != 0;
      })) {
    return result;
  }
  try {
    const auto root = json::parse(metadata);
    if (root.at("schema_version").as_int() != 1) {
      throw std::runtime_error("unsupported dependency schema");
    }
    const auto &files = root.at("files").as_array();
    constexpr std::size_t maximum_dependency_count = 65536;
    constexpr std::size_t maximum_dependency_path_bytes = 65536;
    if (files.size() > maximum_dependency_count) {
      throw std::runtime_error("too many dependency paths");
    }
    result.dependencies.reserve(files.size());
    for (const auto &file : files) {
      const std::string &text = file.as_string();
      const std::filesystem::path dependency(text);
      if (text.empty() || text.size() > maximum_dependency_path_bytes ||
          !dependency.is_absolute()) {
        throw std::runtime_error(
            "dependency paths must be nonempty absolute paths");
      }
      result.dependencies.push_back(dependency.lexically_normal());
    }
    std::sort(result.dependencies.begin(), result.dependencies.end());
    result.dependencies.erase(
        std::unique(result.dependencies.begin(), result.dependencies.end()),
        result.dependencies.end());

    const auto &object = root.as_object();
    const auto completeness = object.find("dependencies_complete");
    if (completeness != object.end()) {
      result.dependencies_complete = completeness->second.as_bool();
    }
    const auto snapshot_schema = object.find("snapshot_schema_version");
    const auto snapshots = object.find("snapshots");
    if ((snapshot_schema == object.end()) != (snapshots == object.end())) {
      throw std::runtime_error(
          "snapshot_schema_version and snapshots must appear together");
    }
    if (snapshots != object.end()) {
      if (snapshot_schema->second.as_int() != 1) {
        throw std::runtime_error("unsupported dependency snapshot schema");
      }
      const auto &entries = snapshots->second.as_array();
      if (entries.size() > maximum_dependency_count) {
        throw std::runtime_error("too many dependency snapshots");
      }
      result.snapshots.reserve(entries.size());
      for (const auto &entry : entries) {
        const std::string &text = entry.at("path").as_string();
        const std::string &fingerprint = entry.at("fingerprint").as_string();
        const std::filesystem::path dependency(text);
        if (text.empty() || text.size() > maximum_dependency_path_bytes ||
            !dependency.is_absolute() || !is_sha256(fingerprint)) {
          throw std::runtime_error(
              "dependency snapshots contain an invalid path or fingerprint");
        }
        std::string content_sha256;
        const auto &snapshot_object = entry.as_object();
        const auto content = snapshot_object.find("content_sha256");
        if (content != snapshot_object.end()) {
          content_sha256 = content->second.as_string();
          if (!is_sha256(content_sha256)) {
            throw std::runtime_error(
                "dependency snapshot content hash is invalid");
          }
        }
        result.snapshots.push_back(
            {dependency.lexically_normal(), fingerprint, content_sha256});
      }
      std::sort(result.snapshots.begin(), result.snapshots.end(),
                [](const auto &left, const auto &right) {
                  return left.path < right.path;
                });
      if (std::adjacent_find(result.snapshots.begin(), result.snapshots.end(),
                             [](const auto &left, const auto &right) {
                               return left.path == right.path;
                             }) != result.snapshots.end()) {
        throw std::runtime_error(
            "dependency snapshots contain duplicate paths");
      }
      if (result.snapshots.size() != result.dependencies.size()) {
        throw std::runtime_error(
            "dependency snapshots do not match dependency paths");
      }
      for (std::size_t index = 0; index < result.dependencies.size(); ++index) {
        if (result.snapshots[index].path != result.dependencies[index]) {
          throw std::runtime_error(
              "dependency snapshots do not match dependency paths");
        }
      }
      result.has_snapshots = true;
    }
    if (result.dependencies_complete && !result.has_snapshots) {
      throw std::runtime_error(
          "complete dependencies require dependency snapshots");
    }
  } catch (const std::exception &error) {
    throw ApiError(
        FLAGDNN_STATUS_COMPILATION_FAILED,
        "external compiler returned invalid identity dependencies: " +
            std::string(error.what()));
  }
  return result;
}

void append_stat_record(std::string &state, const struct stat &status) {
  state += std::to_string(static_cast<std::uintmax_t>(status.st_dev));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_ino));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_mode));
  state.push_back(':');
  state += std::to_string(static_cast<std::uintmax_t>(status.st_size));
  state.push_back(':');
#if defined(__APPLE__)
  const auto &modified = status.st_mtimespec;
  const auto &changed = status.st_ctimespec;
#else
  const auto &modified = status.st_mtim;
  const auto &changed = status.st_ctim;
#endif
  state += std::to_string(modified.tv_sec);
  state.push_back(':');
  state += std::to_string(modified.tv_nsec);
  state.push_back(':');
  state += std::to_string(changed.tv_sec);
  state.push_back(':');
  state += std::to_string(changed.tv_nsec);
}

std::string dependency_fingerprint(const std::filesystem::path &path) {
  std::string state("flagdnn-dependency-state-v1\0", 28);
  struct stat link_status {};
  if (::lstat(path.c_str(), &link_status) != 0) {
    const int error = errno;
    state += "lstat-error:";
    state += std::to_string(error);
    state.push_back('\0');
    return sha256(state);
  }
  state += "lstat:";
  append_stat_record(state, link_status);
  state.push_back('\0');

  if (S_ISLNK(link_status.st_mode)) {
    std::vector<char> link_value(256);
    for (;;) {
      const ssize_t size =
          ::readlink(path.c_str(), link_value.data(), link_value.size());
      if (size < 0) {
        const int error = errno;
        state += "link-error:";
        state += std::to_string(error);
        state.push_back('\0');
        break;
      }
      if (static_cast<std::size_t>(size) < link_value.size()) {
        state += "link:";
        state.append(link_value.data(), static_cast<std::size_t>(size));
        state.push_back('\0');
        break;
      }
      if (link_value.size() >= 65536U) {
        state += "link-error:";
        state += std::to_string(ENAMETOOLONG);
        state.push_back('\0');
        break;
      }
      link_value.resize(link_value.size() * 2U);
    }
  } else {
    state.append("link:\0", 6);
  }

  struct stat status {};
  if (::stat(path.c_str(), &status) != 0) {
    const int error = errno;
    state += "stat-error:";
    state += std::to_string(error);
    state.push_back('\0');
  } else {
    state += "stat:";
    append_stat_record(state, status);
    state.push_back('\0');
  }
  return sha256(state);
}

void append_file_identity(std::string &snapshot,
                          const std::filesystem::path &path) {
  snapshot += path.string();
  snapshot.push_back('\0');
  snapshot += dependency_fingerprint(path);
  snapshot.push_back('\0');
}

std::string compiler_dependency_snapshot(
    const std::vector<std::filesystem::path> &dependencies) {
  std::string snapshot;
  for (const auto &dependency : dependencies) {
    append_file_identity(snapshot, dependency);
  }
  return snapshot;
}

std::string
reported_dependency_snapshot(const CompilerIdentityResponse &response) {
  std::string snapshot;
  for (const auto &entry : response.snapshots) {
    snapshot += entry.path.string();
    snapshot.push_back('\0');
    snapshot += entry.fingerprint;
    snapshot.push_back('\0');
  }
  return snapshot;
}

bool reported_dependency_contents_match(
    const CompilerIdentityResponse &response) {
  for (const auto &entry : response.snapshots) {
    if (entry.content_sha256.empty()) {
      continue;
    }
    try {
      if (sha256_file(entry.path) != entry.content_sha256) {
        return false;
      }
    } catch (const std::exception &) {
      return false;
    }
  }
  return true;
}

std::string compiler_identity_snapshot(const RuntimeContext &context) {
  const std::shared_lock environment_lock(process_environment_mutex());
  std::string snapshot;
  snapshot.reserve(4096);
  snapshot += context.compiler_executable();
  snapshot.push_back('\0');
  snapshot += context.compiler();
  snapshot.push_back('\0');
  snapshot += context.backend_name();
  snapshot.push_back('\0');
  snapshot += context.target_fingerprint();
  snapshot.push_back('\0');
  snapshot += context.execution_engine();
  snapshot.push_back('\0');

  if (context.compiler_executable().find('/') != std::string::npos) {
    append_file_identity(snapshot, context.compiler_executable());
  }
  append_file_identity(snapshot, context.compiler());

  std::error_code current_directory_error;
  const std::filesystem::path current_directory =
      std::filesystem::current_path(current_directory_error);
  snapshot += current_directory_error
                  ? "cwd-error:" + current_directory_error.message()
                  : current_directory.string();
  snapshot.push_back('\0');

  // The identity subprocess inherits the complete environment. Sorting it
  // makes the memo key stable. Provider-reported files are tracked separately
  // so in-place kernel/registry/tuning/JIT changes also invalidate the memo.
  std::vector<std::string> environment;
  if (environ != nullptr) {
    for (char **entry = environ; *entry != nullptr; ++entry) {
      environment.emplace_back(*entry);
    }
  }
  std::sort(environment.begin(), environment.end());
  for (const std::string &entry : environment) {
    snapshot.append(entry);
    snapshot.push_back('\0');
  }
  return snapshot;
}

} // namespace

std::string
query_compiler_identity(RuntimeContext &context,
                        const std::filesystem::path &graph_cache_directory,
                        bool force_refresh) {
  // prepare_artifact_package() owns compiler_configuration_mutex_ for the
  // complete identity/cache/compile transaction. Keeping the lock at that
  // boundary also makes set_compiler() atomic with respect to every compiler
  // configuration read below.
  const std::string snapshot = compiler_identity_snapshot(context);
  if (!force_refresh && !context.compiler_identity_.empty() &&
      context.compiler_identity_snapshot_ == snapshot &&
      context.compiler_identity_dependencies_snapshot_ ==
          compiler_dependency_snapshot(
              context.compiler_identity_dependencies_)) {
    return context.compiler_identity_;
  }
  for (std::size_t attempt = 0; attempt < kIdentityQueryAttempts; ++attempt) {
    const std::string snapshot_before = compiler_identity_snapshot(context);
    const std::uint64_t serial = identity_temporary_counter.fetch_add(1);
    const std::filesystem::path identity_output =
        graph_cache_directory / (".identity.tmp." + std::to_string(getpid()) +
                                 "." + std::to_string(serial));
    TemporaryFile cleanup(identity_output);

    try {
      run_compiler_process(context,
                           {context.compiler_executable(), context.compiler(),
                            "--identify", "--backend", context.backend_name(),
                            "--target", context.target_fingerprint(),
                            "--execution-engine", context.execution_engine(),
                            "--identity-output", identity_output.string(),
                            "--quiet"},
                           "identity query", true);
    } catch (const TemporaryIdentityFailure &) {
      continue;
    }

    CompilerIdentityResponse response = read_compiler_identity(identity_output);
    const std::string dependencies_before =
        compiler_dependency_snapshot(response.dependencies);
    if (response.has_snapshots &&
        dependencies_before != reported_dependency_snapshot(response)) {
      continue;
    }
    if (response.has_snapshots &&
        !reported_dependency_contents_match(response)) {
      continue;
    }
    const std::string snapshot_after = compiler_identity_snapshot(context);
    const std::string dependencies_after =
        compiler_dependency_snapshot(response.dependencies);
    if (snapshot_before != snapshot_after ||
        dependencies_before != dependencies_after) {
      continue;
    }

    if (response.dependencies_complete && response.has_snapshots) {
      context.compiler_identity_snapshot_ = snapshot_after;
      context.compiler_identity_dependencies_ =
          std::move(response.dependencies);
      context.compiler_identity_dependencies_snapshot_ =
          std::move(dependencies_after);
      context.compiler_identity_ = std::move(response.digest);
      return context.compiler_identity_;
    }

    // Digest-only and legacy/incomplete dependency responses stay supported,
    // but must retain the historical query-per-build behavior. Otherwise a
    // provider that hashes unreported resources could return a stale memo.
    context.compiler_identity_snapshot_.clear();
    context.compiler_identity_dependencies_.clear();
    context.compiler_identity_dependencies_snapshot_.clear();
    context.compiler_identity_.clear();
    return response.digest;
  }
  throw ApiError(
      FLAGDNN_STATUS_COMPILATION_FAILED,
      "external compiler identity dependencies did not stabilize after " +
          std::to_string(kIdentityQueryAttempts) + " attempts");
}

void compile_external_artifact(const RuntimeContext &context,
                               const std::filesystem::path &request,
                               const std::filesystem::path &output_directory) {
  run_compiler_process(context,
                       {context.compiler_executable(), context.compiler(),
                        "--request", request.string(), "--output-dir",
                        output_directory.string(), "--execution-engine",
                        context.execution_engine(), "--quiet"},
                       "compile");
}

} // namespace flagdnn::native
