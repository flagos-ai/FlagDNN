/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/ascend/context.hpp"

#include "backends/ascend/error.hpp"
#include "backends/ascend/target_policy.hpp"
#include "runtime/sha256.hpp"

#include <Python.h>
#include <triton_jit/kernel_metadata.h>

#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <runtime/runtime/rt.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef FLAGDNN_ASCEND_CANN_ROOT
#define FLAGDNN_ASCEND_CANN_ROOT ""
#endif

#ifndef FLAGDNN_ASCEND_CANN_VERSION
#define FLAGDNN_ASCEND_CANN_VERSION "unknown"
#endif

#ifndef FLAGDNN_ASCEND_CANN_INNER_VERSION
#define FLAGDNN_ASCEND_CANN_INNER_VERSION "unknown"
#endif

#ifndef FLAGDNN_ASCEND_ASCENDCL_PATH
#define FLAGDNN_ASCEND_ASCENDCL_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_ASCENDCL_SHA256
#define FLAGDNN_ASCEND_ASCENDCL_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_RUNTIME_PATH
#define FLAGDNN_ASCEND_RUNTIME_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_RUNTIME_SHA256
#define FLAGDNN_ASCEND_RUNTIME_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_LIBTRITON_JIT_PATH
#define FLAGDNN_ASCEND_LIBTRITON_JIT_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256
#define FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_PYTHON_LIBRARY_PATH
#define FLAGDNN_ASCEND_PYTHON_LIBRARY_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256
#define FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_PYTHON_MODULE_ROOT
#define FLAGDNN_ASCEND_PYTHON_MODULE_ROOT ""
#endif
#ifndef FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH
#define FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH ""
#endif
#ifndef FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH
#define FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH ""
#endif
#ifndef FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256
#define FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_TORCH_MODULE_PATH
#define FLAGDNN_ASCEND_TORCH_MODULE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_TORCH_MODULE_SHA256
#define FLAGDNN_ASCEND_TORCH_MODULE_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_TRITON_MODULE_PATH
#define FLAGDNN_ASCEND_TRITON_MODULE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_TRITON_MODULE_SHA256
#define FLAGDNN_ASCEND_TRITON_MODULE_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_TORCH_NPU_MODULE_PATH
#define FLAGDNN_ASCEND_TORCH_NPU_MODULE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_TORCH_NPU_MODULE_SHA256
#define FLAGDNN_ASCEND_TORCH_NPU_MODULE_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_YAML_MODULE_PATH
#define FLAGDNN_ASCEND_YAML_MODULE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_YAML_MODULE_SHA256
#define FLAGDNN_ASCEND_YAML_MODULE_SHA256 ""
#endif

#ifndef FLAGDNN_ASCEND_STANDALONE_PATH
#define FLAGDNN_ASCEND_STANDALONE_PATH ""
#endif

#ifndef FLAGDNN_ASCEND_STANDALONE_SHA256
#define FLAGDNN_ASCEND_STANDALONE_SHA256 ""
#endif

#include <stdexcept>
#include <string>

namespace flagdnn::ascend::detail {
namespace {

[[nodiscard]] std::string status_message(const PyStatus &status) {
  std::string result =
      status.err_msg == nullptr ? "unknown Python error" : status.err_msg;
  if (status.func != nullptr) {
    result = std::string(status.func) + ": " + result;
  }
  return result;
}

[[nodiscard]] PyThreadState *&embedded_main_thread_state() noexcept {
  static PyThreadState *state = nullptr;
  return state;
}

[[nodiscard]] bool &owns_embedded_interpreter() noexcept {
  static bool owns = false;
  return owns;
}

void hold_owned_interpreter_gil_at_process_exit() noexcept {
  if (owns_embedded_interpreter() && embedded_main_thread_state() != nullptr &&
      Py_IsInitialized() != 0) {
    /* The pinned extension has process-lifetime C++ static PyObjects. Its
     * later __cxa_atexit handlers DECREF them, so keep the GIL across only the
     * remaining process-exit handlers. */
    (void)PyGILState_Ensure();
  }
}

void register_owned_interpreter_exit_guard() {
  static bool registered = false;
  if (registered) {
    return;
  }
  if (std::atexit(&hold_owned_interpreter_gil_at_process_exit) != 0) {
    throw std::runtime_error(
        "cannot register embedded Python process-exit GIL guard");
  }
  registered = true;
}

} // namespace

void initialize_embedded_python_from_program(const char *program_name) {
  if (program_name == nullptr || program_name[0] == '\0' ||
      Py_IsInitialized() != 0) {
    throw std::invalid_argument(
        "embedded Python requires a program and a clean runtime");
  }

  PyConfig config;
  PyConfig_InitPythonConfig(&config);
  config.install_signal_handlers = 0;
  PyStatus status =
      PyConfig_SetBytesString(&config, &config.program_name, program_name);
  if (PyStatus_Exception(status)) {
    const std::string message = status_message(status);
    PyConfig_Clear(&config);
    throw std::runtime_error("cannot configure embedded Python program: " +
                             message);
  }
  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  if (PyStatus_Exception(status) || Py_IsInitialized() == 0) {
    throw std::runtime_error("cannot initialize embedded Python: " +
                             status_message(status));
  }
  owns_embedded_interpreter() = true;
}

void release_embedded_python_gil_for_process_lifetime() {
  if (!owns_embedded_interpreter() || Py_IsInitialized() == 0 ||
      embedded_main_thread_state() != nullptr) {
    throw std::runtime_error(
        "embedded Python GIL release requires one owned active interpreter");
  }

  /* Register after extension imports. std::atexit is LIFO, so the guard runs
   * before the imported extension's static PyObject destructors. */
  register_owned_interpreter_exit_guard();
  embedded_main_thread_state() = PyEval_SaveThread();
  if (embedded_main_thread_state() == nullptr) {
    throw std::runtime_error(
        "PyEval_SaveThread returned a null main thread state");
  }
}

} // namespace flagdnn::ascend::detail

#ifndef FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE
#define FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE ""
#endif

namespace flagdnn::ascend::detail {
namespace {

const int kRuntimeLayoutAnchor = 0;

[[nodiscard]] bool path_is_within(const std::filesystem::path &child,
                                  const std::filesystem::path &parent) {
  auto child_iterator = child.begin();
  for (auto parent_iterator = parent.begin(); parent_iterator != parent.end();
       ++parent_iterator, ++child_iterator) {
    if (child_iterator == child.end() || *child_iterator != *parent_iterator) {
      return false;
    }
  }
  return true;
}

} // namespace

std::optional<std::filesystem::path>
private_runtime_path(std::string_view relative) {
  if (FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE[0] == '\0' || relative.empty()) {
    return std::nullopt;
  }
  const std::filesystem::path relative_path(relative);
  if (relative_path.is_absolute() || relative_path.has_root_path()) {
    throw std::runtime_error("Ascend private runtime path must be relative");
  }

  Dl_info information{};
  if (::dladdr(&kRuntimeLayoutAnchor, &information) == 0 ||
      information.dli_fname == nullptr) {
    throw std::runtime_error("cannot locate the loaded Ascend plugin");
  }
  std::error_code error;
  const std::filesystem::path plugin =
      std::filesystem::canonical(information.dli_fname, error);
  if (error) {
    throw std::runtime_error("cannot canonicalize the loaded Ascend plugin");
  }
  const std::filesystem::path lexical_root =
      (plugin.parent_path() / FLAGDNN_ASCEND_PRIVATE_RUNTIME_RELATIVE)
          .lexically_normal();
  const bool root_exists = std::filesystem::exists(lexical_root, error);
  if (error) {
    throw std::runtime_error("cannot inspect the Ascend private runtime root");
  }
  if (!root_exists) {
    return std::nullopt;
  }
  const std::filesystem::file_status root_status =
      std::filesystem::symlink_status(lexical_root, error);
  if (error || std::filesystem::is_symlink(root_status) ||
      !std::filesystem::is_directory(root_status)) {
    throw std::runtime_error(
        "Ascend private runtime root is not a canonical directory");
  }
  const std::filesystem::path root =
      std::filesystem::canonical(lexical_root, error);
  if (error || root != lexical_root ||
      !path_is_within(root, plugin.parent_path())) {
    throw std::runtime_error(
        "Ascend private runtime root escapes the plugin directory");
  }
  const std::filesystem::path candidate =
      (root / relative_path).lexically_normal();
  if (!path_is_within(candidate, root)) {
    throw std::runtime_error("Ascend private runtime path escapes its root");
  }
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(candidate, error);
  if (error || !std::filesystem::exists(status)) {
    throw std::runtime_error("Ascend private runtime entry is missing");
  }
  if (std::filesystem::is_symlink(status) ||
      (!std::filesystem::is_regular_file(status) &&
       !std::filesystem::is_directory(status))) {
    throw std::runtime_error(
        "Ascend private runtime entry has an invalid file type");
  }
  const std::filesystem::path canonical =
      std::filesystem::canonical(candidate, error);
  if (error || !path_is_within(canonical, root)) {
    throw std::runtime_error("Ascend private runtime entry is not canonical");
  }
  return canonical;
}

} // namespace flagdnn::ascend::detail

namespace flagdnn::ascend {

namespace detail {

struct ProcessConfigurationSnapshot {
  struct EnvironmentValue {
    const char *name = nullptr;
    bool is_set = false;
    std::string value;
  };

  struct PrivateDirectory {
    std::string path;
    dev_t device = 0;
    ino_t inode = 0;
    uid_t owner = 0;
  };

  struct OwnedDirectories {
    PrivateDirectory root;

    ~OwnedDirectories() noexcept {
      if (root.path.empty()) {
        return;
      }
      struct stat status{};
      struct stat link_status{};
      if (::lstat(root.path.c_str(), &link_status) != 0 ||
          ::stat(root.path.c_str(), &status) != 0 ||
          !S_ISDIR(link_status.st_mode) || !S_ISDIR(status.st_mode) ||
          link_status.st_dev != status.st_dev ||
          link_status.st_ino != status.st_ino || status.st_dev != root.device ||
          status.st_ino != root.inode || status.st_uid != root.owner ||
          (status.st_mode & 07777) != S_IRWXU) {
        return;
      }
      std::error_code ignored;
      std::filesystem::remove_all(root.path, ignored);
    }
  };

  std::string containment_mode;
  std::vector<EnvironmentValue> environment;
  PrivateDirectory cache;
  PrivateDirectory temporary_root;
  std::shared_ptr<const OwnedDirectories> owned_directories;
};

} // namespace detail

namespace {

constexpr std::uint32_t kMaximumAiCoreCount = 65535;

namespace fs = std::filesystem;

enum class ContainmentMode {
  kTrustedLocal,
  kDevelopment,
  kHardened,
};

[[nodiscard]] constexpr const char *
containment_mode_name(ContainmentMode mode) noexcept {
  switch (mode) {
  case ContainmentMode::kTrustedLocal:
    return "trusted-local";
  case ContainmentMode::kDevelopment:
    return "development";
  case ContainmentMode::kHardened:
    return "hardened";
  }
  return "unknown";
}

static constexpr const char *kFrozenEnvironmentNames[] = {
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_AICPU_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "TRITON_JIT_BACKEND",
    "TRITON_ASCEND_ARCH",
    target_policy::kCompatibilityEnvironment,
    "TRITON_NPU_COMPILER_PATH",
    "MLIR_ROOT",
    "LLVM_ROOT",
    "CC",
    "TRITON_BACKEND",
    "TORCH_DEVICE_BACKEND_AUTOLOAD",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONHASHSEED",
    "PYTHONSAFEPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONUSERBASE",
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
    "TRITON_COMPILE_ONLY",
    "TRITON_ALWAYS_COMPILE",
    "TRITON_KERNEL_OVERRIDE",
    "TRITON_STORE_BINARY_ONLY",
    "TRITON_CACHE_MANAGER",
    "TRITON_REMOTE_CACHE_BACKEND",
    "TRITON_OVERRIDE_DIR",
    "PYTHONHOME",
    "PYTHONPATH",
    "PATH",
    "LD_LIBRARY_PATH",
    "TRITON_CACHE_DIR",
    "TMPDIR",
};

struct LoadedObject {
  fs::path path;
  std::string build_id;
};

struct PythonModulePin {
  const char *name;
  const char *path;
  const char *sha256;
};

static constexpr PythonModulePin kPythonModules[] = {
    {"torch", FLAGDNN_ASCEND_TORCH_MODULE_PATH,
     FLAGDNN_ASCEND_TORCH_MODULE_SHA256},
    {"torch_npu", FLAGDNN_ASCEND_TORCH_NPU_MODULE_PATH,
     FLAGDNN_ASCEND_TORCH_NPU_MODULE_SHA256},
    /* torch_npu must finish before Triton discovers the Ascend backend. The
     * pinned backend's benchmark helper imports torch_npu while Triton itself
     * is only partially initialized otherwise. */
    {"triton", FLAGDNN_ASCEND_TRITON_MODULE_PATH,
     FLAGDNN_ASCEND_TRITON_MODULE_SHA256},
    {"yaml", FLAGDNN_ASCEND_YAML_MODULE_PATH,
     FLAGDNN_ASCEND_YAML_MODULE_SHA256},
};

[[nodiscard]] std::string environment_value(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

[[nodiscard]] std::string
configuration_identity(const detail::ProcessConfigurationSnapshot &frozen) {
  std::ostringstream output;
  output << "containment=" << frozen.containment_mode << '\n';
  for (const auto &value : frozen.environment) {
    output << value.name << '=';
    if (!value.is_set) {
      output << "unset";
    } else {
      output << "set:" << value.value.size() << ':' << value.value;
    }
    output << '\n';
  }
  output << "configured_cann_root=" << FLAGDNN_ASCEND_CANN_ROOT << '\n'
         << "configured_cann_version=" << FLAGDNN_ASCEND_CANN_VERSION << '\n'
         << "configured_python_module_root="
         << FLAGDNN_ASCEND_PYTHON_MODULE_ROOT << '\n'
         << "configured_python_program=" << FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH
         << '\n'
         << "configured_python_program_canonical="
         << FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH << '\n'
         << "configured_python_program_sha256="
         << FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256 << '\n';
  return output.str();
}

[[nodiscard]] std::optional<ContainmentMode>
containment_mode(std::string *error) {
  const std::string configured =
      environment_value("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  if (configured.empty() || configured == "trusted-local") {
    return ContainmentMode::kTrustedLocal;
  }
  if (configured == "development") {
    return ContainmentMode::kDevelopment;
  }
  if (configured == "production" || configured == "hardened") {
    return ContainmentMode::kHardened;
  }
  if (error != nullptr) {
    *error = "FLAGDNN_ASCEND_RESOURCE_CONTAINMENT must be exactly "
             "'trusted-local', 'development', 'production', or 'hardened'";
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<fs::path>
canonical_existing_path(const std::string &value, bool directory,
                        std::string *error) {
  if (value.empty()) {
    if (error != nullptr) {
      *error = "required path is not configured";
    }
    return std::nullopt;
  }
  const fs::path path(value);
  if (!path.is_absolute()) {
    if (error != nullptr) {
      *error = "configured path is not absolute: " + value;
    }
    return std::nullopt;
  }
  std::error_code status_error;
  const bool correct_type = directory ? fs::is_directory(path, status_error)
                                      : fs::is_regular_file(path, status_error);
  if (status_error || !correct_type) {
    if (error != nullptr) {
      *error = std::string(directory ? "directory" : "file") +
               " does not exist: " + value;
    }
    return std::nullopt;
  }
  std::error_code canonical_error;
  fs::path canonical = fs::canonical(path, canonical_error);
  if (canonical_error) {
    if (error != nullptr) {
      *error = "cannot canonicalize path: " + value;
    }
    return std::nullopt;
  }
  return canonical;
}

[[nodiscard]] bool path_is_within(const fs::path &child,
                                  const fs::path &parent) {
  auto child_it = child.begin();
  for (auto parent_it = parent.begin(); parent_it != parent.end();
       ++parent_it, ++child_it) {
    if (child_it == child.end() || *child_it != *parent_it) {
      return false;
    }
  }
  return true;
}

struct LocalEnvironment {
  fs::path cache_root;
  fs::path temporary_root;
  fs::path python_module_root;
  std::shared_ptr<const detail::ProcessConfigurationSnapshot::OwnedDirectories>
      owned_directories;
};

enum class FrozenConfigurationStatus {
  kValid,
  kContainmentModeChanged,
  kEnvironmentChanged,
  kCacheChanged,
  kTemporaryRootChanged,
};

[[nodiscard]] fs::path require_canonical_private_directory(const char *name) {
  const std::string value = environment_value(name);
  std::string path_error;
  const std::optional<fs::path> canonical =
      canonical_existing_path(value, true, &path_error);
  if (!canonical.has_value()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string("invalid ") + name + ": " + path_error);
  }
  if (canonical->string() != value) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string(name) +
                          " must contain its canonical absolute path");
  }

  struct stat status{};
  struct stat link_status{};
  if (::lstat(value.c_str(), &link_status) != 0 ||
      ::stat(value.c_str(), &status) != 0 || !S_ISDIR(link_status.st_mode) ||
      !S_ISDIR(status.st_mode) || link_status.st_dev != status.st_dev ||
      link_status.st_ino != status.st_ino || status.st_uid != ::geteuid() ||
      (status.st_mode & 07777) != S_IRWXU ||
      ::access(value.c_str(), R_OK | W_OK | X_OK) != 0) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        std::string(name) +
            " must be a caller-provided, canonical, effective-UID-owned 0700 "
            "directory");
  }
  return *canonical;
}

[[nodiscard]] fs::path require_configured_directory(const char *value,
                                                    const char *description) {
  std::string path_error;
  const std::optional<fs::path> canonical =
      canonical_existing_path(value == nullptr ? "" : value, true, &path_error);
  if (!canonical.has_value()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string("invalid configured ") + description + ": " +
                          path_error);
  }
  return *canonical;
}

void freeze_exact_environment(const char *name, std::string_view expected) {
  const char *current = std::getenv(name);
  if (current != nullptr && std::string_view(current) != expected) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string(name) +
                          " conflicts with the pinned Ascend environment");
  }
  if (current == nullptr &&
      ::setenv(name, std::string(expected).c_str(), 1) != 0) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      std::string("cannot freeze ") + name + ": " +
                          std::strerror(errno));
  }
}

void freeze_canonical_environment_path(const char *name,
                                       const fs::path &expected) {
  const char *current = std::getenv(name);
  if (current != nullptr) {
    std::string path_error;
    const std::optional<fs::path> canonical =
        canonical_existing_path(current, true, &path_error);
    if (!canonical.has_value() || *canonical != expected) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        std::string(name) +
                            " differs from the configured CANN root");
    }
  }
  if ((current == nullptr || std::string_view(current) != expected.string()) &&
      ::setenv(name, expected.c_str(), 1) != 0) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      std::string("cannot freeze ") + name + ": " +
                          std::strerror(errno));
  }
}

void require_unset_environment(const char *name) {
  if (std::getenv(name) != nullptr) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string(name) +
                          " must be unset for embedded Ascend compilation");
  }
}

void configure_exact_local_environment(ContainmentMode mode, const char *name,
                                       std::string_view expected) {
  if (mode == ContainmentMode::kDevelopment) {
    freeze_exact_environment(name, expected);
    return;
  }
  if (::setenv(name, std::string(expected).c_str(), 1) != 0) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      std::string("cannot configure trusted-local ") + name +
                          ": " + std::strerror(errno));
  }
}

void configure_unset_local_environment(ContainmentMode mode, const char *name) {
  if (mode == ContainmentMode::kDevelopment) {
    require_unset_environment(name);
    return;
  }
  if (::unsetenv(name) != 0) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      std::string("cannot sanitize trusted-local ") + name +
                          ": " + std::strerror(errno));
  }
}

[[nodiscard]] detail::ProcessConfigurationSnapshot::PrivateDirectory
freeze_private_directory(const fs::path &path, const char *description) {
  detail::ProcessConfigurationSnapshot::PrivateDirectory result;
  result.path = path.string();

  struct stat status{};
  struct stat link_status{};
  if (::lstat(result.path.c_str(), &link_status) != 0 ||
      ::stat(result.path.c_str(), &status) != 0 ||
      !S_ISDIR(link_status.st_mode) || !S_ISDIR(status.st_mode) ||
      link_status.st_dev != status.st_dev ||
      link_status.st_ino != status.st_ino || status.st_uid != ::geteuid() ||
      (status.st_mode & 07777) != S_IRWXU ||
      ::access(result.path.c_str(), R_OK | W_OK | X_OK) != 0) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        std::string(description) +
            " changed while freezing the Ascend process configuration");
  }
  result.device = status.st_dev;
  result.inode = status.st_ino;
  result.owner = status.st_uid;
  return result;
}

[[nodiscard]] LocalEnvironment create_trusted_local_directories() {
  std::string path_error;
  const std::optional<fs::path> system_temporary =
      canonical_existing_path("/tmp", true, &path_error);
  if (!system_temporary.has_value()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "trusted-local Ascend runtime cannot use /tmp: " +
                          path_error);
  }

  std::string pattern =
      (*system_temporary /
       ("flagdnn-ascend-" + std::to_string(::geteuid()) + "-XXXXXX"))
          .string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  char *created = ::mkdtemp(writable.data());
  if (created == nullptr) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "cannot create trusted-local Ascend process root: " +
                          std::string(std::strerror(errno)));
  }

  const fs::path root(created);
  try {
    auto owner = std::make_shared<
        detail::ProcessConfigurationSnapshot::OwnedDirectories>();
    owner->root =
        freeze_private_directory(root, "trusted-local Ascend process root");
    const fs::path cache = root / "cache";
    const fs::path temporary = root / "tmp";
    if (::mkdir(cache.c_str(), 0700) != 0 ||
        ::mkdir(temporary.c_str(), 0700) != 0 ||
        ::chmod(cache.c_str(), 0700) != 0 ||
        ::chmod(temporary.c_str(), 0700) != 0) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                        "cannot create trusted-local Ascend cache/tmp: " +
                            std::string(std::strerror(errno)));
    }

    LocalEnvironment result;
    result.cache_root =
        freeze_private_directory(cache, "trusted-local TRITON_CACHE_DIR").path;
    result.temporary_root =
        freeze_private_directory(temporary, "trusted-local TMPDIR").path;
    result.owned_directories = std::move(owner);
    return result;
  } catch (...) {
    std::error_code ignored;
    fs::remove_all(root, ignored);
    throw;
  }
}

void replace_environment_path(const char *name, const fs::path &value) {
  if (::setenv(name, value.c_str(), 1) != 0) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      std::string("cannot set trusted-local ") + name + ": " +
                          std::strerror(errno));
  }
}

[[nodiscard]] LocalEnvironment
prepare_local_environment(ContainmentMode mode,
                          const target_policy::Resolution &target) {
  LocalEnvironment result;
  if (mode == ContainmentMode::kDevelopment) {
    result.cache_root = require_canonical_private_directory("TRITON_CACHE_DIR");
    result.temporary_root = require_canonical_private_directory("TMPDIR");
  } else if (mode == ContainmentMode::kTrustedLocal) {
    result = create_trusted_local_directories();
    replace_environment_path("TRITON_CACHE_DIR", result.cache_root);
    replace_environment_path("TMPDIR", result.temporary_root);
    if (::setenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT",
                 containment_mode_name(mode), 1) != 0) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "cannot freeze trusted-local Ascend containment mode: " +
              std::string(std::strerror(errno)));
    }
  } else {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "hardened Ascend environment preparation is disabled");
  }
  if (result.cache_root == result.temporary_root) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "TRITON_CACHE_DIR and TMPDIR must be distinct private "
                      "directories");
  }

  result.python_module_root = require_configured_directory(
      FLAGDNN_ASCEND_PYTHON_MODULE_ROOT, "Python module root");
  const fs::path cann_root =
      require_configured_directory(FLAGDNN_ASCEND_CANN_ROOT, "CANN root");

  freeze_canonical_environment_path("ASCEND_HOME_PATH", cann_root);
  freeze_canonical_environment_path("ASCEND_TOOLKIT_HOME", cann_root);
  configure_exact_local_environment(mode, "TRITON_JIT_BACKEND", "NPU");
  if (target.detects_codegen_arch_from_runtime) {
    configure_unset_local_environment(mode, "TRITON_ASCEND_ARCH");
  } else {
    configure_exact_local_environment(mode, "TRITON_ASCEND_ARCH",
                                      target.codegen_arch);
  }
  configure_exact_local_environment(mode, "TRITON_BACKEND", "torch_npu");
  configure_exact_local_environment(mode, "TRITON_ALL_BLOCKS_PARALLEL",
                                    "false");
  configure_exact_local_environment(mode, "TORCH_DEVICE_BACKEND_AUTOLOAD", "0");
  configure_exact_local_environment(mode, "PYTHONDONTWRITEBYTECODE", "1");
  configure_exact_local_environment(mode, "PYTHONNOUSERSITE", "1");
  configure_exact_local_environment(mode, "PYTHONHASHSEED", "0");
  configure_exact_local_environment(mode, "PYTHONSAFEPATH", "1");
  configure_exact_local_environment(mode, "PYTHONPATH",
                                    result.python_module_root.string());

  for (const char *name : {
           "PYTHONHOME",
           "PYTHONPYCACHEPREFIX",
           "PYTHONUSERBASE",
           "TRITON_COMPILE_ONLY",
           "TRITON_ALWAYS_COMPILE",
           "TRITON_KERNEL_OVERRIDE",
           "TRITON_STORE_BINARY_ONLY",
           "TRITON_CACHE_MANAGER",
           "TRITON_REMOTE_CACHE_BACKEND",
           "TRITON_OVERRIDE_DIR",
       }) {
    configure_unset_local_environment(mode, name);
  }
  return result;
}

[[nodiscard]] std::shared_ptr<const detail::ProcessConfigurationSnapshot>
freeze_process_configuration(ContainmentMode mode,
                             const LocalEnvironment &environment) {
  auto result = std::make_shared<detail::ProcessConfigurationSnapshot>();
  result->containment_mode = containment_mode_name(mode);
  result->environment.reserve(sizeof(kFrozenEnvironmentNames) /
                              sizeof(kFrozenEnvironmentNames[0]));
  for (const char *name : kFrozenEnvironmentNames) {
    const char *value = std::getenv(name);
    detail::ProcessConfigurationSnapshot::EnvironmentValue frozen;
    frozen.name = name;
    frozen.is_set = value != nullptr;
    if (value != nullptr) {
      frozen.value = value;
    }
    result->environment.push_back(std::move(frozen));
  }
  result->cache =
      freeze_private_directory(environment.cache_root, "TRITON_CACHE_DIR");
  result->temporary_root =
      freeze_private_directory(environment.temporary_root, "TMPDIR");
  result->owned_directories = environment.owned_directories;
  return result;
}

[[nodiscard]] bool private_directory_matches(
    const detail::ProcessConfigurationSnapshot::PrivateDirectory
        &frozen) noexcept {
  struct stat status{};
  struct stat link_status{};
  return ::geteuid() == frozen.owner &&
         ::lstat(frozen.path.c_str(), &link_status) == 0 &&
         ::stat(frozen.path.c_str(), &status) == 0 &&
         S_ISDIR(link_status.st_mode) && S_ISDIR(status.st_mode) &&
         link_status.st_dev == status.st_dev &&
         link_status.st_ino == status.st_ino &&
         status.st_dev == frozen.device && status.st_ino == frozen.inode &&
         status.st_uid == frozen.owner && (status.st_mode & 07777) == S_IRWXU &&
         ::access(frozen.path.c_str(), R_OK | W_OK | X_OK) == 0;
}

[[nodiscard]] FrozenConfigurationStatus frozen_configuration_status(
    const detail::ProcessConfigurationSnapshot &frozen) noexcept {
  const char *mode = std::getenv("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT");
  if (mode == nullptr ||
      std::strcmp(mode, frozen.containment_mode.c_str()) != 0) {
    return FrozenConfigurationStatus::kContainmentModeChanged;
  }

  for (const auto &expected : frozen.environment) {
    const char *current = std::getenv(expected.name);
    if ((current != nullptr) != expected.is_set ||
        (current != nullptr &&
         std::strcmp(current, expected.value.c_str()) != 0)) {
      return FrozenConfigurationStatus::kEnvironmentChanged;
    }
  }
  if (!private_directory_matches(frozen.cache)) {
    return FrozenConfigurationStatus::kCacheChanged;
  }
  if (!private_directory_matches(frozen.temporary_root)) {
    return FrozenConfigurationStatus::kTemporaryRootChanged;
  }
  return FrozenConfigurationStatus::kValid;
}

[[nodiscard]] const char *
frozen_configuration_error(FrozenConfigurationStatus status) noexcept {
  switch (status) {
  case FrozenConfigurationStatus::kValid:
    return nullptr;
  case FrozenConfigurationStatus::kContainmentModeChanged:
  case FrozenConfigurationStatus::kEnvironmentChanged:
    return "Ascend process configuration changed after the domain was bound";
  case FrozenConfigurationStatus::kCacheChanged:
    return "Ascend Triton cache root changed after domain binding";
  case FrozenConfigurationStatus::kTemporaryRootChanged:
    return "Ascend temporary root changed after domain binding";
  }
  return "Ascend process configuration validation returned an unknown state";
}

[[nodiscard]] std::size_t align_note(std::size_t size) {
  constexpr std::size_t kAlignment = 4;
  return (size + kAlignment - 1) & ~(kAlignment - 1);
}

[[nodiscard]] std::string loaded_build_id(const dl_phdr_info &information) {
  for (std::size_t index = 0; index < information.dlpi_phnum; ++index) {
    const ElfW(Phdr) &header = information.dlpi_phdr[index];
    if (header.p_type != PT_NOTE || header.p_memsz < sizeof(ElfW(Nhdr))) {
      continue;
    }
    const auto *cursor = reinterpret_cast<const std::byte *>(
        information.dlpi_addr + header.p_vaddr);
    const auto *end = cursor + header.p_memsz;
    while (static_cast<std::size_t>(end - cursor) >= sizeof(ElfW(Nhdr))) {
      const auto *note = reinterpret_cast<const ElfW(Nhdr) *>(cursor);
      cursor += sizeof(ElfW(Nhdr));
      const std::size_t name_size = align_note(note->n_namesz);
      const std::size_t description_size = align_note(note->n_descsz);
      if (name_size > static_cast<std::size_t>(end - cursor)) {
        break;
      }
      const auto *name = reinterpret_cast<const char *>(cursor);
      cursor += name_size;
      if (description_size > static_cast<std::size_t>(end - cursor)) {
        break;
      }
      const auto *description = cursor;
      cursor += description_size;
      if (note->n_type != NT_GNU_BUILD_ID || note->n_namesz < 3 ||
          std::string_view(name, 3) != "GNU") {
        continue;
      }
      std::ostringstream output;
      output << std::hex << std::setfill('0');
      for (std::size_t byte = 0; byte < note->n_descsz; ++byte) {
        output << std::setw(2)
               << std::to_integer<unsigned int>(description[byte]);
      }
      return output.str();
    }
  }
  return {};
}

struct LoadedObjectQuery {
  std::string basename;
  std::vector<LoadedObject> matches;
};

struct LoadedPythonQuery {
  std::vector<LoadedObject> matches;
};

int collect_loaded_object(dl_phdr_info *information, std::size_t,
                          void *opaque) {
  auto &query = *static_cast<LoadedObjectQuery *>(opaque);
  if (information == nullptr || information->dlpi_name == nullptr ||
      information->dlpi_name[0] == '\0') {
    return 0;
  }
  std::error_code error;
  const fs::path candidate = fs::canonical(information->dlpi_name, error);
  if (error || candidate.filename() != query.basename) {
    return 0;
  }
  const auto duplicate = std::find_if(
      query.matches.begin(), query.matches.end(),
      [&](const LoadedObject &item) { return item.path == candidate; });
  if (duplicate == query.matches.end()) {
    query.matches.push_back({candidate, loaded_build_id(*information)});
  }
  return 0;
}

int collect_loaded_python_object(dl_phdr_info *information, std::size_t,
                                 void *opaque) {
  auto &query = *static_cast<LoadedPythonQuery *>(opaque);
  if (information == nullptr || information->dlpi_name == nullptr ||
      information->dlpi_name[0] == '\0') {
    return 0;
  }
  std::error_code error;
  const fs::path candidate = fs::canonical(information->dlpi_name, error);
  if (error) {
    return 0;
  }
  const std::string filename = candidate.filename().string();
  if (filename.rfind("libpython", 0) != 0 ||
      filename.find(".so") == std::string::npos) {
    return 0;
  }
  const auto duplicate = std::find_if(
      query.matches.begin(), query.matches.end(),
      [&](const LoadedObject &item) { return item.path == candidate; });
  if (duplicate == query.matches.end()) {
    query.matches.push_back({candidate, loaded_build_id(*information)});
  }
  return 0;
}

[[nodiscard]] std::vector<LoadedObject> loaded_python_objects() {
  LoadedPythonQuery query;
  (void)::dl_iterate_phdr(&collect_loaded_python_object, &query);
  return query.matches;
}

[[nodiscard]] LoadedObject
unique_loaded_python_object(const LoadedObject &symbol_object) {
  std::vector<LoadedObject> matches = loaded_python_objects();
  if (matches.size() != 1 || matches.front().path != symbol_object.path) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "expected exactly one canonical loaded libpython object matching the "
        "CPython API, found " +
            std::to_string(matches.size()));
  }
  return matches.front();
}

[[nodiscard]] LoadedObject unique_loaded_object(const std::string &basename) {
  LoadedObjectQuery query{basename, {}};
  (void)::dl_iterate_phdr(&collect_loaded_object, &query);
  if (query.matches.size() != 1) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "expected exactly one loaded " + basename + ", found " +
                          std::to_string(query.matches.size()));
  }
  return query.matches.front();
}

template <typename Function>
[[nodiscard]] LoadedObject object_containing(Function function,
                                             const char *description) {
  Dl_info information{};
  const auto address =
      reinterpret_cast<void *>(reinterpret_cast<std::uintptr_t>(function));
  if (::dladdr(address, &information) == 0 ||
      information.dli_fname == nullptr) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "cannot locate loaded " + std::string(description));
  }
  std::error_code error;
  const fs::path canonical = fs::canonical(information.dli_fname, error);
  if (error) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "cannot canonicalize loaded " + std::string(description));
  }
  LoadedObjectQuery query{canonical.filename().string(), {}};
  (void)::dl_iterate_phdr(&collect_loaded_object, &query);
  for (const LoadedObject &object : query.matches) {
    if (object.path == canonical) {
      return object;
    }
  }
  return {canonical, {}};
}

void verify_object(const LoadedObject &object, const char *configured_path,
                   const char *configured_sha256, const char *description,
                   std::string_view private_relative = {}) {
  std::error_code error;
  fs::path expected;
  if (!private_relative.empty()) {
    try {
      const std::optional<fs::path> private_path =
          detail::private_runtime_path(private_relative);
      if (private_path.has_value()) {
        expected = *private_path;
      }
    } catch (const std::exception &exception) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        std::string("invalid private ") + description + ": " +
                            exception.what());
    }
  }
  if (expected.empty()) {
    expected = fs::canonical(configured_path, error);
  }
  if (error || expected != object.path) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        std::string("loaded ") + description +
            " does not match its configured or plugin-private file: " +
            object.path.string());
  }
  const std::string digest = flagdnn::native::sha256_file(object.path);
  if (configured_sha256 == nullptr || configured_sha256[0] == '\0' ||
      digest != configured_sha256) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string("loaded ") + description +
                          " content hash differs from configure time");
  }
}

[[nodiscard]] std::string
loaded_object_identity(const LoadedObject &object,
                       const char *configured_sha256) {
  if (!object.build_id.empty()) {
    return "build-id:" + object.build_id;
  }
  /* Some CANN 9 runtime DSOs do not carry an NT_GNU_BUILD_ID note. The
   * configure-time and loaded-file SHA-256 equality checked above remains a
   * complete content identity and matches the validation-side fallback. */
  return "sha256:" + std::string(configured_sha256);
}

[[nodiscard]] fs::path
verify_configured_file(const char *configured_path,
                       const char *configured_sha256, const char *description,
                       std::string_view private_relative = {}) {
  if (!private_relative.empty()) {
    try {
      const std::optional<fs::path> private_path =
          detail::private_runtime_path(private_relative);
      if (private_path.has_value()) {
        std::error_code error;
        if (!fs::is_regular_file(*private_path, error) || error) {
          throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                            std::string("plugin-private ") + description +
                                " is not a regular file");
        }
        const std::string digest = flagdnn::native::sha256_file(*private_path);
        if (configured_sha256 == nullptr || configured_sha256[0] == '\0' ||
            digest != configured_sha256) {
          throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                            std::string("plugin-private ") + description +
                                " content hash differs from configure time");
        }
        return *private_path;
      }
    } catch (const AscendError &) {
      throw;
    } catch (const std::exception &exception) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        std::string("invalid plugin-private ") + description +
                            ": " + exception.what());
    }
  }
  std::string path_error;
  const std::optional<fs::path> path = canonical_existing_path(
      configured_path == nullptr ? "" : configured_path, false, &path_error);
  if (!path.has_value()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string("invalid configured ") + description + ": " +
                          path_error);
  }
  const std::string digest = flagdnn::native::sha256_file(*path);
  if (configured_sha256 == nullptr || configured_sha256[0] == '\0' ||
      digest != configured_sha256) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      std::string(description) +
                          " content hash differs from configure time");
  }
  return *path;
}

void verify_pinned_python_module_files() {
  for (const PythonModulePin &module : kPythonModules) {
    const std::string description = std::string(module.name) + " module";
    (void)verify_configured_file(module.path, module.sha256,
                                 description.c_str());
  }
}

[[nodiscard]] void *&process_python_global_handle() noexcept {
  static void *handle = nullptr;
  return handle;
}

void promote_loaded_python(const LoadedObject &python) {
  if (process_python_global_handle() != nullptr) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                      "libpython global promotion ran more than once");
  }

  (void)::dlerror();
  void *handle =
      ::dlopen(python.path.c_str(), RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
  if (handle == nullptr) {
    const char *error = ::dlerror();
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "cannot promote the pinned loaded libpython into the global symbol "
        "scope: " +
            std::string(error == nullptr ? "unknown dlopen failure" : error));
  }
  /* Even a later readback failure has changed process-global loader state and
   * makes the domain terminal. Retain the reference on every successful
   * dlopen path; never attempt to demote libpython with dlclose. */
  process_python_global_handle() = handle;

  (void)::dlerror();
  void *symbol = ::dlsym(handle, "Py_IsInitialized");
  const char *symbol_error = ::dlerror();
  Dl_info information{};
  std::error_code canonical_error;
  fs::path symbol_object;
  if (symbol != nullptr && symbol_error == nullptr &&
      ::dladdr(symbol, &information) != 0 && information.dli_fname != nullptr) {
    symbol_object = fs::canonical(information.dli_fname, canonical_error);
  }
  if (symbol == nullptr || symbol_error != nullptr || canonical_error ||
      symbol_object != python.path) {
    const std::string diagnostic = symbol_error == nullptr
                                       ? "symbol resolved from the wrong object"
                                       : std::string(symbol_error);
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "pinned libpython global promotion readback failed: " +
                          diagnostic);
  }

  (void)::dlerror();
  void *global_symbol = ::dlsym(RTLD_DEFAULT, "PyTuple_Type");
  const char *global_error = ::dlerror();
  Dl_info global_information{};
  std::error_code global_canonical_error;
  fs::path global_object;
  if (global_symbol != nullptr && global_error == nullptr &&
      ::dladdr(global_symbol, &global_information) != 0 &&
      global_information.dli_fname != nullptr) {
    global_object =
        fs::canonical(global_information.dli_fname, global_canonical_error);
  }
  if (global_symbol == nullptr || global_error != nullptr ||
      global_canonical_error || global_object != python.path) {
    const std::string diagnostic =
        global_error == nullptr ? "global symbol resolved from the wrong object"
                                : std::string(global_error);
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "pinned libpython is not globally visible after promotion: " +
            diagnostic);
  }

  /* The extra NOLOAD reference is deliberately process-lifetime. CPython and
   * extension modules may continue to resolve its symbols after all FlagDNN
   * handles have been destroyed, so this handle must never be dlclose'd. */
}

struct PyObjectDeleter {
  void operator()(PyObject *object) const noexcept { Py_XDECREF(object); }
};

using OwnedPyObject = std::unique_ptr<PyObject, PyObjectDeleter>;

[[nodiscard]] std::string consume_python_error() {
  if (PyErr_Occurred() == nullptr) {
    return "unknown Python error";
  }
  PyObject *type_raw = nullptr;
  PyObject *value_raw = nullptr;
  PyObject *traceback_raw = nullptr;
  PyErr_Fetch(&type_raw, &value_raw, &traceback_raw);
  PyErr_NormalizeException(&type_raw, &value_raw, &traceback_raw);
  OwnedPyObject type(type_raw);
  OwnedPyObject value(value_raw);
  OwnedPyObject traceback(traceback_raw);

  PyObject *source = value != nullptr ? value.get() : type.get();
  if (source == nullptr) {
    PyErr_Clear();
    return "unknown Python exception";
  }
  OwnedPyObject rendered(PyObject_Str(source));
  if (rendered == nullptr) {
    PyErr_Clear();
    return "unprintable Python exception";
  }
  const char *text = PyUnicode_AsUTF8(rendered.get());
  if (text == nullptr) {
    PyErr_Clear();
    return "non-UTF-8 Python exception";
  }
  std::string result(text);
  PyErr_Clear();
  return result;
}

[[nodiscard]] std::string python_utf8(PyObject *object,
                                      const char *description) {
  if (object == nullptr) {
    throw std::runtime_error(std::string("missing ") + description + ": " +
                             consume_python_error());
  }
  OwnedPyObject rendered;
  PyObject *text_object = object;
  if (!PyUnicode_Check(object)) {
    rendered.reset(PyObject_Str(object));
    if (rendered == nullptr) {
      throw std::runtime_error(std::string("cannot render ") + description +
                               ": " + consume_python_error());
    }
    text_object = rendered.get();
  }
  const char *text = PyUnicode_AsUTF8(text_object);
  if (text == nullptr) {
    throw std::runtime_error(std::string("cannot decode ") + description +
                             ": " + consume_python_error());
  }
  return text;
}

[[nodiscard]] fs::path python_import_root(const fs::path &module_file) {
  PyObject *sys_path = PySys_GetObject("path"); // Borrowed reference.
  if (sys_path == nullptr || !PyList_Check(sys_path)) {
    throw std::runtime_error("embedded Python sys.path is unavailable");
  }
  fs::path best;
  const Py_ssize_t count = PyList_Size(sys_path);
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObject *entry = PyList_GetItem(sys_path, index); // Borrowed reference.
    if (entry == nullptr || !PyUnicode_Check(entry)) {
      continue;
    }
    const char *value = PyUnicode_AsUTF8(entry);
    if (value == nullptr) {
      throw std::runtime_error("cannot decode embedded Python sys.path: " +
                               consume_python_error());
    }
    if (value[0] == '\0') {
      throw std::runtime_error(
          "embedded Python sys.path contains an unsafe empty entry");
    }
    std::error_code error;
    const fs::path canonical = fs::canonical(value, error);
    if (!error && fs::is_directory(canonical, error) && !error &&
        path_is_within(module_file, canonical) &&
        canonical.string().size() > best.string().size()) {
      best = canonical;
    }
  }
  if (best.empty()) {
    throw std::runtime_error(
        "Python module was loaded outside canonical embedded sys.path");
  }
  return best;
}

[[nodiscard]] std::string read_python_module(const char *name,
                                             const char *configured_path,
                                             const char *configured_sha256) {
  OwnedPyObject module(PyImport_ImportModule(name));
  if (module == nullptr) {
    throw std::runtime_error(std::string("cannot import ") + name + ": " +
                             consume_python_error());
  }
  OwnedPyObject file(PyObject_GetAttrString(module.get(), "__file__"));
  const std::string file_value =
      python_utf8(file.get(), (std::string(name) + ".__file__").c_str());
  std::error_code error;
  const fs::path canonical_file = fs::canonical(file_value, error);
  if (error || !fs::is_regular_file(canonical_file, error) || error) {
    throw std::runtime_error(std::string(name) +
                             " has no canonical regular module file");
  }
  const std::string description = std::string(name) + " module";
  const fs::path expected_file = verify_configured_file(
      configured_path, configured_sha256, description.c_str());
  if (canonical_file != expected_file) {
    throw std::runtime_error(std::string(name) +
                             " was not imported from its configured module "
                             "path");
  }
  const fs::path import_root = python_import_root(canonical_file);

  OwnedPyObject version(PyObject_GetAttrString(module.get(), "__version__"));
  const std::string version_value =
      python_utf8(version.get(), (std::string(name) + ".__version__").c_str());
  std::ostringstream identity;
  identity << name << "_root=" << import_root.string() << ';' << name
           << "_path=" << canonical_file.string() << ';' << name
           << "_sha256=" << configured_sha256 << ';' << name
           << "_version=" << version_value;
  return identity.str();
}

void verify_python_module_root(const fs::path &module_root) {
  OwnedPyObject sysconfig(PyImport_ImportModule("sysconfig"));
  if (sysconfig == nullptr) {
    throw std::runtime_error("cannot import sysconfig: " +
                             consume_python_error());
  }
  OwnedPyObject get_paths(PyObject_GetAttrString(sysconfig.get(), "get_paths"));
  if (get_paths == nullptr || !PyCallable_Check(get_paths.get())) {
    throw std::runtime_error("sysconfig.get_paths is unavailable");
  }
  OwnedPyObject paths(PyObject_CallNoArgs(get_paths.get()));
  if (paths == nullptr || !PyDict_Check(paths.get())) {
    throw std::runtime_error("sysconfig.get_paths failed: " +
                             consume_python_error());
  }
  PyObject *purelib = PyDict_GetItemString(paths.get(), "purelib");
  const std::string purelib_value = python_utf8(purelib, "sysconfig purelib");
  std::error_code error;
  const fs::path canonical_purelib = fs::canonical(purelib_value, error);
  if (error || canonical_purelib != module_root) {
    throw std::runtime_error(
        "embedded Python default purelib differs from the configured module "
        "root");
  }

  PyObject *sys_path = PySys_GetObject("path"); // Borrowed reference.
  if (sys_path == nullptr || !PyList_Check(sys_path)) {
    throw std::runtime_error("embedded Python sys.path is unavailable");
  }
  bool found = false;
  const Py_ssize_t count = PyList_Size(sys_path);
  for (Py_ssize_t index = 0; index < count; ++index) {
    PyObject *entry = PyList_GetItem(sys_path, index); // Borrowed reference.
    if (entry == nullptr || !PyUnicode_Check(entry)) {
      continue;
    }
    const char *value = PyUnicode_AsUTF8(entry);
    if (value == nullptr) {
      throw std::runtime_error("cannot decode embedded Python sys.path: " +
                               consume_python_error());
    }
    std::error_code path_error;
    const fs::path canonical = fs::canonical(value, path_error);
    if (!path_error && canonical == module_root) {
      found = true;
    }
  }
  if (!found) {
    throw std::runtime_error(
        "configured Python module root is absent from embedded sys.path");
  }
}

[[nodiscard]] std::string
readback_python_environment(const fs::path &module_root) {
  verify_python_module_root(module_root);
  std::ostringstream identity;
  bool first = true;
  for (const PythonModulePin &module : kPythonModules) {
    if (!first) {
      identity << ';';
    }
    first = false;
    identity << read_python_module(module.name, module.path, module.sha256);
  }
  return identity.str();
}

[[nodiscard]] std::string
initialize_embedded_python(const fs::path &module_root,
                           const char *program_name) {
  if (Py_IsInitialized() != 0) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Ascend requires a clean parent with no initialized CPython runtime");
  }

  try {
    detail::initialize_embedded_python_from_program(program_name);
  } catch (const std::exception &error) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "embedded Python initialization failed: " +
                          std::string(error.what()));
  }

  std::string identity;
  std::string failure;
  try {
    identity = readback_python_environment(module_root);
    if (PyErr_Occurred() != nullptr) {
      failure = "embedded Python left an exception pending: " +
                consume_python_error();
    }
  } catch (const std::exception &error) {
    failure = error.what();
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
    }
  } catch (...) {
    failure = "unknown embedded Python dependency readback failure";
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
    }
  }

  /* No owned PyObject crosses this point. The initial GIL is released exactly
   * once and the saved main thread state remains process-lifetime state. */
  try {
    detail::release_embedded_python_gil_for_process_lifetime();
  } catch (const std::exception &error) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                      "cannot release embedded Python GIL: " +
                          std::string(error.what()));
  }
  if (!failure.empty()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "embedded Python dependency validation failed: " +
                          failure);
  }
  return identity;
}

[[nodiscard]] std::string
sanitize_fingerprint_component(std::string_view value) {
  std::string result;
  result.reserve(value.size());
  for (unsigned char character : value) {
    if (std::isalnum(character) != 0 || character == '.' || character == '_' ||
        character == '-') {
      result.push_back(static_cast<char>(character));
    } else {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        "Ascend runtime identity is not file-system safe");
    }
  }
  if (result.empty()) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                      "aclrtGetSocName returned an empty SoC identity");
  }
  return result;
}

[[nodiscard]] std::string acl_failure(const char *operation, aclError result) {
  std::ostringstream output;
  output << operation << " failed (aclError " << result << ')';
  return output.str();
}

[[nodiscard]] detail::DomainInitialization
initialize_domain_impl(std::int32_t device_ordinal,
                       const detail::AclRuntimeApi &api) {
  std::string mode_error;
  const std::optional<ContainmentMode> mode = containment_mode(&mode_error);
  if (!mode.has_value()) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED, std::move(mode_error));
  }

  std::lock_guard<std::mutex> ltj_lock(process_ltj_mutex());
  if (detail::process_domain().terminal_failure_latched()) {
    return detail::DomainInitialization::terminal(
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
        "Ascend process domain became terminal before initialization");
  }

  if (*mode == ContainmentMode::kHardened) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "Ascend production/hardened containment is not implemented; use "
        "trusted-local or development mode");
  }

  if (api.get_current_context == nullptr || api.get_device == nullptr ||
      api.get_ai_core_count == nullptr || api.get_soc_name == nullptr ||
      api.get_version_string == nullptr || api.get_version_number == nullptr) {
    return detail::DomainInitialization::terminal(
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
        "Ascend runtime probe API is incomplete");
  }

  aclrtContext current = nullptr;
  const aclError context_result = api.get_current_context(&current);
  if (context_result != ACL_SUCCESS) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        acl_failure("aclrtGetCurrentContext", context_result) +
            "; caller must call aclInit and aclrtSetDevice first");
  }
  if (current == nullptr) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "caller has no current Ascend default context; call aclInit and "
        "aclrtSetDevice before creating the handle");
  }

  std::int32_t current_device = -1;
  const aclError device_result = api.get_device(&current_device);
  if (device_result != ACL_SUCCESS) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        acl_failure("aclrtGetDevice", device_result));
  }
  if (current_device != device_ordinal) {
    return detail::DomainInitialization::retryable(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "current Ascend device does not match requested device ordinal");
  }

  std::uint32_t ai_core_count = 0;
  try {
    const char *soc_name = api.get_soc_name();
    if (soc_name == nullptr || soc_name[0] == '\0') {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        "aclrtGetSocName did not return a SoC identity");
    }
    const std::string soc = sanitize_fingerprint_component(soc_name);
    const std::string compatibility_setting =
        environment_value(target_policy::kCompatibilityEnvironment);
    const target_policy::Resolution target =
        target_policy::resolve_codegen_arch(soc, compatibility_setting);
    if (target.status != target_policy::ResolutionStatus::kSupported) {
      std::string message;
      if (target.status ==
          target_policy::ResolutionStatus::kInvalidCompatibilitySetting) {
        message = std::string(target_policy::kCompatibilityEnvironment) +
                  " must be unset, '0', or '1', and '1' is valid only on " +
                  std::string(target_policy::kCompatibilityRuntimeSoc);
      } else {
        message =
            "aclrtGetSocName returned an unsupported code-generation class: " +
            soc;
      }
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        std::move(message));
    }
    const std::string codegen_arch(target.codegen_arch);
    const std::int32_t ai_core_result = api.get_ai_core_count(&ai_core_count);
    if (ai_core_result != RT_ERROR_NONE) {
      std::ostringstream message;
      message << "rtGetAiCoreCount failed (rtError " << ai_core_result << ')';
      return detail::DomainInitialization::retryable(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED, message.str());
    }
    if (ai_core_count == 0 || ai_core_count > kMaximumAiCoreCount) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "rtGetAiCoreCount returned a value outside the Ascend capability "
          "envelope");
    }
    configure_exact_local_environment(*mode, "TRITON_ALL_BLOCKS_PARALLEL",
                                      "false");
    if (Py_IsInitialized() != 0) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "Ascend requires a clean parent with no initialized CPython runtime");
    }

    std::array<char, ACL_PKG_VERSION_MAX_SIZE> package_version{};
    std::array<char, 8> package_name = {'r', 'u', 'n', 't', 'i', 'm', 'e', 0};
    check_acl(
        api.get_version_string(package_name.data(), package_version.data()),
        "aclsysGetVersionStr(runtime)");
    std::int32_t package_version_number = 0;
    check_acl(
        api.get_version_number(package_name.data(), &package_version_number),
        "aclsysGetVersionNum(runtime)");
    if (std::string_view(package_version.data()) !=
        FLAGDNN_ASCEND_CANN_VERSION) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        "loaded CANN runtime version '" +
                            std::string(package_version.data()) +
                            "' differs from configured version '" +
                            std::string(FLAGDNN_ASCEND_CANN_VERSION) + "'");
    }

    const LoadedObject ascendcl =
        object_containing(api.get_current_context, "libascendcl.so");
    const LoadedObject runtime = unique_loaded_object(
        fs::path(FLAGDNN_ASCEND_RUNTIME_PATH).filename().string());
    verify_object(ascendcl, FLAGDNN_ASCEND_ASCENDCL_PATH,
                  FLAGDNN_ASCEND_ASCENDCL_SHA256, "libascendcl.so");
    verify_object(runtime, FLAGDNN_ASCEND_RUNTIME_PATH,
                  FLAGDNN_ASCEND_RUNTIME_SHA256, "libruntime.so");

    const LoadedObject triton_jit_symbol = object_containing(
        &triton_jit::load_npu_metadata, "libtriton_jit public metadata loader");
    const LoadedObject triton_jit = unique_loaded_object(
        fs::path(FLAGDNN_ASCEND_LIBTRITON_JIT_PATH).filename().string());
    if (triton_jit_symbol.path != triton_jit.path) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "libtriton_jit public metadata loader resolves from another DSO");
    }
    verify_object(triton_jit, FLAGDNN_ASCEND_LIBTRITON_JIT_PATH,
                  FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256, "libtriton_jit.so",
                  "libtriton_jit.so");
    const LoadedObject python_symbol =
        object_containing(&Py_IsInitialized, "libpython");
    const LoadedObject python = unique_loaded_python_object(python_symbol);
    verify_object(python, FLAGDNN_ASCEND_PYTHON_LIBRARY_PATH,
                  FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256, "libpython");
    const fs::path standalone = verify_configured_file(
        FLAGDNN_ASCEND_STANDALONE_PATH, FLAGDNN_ASCEND_STANDALONE_SHA256,
        "standalone_compile.py", "standalone_compile.py");
    const fs::path python_program = verify_configured_file(
        FLAGDNN_ASCEND_PYTHON_PROGRAM_CANONICAL_PATH,
        FLAGDNN_ASCEND_PYTHON_PROGRAM_SHA256, "Python program");
    std::error_code python_program_error;
    const fs::path resolved_python_program =
        fs::canonical(FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH, python_program_error);
    if (python_program_error || resolved_python_program != python_program) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "configured Python program no longer resolves to its pinned file");
    }
    verify_pinned_python_module_files();

    const LocalEnvironment local_environment =
        prepare_local_environment(*mode, target);
    const auto configuration_snapshot =
        freeze_process_configuration(*mode, local_environment);
    const std::string frozen_configuration =
        configuration_identity(*configuration_snapshot);
    promote_loaded_python(python);
    const std::string python_identity =
        initialize_embedded_python(local_environment.python_module_root,
                                   FLAGDNN_ASCEND_PYTHON_PROGRAM_PATH);
    if (frozen_configuration_status(*configuration_snapshot) !=
        FrozenConfigurationStatus::kValid) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
          "Ascend compiler environment changed during embedded Python import");
    }

    aclrtContext verified_context = nullptr;
    const aclError verified_context_result =
        api.get_current_context(&verified_context);
    if (verified_context_result != ACL_SUCCESS || verified_context != current) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          verified_context_result == ACL_SUCCESS
              ? "current Ascend context changed during domain initialization"
              : acl_failure("final aclrtGetCurrentContext",
                            verified_context_result));
    }
    std::int32_t verified_device = -1;
    const aclError verified_device_result = api.get_device(&verified_device);
    if (verified_device_result != ACL_SUCCESS ||
        verified_device != device_ordinal) {
      throw AscendError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          verified_device_result == ACL_SUCCESS
              ? "current Ascend device changed during domain initialization"
              : acl_failure("final aclrtGetDevice", verified_device_result));
    }

    const std::string version =
        sanitize_fingerprint_component(package_version.data());
    detail::DomainBinding binding;
    binding.device_ordinal = device_ordinal;
    binding.default_context = current;
    binding.ai_core_count = ai_core_count;
    binding.codegen_arch = codegen_arch;
    binding.target_fingerprint = "ascend_" + soc + "_cann_" + version +
                                 "_aic_" + std::to_string(ai_core_count);
    if (binding.target_fingerprint.size() + 1 >
        FLAGDNN_BACKEND_MAX_TARGET_FINGERPRINT) {
      throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                        "Ascend target fingerprint exceeds the backend ABI "
                        "limit");
    }
    std::ostringstream identity;
    identity
        << "runtime_soc=" << soc << ";codegen_arch=" << codegen_arch
        << ";codegen_compatibility_alias="
        << (target.uses_compatibility_alias ? "enabled" : "disabled")
        << ";codegen_arch_source="
        << (target.detects_codegen_arch_from_runtime ? "runtime_detection"
                                                     : "environment")
        << ";ai_core_count=" << ai_core_count
        << ";cann_version=" << package_version.data()
        << ";cann_version_num=" << package_version_number
        << ";cann_inner_version=" << FLAGDNN_ASCEND_CANN_INNER_VERSION
        << ";ascendcl_path=" << ascendcl.path.string() << ";ascendcl_object_id="
        << loaded_object_identity(ascendcl, FLAGDNN_ASCEND_ASCENDCL_SHA256)
        << ";ascendcl_sha256=" << FLAGDNN_ASCEND_ASCENDCL_SHA256
        << ";runtime_path=" << runtime.path.string() << ";runtime_object_id="
        << loaded_object_identity(runtime, FLAGDNN_ASCEND_RUNTIME_SHA256)
        << ";runtime_sha256=" << FLAGDNN_ASCEND_RUNTIME_SHA256
        << ";triton_jit_path=" << triton_jit.path.string()
        << ";triton_jit_object_id="
        << loaded_object_identity(triton_jit,
                                  FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256)
        << ";triton_jit_sha256=" << FLAGDNN_ASCEND_LIBTRITON_JIT_SHA256
        << ";python_library_path=" << python.path.string()
        << ";python_library_object_id="
        << loaded_object_identity(python, FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256)
        << ";python_library_sha256=" << FLAGDNN_ASCEND_PYTHON_LIBRARY_SHA256
        << ";python_global_scope=promoted"
        << ";python_module_root="
        << local_environment.python_module_root.string()
        << ";standalone_path=" << standalone.string()
        << ";standalone_sha256=" << FLAGDNN_ASCEND_STANDALONE_SHA256 << ';'
        << python_identity;
    binding.runtime_identity = identity.str();
    binding.cache_root = local_environment.cache_root.string();
    binding.configuration_identity = frozen_configuration;
    binding.configuration_snapshot = configuration_snapshot;
    if (detail::process_domain().terminal_failure_latched()) {
      return detail::DomainInitialization::terminal(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "Ascend process domain became terminal during initialization");
    }
    return detail::DomainInitialization::bound(std::move(binding));
  } catch (const AscendError &error) {
    return detail::DomainInitialization::terminal(error.result(), error.what());
  } catch (const std::exception &error) {
    return detail::DomainInitialization::terminal(
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR, error.what());
  } catch (...) {
    return detail::DomainInitialization::terminal(
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
        "unknown failure while binding the Ascend process domain");
  }
}

[[nodiscard]] detail::DomainInitialization
initialize_real_domain(std::int32_t device_ordinal) {
  return initialize_domain_impl(device_ordinal,
                                detail::production_acl_runtime_api());
}

} // namespace

namespace detail {

DomainInitialization DomainInitialization::bound(DomainBinding binding) {
  DomainInitialization result;
  result.disposition = InitializationDisposition::kBound;
  result.binding = std::move(binding);
  result.result = FLAGDNN_BACKEND_RESULT_SUCCESS;
  return result;
}

DomainInitialization
DomainInitialization::retryable(flagdnnBackendResult_t result,
                                std::string message) {
  DomainInitialization initialization;
  initialization.disposition = InitializationDisposition::kRetryableFailure;
  initialization.result = result;
  initialization.message = std::move(message);
  return initialization;
}

DomainInitialization
DomainInitialization::terminal(flagdnnBackendResult_t result,
                               std::string message) {
  DomainInitialization initialization;
  initialization.disposition = InitializationDisposition::kTerminalFailure;
  initialization.result = result;
  initialization.message = std::move(message);
  return initialization;
}

std::shared_ptr<const DomainBinding>
DomainCoordinator::acquire(std::int32_t device_ordinal,
                           const Initializer &initializer) {
  require(device_ordinal >= 0, "device ordinal must be nonnegative");
  require(static_cast<bool>(initializer), "Ascend domain initializer is empty",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

  for (;;) {
    std::uint64_t claimed_initializer = 0;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (terminal_failure_latch_.load(std::memory_order_acquire) ||
          phase_ == DomainPhase::kFailed) {
        throw_failure_locked();
      }
      if (phase_ == DomainPhase::kBound) {
        if (binding_ == nullptr) {
          throw AscendError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                            "Ascend domain is bound without a binding");
        }
        if (binding_->device_ordinal != device_ordinal) {
          throw AscendError(
              FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
              "Ascend process domain is already bound to device " +
                  std::to_string(binding_->device_ordinal));
        }
        return binding_;
      }
      if (phase_ == DomainPhase::kInitializing) {
        condition_.wait(lock, [this] {
          return phase_ != DomainPhase::kInitializing ||
                 terminal_failure_latch_.load(std::memory_order_acquire);
        });
        continue;
      }

      phase_ = DomainPhase::kInitializing;
      requested_device_ = device_ordinal;
      claimed_initializer = ++initializer_id_;
    }

    DomainInitialization initialization;
    try {
      initialization = initializer(device_ordinal);
    } catch (const AscendError &error) {
      mark_terminal(error.result(), error.what());
      ensure_healthy();
    } catch (const std::bad_alloc &) {
      mark_terminal(
          FLAGDNN_BACKEND_RESULT_ALLOC_FAILED,
          "host memory allocation failed during Ascend domain initialization");
      ensure_healthy();
    } catch (const std::exception &error) {
      mark_terminal(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR, error.what());
      ensure_healthy();
    } catch (...) {
      mark_terminal(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                    "unknown Ascend domain initialization failure");
      ensure_healthy();
    }

    std::shared_ptr<const DomainBinding> pending_binding;
    if (initialization.disposition == InitializationDisposition::kBound) {
      if (initialization.binding.device_ordinal != device_ordinal ||
          initialization.binding.default_context == nullptr ||
          initialization.binding.ai_core_count == 0 ||
          initialization.binding.target_fingerprint.empty() ||
          initialization.binding.codegen_arch.empty() ||
          initialization.binding.cache_root.empty() ||
          initialization.binding.configuration_identity.empty()) {
        mark_terminal(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                      "Ascend initializer returned an incomplete binding");
        ensure_healthy();
      }
      try {
        pending_binding = std::make_shared<const DomainBinding>(
            std::move(initialization.binding));
      } catch (const std::bad_alloc &) {
        mark_terminal(FLAGDNN_BACKEND_RESULT_ALLOC_FAILED,
                      "cannot allocate the Ascend domain binding");
        ensure_healthy();
      }
    }

    if (initialization.disposition ==
        InitializationDisposition::kTerminalFailure) {
      terminal_failure_latch_.store(true, std::memory_order_release);
    }

    std::shared_ptr<const DomainBinding> committed;
    flagdnnBackendResult_t error_result = initialization.result;
    std::string error_message = std::move(initialization.message);
    {
      std::unique_lock<std::mutex> lock(mutex_);
      if (phase_ != DomainPhase::kInitializing ||
          initializer_id_ != claimed_initializer ||
          requested_device_ != device_ordinal) {
        if (phase_ == DomainPhase::kFailed ||
            terminal_failure_latch_.load(std::memory_order_acquire)) {
          throw_failure_locked();
        }
        terminal_failure_latch_.store(true, std::memory_order_release);
        phase_ = DomainPhase::kFailed;
        failure_result_ = FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
        try {
          failure_message_ = "Ascend domain initialization ownership changed";
        } catch (...) {
          failure_message_.clear();
        }
        condition_.notify_all();
        throw_failure_locked();
      }

      switch (initialization.disposition) {
      case InitializationDisposition::kBound:
        if (terminal_failure_latch_.load(std::memory_order_acquire)) {
          phase_ = DomainPhase::kFailed;
        } else {
          binding_ = std::move(pending_binding);
          committed = binding_;
          phase_ = DomainPhase::kBound;
          requested_device_ = -1;
        }
        break;
      case InitializationDisposition::kRetryableFailure:
        phase_ = DomainPhase::kUnbound;
        requested_device_ = -1;
        break;
      case InitializationDisposition::kTerminalFailure:
        phase_ = DomainPhase::kFailed;
        failure_result_ = error_result == FLAGDNN_BACKEND_RESULT_SUCCESS
                              ? FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR
                              : error_result;
        try {
          failure_message_ = error_message.empty()
                                 ? "Ascend process domain initialization "
                                   "failed terminally"
                                 : error_message;
        } catch (...) {
          failure_message_.clear();
        }
        break;
      }
      condition_.notify_all();
      if (phase_ == DomainPhase::kFailed) {
        throw_failure_locked();
      }
    }

    if (committed != nullptr) {
      return committed;
    }
    if (error_result == FLAGDNN_BACKEND_RESULT_SUCCESS) {
      error_result = FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
    }
    if (error_message.empty()) {
      error_message = "Ascend domain initialization did not complete";
    }
    throw AscendError(error_result, std::move(error_message));
  }
}

void DomainCoordinator::ensure_healthy() const {
  if (!terminal_failure_latch_.load(std::memory_order_acquire)) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  throw_failure_locked();
}

void DomainCoordinator::latch_terminal() noexcept {
  terminal_failure_latch_.store(true, std::memory_order_release);
}

void DomainCoordinator::mark_terminal(flagdnnBackendResult_t result,
                                      std::string message) noexcept {
  latch_terminal();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = DomainPhase::kFailed;
    binding_.reset();
    requested_device_ = -1;
    failure_result_ = result == FLAGDNN_BACKEND_RESULT_SUCCESS
                          ? FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR
                          : result;
    try {
      failure_message_ = message.empty() ? "Ascend process domain is terminal"
                                         : std::move(message);
    } catch (...) {
      failure_message_.clear();
    }
  }
  condition_.notify_all();
}

bool DomainCoordinator::terminal_failure_latched() const noexcept {
  return terminal_failure_latch_.load(std::memory_order_acquire);
}

DomainSnapshot DomainCoordinator::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return {phase_, requested_device_, initializer_id_,
          terminal_failure_latch_.load(std::memory_order_acquire)};
}

std::int32_t
DomainCoordinator::bound_device_ordinal(aclrtContext expected_context) const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (terminal_failure_latch_.load(std::memory_order_acquire) ||
      phase_ == DomainPhase::kFailed) {
    throw_failure_locked();
  }
  if (phase_ != DomainPhase::kBound || binding_ == nullptr ||
      binding_->default_context != expected_context) {
    throw AscendError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
                      "Ascend context is not part of the bound process domain");
  }
  return binding_->device_ordinal;
}

void DomainCoordinator::throw_failure_locked() const {
  throw AscendError(failure_result_, failure_message_.empty()
                                         ? "Ascend process domain is terminal"
                                         : failure_message_);
}

const AclRuntimeApi &production_acl_runtime_api() noexcept {
  static const AclRuntimeApi api = {
      &aclrtGetCurrentContext, &aclrtSetCurrentContext, &aclrtGetDevice,
      &rtGetAiCoreCount,       &aclrtGetSocName,        &aclsysGetVersionStr,
      &aclsysGetVersionNum,
  };
  return api;
}

DomainCoordinator &process_domain() noexcept {
  static DomainCoordinator domain;
  return domain;
}

DomainInitialization initialize_domain_with_api(std::int32_t device_ordinal,
                                                const AclRuntimeApi &api) {
  return initialize_domain_impl(device_ordinal, api);
}

std::size_t loaded_python_object_count_for_test() {
  return loaded_python_objects().size();
}

void verify_pinned_python_files_for_test() {
  verify_pinned_python_module_files();
}

} // namespace detail

ContextGuard::ContextGuard(aclrtContext expected_context)
    : ContextGuard(
          expected_context,
          detail::process_domain().bound_device_ordinal(expected_context),
          detail::process_domain(), detail::production_acl_runtime_api()) {}

ContextGuard::ContextGuard(aclrtContext expected_context,
                           std::int32_t expected_device,
                           detail::DomainCoordinator &domain,
                           const detail::AclRuntimeApi &api) {
  require(expected_context != nullptr, "expected Ascend context is null",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  require(expected_device >= 0, "expected Ascend device is invalid",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  require(api.get_current_context != nullptr &&
              api.set_current_context != nullptr && api.get_device != nullptr,
          "Ascend runtime context API is incomplete",
          FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
  domain.ensure_healthy();

  aclrtContext current = nullptr;
  aclError result = api.get_current_context(&current);
  const bool context_is_empty = result == ACL_ERROR_RT_CONTEXT_NULL ||
                                (result == ACL_SUCCESS && current == nullptr);
  if (result != ACL_SUCCESS && result != ACL_ERROR_RT_CONTEXT_NULL) {
    const std::string message = acl_failure("aclrtGetCurrentContext", result);
    domain.mark_terminal(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
    throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
  }
  if (current != nullptr && current != expected_context) {
    throw AscendError(
        FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
        "current Ascend context differs from the process-bound default "
        "context; explicit or alternate contexts are not supported");
  }
  if (context_is_empty) {
    result = api.set_current_context(expected_context);
    if (result != ACL_SUCCESS) {
      const std::string message =
          acl_failure("aclrtSetCurrentContext(bound default)", result);
      domain.mark_terminal(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
    }

    current = nullptr;
    result = api.get_current_context(&current);
    if (result != ACL_SUCCESS || current != expected_context) {
      const std::string message =
          result == ACL_SUCCESS
              ? "aclrtSetCurrentContext did not restore the process-bound "
                "default context"
              : acl_failure("aclrtGetCurrentContext(readback)", result);
      domain.mark_terminal(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
    }

    std::int32_t current_device = -1;
    result = api.get_device(&current_device);
    if (result != ACL_SUCCESS || current_device != expected_device) {
      const std::string message =
          result == ACL_SUCCESS
              ? "restored Ascend context has the wrong device ordinal"
              : acl_failure("aclrtGetDevice(restored context)", result);
      domain.mark_terminal(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
      throw AscendError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR, message);
    }
  }
  domain.ensure_healthy();
}

AscendContext::AscendContext(std::int32_t device_ordinal) {
  binding_ =
      detail::process_domain().acquire(device_ordinal, &initialize_real_domain);
  const FrozenConfigurationStatus status =
      binding_->configuration_snapshot == nullptr
          ? FrozenConfigurationStatus::kEnvironmentChanged
          : frozen_configuration_status(*binding_->configuration_snapshot);
  if (status != FrozenConfigurationStatus::kValid) {
    const char *message = frozen_configuration_error(status);
    detail::process_domain().mark_terminal(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
                                           message);
    throw AscendError(FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED, message);
  }
  ContextGuard guard(binding_->default_context);
}

const std::string &AscendContext::target_fingerprint() const noexcept {
  return binding_->target_fingerprint;
}

EngineBuildContext AscendContext::engine_build_context() const {
  ensure_process_healthy();
  return {binding_->device_ordinal,
          binding_->default_context,
          binding_->ai_core_count,
          binding_->target_fingerprint,
          binding_->codegen_arch,
          binding_->runtime_identity,
          binding_->cache_root,
          binding_->configuration_identity,
          binding_->configuration_snapshot};
}

std::mutex &process_ltj_mutex() noexcept {
  static std::mutex mutex;
  return mutex;
}

void ensure_process_healthy() { detail::process_domain().ensure_healthy(); }

void ensure_process_configuration(const EngineBuildContext &context) {
  ensure_process_healthy();
  try {
    require(!context.cache_root.empty() &&
                !context.configuration_identity.empty() &&
                context.configuration_snapshot != nullptr &&
                context.ai_core_count != 0 &&
                context.ai_core_count <= kMaximumAiCoreCount,
            "Ascend engine context has no frozen process configuration",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    require(context.configuration_snapshot->cache.path == context.cache_root,
            "Ascend engine context differs from its process configuration",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    const FrozenConfigurationStatus status =
        frozen_configuration_status(*context.configuration_snapshot);
    require(status == FrozenConfigurationStatus::kValid,
            frozen_configuration_error(status),
            FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED);
  } catch (const AscendError &error) {
    detail::process_domain().mark_terminal(error.result(), error.what());
    throw;
  } catch (const std::exception &error) {
    detail::process_domain().mark_terminal(
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR, error.what());
    throw AscendError(FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR, error.what());
  }
  ensure_process_healthy();
}

bool process_configuration_matches(const EngineBuildContext &context) noexcept {
  return !context.cache_root.empty() &&
         !context.configuration_identity.empty() &&
         context.configuration_snapshot != nullptr &&
         context.ai_core_count != 0 &&
         context.ai_core_count <= kMaximumAiCoreCount &&
         context.configuration_snapshot->cache.path == context.cache_root &&
         frozen_configuration_status(*context.configuration_snapshot) ==
             FrozenConfigurationStatus::kValid;
}

void latch_process_terminal() noexcept {
  detail::process_domain().latch_terminal();
}

void mark_process_terminal(flagdnnBackendResult_t result,
                           std::string message) noexcept {
  detail::process_domain().mark_terminal(result, std::move(message));
}

} // namespace flagdnn::ascend
