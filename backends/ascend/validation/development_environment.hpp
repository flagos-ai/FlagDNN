/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ASCEND_VALIDATION_DEVELOPMENT_ENVIRONMENT_HPP_
#define FLAGDNN_BACKENDS_ASCEND_VALIDATION_DEVELOPMENT_ENVIRONMENT_HPP_

#include <sys/stat.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef FLAGDNN_ASCEND_VALIDATION_CANN_ROOT
#define FLAGDNN_ASCEND_VALIDATION_CANN_ROOT ""
#endif

#ifndef FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT
#define FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT ""
#endif

namespace flagdnn::validation::ascend {

/*
 * Validation executables are deliberately development-only processes. They
 * establish their complete process environment before the first backend
 * Handle and keep the owned root alive until ACL, streams, executables, and
 * handles have all been torn down.
 */
class DevelopmentEnvironment {
 public:
  explicit DevelopmentEnvironment(std::string_view purpose) {
    if (purpose.empty()) {
      throw std::invalid_argument(
          "Ascend validation environment purpose is empty");
    }
    const std::filesystem::path temporary_base =
        std::filesystem::temp_directory_path();
    std::string pattern =
        (temporary_base /
         ("flagdnn-ascend-" + std::string(purpose) + "-XXXXXX"))
            .string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* created = ::mkdtemp(writable.data());
    if (created == nullptr) {
      throw std::runtime_error(
          "mkdtemp failed for Ascend validation environment: " +
          std::string(std::strerror(errno)));
    }
    root_ = created;
    try {
      require_mode_0700(root_);
      graph_cache_ = create_private_directory("graph-cache");
      triton_cache_ = create_private_directory("triton-cache");
      temporary_ = create_private_directory("tmp");
      prepare_environment();
    } catch (...) {
      cleanup_noexcept();
      throw;
    }
  }

  ~DevelopmentEnvironment() { cleanup_noexcept(); }

  DevelopmentEnvironment(const DevelopmentEnvironment&) = delete;
  DevelopmentEnvironment& operator=(const DevelopmentEnvironment&) = delete;

  [[nodiscard]] const std::filesystem::path& root() const noexcept {
    return root_;
  }

  [[nodiscard]] const std::filesystem::path& graph_cache() const noexcept {
    return graph_cache_;
  }

  void prepare_target(std::string_view soc_name) {
    if (soc_name.empty()) {
      throw std::invalid_argument("Ascend validation SoC name is empty");
    }
    for (const unsigned char character : soc_name) {
      if ((character < '0' || character > '9') &&
          (character < 'A' || character > 'Z') &&
          (character < 'a' || character > 'z') && character != '_' &&
          character != '.' && character != '-') {
        throw std::invalid_argument(
            "Ascend validation SoC name is not environment-safe");
      }
    }
    if (target_prepared_ && target_ != soc_name) {
      throw std::logic_error(
          "Ascend validation target changed within one process");
    }
    target_ = soc_name;
    set_environment("TRITON_ASCEND_ARCH", target_);
    target_prepared_ = true;
  }

  void cleanup() {
    if (cleaned_) {
      return;
    }
    std::error_code error;
    const std::uintmax_t removed = std::filesystem::remove_all(root_, error);
    if (error || removed == 0) {
      throw std::runtime_error(
          "cannot remove Ascend validation environment root: " +
          (error ? error.message() : std::string("root was missing")));
    }
    cleaned_ = true;
  }

 private:
  static void set_environment(const char* name, std::string_view value) {
    const std::string owned(value);
    if (::setenv(name, owned.c_str(), 1) != 0) {
      throw std::runtime_error(std::string("cannot set ") + name + ": " +
                               std::strerror(errno));
    }
  }

  static void unset_environment(const char* name) {
    if (::unsetenv(name) != 0) {
      throw std::runtime_error(std::string("cannot unset ") + name + ": " +
                               std::strerror(errno));
    }
  }

  static void require_mode_0700(const std::filesystem::path& path) {
    if (::chmod(path.c_str(), S_IRWXU) != 0) {
      throw std::runtime_error("cannot restrict validation directory " +
                               path.string() + ": " +
                               std::strerror(errno));
    }
    struct stat status {};
    if (::stat(path.c_str(), &status) != 0 || !S_ISDIR(status.st_mode) ||
        (status.st_mode & 0777) != S_IRWXU) {
      throw std::runtime_error(
          "Ascend validation directory is not a 0700 directory: " +
          path.string());
    }
  }

  [[nodiscard]] std::filesystem::path create_private_directory(
      std::string_view name) const {
    const std::filesystem::path result = root_ / std::string(name);
    std::error_code error;
    if (!std::filesystem::create_directory(result, error) || error) {
      throw std::runtime_error(
          "cannot create Ascend validation directory " + result.string() +
          ": " + error.message());
    }
    require_mode_0700(result);
    return result;
  }

  void prepare_environment() {
    const std::filesystem::path cann_root =
        std::filesystem::canonical(FLAGDNN_ASCEND_VALIDATION_CANN_ROOT);
    const std::filesystem::path python_module_root =
        std::filesystem::canonical(
            FLAGDNN_ASCEND_VALIDATION_PYTHON_MODULE_ROOT);
    if (!std::filesystem::is_directory(cann_root) ||
        !std::filesystem::is_directory(python_module_root)) {
      throw std::runtime_error(
          "Ascend validation configure-time environment paths are invalid");
    }

    set_environment("FLAGDNN_ASCEND_RESOURCE_CONTAINMENT", "development");
    set_environment("ASCEND_HOME_PATH", cann_root.string());
    set_environment("ASCEND_TOOLKIT_HOME", cann_root.string());
    set_environment("TRITON_JIT_BACKEND", "NPU");
    set_environment("TRITON_BACKEND", "torch_npu");
    set_environment("TORCH_DEVICE_BACKEND_AUTOLOAD", "0");
    set_environment("PYTHONDONTWRITEBYTECODE", "1");
    set_environment("PYTHONNOUSERSITE", "1");
    set_environment("PYTHONHASHSEED", "0");
    set_environment("PYTHONSAFEPATH", "1");
    set_environment("PYTHONPATH", python_module_root.string());
    set_environment("TRITON_CACHE_DIR", triton_cache_.string());
    set_environment("TMPDIR", temporary_.string());

    for (const char* name : {
             "PYTHONHOME",
             "PYTHONPYCACHEPREFIX",
             "PYTHONUSERBASE",
             "TRITON_ASCEND_ARCH",
             "TRITON_COMPILE_ONLY",
             "TRITON_ALWAYS_COMPILE",
             "TRITON_KERNEL_OVERRIDE",
             "TRITON_STORE_BINARY_ONLY",
             "TRITON_CACHE_MANAGER",
             "TRITON_REMOTE_CACHE_BACKEND",
             "TRITON_OVERRIDE_DIR",
         }) {
      unset_environment(name);
    }
  }

  void cleanup_noexcept() noexcept {
    if (cleaned_ || root_.empty()) {
      return;
    }
    std::error_code error;
    (void)std::filesystem::remove_all(root_, error);
    if (error) {
      std::cerr << "Ascend validation cleanup: cannot remove " << root_
                << ": " << error.message() << '\n';
    }
    cleaned_ = true;
  }

  std::filesystem::path root_;
  std::filesystem::path graph_cache_;
  std::filesystem::path triton_cache_;
  std::filesystem::path temporary_;
  std::string target_;
  bool target_prepared_ = false;
  bool cleaned_ = false;
};

}  // namespace flagdnn::validation::ascend

#endif  // FLAGDNN_BACKENDS_ASCEND_VALIDATION_DEVELOPMENT_ENVIRONMENT_HPP_
