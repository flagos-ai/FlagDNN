/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/hygon/engines/engine.hpp"
#include "backends/hygon/engines/jit_compatibility.hpp"
#include "backends/hygon/error.hpp"
#include "runtime/sha256.hpp"

#if defined(FLAGDNN_HAS_LIBTRITON_JIT)

#include <Python.h>
#include <dlfcn.h>
#include <triton_jit/kernel_metadata.h>
#include <triton_jit/triton_jit_function.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "backends/autotune_policy.hpp"

#ifndef FLAGDNN_LIBTRITON_JIT_PYTHONPATH
#define FLAGDNN_LIBTRITON_JIT_PYTHONPATH ""
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY
#define FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_SHA256
#define FLAGDNN_LIBTRITON_JIT_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_SHA256
#define FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_HELPER_SHA256
#define FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_HELPER_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256
#define FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256
#define FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_TORCH_SHA256
#define FLAGDNN_LIBTRITON_JIT_TORCH_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_TORCH_C_SHA256
#define FLAGDNN_LIBTRITON_JIT_TORCH_C_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_TRITON_SHA256
#define FLAGDNN_LIBTRITON_JIT_TRITON_SHA256 "unknown"
#endif

#ifndef FLAGDNN_LIBTRITON_JIT_YAML_SHA256
#define FLAGDNN_LIBTRITON_JIT_YAML_SHA256 "unknown"
#endif

namespace flagdnn::hygon {
namespace {

using DeviceAddress = std::uintptr_t;
constexpr std::size_t kWorkspaceAlignment = 256;

std::size_t aligned_allocation_size(std::size_t logical_size,
                                    std::size_t alignment,
                                    flagdnnBackendResult_t result,
                                    const char *message) {
  require(alignment != 0 && (alignment & (alignment - 1)) == 0, message,
          result);
  if (logical_size == 0) {
    return 0;
  }
  require(logical_size <=
              std::numeric_limits<std::size_t>::max() - (alignment - 1),
          message, result);
  return logical_size + alignment - 1;
}

DeviceAddress aligned_suballocation_address(DeviceAddress allocation,
                                            std::size_t allocation_size,
                                            std::size_t logical_size,
                                            std::size_t alignment,
                                            flagdnnBackendResult_t result,
                                            const char *message) {
  require(allocation != 0 && logical_size != 0, message, result);
  require(alignment != 0 && (alignment & (alignment - 1)) == 0, message,
          result);
  require(allocation <=
              std::numeric_limits<DeviceAddress>::max() - (alignment - 1),
          message, result);
  const DeviceAddress aligned = (allocation + alignment - 1) & ~(alignment - 1);
  const std::size_t adjustment = static_cast<std::size_t>(aligned - allocation);
  require(adjustment <= allocation_size &&
              logical_size <= allocation_size - adjustment,
          message, result);
  require(aligned <= std::numeric_limits<DeviceAddress>::max() - logical_size,
          message, result);
  return aligned;
}

hipError_t allocate_device(DeviceAddress *address, std::size_t size) {
  if (address == nullptr) {
    return hipErrorInvalidValue;
  }
  void *allocation = nullptr;
  const hipError_t status = hipMalloc(&allocation, size);
  if (status == hipSuccess) {
    *address = reinterpret_cast<DeviceAddress>(allocation);
  }
  return status;
}

hipError_t free_device(DeviceAddress address) {
  return hipFree(reinterpret_cast<void *>(address));
}

std::shared_mutex libtriton_jit_mutex;
std::once_flag libtriton_jit_identity_once;
std::once_flag libtriton_jit_python_modules_once;
std::once_flag python_runtime_once;
void *python_global_handle = nullptr;

constexpr std::string_view kHygonTritonJitBackend = "HCU";

bool logging_enabled() noexcept {
  const char *value = std::getenv("FLAGDNN_PRINT_AUTOTUNING");
  return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
}

std::optional<std::string>
active_python_triton_jit_backend() {
  require(Py_IsInitialized(),
          "cannot inspect the Python Triton JIT backend before Python is "
          "initialized",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  const PyGILState_STATE gil = PyGILState_Ensure();
  std::optional<std::string> result;
  std::string failure;
  PyObject *os = PyImport_ImportModule("os");
  PyObject *environment =
      os == nullptr ? nullptr : PyObject_GetAttrString(os, "environ");
  PyObject *value = nullptr;
  if (environment == nullptr) {
    failure = "cannot access Python os.environ for the Hygon Triton backend";
  } else {
    value = PyMapping_GetItemString(environment, "TRITON_JIT_BACKEND");
    if (value == nullptr) {
      if (PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Clear();
      } else {
        failure = "cannot read the Python Triton JIT backend";
      }
    } else if (!PyUnicode_Check(value)) {
      failure = "Python TRITON_JIT_BACKEND is not a string";
    } else if (const char *backend = PyUnicode_AsUTF8(value);
               backend != nullptr) {
      result = backend;
    } else {
      failure = "cannot decode the Python Triton JIT backend";
    }
  }
  Py_XDECREF(value);
  Py_XDECREF(environment);
  Py_XDECREF(os);
  if (!failure.empty() || PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);

  if (!failure.empty()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, failure);
  }
  return result;
}

void verify_triton_jit_backend_compatibility() {
  const char *backend = std::getenv("TRITON_JIT_BACKEND");
  require(backend == nullptr ||
              std::string_view(backend) == kHygonTritonJitBackend,
          "C environment TRITON_JIT_BACKEND must be unset or exactly HCU",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  if (!Py_IsInitialized()) {
    return;
  }
  const std::optional<std::string> python_backend =
      active_python_triton_jit_backend();
  require(!python_backend || *python_backend == kHygonTritonJitBackend,
          "Python os.environ TRITON_JIT_BACKEND must be unset or exactly HCU",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
}

bool is_candidate_compatibility_error(const std::exception &error) noexcept {
  return detail::is_jit_candidate_compatibility_error(error);
}

class ScopedEnvironmentVariable {
public:
  ScopedEnvironmentVariable(const char *name, const std::string &value,
                            const char *failure_message)
      : name_(name), failure_message_(failure_message) {
    const char *current = std::getenv(name_.c_str());
    if (current != nullptr) {
      original_c_ = current;
    }
    bool failed = false;
    if (Py_IsInitialized()) {
      const PyGILState_STATE gil = PyGILState_Ensure();
      PyObject *os = PyImport_ImportModule("os");
      PyObject *environment =
          os == nullptr ? nullptr : PyObject_GetAttrString(os, "environ");
      PyObject *original = environment == nullptr
                               ? nullptr
                               : PyMapping_GetItemString(environment,
                                                         name_.c_str());
      if (original == nullptr && PyErr_ExceptionMatches(PyExc_KeyError)) {
        PyErr_Clear();
      } else if (original == nullptr || !PyUnicode_Check(original)) {
        failed = true;
      } else if (const char *text = PyUnicode_AsUTF8(original);
                 text != nullptr) {
        original_python_ = text;
      } else {
        failed = true;
      }
      PyObject *replacement = PyUnicode_FromString(value.c_str());
      if (!failed &&
          (environment == nullptr || replacement == nullptr ||
           PyMapping_SetItemString(environment, name_.c_str(), replacement) !=
               0)) {
        failed = true;
      }
      Py_XDECREF(replacement);
      Py_XDECREF(original);
      Py_XDECREF(environment);
      Py_XDECREF(os);
      if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
        failed = true;
      }
      PyGILState_Release(gil);
    } else if (setenv(name_.c_str(), value.c_str(), 1) != 0) {
      failed = true;
    }
    active_ = true;
    if (failed) {
      restore();
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       failure_message_);
    }
  }

  ~ScopedEnvironmentVariable() { restore(); }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable &) = delete;
  ScopedEnvironmentVariable &
  operator=(const ScopedEnvironmentVariable &) = delete;

private:
  void restore() noexcept {
    if (!active_) {
      return;
    }
    active_ = false;
    if (Py_IsInitialized()) {
      const PyGILState_STATE gil = PyGILState_Ensure();
      PyObject *os = PyImport_ImportModule("os");
      PyObject *environment =
          os == nullptr ? nullptr : PyObject_GetAttrString(os, "environ");
      if (environment != nullptr) {
        if (original_python_) {
          PyObject *value = PyUnicode_FromString(original_python_->c_str());
          if (value != nullptr) {
            (void)PyMapping_SetItemString(environment, name_.c_str(), value);
          }
          Py_XDECREF(value);
        } else if (PyMapping_DelItemString(environment, name_.c_str()) != 0 &&
                   PyErr_ExceptionMatches(PyExc_KeyError)) {
          PyErr_Clear();
        }
      }
      Py_XDECREF(environment);
      Py_XDECREF(os);
      if (PyErr_Occurred() != nullptr) {
        PyErr_Clear();
      }
      PyGILState_Release(gil);
    }
    // Restore the exact C view as well: an embedding application can have
    // initialized Python before changing its native environment.
    if (original_c_) {
      (void)setenv(name_.c_str(), original_c_->c_str(), 1);
    } else {
      (void)unsetenv(name_.c_str());
    }
  }

  std::string name_;
  std::string failure_message_;
  std::optional<std::string> original_c_;
  std::optional<std::string> original_python_;
  bool active_ = false;
};

std::string read_regular_file(const std::filesystem::path &path,
                              std::size_t maximum_size,
                              std::string_view description) {
  std::error_code error;
  const auto status = std::filesystem::symlink_status(path, error);
  if (error || !std::filesystem::is_regular_file(status)) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string(description) + " is not a regular file");
  }
  const std::uintmax_t size = std::filesystem::file_size(path, error);
  if (error || size == 0 || size > maximum_size) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     std::string(description) + " size is invalid");
  }
  std::string result(static_cast<std::size_t>(size), '\0');
  std::ifstream input(path, std::ios::binary);
  input.read(result.data(), static_cast<std::streamsize>(result.size()));
  if (!input) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot read " + std::string(description));
  }
  return result;
}

std::string read_source_snapshot(const std::filesystem::path &path,
                                 std::size_t maximum_size) {
  return read_regular_file(path, maximum_size,
                           "per-device JIT source snapshot");
}

void verify_jit_python_module(std::string_view name,
                              std::string_view expected_sha256,
                              bool import_if_missing) {
  if (!Py_IsInitialized()) {
    require(!import_if_missing,
            "cannot import a JIT helper before Python is initialized",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    return;
  }

  const std::string module_name(name);
  const PyGILState_STATE gil = PyGILState_Ensure();
  std::string origin;
  std::string failure;
  PyObject *modules = PyImport_GetModuleDict();
  PyObject *module = modules == nullptr
                         ? nullptr
                         : PyDict_GetItemString(modules, module_name.c_str());
  PyObject *imported = nullptr;
  if (module == nullptr && import_if_missing) {
    imported = PyImport_ImportModule(module_name.c_str());
    module = imported;
  }
  if (module == nullptr) {
    if (import_if_missing) {
      failure = "cannot import libtriton_jit Python helper " + module_name;
    }
  } else {
    PyObject *file = PyObject_GetAttrString(module, "__file__");
    if (file == nullptr || !PyUnicode_Check(file)) {
      failure =
          "libtriton_jit Python helper has no file origin: " + module_name;
    } else if (const char *value = PyUnicode_AsUTF8(file); value != nullptr) {
      origin = value;
    } else {
      failure =
          "cannot read libtriton_jit Python helper origin: " + module_name;
    }
    Py_XDECREF(file);
  }
  Py_XDECREF(imported);
  if (!failure.empty() || PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);

  if (!failure.empty()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, failure);
  }
  if (origin.empty()) {
    return;
  }
  if (flagdnn::native::sha256(read_regular_file(
          origin, std::size_t{4} << 20,
          "loaded libtriton_jit Python helper")) != expected_sha256) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "loaded libtriton_jit Python helper differs from the "
                     "CMake selection: " +
                         module_name);
  }
}

void verify_active_python_environment() {
  require(
      Py_IsInitialized(),
      "cannot verify the Hygon JIT environment before Python is initialized",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  const std::filesystem::path helper =
      std::filesystem::path(triton_jit::get_script_dir()) /
      "flagdnn_python_environment_identity.py";
  const std::string helper_source = read_regular_file(
      helper, std::size_t{1} << 20, "Hygon Python environment identity helper");
  require(flagdnn::native::sha256(helper_source) ==
              FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_HELPER_SHA256,
          "Hygon Python environment identity helper differs from the CMake "
          "selection",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  const PyGILState_STATE gil = PyGILState_Ensure();
  std::string canonical_environment;
  std::string failure;
  PyObject *globals = PyDict_New();
  PyObject *module_name =
      PyUnicode_FromString("_flagdnn_hygon_runtime_environment_identity");
  PyObject *module_file = PyUnicode_DecodeFSDefault(helper.c_str());
  PyObject *code = nullptr;
  PyObject *evaluated = nullptr;
  PyObject *result = nullptr;
  if (globals == nullptr || module_name == nullptr || module_file == nullptr ||
      PyDict_SetItemString(globals, "__builtins__", PyEval_GetBuiltins()) !=
          0 ||
      PyDict_SetItemString(globals, "__name__", module_name) != 0 ||
      PyDict_SetItemString(globals, "__file__", module_file) != 0) {
    failure = "cannot create the isolated Hygon Python identity namespace";
  } else {
    code =
        Py_CompileString(helper_source.c_str(), helper.c_str(), Py_file_input);
    if (code == nullptr) {
      failure = "cannot compile the Hygon Python environment identity helper";
    } else {
      evaluated = PyEval_EvalCode(code, globals, globals);
      if (evaluated == nullptr) {
        failure = "cannot execute the Hygon Python environment identity helper";
      } else {
        PyObject *function =
            PyDict_GetItemString(globals, "canonical_python_environment_json");
        if (function == nullptr || !PyCallable_Check(function)) {
          failure = "Hygon Python environment identity helper has no canonical "
                    "entry point";
        } else {
          result = PyObject_CallNoArgs(function);
          if (result == nullptr || !PyUnicode_Check(result)) {
            failure = "Hygon Python environment identity helper returned an "
                      "invalid result";
          } else if (const char *value = PyUnicode_AsUTF8(result);
                     value != nullptr) {
            canonical_environment = value;
          } else {
            failure = "cannot read the Hygon Python environment identity";
          }
        }
      }
    }
  }
  Py_XDECREF(result);
  Py_XDECREF(evaluated);
  Py_XDECREF(code);
  Py_XDECREF(module_file);
  Py_XDECREF(module_name);
  Py_XDECREF(globals);
  if (!failure.empty() || PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);

  if (!failure.empty()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, failure);
  }
  require(!canonical_environment.empty() &&
              flagdnn::native::sha256(canonical_environment) ==
                  FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_SHA256,
          "active Python/Torch/Triton environment differs from the Hygon "
          "backend CMake selection",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
}

void initialize_verified_jit_python_modules() {
  std::call_once(libtriton_jit_python_modules_once, [] {
    // Validate the complete active interpreter before importing either of the
    // libtriton_jit helper modules below.
    verify_active_python_environment();
    verify_jit_python_module("torch", FLAGDNN_LIBTRITON_JIT_TORCH_SHA256, true);
    verify_jit_python_module("torch._C", FLAGDNN_LIBTRITON_JIT_TORCH_C_SHA256,
                             true);
    verify_jit_python_module("triton", FLAGDNN_LIBTRITON_JIT_TRITON_SHA256,
                             true);
    verify_jit_python_module("yaml", FLAGDNN_LIBTRITON_JIT_YAML_SHA256, true);
    verify_jit_python_module("gen_ssig", FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256,
                             true);
    verify_jit_python_module("standalone_compile",
                             FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256, true);
  });
}

void verify_libtriton_jit_runtime_identity() {
  std::call_once(libtriton_jit_identity_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void *>(
        reinterpret_cast<std::uintptr_t>(&triton_jit::get_script_dir));
    require(dladdr(address, &information) != 0 &&
                information.dli_fname != nullptr,
            "cannot locate the mapped libtriton_jit image",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::error_code error;
    const std::filesystem::path library =
        std::filesystem::canonical(information.dli_fname, error);
    require(!error, "cannot resolve the mapped libtriton_jit image",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const std::string library_bytes = read_regular_file(
        library, std::size_t{512} << 20, "mapped libtriton_jit image");
    require(flagdnn::native::sha256(library_bytes) ==
                FLAGDNN_LIBTRITON_JIT_SHA256,
            "mapped libtriton_jit image differs from the CMake selection",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    const std::filesystem::path script_directory = triton_jit::get_script_dir();
    const std::filesystem::path standalone =
        script_directory / "standalone_compile.py";
    const std::filesystem::path gen_ssig = script_directory / "gen_ssig.py";
    const std::filesystem::path environment_helper =
        script_directory / "flagdnn_python_environment_identity.py";
    require(flagdnn::native::sha256(
                read_regular_file(standalone, std::size_t{4} << 20,
                                  "libtriton_jit standalone_compile.py")) ==
                FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256,
            "libtriton_jit standalone_compile.py differs from the CMake "
            "selection",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(flagdnn::native::sha256(read_regular_file(
                gen_ssig, std::size_t{4} << 20, "libtriton_jit gen_ssig.py")) ==
                FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256,
            "libtriton_jit gen_ssig.py differs from the CMake selection",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(flagdnn::native::sha256(read_regular_file(
                environment_helper, std::size_t{1} << 20,
                "Hygon Python environment identity helper")) ==
                FLAGDNN_LIBTRITON_JIT_PYTHON_ENVIRONMENT_HELPER_SHA256,
            "Hygon Python environment identity helper differs from the CMake "
            "selection",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    // Reject generic helper modules imported by the embedding application
    // before libtriton_jit can reuse them from sys.modules.
    verify_jit_python_module("torch", FLAGDNN_LIBTRITON_JIT_TORCH_SHA256,
                             false);
    verify_jit_python_module("torch._C", FLAGDNN_LIBTRITON_JIT_TORCH_C_SHA256,
                             false);
    verify_jit_python_module("triton", FLAGDNN_LIBTRITON_JIT_TRITON_SHA256,
                             false);
    verify_jit_python_module("yaml", FLAGDNN_LIBTRITON_JIT_YAML_SHA256, false);
    verify_jit_python_module("gen_ssig", FLAGDNN_LIBTRITON_JIT_GEN_SSIG_SHA256,
                             false);
    verify_jit_python_module("standalone_compile",
                             FLAGDNN_LIBTRITON_JIT_STANDALONE_SHA256, false);
  });
}

std::filesystem::path publish_device_source_snapshot(
    const HygonStageArtifact &stage, const EngineBuildContext &context,
    const std::filesystem::path &device_cache_directory) {
  require(!stage.materialized_source.empty() &&
              flagdnn::native::sha256(stage.materialized_source) ==
                  stage.materialized_source_sha256,
          "validated JIT source snapshot is inconsistent",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  const std::filesystem::path destination =
      device_cache_directory /
      ("flagdnn-source-" + stage.materialized_source_sha256 + "-" +
       context.device_identity + ".py");
  std::error_code error;
  if (std::filesystem::exists(destination, error)) {
    if (error || read_source_snapshot(destination, 1U << 20) !=
                     stage.materialized_source) {
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "existing per-device JIT source snapshot is inconsistent");
    }
    return destination;
  }
  if (error) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot inspect per-device JIT source snapshot");
  }

  static std::atomic<std::uint64_t> snapshot_sequence{0};
  const std::filesystem::path temporary =
      destination.string() + ".tmp." + std::to_string(getpid()) + "." +
      std::to_string(snapshot_sequence.fetch_add(1));
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "cannot create per-device JIT source snapshot");
    }
    output.write(
        stage.materialized_source.data(),
        static_cast<std::streamsize>(stage.materialized_source.size()));
    output.close();
    if (!output) {
      std::filesystem::remove(temporary, error);
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "cannot write per-device JIT source snapshot");
    }
  }
  std::filesystem::rename(temporary, destination, error);
  if (error) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    if (!std::filesystem::exists(destination, ignored) || ignored ||
        read_source_snapshot(destination, 1U << 20) !=
            stage.materialized_source) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "cannot publish per-device JIT source snapshot");
    }
  }
  if (read_source_snapshot(destination, 1U << 20) !=
      stage.materialized_source) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "published per-device JIT source snapshot changed");
  }
  return destination;
}

std::filesystem::path
device_triton_cache_directory(const EngineBuildContext &context) {
  std::filesystem::path root;
  if (const char *configured = std::getenv("TRITON_CACHE_DIR");
      configured != nullptr && configured[0] != '\0') {
    root = configured;
  } else {
    const char *triton_home = std::getenv("TRITON_HOME");
    if (triton_home == nullptr || triton_home[0] == '\0') {
      triton_home = std::getenv("HOME");
    }
    if (triton_home == nullptr || triton_home[0] == '\0') {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "TRITON_CACHE_DIR, TRITON_HOME, and HOME are all unset");
    }
    root = std::filesystem::path(triton_home) / ".triton" / "cache";
  }
  // Triton's native cache key does not know which prepared standalone helper
  // fed it.  Include the CMake-verified JIT/script/environment identity so a
  // helper compatibility change can never reuse code produced under another
  // compiler contract.
  return root / ("flagdnn-hygon-v2-" +
                 std::string(FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY) + "-" +
                 context.device_identity);
}

void promote_python_runtime() {
  std::call_once(python_runtime_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void *>(
        reinterpret_cast<std::uintptr_t>(&Py_IsInitialized));
    if (dladdr(address, &information) == 0 ||
        information.dli_fname == nullptr) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "cannot locate the embedded Python runtime");
    }
    python_global_handle =
        dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (python_global_handle == nullptr) {
      python_global_handle =
          dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (python_global_handle == nullptr) {
      const char *detail = dlerror();
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot expose embedded Python symbols to Triton extensions: " +
              std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
  });
}

void configure_active_python_path(
    const std::vector<std::string> &required_paths) {
  if (!Py_IsInitialized() || required_paths.empty()) {
    return;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  std::string failure;
  PyObject *sys = PyImport_ImportModule("sys");
  PyObject *sys_path =
      sys == nullptr ? nullptr : PyObject_GetAttrString(sys, "path");
  if (sys_path == nullptr || !PyList_Check(sys_path)) {
    failure = "cannot access sys.path for embedded Hygon JIT";
  } else {
    // Insert in reverse so the configured order is preserved at the front.
    // PySequence_Contains keeps repeated engine builds and an overlap between
    // the configured module path and libtriton_jit's script directory from
    // introducing duplicate sys.path entries.
    for (auto path = required_paths.rbegin(); path != required_paths.rend();
         ++path) {
      PyObject *value = PyUnicode_DecodeFSDefault(path->c_str());
      const int present =
          value == nullptr ? -1 : PySequence_Contains(sys_path, value);
      if (present < 0 ||
          (present == 0 && PyList_Insert(sys_path, 0, value) != 0)) {
        Py_XDECREF(value);
        failure = "cannot prepend the configured Hygon JIT Python path";
        break;
      }
      Py_DECREF(value);
    }
  }
  Py_XDECREF(sys_path);
  Py_XDECREF(sys);
  if (!failure.empty() || PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);
  if (!failure.empty()) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED, failure);
  }
}

class ScopedPythonPath {
public:
  ScopedPythonPath() {
    require(Py_IsInitialized(),
            "cannot scope sys.path before Python initialization",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *sys = PyImport_ImportModule("sys");
    PyObject *current =
        sys == nullptr ? nullptr : PyObject_GetAttrString(sys, "path");
    if (current != nullptr && PyList_Check(current)) {
      original_ = current;
      snapshot_ = PySequence_List(original_);
      current = nullptr;
    }
    Py_XDECREF(current);
    Py_XDECREF(sys);
    const bool failed = snapshot_ == nullptr || PyErr_Occurred() != nullptr;
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
    }
    PyGILState_Release(gil);
    if (failed) {
      Py_XDECREF(original_);
      Py_XDECREF(snapshot_);
      original_ = nullptr;
      snapshot_ = nullptr;
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "cannot snapshot embedded Python sys.path");
    }
  }

  ~ScopedPythonPath() {
    if (original_ == nullptr || snapshot_ == nullptr || !Py_IsInitialized()) {
      return;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject *sys = PyImport_ImportModule("sys");
    if (sys != nullptr &&
        PyList_SetSlice(original_, 0, PyList_Size(original_), snapshot_) == 0) {
      (void)PyObject_SetAttrString(sys, "path", original_);
    }
    Py_XDECREF(sys);
    Py_DECREF(original_);
    Py_DECREF(snapshot_);
    original_ = nullptr;
    snapshot_ = nullptr;
    if (PyErr_Occurred() != nullptr) {
      PyErr_Clear();
    }
    PyGILState_Release(gil);
  }

  ScopedPythonPath(const ScopedPythonPath &) = delete;
  ScopedPythonPath &operator=(const ScopedPythonPath &) = delete;

private:
  PyObject *original_ = nullptr;
  PyObject *snapshot_ = nullptr;
};

void configure_python_path() {
  const char *runtime_override = std::getenv("FLAGDNN_HYGON_PYTHONPATH");
  const std::string required =
      runtime_override == nullptr
          ? std::string(FLAGDNN_LIBTRITON_JIT_PYTHONPATH)
          : std::string(runtime_override);
  if (required.empty()) {
    return;
  }
  std::vector<std::string> required_paths;
  const auto append_paths = [&](std::string_view value,
                                std::vector<std::string> &destination) {
      if (value.empty()) {
        return;
      }
      std::size_t offset = 0;
      while (offset <= value.size()) {
        const std::size_t separator = value.find(':', offset);
        const std::size_t length = separator == std::string_view::npos
                                       ? value.size() - offset
                                       : separator - offset;
        const std::string entry(value.substr(offset, length));
        if (!entry.empty() &&
            std::find(destination.begin(), destination.end(), entry) ==
                destination.end()) {
          destination.push_back(entry);
        }
        if (separator == std::string_view::npos) {
          break;
        }
        offset = separator + 1;
      }
  };
  append_paths(required, required_paths);
  configure_active_python_path(required_paths);
}

void initialize_python_runtime() {
  if (Py_IsInitialized()) {
    return;
  }
  Py_InitializeEx(0);
  require(Py_IsInitialized(), "cannot initialize Python for Hygon JIT",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  // Leave the interpreter initialized, but release the bootstrap thread's GIL
  // so engine builds on other host threads can acquire it.
  (void)PyEval_SaveThread();
}

struct TuningAllocation {
  std::int64_t uid = 0;
  std::size_t size = 0;
  std::size_t alignment = 1;
  DeviceAddress allocation = 0;
  DeviceAddress pointer = 0;
};

class JitTuningResources {
public:
  JitTuningResources(const HygonKernelArtifact &kernel,
                     std::size_t workspace_size,
                     std::size_t workspace_alignment)
      : JitTuningResources(std::vector<const HygonKernelArtifact *>{&kernel},
                           workspace_size, workspace_alignment) {}

  JitTuningResources(const std::vector<const HygonKernelArtifact *> &kernels,
                     std::size_t workspace_size,
                     std::size_t workspace_alignment) {
    for (const HygonKernelArtifact *kernel : kernels) {
      require(kernel != nullptr, "libtriton_jit autotune kernel is null",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
      for (const ArgumentSpec &argument : kernel->arguments) {
        if (argument.kind != ArgumentKind::kTensor) {
          continue;
        }
        auto existing = std::find_if(allocations_.begin(), allocations_.end(),
                                     [&](const TuningAllocation &allocation) {
                                       return allocation.uid == argument.uid;
                                     });
        if (existing == allocations_.end()) {
          allocations_.push_back(
              {argument.uid, argument.storage_size, argument.alignment, 0, 0});
        } else {
          existing->size = std::max(existing->size, argument.storage_size);
          existing->alignment =
              std::max(existing->alignment, argument.alignment);
        }
      }
    }
    try {
      for (TuningAllocation &allocation : allocations_) {
        const std::size_t allocation_size = aligned_allocation_size(
            allocation.size, allocation.alignment,
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "libtriton_jit autotune tensor allocation overflows");
        check_hip(allocate_device(&allocation.allocation, allocation_size),
                  "hipMalloc(libtriton_jit autotune tensor)");
        allocation.pointer = aligned_suballocation_address(
            allocation.allocation, allocation_size, allocation.size,
            allocation.alignment, FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "libtriton_jit autotune tensor alignment is invalid");
      }
      if (workspace_size != 0) {
        const std::size_t allocation_size = aligned_allocation_size(
            workspace_size, workspace_alignment,
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "libtriton_jit autotune workspace allocation overflows");
        check_hip(allocate_device(&workspace_allocation_, allocation_size),
                  "hipMalloc(libtriton_jit autotune workspace)");
        workspace_ = aligned_suballocation_address(
            workspace_allocation_, allocation_size, workspace_size,
            workspace_alignment, FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
            "libtriton_jit autotune workspace alignment is invalid");
      }
      check_hip(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking),
                "hipStreamCreateWithFlags(libtriton_jit autotune)");
      check_hip(hipEventCreateWithFlags(&start_, hipEventDefault),
                "hipEventCreateWithFlags(libtriton_jit autotune start)");
      check_hip(hipEventCreateWithFlags(&stop_, hipEventDefault),
                "hipEventCreateWithFlags(libtriton_jit autotune stop)");
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~JitTuningResources() { cleanup(); }

  JitTuningResources(const JitTuningResources &) = delete;
  JitTuningResources &operator=(const JitTuningResources &) = delete;

  [[nodiscard]] const std::vector<TuningAllocation> &allocations() const {
    return allocations_;
  }
  [[nodiscard]] DeviceAddress workspace() const noexcept { return workspace_; }
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  [[nodiscard]] hipEvent_t start() const noexcept { return start_; }
  [[nodiscard]] hipEvent_t stop() const noexcept { return stop_; }

private:
  void cleanup() noexcept {
    if (start_ != nullptr) {
      (void)hipEventDestroy(start_);
      start_ = nullptr;
    }
    if (stop_ != nullptr) {
      (void)hipEventDestroy(stop_);
      stop_ = nullptr;
    }
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (workspace_allocation_ != 0) {
      (void)free_device(workspace_allocation_);
      workspace_allocation_ = 0;
      workspace_ = 0;
    }
    for (TuningAllocation &allocation : allocations_) {
      if (allocation.allocation != 0) {
        (void)free_device(allocation.allocation);
        allocation.allocation = 0;
        allocation.pointer = 0;
      }
    }
  }

  std::vector<TuningAllocation> allocations_;
  DeviceAddress workspace_allocation_ = 0;
  DeviceAddress workspace_ = 0;
  hipStream_t stream_ = nullptr;
  hipEvent_t start_ = nullptr;
  hipEvent_t stop_ = nullptr;
};

class CapturedLaunchBatch {
public:
  template <typename Function>
  CapturedLaunchBatch(hipStream_t stream, unsigned int execution_count,
                      Function &&launch)
      : execution_count_(execution_count) {
    require(execution_count_ != 0,
            "libtriton_jit autotune batch cannot be empty",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    check_hip(hipStreamBeginCapture(stream, hipStreamCaptureModeRelaxed),
              "hipStreamBeginCapture(libtriton_jit autotune)");
    try {
      for (unsigned int iteration = 0; iteration < execution_count_;
           ++iteration) {
        launch();
      }
    } catch (...) {
      hipGraph_t abandoned = nullptr;
      if (hipStreamEndCapture(stream, &abandoned) == hipSuccess &&
          abandoned != nullptr) {
        (void)hipGraphDestroy(abandoned);
      }
      throw;
    }

    hipGraph_t graph = nullptr;
    check_hip(hipStreamEndCapture(stream, &graph),
              "hipStreamEndCapture(libtriton_jit autotune)");
    try {
      check_hip(hipGraphInstantiateWithFlags(&executable_, graph, 0),
                "hipGraphInstantiateWithFlags(libtriton_jit autotune)");
      check_hip(hipGraphDestroy(graph),
                "hipGraphDestroy(libtriton_jit autotune source)");
    } catch (...) {
      if (graph != nullptr) {
        (void)hipGraphDestroy(graph);
      }
      cleanup();
      throw;
    }
  }

  ~CapturedLaunchBatch() { cleanup(); }

  CapturedLaunchBatch(const CapturedLaunchBatch &) = delete;
  CapturedLaunchBatch &operator=(const CapturedLaunchBatch &) = delete;

  void launch(hipStream_t stream) const {
    check_hip(hipGraphLaunch(executable_, stream),
              "hipGraphLaunch(libtriton_jit autotune)");
  }

  [[nodiscard]] unsigned int execution_count() const noexcept {
    return execution_count_;
  }

private:
  void cleanup() noexcept {
    if (executable_ != nullptr) {
      (void)hipGraphExecDestroy(executable_);
      executable_ = nullptr;
    }
  }

  hipGraphExec_t executable_ = nullptr;
  unsigned int execution_count_ = 0;
};

struct ArgumentValue {
  DeviceAddress pointer = 0;
  std::int32_t scalar_i32 = 0;
  float scalar_f32 = 0.0F;
};

class RawArguments {
public:
  RawArguments(const HygonKernelArtifact &kernel,
               const std::vector<TuningAllocation> &allocations,
               DeviceAddress workspace, DeviceAddress global_scratch) {
    initialize(kernel);
    for (std::size_t index = 0; index < kernel.arguments.size(); ++index) {
      const ArgumentSpec &argument = kernel.arguments[index];
      if (argument.kind == ArgumentKind::kTensor) {
        const auto allocation =
            std::find_if(allocations.begin(), allocations.end(),
                         [&](const TuningAllocation &value) {
                           return value.uid == argument.uid;
                         });
        if (allocation == allocations.end()) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "libtriton_jit autotune allocation is missing");
        }
        values_[index].pointer = allocation->pointer;
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kWorkspaceTensor) {
        if (workspace == 0) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "libtriton_jit autotune workspace is missing");
        }
        values_[index].pointer = workspace + argument.workspace_offset;
        if (values_[index].pointer % argument.alignment != 0) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
              "libtriton_jit autotune workspace tensor is misaligned");
        }
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kScalarI32) {
        values_[index].scalar_i32 = argument.scalar_i32;
        parameters_[index] = &values_[index].scalar_i32;
      } else if (argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_f32 = argument.scalar_f32;
        parameters_[index] = &values_[index].scalar_f32;
      } else {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "libtriton_jit argument kind is unsupported");
      }
    }
    finish(kernel.arguments.size(), global_scratch);
  }

  RawArguments(const HygonKernelArtifact &kernel,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count, void *workspace,
               DeviceAddress global_scratch) {
    initialize(kernel);
    for (std::size_t index = 0; index < kernel.arguments.size(); ++index) {
      const ArgumentSpec &argument = kernel.arguments[index];
      if (argument.kind == ArgumentKind::kTensor) {
        bool found = false;
        for (std::size_t supplied = 0; supplied < binding_count; ++supplied) {
          if (bindings[supplied].uid == argument.uid) {
            values_[index].pointer =
                static_cast<DeviceAddress>(reinterpret_cast<std::uintptr_t>(
                    bindings[supplied].device_pointer));
            found = true;
            break;
          }
        }
        require(found, "a required tensor UID is missing from bindings");
        require(values_[index].pointer % argument.alignment == 0,
                "a tensor binding does not satisfy its declared alignment");
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kWorkspaceTensor) {
        values_[index].pointer = static_cast<DeviceAddress>(
            reinterpret_cast<std::uintptr_t>(workspace) +
            argument.workspace_offset);
        require(values_[index].pointer % argument.alignment == 0,
                "a workspace tensor does not satisfy its declared "
                "alignment");
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kScalarI32) {
        values_[index].scalar_i32 = argument.scalar_i32;
        parameters_[index] = &values_[index].scalar_i32;
      } else if (argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_f32 = argument.scalar_f32;
        parameters_[index] = &values_[index].scalar_f32;
      } else {
        throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                         "libtriton_jit argument kind is unsupported");
      }
    }
    finish(kernel.arguments.size(), global_scratch);
  }

  [[nodiscard]] void **data() noexcept { return parameters_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return parameter_count_; }

private:
  void initialize(const HygonKernelArtifact &kernel) {
    const std::size_t argument_count = kernel.arguments.size();
    parameter_count_ = argument_count + 2;
    require(argument_count <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
            "libtriton_jit kernel has too many arguments",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    std::fill_n(parameters_.data(), parameter_count_, nullptr);
  }

  void finish(std::size_t visible_argument_count,
              DeviceAddress global_scratch) {
    global_scratch_ = global_scratch;
    parameters_[visible_argument_count] = &global_scratch_;
    parameters_[visible_argument_count + 1] = &profile_scratch_;
  }

  std::array<ArgumentValue, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> values_{};
  std::array<void *, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2> parameters_{};
  std::size_t parameter_count_ = 0;
  DeviceAddress global_scratch_ = 0;
  DeviceAddress profile_scratch_ = 0;
};

using JitFunction = triton_jit::TritonJITFunction;

void preflight_jit_candidate(const std::filesystem::path &source,
                             const std::string &function_name,
                             const HygonKernelArtifact &kernel, int device) {
  std::string cache_directory;
  {
    struct PythonScope {
      PyGILState_STATE gil = PyGILState_Ensure();
      ~PythonScope() {
        PyErr_Clear();
        PyGILState_Release(gil);
      }
    } python_scope;
    using PythonObject = std::unique_ptr<PyObject, decltype(&Py_DecRef)>;
    const auto fail = []() {
      PyObject *type = nullptr;
      PyObject *value = nullptr;
      PyObject *traceback = nullptr;
      PyErr_Fetch(&type, &value, &traceback);
      PythonObject error_type(type, Py_DecRef);
      PythonObject error_value(value, Py_DecRef);
      PythonObject error_traceback(traceback, Py_DecRef);
      PythonObject description(value == nullptr ? nullptr : PyObject_Str(value),
                               Py_DecRef);
      const char *text = description ? PyUnicode_AsUTF8(description.get())
                                     : nullptr;
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "libtriton_jit candidate preflight failed: " +
              std::string(text == nullptr ? "invalid compiler result" : text));
    };
    // Use the same verified helper and arguments as upstream's private
    // get_kernel(). Compile before loading a module so an oversized candidate
    // can be rejected before upstream allocates a module for it.
    PythonObject module(PyImport_ImportModule("standalone_compile"), Py_DecRef);
    PythonObject compile(module ? PyObject_GetAttrString(module.get(),
                                                         "compile_a_kernel")
                                : nullptr,
                         Py_DecRef);
    if (!compile) {
      fail();
    }
    PythonObject result(
        PyObject_CallFunction(compile.get(), "sssiii", source.c_str(),
                              function_name.c_str(), kernel.full_signature.c_str(),
                              static_cast<int>(kernel.num_warps),
                              static_cast<int>(kernel.num_stages), device),
        Py_DecRef);
    const char *directory = result && PyUnicode_Check(result.get())
                                ? PyUnicode_AsUTF8(result.get())
                                : nullptr;
    if (directory == nullptr || directory[0] == '\0') {
      fail();
    }
    cache_directory = directory;
  }

  triton_jit::HcuKernelMetadata metadata;
  try {
    metadata = triton_jit::load_hcu_metadata(cache_directory, function_name);
  } catch (const std::exception &error) {
    throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                     "cannot read HCU kernel metadata: " +
                         std::string(error.what()));
  }
  hipDeviceProp_t properties{};
  check_hip(hipGetDeviceProperties(&properties, device),
            "hipGetDeviceProperties(libtriton_jit preflight)");
  detail::validate_jit_kernel_resources(metadata.arch, metadata.shared,
                                         properties);
}

void launch_jit(const JitFunction &function, const HygonKernelArtifact &kernel,
                hipStream_t stream, RawArguments &arguments) {
  function.launch_with_raw_args(stream, kernel.grid[0], kernel.grid[1],
                                kernel.grid[2], kernel.num_warps,
                                kernel.num_stages, kernel.full_signature,
                                arguments.data(), arguments.size());
}

struct PreparedHygonLaunch {
  hipFunction_t function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

PreparedHygonLaunch prepare_hip_launch(const JitFunction &function,
                                       const HygonKernelArtifact &kernel,
                                       std::size_t workspace_size,
                                       std::size_t workspace_alignment,
                                       std::size_t global_scratch_offset) {
  JitTuningResources resources(kernel, workspace_size, workspace_alignment);
  RawArguments arguments(kernel, resources.allocations(), resources.workspace(),
                         resources.workspace() + global_scratch_offset);

  hipGraph_t graph = nullptr;
  bool capture_active = false;
  try {
    check_hip(
        hipStreamBeginCapture(resources.stream(), hipStreamCaptureModeRelaxed),
        "hipStreamBeginCapture(libtriton_jit prepared launch)");
    capture_active = true;
    launch_jit(function, kernel, resources.stream(), arguments);
    check_hip(hipStreamEndCapture(resources.stream(), &graph),
              "hipStreamEndCapture(libtriton_jit prepared launch)");
    capture_active = false;

    std::size_t node_count = 0;
    check_hip(hipGraphGetNodes(graph, nullptr, &node_count),
              "hipGraphGetNodes(libtriton_jit prepared launch count)");
    std::vector<hipGraphNode_t> nodes(node_count);
    check_hip(hipGraphGetNodes(graph, nodes.data(), &node_count),
              "hipGraphGetNodes(libtriton_jit prepared launch)");

    require(node_count == 1,
            "libtriton_jit prepared launch must capture exactly one HIP "
            "kernel node",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const hipGraphNode_t kernel_node = nodes.front();
    hipGraphNodeType node_type = hipGraphNodeTypeEmpty;
    check_hip(hipGraphNodeGetType(kernel_node, &node_type),
              "hipGraphNodeGetType(libtriton_jit prepared launch)");
    require(node_type == hipGraphNodeTypeKernel,
            "libtriton_jit prepared launch captured a non-kernel HIP node",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    hipKernelNodeParams parameters{};
    check_hip(hipGraphKernelNodeGetParams(kernel_node, &parameters),
              "hipGraphKernelNodeGetParams(libtriton_jit prepared launch)");
    require(parameters.func != nullptr,
            "libtriton_jit captured kernel has no hipFunction_t",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    PreparedHygonLaunch prepared{
        reinterpret_cast<hipFunction_t>(parameters.func),
        {parameters.gridDim.x, parameters.gridDim.y, parameters.gridDim.z},
        {parameters.blockDim.x, parameters.blockDim.y, parameters.blockDim.z},
        parameters.sharedMemBytes};
    check_hip(hipGraphDestroy(graph),
              "hipGraphDestroy(libtriton_jit prepared launch)");
    graph = nullptr;
    return prepared;
  } catch (...) {
    if (capture_active) {
      hipGraph_t abandoned = nullptr;
      if (hipStreamEndCapture(resources.stream(), &abandoned) == hipSuccess &&
          abandoned != nullptr) {
        (void)hipGraphDestroy(abandoned);
      }
    } else if (graph != nullptr) {
      (void)hipGraphDestroy(graph);
    }
    throw;
  }
}

void launch_prepared_hip(const PreparedHygonLaunch &prepared,
                         hipStream_t stream, RawArguments &arguments) {
  check_hip(hipModuleLaunchKernel(prepared.function, prepared.grid[0],
                                  prepared.grid[1], prepared.grid[2],
                                  prepared.block[0], prepared.block[1],
                                  prepared.block[2], prepared.shared_memory,
                                  stream, arguments.data(), nullptr),
            "hipModuleLaunchKernel(libtriton_jit prepared launch)");
}

struct LoadedJitKernel {
  const JitFunction *function = nullptr;
  HygonKernelArtifact specification;
  PreparedHygonLaunch prepared;
};

struct ExternalTensorBindingSpec {
  std::int64_t uid = 0;
  std::size_t storage_size = 0;
};

class LibTritonJitEngine final : public ExecutionEngine {
public:
  LibTritonJitEngine(const EngineBuildContext &context, HygonArtifact artifact)
      : context_(context), binding_uids_(std::move(artifact.binding_uids)),
        logical_workspace_size_(artifact.workspace_size),
        workspace_alignment_(artifact.workspace_alignment) {
    require(artifact.engine == EngineKind::kLibTritonJit,
            "libtriton_jit engine received another artifact",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    require(workspace_alignment_ >= kWorkspaceAlignment &&
                (workspace_alignment_ & (workspace_alignment_ - 1)) == 0,
            "libtriton_jit workspace alignment is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    workspace_size_ = aligned_allocation_size(
        logical_workspace_size_, workspace_alignment_,
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "libtriton_jit workspace allocation size overflows");

    external_bindings_.reserve(binding_uids_.size());
    for (const std::int64_t uid : binding_uids_) {
      std::size_t storage_size = 0;
      for (const HygonStageArtifact &stage : artifact.stages) {
        for (const HygonKernelArtifact &variant : stage.variants) {
          for (const ArgumentSpec &argument : variant.arguments) {
            if (argument.kind != ArgumentKind::kTensor || argument.uid != uid) {
              continue;
            }
            require(argument.storage_size != 0,
                    "libtriton_jit external tensor storage size is zero",
                    FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
            require(storage_size == 0 || storage_size == argument.storage_size,
                    "libtriton_jit external tensor storage sizes disagree",
                    FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
            storage_size = argument.storage_size;
          }
        }
      }
      require(storage_size != 0,
              "libtriton_jit external tensor binding has no Graph ABI",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      external_bindings_.push_back({uid, storage_size});
    }
    std::sort(external_bindings_.begin(), external_bindings_.end(),
              [](const ExternalTensorBindingSpec &left,
                 const ExternalTensorBindingSpec &right) {
                return left.uid < right.uid;
              });

    for (const HygonStageArtifact &stage : artifact.stages) {
      for (const HygonKernelArtifact &variant : stage.variants) {
        global_scratch_size_ =
            std::max(global_scratch_size_, variant.global_scratch_size);
      }
    }
    require(global_scratch_size_ != 0 &&
                global_scratch_size_ <= logical_workspace_size_,
            "libtriton_jit global scratch exceeds workspace",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    global_scratch_offset_ = logical_workspace_size_ - global_scratch_size_;
    require(logical_workspace_size_ % kWorkspaceAlignment == 0 &&
                global_scratch_size_ % kWorkspaceAlignment == 0 &&
                global_scratch_offset_ % kWorkspaceAlignment == 0,
            "libtriton_jit workspace/scratch alignment is invalid",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    for (const HygonStageArtifact &stage : artifact.stages) {
      for (const HygonKernelArtifact &variant : stage.variants) {
        for (const ArgumentSpec &argument : variant.arguments) {
          require(argument.kind != ArgumentKind::kWorkspaceTensor ||
                      (argument.workspace_offset <= global_scratch_offset_ &&
                       argument.storage_size <=
                           global_scratch_offset_ - argument.workspace_offset),
                  "libtriton_jit workspace tensor overlaps global scratch",
                  FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        }
      }
    }

    hipCtx_t retained_context = nullptr;
    check_hip(hipDevicePrimaryCtxRetain(&retained_context, context_.device),
              "hipDevicePrimaryCtxRetain(libtriton_jit engine)");
    if (retained_context != context_.context) {
      (void)hipDevicePrimaryCtxRelease(context_.device);
      throw HygonError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "HIP primary context changed while building JIT executable");
    }
    retained_ = true;

    try {
      std::unique_lock lock(libtriton_jit_mutex);
      // Reject a foreign process-global JIT selection before FlagDNN mutates
      // Python linkage, PYTHONPATH, sys.path, or the backend selection itself.
      verify_triton_jit_backend_compatibility();
      verify_libtriton_jit_runtime_identity();
      promote_python_runtime();
      ContextGuard guard(context_.device, context_.context);
      initialize_python_runtime();
      const ScopedEnvironmentVariable backend_selection(
          "TRITON_JIT_BACKEND", "HCU",
          "cannot scope the Hygon Triton JIT backend");
      verify_triton_jit_backend_compatibility();
      const ScopedPythonPath python_path;
      configure_python_path();
      // get_instance() normally prepends this directory immediately before it
      // imports gen_ssig. Verification intentionally runs earlier, so expose
      // the already hash-validated directory here and make the first import
      // obey the same exact-origin contract.
      configure_active_python_path(
          {std::filesystem::path(triton_jit::get_script_dir()).string()});
      initialize_verified_jit_python_modules();
      require(!artifact.stages.empty(),
              "libtriton_jit artifact has no execution stages",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      const std::filesystem::path device_cache_directory =
          device_triton_cache_directory(context_);
      std::error_code cache_error;
      std::filesystem::create_directories(device_cache_directory, cache_error);
      const auto cache_status =
          std::filesystem::symlink_status(device_cache_directory, cache_error);
      if (cache_error || !std::filesystem::is_directory(cache_status)) {
        throw HygonError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "cannot create a regular per-device Triton cache directory");
      }
      const ScopedEnvironmentVariable device_cache(
          "TRITON_CACHE_DIR", device_cache_directory.string(),
          "cannot scope the per-device Triton cache directory");
      kernels_.reserve(artifact.stages.size());
      for (const HygonStageArtifact &stage : artifact.stages) {
        if (stage.source.empty() || stage.function_name.empty() ||
            stage.variants.empty()) {
          throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                           "libtriton_jit stage is incomplete");
        }
        const std::filesystem::path source_snapshot =
            publish_device_source_snapshot(stage, context_,
                                           device_cache_directory);
        source_snapshot_paths_.push_back(source_snapshot);
        const JitFunction &function = JitFunction::get_instance(
            source_snapshot.string(), stage.function_name);
        const std::size_t selected =
            stage.autotune ? select_candidate(function, stage) : 0;
        if (selected >= stage.variants.size()) {
          throw HygonError(
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
              "libtriton_jit autotune selected an invalid candidate");
        }
        if (!stage.autotune) {
          prepare_candidate(function, stage, stage.variants[selected]);
        }
        HygonKernelArtifact specification = stage.variants[selected];
        PreparedHygonLaunch prepared =
            prepare_hip_launch(function, specification, logical_workspace_size_,
                               workspace_alignment_, global_scratch_offset_);
        kernels_.push_back({&function, std::move(specification), prepared});
      }
    } catch (const HygonError &) {
      release_context();
      throw;
    } catch (const std::exception &error) {
      release_context();
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "libtriton_jit executable build failed: " +
                           std::string(error.what()));
    } catch (...) {
      release_context();
      throw;
    }
  }

  ~LibTritonJitEngine() override { release_context(); }

  LibTritonJitEngine(const LibTritonJitEngine &) = delete;
  LibTritonJitEngine &operator=(const LibTritonJitEngine &) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(hipStream_t stream, const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count, void *workspace,
               std::size_t workspace_size) const override {
    require(workspace_size >= workspace_size_,
            "workspace is smaller than HIP executable requirement");
    require(workspace_size_ == 0 || workspace != nullptr,
            "HIP executable workspace is null");
    DeviceAddress logical_workspace = 0;
    if (logical_workspace_size_ != 0) {
      logical_workspace = aligned_suballocation_address(
          reinterpret_cast<DeviceAddress>(workspace), workspace_size,
          logical_workspace_size_, workspace_alignment_,
          FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
          "libtriton_jit workspace cannot satisfy its alignment contract");
    }
    require(binding_count == binding_uids_.size(),
            "binding count does not match HIP executable");
    require(binding_count == 0 || bindings != nullptr, "binding array is null");
    for (std::size_t index = 0; index < binding_count; ++index) {
      require(bindings[index].device_pointer != nullptr,
              "tensor binding device pointer is null");
      const ExternalTensorBindingSpec *specification =
          external_binding(bindings[index].uid);
      require(specification != nullptr,
              "binding contains a UID not required by HIP executable");
      const DeviceAddress begin =
          reinterpret_cast<DeviceAddress>(bindings[index].device_pointer);
      require(specification->storage_size <=
                  std::numeric_limits<DeviceAddress>::max() - begin,
              "tensor binding device address range overflows");
      for (std::size_t previous = 0; previous < index; ++previous) {
        require(bindings[previous].uid != bindings[index].uid,
                "binding array contains a duplicate tensor UID");
      }
    }

    try {
      ContextGuard guard(context_.device, context_.context);
      if (stream != nullptr) {
        hipDevice_t stream_device = 0;
        check_hip(hipStreamGetDevice(stream, &stream_device),
                  "hipStreamGetDevice(libtriton_jit execute)");
        require(stream_device == context_.device,
                "HIP stream belongs to another device");
        hipCtx_t stream_context = nullptr;
        check_hip(hipStreamGetCtx(stream, &stream_context),
                  "hipStreamGetCtx(libtriton_jit execute)");
        require(stream_context == context_.context,
                "HIP stream belongs to another context");
      }
      require(scratch_pointer(logical_workspace) % kWorkspaceAlignment == 0,
              "libtriton_jit global scratch pointer is misaligned");
      for (const LoadedJitKernel &kernel : kernels_) {
        RawArguments arguments(kernel.specification, bindings, binding_count,
                               reinterpret_cast<void *>(logical_workspace),
                               scratch_pointer(logical_workspace));
        launch_prepared_hip(kernel.prepared, stream, arguments);
      }
    } catch (const HygonError &) {
      throw;
    } catch (const std::exception &error) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                       "libtriton_jit kernel launch failed: " +
                           std::string(error.what()));
    }
  }

private:
  [[nodiscard]] const ExternalTensorBindingSpec *
  external_binding(std::int64_t uid) const noexcept {
    const auto candidate = std::lower_bound(
        external_bindings_.begin(), external_bindings_.end(), uid,
        [](const ExternalTensorBindingSpec &value, std::int64_t expected_uid) {
          return value.uid < expected_uid;
        });
    return candidate != external_bindings_.end() && candidate->uid == uid
               ? &*candidate
               : nullptr;
  }

  [[nodiscard]] std::size_t
  select_candidate(const JitFunction &function,
                   const HygonStageArtifact &stage) const {
    require(stage.autotune && stage.variants.size() >= 2,
            "invalid libtriton_jit autotune stage",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    backend::autotune::SelectionRequest full_request;
    full_request.candidate_identity = stage.candidate_identity;
    full_request.device_identity = context_.device_identity;
    full_request.measurement_identity =
        "hygon-libtriton-jit-stage-hip-graph-v3-build-" +
        std::string(FLAGDNN_LIBTRITON_JIT_BUILD_IDENTITY) + "-stage-" +
        std::to_string(kernels_.size());
    full_request.cache_path = stage.selection_cache;
    full_request.warmup_milliseconds = stage.warmup;
    full_request.benchmark_milliseconds = stage.repetitions;
    full_request.candidate_ids.reserve(stage.variants.size());
    for (const HygonKernelArtifact &variant : stage.variants) {
      full_request.candidate_ids.push_back(variant.variant_id);
    }

    if (const auto cached =
            backend::autotune::find_cached_candidate(full_request)) {
      try {
        prepare_candidate(function, stage, stage.variants[*cached]);
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] cache hit "
                    << stage.candidate_identity.substr(0, 12) << " -> "
                    << stage.variants[*cached].variant_id << '\n';
        }
        return *cached;
      } catch (const std::exception &error) {
        if (!is_candidate_compatibility_error(error)) {
          throw;
        }
        backend::autotune::discard_cached_candidate(full_request);
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] discarded cached "
                    << stage.variants[*cached].variant_id << ": "
                    << error.what() << '\n';
        }
      }
    }

    // Triton tuning spaces can legitimately contain configurations that do
    // not fit a particular device (for example, a convolution tile whose
    // shared-memory requirement is too large). libtriton_jit currently
    // reports such failures while lazily compiling on the first launch. A
    // single device-incompatible configuration must not reject the whole
    // graph, so establish the runnable subset before invoking the shared
    // timing/cache policy.
    std::vector<std::size_t> runnable_indices;
    runnable_indices.reserve(stage.variants.size());
    std::string rejected_candidates;
    for (std::size_t index = 0; index < stage.variants.size(); ++index) {
      try {
        prepare_candidate(function, stage, stage.variants[index]);
        runnable_indices.push_back(index);
      } catch (const std::exception &error) {
        if (!is_candidate_compatibility_error(error)) {
          throw;
        }
        if (!rejected_candidates.empty()) {
          rejected_candidates += "; ";
        }
        rejected_candidates +=
            stage.variants[index].variant_id + ": " + std::string(error.what());
        if (logging_enabled()) {
          std::cerr << "[FlagDNN autotune/JIT] skipped "
                    << stage.variants[index].variant_id << ": " << error.what()
                    << '\n';
        }
      }
    }
    if (runnable_indices.empty()) {
      throw HygonError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                       "libtriton_jit autotune has no runnable candidates" +
                           (rejected_candidates.empty()
                                ? std::string{}
                                : std::string(": ") + rejected_candidates));
    }
    if (runnable_indices.size() == 1) {
      if (logging_enabled()) {
        std::cerr << "[FlagDNN autotune/JIT] only runnable candidate "
                  << stage.variants[runnable_indices.front()].variant_id
                  << '\n';
      }
      return runnable_indices.front();
    }

    backend::autotune::SelectionRequest request;
    request.candidate_identity = stage.candidate_identity;
    request.device_identity = context_.device_identity;
    request.measurement_identity = full_request.measurement_identity;
    request.cache_path = stage.selection_cache;
    request.warmup_milliseconds = stage.warmup;
    request.benchmark_milliseconds = stage.repetitions;
    request.candidate_ids.reserve(runnable_indices.size());
    for (const std::size_t index : runnable_indices) {
      request.candidate_ids.push_back(stage.variants[index].variant_id);
    }

    std::vector<const HygonKernelArtifact *> tuning_kernels;
    tuning_kernels.reserve(kernels_.size() + 1);
    for (const LoadedJitKernel &kernel : kernels_) {
      tuning_kernels.push_back(&kernel.specification);
    }
    tuning_kernels.push_back(&stage.variants[runnable_indices.front()]);
    JitTuningResources resources(tuning_kernels, logical_workspace_size_,
                                 workspace_alignment_);
    const auto initialize_candidate_inputs = [&]() {
      for (const LoadedJitKernel &prefix : kernels_) {
        RawArguments prefix_arguments(
            prefix.specification, resources.allocations(),
            resources.workspace(), scratch_pointer(resources.workspace()));
        launch_prepared_hip(prefix.prepared, resources.stream(),
                            prefix_arguments);
      }
    };
    const auto launch_candidate = [&](std::size_t index) {
      const HygonKernelArtifact &variant =
          stage.variants[runnable_indices[index]];
      RawArguments arguments(variant, resources.allocations(),
                             resources.workspace(),
                             scratch_pointer(resources.workspace()));
      launch_jit(function, variant, resources.stream(), arguments);
    };
    constexpr unsigned int kGraphBatchSize = 32;
    std::unique_ptr<CapturedLaunchBatch> captured_batch;
    std::size_t captured_candidate = runnable_indices.size();
    const auto batch_for = [&](std::size_t index) -> CapturedLaunchBatch & {
      if (captured_batch == nullptr || captured_candidate != index) {
        captured_batch.reset();
        captured_batch = std::make_unique<CapturedLaunchBatch>(
            resources.stream(), kGraphBatchSize,
            [&] { launch_candidate(index); });
        captured_candidate = index;
      }
      return *captured_batch;
    };
    const auto replay_count = [](unsigned int requested,
                                 unsigned int batch_size) {
      return (requested + batch_size - 1U) / batch_size;
    };

    const backend::autotune::SelectionResult result =
        backend::autotune::select_best_candidate(
            request,
            [&](std::size_t index, unsigned int iterations) {
              CapturedLaunchBatch &batch = batch_for(index);
              initialize_candidate_inputs();
              const unsigned int replays =
                  replay_count(iterations, batch.execution_count());
              for (unsigned int replay = 0; replay < replays; ++replay) {
                batch.launch(resources.stream());
              }
              check_hip(hipStreamSynchronize(resources.stream()),
                        "hipStreamSynchronize(libtriton_jit autotune warmup)");
            },
            [&](std::size_t index, unsigned int iterations) {
              CapturedLaunchBatch &batch = batch_for(index);
              initialize_candidate_inputs();
              const unsigned int replays =
                  replay_count(iterations, batch.execution_count());
              check_hip(hipEventRecord(resources.start(), resources.stream()),
                        "hipEventRecord(libtriton_jit autotune start)");
              for (unsigned int replay = 0; replay < replays; ++replay) {
                batch.launch(resources.stream());
              }
              check_hip(hipEventRecord(resources.stop(), resources.stream()),
                        "hipEventRecord(libtriton_jit autotune stop)");
              check_hip(hipEventSynchronize(resources.stop()),
                        "hipEventSynchronize(libtriton_jit autotune)");
              float milliseconds = 0.0F;
              check_hip(hipEventElapsedTime(&milliseconds, resources.start(),
                                            resources.stop()),
                        "hipEventElapsedTime(libtriton_jit autotune)");
              const unsigned int measured_iterations =
                  replays * batch.execution_count();
              return milliseconds / static_cast<float>(measured_iterations);
            });

    if (logging_enabled()) {
      const std::string &selected =
          request.candidate_ids[result.candidate_index];
      if (result.cache_hit) {
        std::cerr << "[FlagDNN autotune/JIT] cache hit "
                  << stage.candidate_identity.substr(0, 12) << " -> "
                  << selected << '\n';
      } else {
        for (std::size_t index = 0; index < result.median_milliseconds.size();
             ++index) {
          std::cerr << "[FlagDNN autotune/JIT] " << request.candidate_ids[index]
                    << " median_ms=" << result.median_milliseconds[index]
                    << '\n';
        }
        std::cerr << "[FlagDNN autotune/JIT] selected " << selected << '\n';
      }
    }
    return runnable_indices[result.candidate_index];
  }

  void prepare_candidate(const JitFunction &function,
                         const HygonStageArtifact &stage,
                         const HygonKernelArtifact &kernel) const {
    preflight_jit_candidate(source_snapshot_paths_.back(), stage.function_name,
                             kernel, context_.device);
    JitTuningResources resources(kernel, logical_workspace_size_,
                                 workspace_alignment_);
    RawArguments arguments(kernel, resources.allocations(),
                           resources.workspace(),
                           scratch_pointer(resources.workspace()));
    launch_jit(function, kernel, resources.stream(), arguments);
    check_hip(hipStreamSynchronize(resources.stream()),
              "hipStreamSynchronize(libtriton_jit prepare)");
  }

  void release_context() noexcept {
    if (retained_) {
      (void)hipDevicePrimaryCtxRelease(context_.device);
      retained_ = false;
    }
  }

  [[nodiscard]] DeviceAddress
  scratch_pointer(DeviceAddress workspace) const noexcept {
    return workspace + global_scratch_offset_;
  }

  EngineBuildContext context_;
  std::vector<std::int64_t> binding_uids_;
  std::vector<ExternalTensorBindingSpec> external_bindings_;
  std::vector<std::filesystem::path> source_snapshot_paths_;
  std::vector<LoadedJitKernel> kernels_;
  std::size_t logical_workspace_size_ = 0;
  std::size_t workspace_alignment_ = 1;
  std::size_t workspace_size_ = 0;
  std::size_t global_scratch_offset_ = 0;
  std::size_t global_scratch_size_ = 0;
  bool retained_ = false;
};

} // namespace

bool libtriton_jit_engine_available() noexcept { return true; }

std::unique_ptr<ExecutionEngine>
create_libtriton_jit_engine(const EngineBuildContext &context,
                            HygonArtifact artifact) {
  return std::make_unique<LibTritonJitEngine>(context, std::move(artifact));
}

} // namespace flagdnn::hygon

#else

namespace flagdnn::hygon {

bool libtriton_jit_engine_available() noexcept { return false; }

std::unique_ptr<ExecutionEngine>
create_libtriton_jit_engine(const EngineBuildContext &, HygonArtifact) {
  throw HygonError(
      FLAGDNN_BACKEND_RESULT_NOT_SUPPORTED,
      "libtriton_jit execution engine is not enabled in this Hygon plugin");
}

} // namespace flagdnn::hygon

#endif
