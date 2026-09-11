// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/engines/engine.hpp"

#include "backends/autotune_policy.hpp"
#include "backends/thead/engines/jit_candidate_compatibility.hpp"
#include "backends/thead/error.hpp"
#include "runtime/sha256.hpp"

#include <triton_jit/backend_config.h>
#include <triton_jit/backends/cuda_backend.h>
#include <triton_jit/jit_utils.h>
#include <triton_jit/triton_jit_function.h>

#include <Python.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef BACKEND_CUDA
#error "The THead execution engine requires CUDA-backend libtriton_jit"
#endif

#ifndef FLAGDNN_THEAD_TRITON_ROOT
#error "The THead execution engine requires a configured Triton root"
#endif

#ifndef FLAGDNN_THEAD_TRITON_METADATA
#error "The THead execution engine requires Triton distribution metadata"
#endif

#ifndef FLAGDNN_THEAD_PPU_SDK_ROOT
#error "The THead execution engine requires a configured PPU SDK root"
#endif

#ifndef FLAGDNN_THEAD_JIT_SHA256
#error "The THead execution engine requires a libtriton_jit hash"
#endif

#ifndef FLAGDNN_THEAD_JIT_STANDALONE_SHA256
#error "The THead execution engine requires a JIT compiler-script hash"
#endif

#ifndef FLAGDNN_THEAD_JIT_GEN_SSIG_SHA256
#error "The THead execution engine requires a JIT signature-script hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_INIT_SHA256
#error "The THead execution engine requires a Triton package hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_COMPILER_SHA256
#error "The THead execution engine requires a Triton CUDA compiler hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_DRIVER_SHA256
#error "The THead execution engine requires a Triton CUDA driver hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_FRONTEND_SHA256
#error "The THead execution engine requires a Triton frontend hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_METADATA_SHA256
#error "The THead execution engine requires a Triton metadata hash"
#endif

#ifndef FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256
#error "The THead execution engine requires full libtriton_jit provenance"
#endif

static_assert(triton_jit::CudaBackend::WARP_SIZE == 32);
static_assert(
    std::is_same_v<triton_jit::DefaultBackend, triton_jit::CudaBackend>);

namespace flagdnn::thead {
namespace {

using JitFunction = triton_jit::TritonJITFunction;

std::mutex jit_build_mutex;
std::once_flag python_runtime_once;
void* python_runtime_handle = nullptr;

[[noreturn]] void compilation_error(std::string message) {
  throw TheadError(FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
                   std::move(message));
}

void check_preparation(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    compilation_error(driver_error(result, operation));
  }
}

void set_required_environment(const char* name, const std::string& value) {
  if (setenv(name, value.c_str(), 1) != 0) {
    compilation_error("cannot configure " + std::string(name) +
                      " for THead libtriton_jit");
  }
}

void configure_environment(int capability) {
  const char* configured_triton_root =
      std::getenv("FLAGDNN_THEAD_TRITON_ROOT");
  const std::string selected_triton_root =
      configured_triton_root != nullptr && configured_triton_root[0] != '\0'
          ? configured_triton_root
          : FLAGDNN_THEAD_TRITON_ROOT;
  set_required_environment("TRITON_JIT_BACKEND", "CUDA");
  set_required_environment("FLAGDNN_THEAD_TRITON_ROOT",
                           selected_triton_root);
  set_required_environment("FLAGDNN_THEAD_PPU_SDK_ROOT",
                           FLAGDNN_THEAD_PPU_SDK_ROOT);
  set_required_environment("PPU_SDK", FLAGDNN_THEAD_PPU_SDK_ROOT);
  set_required_environment("PPU_HOME", FLAGDNN_THEAD_PPU_SDK_ROOT);
  set_required_environment(
      "CUDA_PATH",
      std::string(FLAGDNN_THEAD_PPU_SDK_ROOT) + "/CUDA_SDK");
  set_required_environment(
      "TRITON_PTXAS_PATH",
      std::string(FLAGDNN_THEAD_PPU_SDK_ROOT) + "/CUDA_SDK/bin/ptxas");
  set_required_environment(
      "TRITON_IR_FORMATTER_PATH",
      std::string(FLAGDNN_THEAD_PPU_SDK_ROOT) +
          "/bin/llvm-irformatter");
  if (std::string(FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND) == "ppu") {
    set_required_environment(
        "TRITON_PPU_LLC_PATH",
        std::string(FLAGDNN_THEAD_PPU_SDK_ROOT) + "/bin/ppu-llc");
  }
  set_required_environment("TRITON_OVERRIDE_ARCH",
                           "sm" + std::to_string(capability));
  set_required_environment("PYTHONDONTWRITEBYTECODE", "1");
}

void promote_python_runtime() {
  std::call_once(python_runtime_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(&Py_IsInitialized));
    if (dladdr(address, &information) == 0 ||
        information.dli_fname == nullptr) {
      compilation_error("cannot locate the embedded Python runtime");
    }
    python_runtime_handle =
        dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (python_runtime_handle == nullptr) {
      python_runtime_handle =
          dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (python_runtime_handle == nullptr) {
      const char* detail = dlerror();
      compilation_error(
          "cannot expose embedded Python symbols to Triton extensions: " +
          std::string(detail == nullptr ? "unknown dlopen error" : detail));
    }
  });
}

struct PythonObjectDeleter {
  void operator()(PyObject* object) const noexcept { Py_XDECREF(object); }
};

using PythonObject = std::unique_ptr<PyObject, PythonObjectDeleter>;

std::string python_exception() {
  if (PyErr_Occurred() == nullptr) {
    return "unknown Python error";
  }
  PyObject* type = nullptr;
  PyObject* value = nullptr;
  PyObject* traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PythonObject owned_type(type);
  PythonObject owned_value(value);
  PythonObject owned_traceback(traceback);
  PythonObject rendered(value == nullptr ? nullptr : PyObject_Str(value));
  if (rendered == nullptr) {
    PyErr_Clear();
    return "unprintable Python error";
  }
  const char* text = PyUnicode_AsUTF8(rendered.get());
  if (text == nullptr) {
    PyErr_Clear();
    return "non-UTF-8 Python error";
  }
  return text;
}

[[noreturn]] void python_failure(std::string_view operation) {
  compilation_error(std::string(operation) + ": " + python_exception());
}

PythonObject import_module(const char* name) {
  PythonObject module(PyImport_ImportModule(name));
  if (module == nullptr) {
    python_failure("cannot import Python module " + std::string(name));
  }
  return module;
}

PythonObject get_attribute(PyObject* object,
                           const char* name,
                           std::string_view description) {
  PythonObject value(PyObject_GetAttrString(object, name));
  if (value == nullptr) {
    python_failure("cannot read " + std::string(description));
  }
  return value;
}

std::filesystem::path python_module_path(PyObject* module,
                                         std::string_view name) {
  PythonObject origin = get_attribute(module, "__file__", name);
  if (!PyUnicode_Check(origin.get())) {
    compilation_error("Python module " + std::string(name) +
                      " has no string file origin");
  }
  const char* text = PyUnicode_AsUTF8(origin.get());
  if (text == nullptr) {
    python_failure("cannot decode Python module origin");
  }
  std::error_code error;
  const std::filesystem::path result =
      std::filesystem::canonical(text, error);
  if (error) {
    compilation_error("cannot resolve Python module " + std::string(name) +
                      " at " + text);
  }
  return result;
}

bool path_below(const std::filesystem::path& root,
                const std::filesystem::path& path) {
  const std::filesystem::path relative = path.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() && relative != ".." &&
         *relative.begin() != "..";
}

void require_file_hash(const std::filesystem::path& path,
                       std::string_view expected,
                       std::string_view description) {
  try {
    if (flagdnn::native::sha256_file(path) != expected) {
      compilation_error(std::string(description) + " hash mismatch: " +
                        path.string());
    }
  } catch (const TheadError&) {
    throw;
  } catch (const std::exception& error) {
    compilation_error("cannot hash " + std::string(description) + ": " +
                      error.what());
  }
}

std::filesystem::path mapped_jit_library() {
  Dl_info information{};
  const auto address = reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(
      &triton_jit::get_script_dir));
  if (dladdr(address, &information) == 0 ||
      information.dli_fname == nullptr) {
    compilation_error("cannot locate the mapped libtriton_jit image");
  }
  std::error_code error;
  const std::filesystem::path result =
      std::filesystem::canonical(information.dli_fname, error);
  if (error) {
    compilation_error("cannot resolve the mapped libtriton_jit image");
  }
  return result;
}

void prepend_python_path(PyObject* path, const std::string& component) {
  PythonObject value(PyUnicode_FromString(component.c_str()));
  if (value == nullptr) {
    python_failure("cannot encode a THead Python search path");
  }
  const int present = PySequence_Contains(path, value.get());
  if (present < 0) {
    python_failure("cannot inspect sys.path");
  }
  if (present == 0 && PyList_Insert(path, 0, value.get()) != 0) {
    python_failure("cannot prepend the THead Python search path");
  }
}

void require_module_file(PyObject* module,
                         std::string_view name,
                         const std::filesystem::path& expected,
                         std::string_view expected_sha256) {
  const std::filesystem::path actual = python_module_path(module, name);
  if (actual != expected) {
    compilation_error("Python module " + std::string(name) +
                      " was loaded from an unexpected path: " +
                      actual.string());
  }
  require_file_hash(actual, expected_sha256,
                    "Python module " + std::string(name));
}

void validate_python_environment(int capability,
                                 const std::filesystem::path& script_root) {
  const bool initialized_here = Py_IsInitialized() == 0;
  if (initialized_here) {
    Py_InitializeEx(0);
    if (Py_IsInitialized() == 0) {
      compilation_error("cannot initialize embedded Python");
    }
  }
  PyGILState_STATE gil_state{};
  if (!initialized_here) {
    gil_state = PyGILState_Ensure();
  }

  const auto release_gil = [initialized_here, gil_state] {
    if (initialized_here) {
      (void)PyEval_SaveThread();
    } else {
      PyGILState_Release(gil_state);
    }
  };

  try {
    {
      PythonObject sys = import_module("sys");
      PythonObject search_path = get_attribute(sys.get(), "path", "sys.path");
      if (!PyList_Check(search_path.get())) {
        compilation_error("embedded Python sys.path is not a list");
      }
      const char* configured_triton_root =
          std::getenv("FLAGDNN_THEAD_TRITON_ROOT");
      const std::string selected_triton_root =
          configured_triton_root != nullptr &&
                  configured_triton_root[0] != '\0'
              ? configured_triton_root
              : FLAGDNN_THEAD_TRITON_ROOT;
      prepend_python_path(search_path.get(), script_root.string());

      PythonObject bridge = import_module("flagdnn_thead_jit_compat");
      require_module_file(bridge.get(), "flagdnn_thead_jit_compat",
                          script_root / "flagdnn_thead_jit_compat.py",
                          FLAGDNN_THEAD_JIT_COMPAT_SHA256);
      PythonObject configure_path = get_attribute(
          bridge.get(), "configure_triton_path", "THead Python package path");
      PythonObject package_root(PyUnicode_FromString(selected_triton_root.c_str()));
      if (package_root == nullptr) {
        python_failure("cannot encode the configured Triton root");
      }
      PythonObject configured_path(PyObject_CallFunctionObjArgs(
          configure_path.get(), package_root.get(), nullptr));
      if (configured_path == nullptr) {
        python_failure("cannot configure the THead Python package path");
      }

      std::error_code error;
      const std::filesystem::path triton_root =
          std::filesystem::canonical(selected_triton_root, error);
      if (error) {
        compilation_error("cannot resolve configured Triton root");
      }
      const std::filesystem::path package_init =
          std::filesystem::canonical(triton_root / "triton/__init__.py", error);
      if (error) {
        compilation_error("configured Triton package is incomplete");
      }
      const std::filesystem::path cuda_compiler = std::filesystem::canonical(
          triton_root / ("triton/backends/"
                         FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND "/compiler.py"),
          error);
      if (error) {
        compilation_error("configured Triton CUDA compiler is missing");
      }
      const std::filesystem::path cuda_driver = std::filesystem::canonical(
          triton_root / ("triton/backends/"
                         FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND "/driver.py"),
          error);
      if (error) {
        compilation_error("configured Triton CUDA driver is missing");
      }
      const std::filesystem::path compiler_frontend =
          std::filesystem::canonical(
              triton_root / FLAGDNN_THEAD_TRITON_FRONTEND_RELATIVE, error);
      if (error) {
        compilation_error("configured Triton compiler frontend is missing");
      }
      const std::filesystem::path configured_metadata =
          std::filesystem::path(FLAGDNN_THEAD_TRITON_METADATA);
      const std::filesystem::path compiled_root =
          std::filesystem::path(FLAGDNN_THEAD_TRITON_ROOT);
      const std::filesystem::path metadata_relative =
          configured_metadata.lexically_relative(compiled_root);
      if (metadata_relative.empty() || metadata_relative.is_absolute() ||
          *metadata_relative.begin() == "..") {
        compilation_error("compiled Triton metadata path escaped its root");
      }
      const std::filesystem::path distribution_metadata =
          std::filesystem::canonical(triton_root / metadata_relative, error);
      if (error) {
        compilation_error("configured Triton distribution metadata is missing");
      }

      PythonObject triton = import_module("triton");
      // FlagTree owns static pybind objects in libtriton. They are destroyed
      // by the C runtime after main(), when an embedded interpreter otherwise
      // has no GIL. Register after importing that extension so this callback
      // runs before its destructors. A caller-owned interpreter retains its
      // own shutdown policy. The plugin is linked NODELETE for callback safety.
      if (initialized_here && std::atexit([] {
            if (Py_IsInitialized()) {
              (void)PyGILState_Ensure();
            }
          }) != 0) {
        compilation_error("cannot register embedded Python shutdown handling");
      }
      require_module_file(triton.get(), "triton", package_init,
                          FLAGDNN_THEAD_TRITON_INIT_SHA256);
      if (!path_below(triton_root,
                      python_module_path(triton.get(), "triton"))) {
        compilation_error("imported Triton package escaped its configured root");
      }
      require_file_hash(distribution_metadata,
                        FLAGDNN_THEAD_TRITON_METADATA_SHA256,
                        "Triton distribution metadata");

      PythonObject backend_module = import_module("triton.backends");
      PythonObject catalog =
          get_attribute(backend_module.get(), "backends", "Triton backends");
      if (!PyDict_Check(catalog.get())) {
        compilation_error("Triton backend catalog is not a dictionary");
      }
      PyObject* backend = PyDict_GetItemString(
          catalog.get(), FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND);
      if (backend == nullptr) {
        compilation_error("Triton backend catalog has no CUDA codegen backend");
      }

      PythonObject cuda_compiler_module =
          import_module("triton.backends."
                        FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND ".compiler");
      require_module_file(cuda_compiler_module.get(),
                          "triton.backends."
                          FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND ".compiler",
                          cuda_compiler,
                          FLAGDNN_THEAD_TRITON_COMPILER_SHA256);
      PythonObject cuda_driver_module =
          import_module("triton.backends."
                        FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND ".driver");
      require_module_file(cuda_driver_module.get(),
                          "triton.backends."
                          FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND ".driver",
                          cuda_driver,
                          FLAGDNN_THEAD_TRITON_DRIVER_SHA256);
      PythonObject frontend_module = import_module("triton.compiler.compiler");
      require_module_file(frontend_module.get(), "triton.compiler.compiler",
                          compiler_frontend,
                          FLAGDNN_THEAD_TRITON_FRONTEND_SHA256);

      PythonObject compiler_class =
          get_attribute(backend, "compiler", "Triton CUDA compiler class");
      PythonObject compiler_api = import_module("triton.backends.compiler");
      PythonObject target_class = get_attribute(
          compiler_api.get(), "GPUTarget", "Triton GPUTarget class");
      PythonObject backend_name(PyUnicode_FromString("cuda"));
      PythonObject architecture(PyLong_FromLong(capability));
      PythonObject warp_size(PyLong_FromLong(32));
      if (backend_name == nullptr || architecture == nullptr ||
          warp_size == nullptr) {
        python_failure("cannot build the Triton CUDA target identity");
      }
      PythonObject target(PyObject_CallFunctionObjArgs(target_class.get(),
                                                       backend_name.get(),
                                                       architecture.get(),
                                                       warp_size.get(),
                                                       nullptr));
      if (target == nullptr) {
        python_failure("cannot construct the Triton CUDA target");
      }
      PythonObject compiler(PyObject_CallFunctionObjArgs(
          compiler_class.get(), target.get(), nullptr));
      if (compiler == nullptr) {
        python_failure("cannot instantiate the Triton CUDA compiler");
      }
      PythonObject binary_extension = get_attribute(
          compiler.get(), "binary_ext", "Triton CUDA binary extension");
      const char* extension = PyUnicode_Check(binary_extension.get())
                                  ? PyUnicode_AsUTF8(binary_extension.get())
                                  : nullptr;
      if (extension == nullptr ||
          std::string_view(extension) !=
              FLAGDNN_THEAD_TRITON_BINARY_EXTENSION) {
        PyErr_Clear();
        compilation_error("unexpected Triton PPU binary extension");
      }

      const std::filesystem::path standalone = std::filesystem::canonical(
          script_root / "standalone_compile.py", error);
      if (error) {
        compilation_error("libtriton_jit standalone compiler is missing");
      }
      const std::filesystem::path signature =
          std::filesystem::canonical(script_root / "gen_ssig.py", error);
      if (error) {
        compilation_error("libtriton_jit signature helper is missing");
      }
      PythonObject standalone_module = import_module("standalone_compile");
      require_module_file(standalone_module.get(), "standalone_compile",
                          standalone,
                          FLAGDNN_THEAD_JIT_STANDALONE_SHA256);
      if (std::string_view(FLAGDNN_THEAD_TRITON_CODEGEN_BACKEND) == "ppu") {
        PythonObject install = get_attribute(
            bridge.get(), "install_cuda_jit_bridge", "PPU CUDA JIT bridge");
        PythonObject installed(PyObject_CallFunctionObjArgs(
            install.get(), standalone_module.get(), nullptr));
        if (installed == nullptr) {
          python_failure("cannot install the PPU CUDA JIT bridge");
        }
      }
      PythonObject signature_module = import_module("gen_ssig");
      require_module_file(signature_module.get(), "gen_ssig", signature,
                          FLAGDNN_THEAD_JIT_GEN_SSIG_SHA256);
    }
    release_gil();
  } catch (...) {
    release_gil();
    throw;
  }
}

int target_capability(std::string_view target) {
  const std::size_t marker = target.rfind("_cc");
  if (marker == std::string_view::npos || marker + 3 == target.size()) {
    compilation_error("THead target has no compatibility capability");
  }
  const std::string_view digits = target.substr(marker + 3);
  int result = 0;
  const auto [end, error] =
      std::from_chars(digits.data(), digits.data() + digits.size(), result);
  if (error != std::errc{} || end != digits.data() + digits.size() ||
      result <= 0) {
    compilation_error("THead target compatibility capability is invalid");
  }
  return result;
}

void validate_jit_runtime(int capability) {
  configure_environment(capability);
  promote_python_runtime();
  std::filesystem::path library;
  std::filesystem::path script_root;
  try {
    library = mapped_jit_library();
    script_root = triton_jit::get_script_dir();
  } catch (const std::exception& error) {
    compilation_error("cannot resolve libtriton_jit runtime resources: " +
                      std::string(error.what()));
  }
  require_file_hash(library, FLAGDNN_THEAD_JIT_SHA256,
                    "mapped libtriton_jit");
  validate_python_environment(capability, script_root);
}

struct TemporaryAllocation {
  std::int64_t uid = 0;
  std::size_t alignment = 1;
  CUdeviceptr pointer = 0;
};

class PreparationResources final {
 public:
  explicit PreparationResources(const ExecutionProgramArtifact& artifact) {
    try {
      check_preparation(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                        "cuStreamCreate(THead JIT preparation)");
      allocations_.reserve(artifact.tensors.size());
      for (const TensorArtifact& tensor : artifact.tensors) {
        if (tensor.is_virtual) {
          continue;
        }
        TemporaryAllocation allocation{tensor.uid, tensor.alignment, 0};
        check_preparation(
            cuMemAlloc(&allocation.pointer, tensor.storage_size),
            "cuMemAlloc(THead JIT preparation tensor)");
        if (allocation.pointer % allocation.alignment != 0) {
          compilation_error(
              "THead JIT preparation tensor does not satisfy alignment");
        }
        allocations_.push_back(allocation);
      }
      if (artifact.workspace_size != 0) {
        check_preparation(cuMemAlloc(&workspace_, artifact.workspace_size),
                          "cuMemAlloc(THead JIT preparation workspace)");
        if (workspace_ % artifact.workspace_alignment != 0) {
          compilation_error(
              "THead JIT preparation workspace does not satisfy alignment");
        }
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~PreparationResources() { cleanup(); }

  PreparationResources(const PreparationResources&) = delete;
  PreparationResources& operator=(const PreparationResources&) = delete;

  [[nodiscard]] CUdeviceptr tensor(std::int64_t uid) const {
    const auto found = std::find_if(
        allocations_.begin(), allocations_.end(),
        [uid](const TemporaryAllocation& allocation) {
          return allocation.uid == uid;
        });
    if (found == allocations_.end()) {
      compilation_error(
          "THead JIT preparation is missing a tensor allocation");
    }
    return found->pointer;
  }

  [[nodiscard]] CUdeviceptr workspace() const noexcept { return workspace_; }
  [[nodiscard]] CUstream stream() const noexcept { return stream_; }

 private:
  void cleanup() noexcept {
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
      stream_ = nullptr;
    }
    if (workspace_ != 0) {
      (void)cuMemFree(workspace_);
      workspace_ = 0;
    }
    for (TemporaryAllocation& allocation : allocations_) {
      if (allocation.pointer != 0) {
        (void)cuMemFree(allocation.pointer);
        allocation.pointer = 0;
      }
    }
  }

  std::vector<TemporaryAllocation> allocations_;
  CUdeviceptr workspace_ = 0;
  CUstream stream_ = nullptr;
};

class TimingEvents final {
 public:
  TimingEvents() {
    check_preparation(cuEventCreate(&start_, CU_EVENT_DEFAULT),
                      "cuEventCreate(THead autotune start)");
    try {
      check_preparation(cuEventCreate(&stop_, CU_EVENT_DEFAULT),
                        "cuEventCreate(THead autotune stop)");
    } catch (...) {
      (void)cuEventDestroy(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~TimingEvents() {
    if (stop_ != nullptr) {
      (void)cuEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cuEventDestroy(start_);
    }
  }

  TimingEvents(const TimingEvents&) = delete;
  TimingEvents& operator=(const TimingEvents&) = delete;

  [[nodiscard]] CUevent start() const noexcept { return start_; }
  [[nodiscard]] CUevent stop() const noexcept { return stop_; }

 private:
  CUevent start_ = nullptr;
  CUevent stop_ = nullptr;
};

unsigned int device_limit(CUdevice device,
                          CUdevice_attribute attribute,
                          const char* operation) {
  int value = 0;
  check_preparation(cuDeviceGetAttribute(&value, attribute, device), operation);
  if (value <= 0) {
    compilation_error(std::string(operation) +
                      " returned a nonpositive limit");
  }
  return static_cast<unsigned int>(value);
}

CandidateDeviceLimits query_device_limits(CUdevice device) {
  return {
      device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                   "cuDeviceGetAttribute(max threads per block)"),
      {device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X,
                    "cuDeviceGetAttribute(max block X)"),
       device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y,
                    "cuDeviceGetAttribute(max block Y)"),
       device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z,
                    "cuDeviceGetAttribute(max block Z)")},
      {device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
                    "cuDeviceGetAttribute(max grid X)"),
       device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
                    "cuDeviceGetAttribute(max grid Y)"),
       device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z,
                    "cuDeviceGetAttribute(max grid Z)")},
      device_limit(device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
                   "cuDeviceGetAttribute(max shared memory per block)"),
  };
}

struct ArgumentValue {
  CUdeviceptr pointer;
  std::int32_t scalar_i32;
  float scalar_f32;
};

class LaunchArguments final {
 public:
  template <typename PointerResolver>
  LaunchArguments(const KernelVariant& variant,
                  PointerResolver&& resolve_pointer) {
    const std::size_t visible_count = variant.arguments.size();
    require(visible_count <= FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS,
            "THead kernel has too many runtime arguments",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    parameter_count_ = visible_count + 2;
    for (std::size_t index = 0; index < visible_count; ++index) {
      const KernelArgument& argument = variant.arguments[index];
      if (argument.kind == ArgumentKind::kTensor ||
          argument.kind == ArgumentKind::kWorkspaceTensor) {
        values_[index].pointer = resolve_pointer(argument);
        parameters_[index] = &values_[index].pointer;
      } else if (argument.kind == ArgumentKind::kScalarI32) {
        values_[index].scalar_i32 = argument.scalar_i32;
        parameters_[index] = &values_[index].scalar_i32;
      } else if (argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_f32 = argument.scalar_f32;
        parameters_[index] = &values_[index].scalar_f32;
      } else {
        compilation_error("THead kernel argument kind is unsupported");
      }
    }
    parameters_[visible_count] = &global_scratch_;
    parameters_[visible_count + 1] = &profile_scratch_;
  }

  [[nodiscard]] void** data() noexcept { return parameters_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return parameter_count_; }

 private:
  // The ABI allows 4096 arguments, but ordinary kernels use only a handful.
  // Initialize each exposed value/pointer in the constructor, avoiding a 96 KiB
  // memset on every launch. Unused array entries are never passed to the driver.
  std::array<ArgumentValue, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS> values_;
  std::array<void*, FLAGDNN_BACKEND_MAX_KERNEL_ARGUMENTS + 2> parameters_;
  std::size_t parameter_count_ = 0;
  CUdeviceptr global_scratch_ = 0;
  CUdeviceptr profile_scratch_ = 0;
};

std::vector<std::string_view> signature_tokens(std::string_view signature) {
  std::vector<std::string_view> result;
  std::size_t start = 0;
  unsigned int nesting = 0;
  for (std::size_t index = 0; index <= signature.size(); ++index) {
    const bool at_end = index == signature.size();
    const char character = at_end ? ',' : signature[index];
    if (!at_end && character == '(') {
      ++nesting;
    } else if (!at_end && character == ')') {
      if (nesting == 0) {
        compilation_error("THead JIT signature has unbalanced parentheses");
      }
      --nesting;
    }
    if (character == ',' && nesting == 0) {
      const std::string_view token = signature.substr(start, index - start);
      if (token.empty()) {
        compilation_error("THead JIT signature contains an empty token");
      }
      result.push_back(token);
      start = index + 1;
    }
  }
  if (nesting != 0) {
    compilation_error("THead JIT signature has unbalanced parentheses");
  }
  return result;
}

void validate_jit_abi(const JitFunction& function,
                      const KernelVariant& variant) {
  const triton_jit::StaticSignature& static_signature =
      function.get_static_sig();
  const std::vector<std::string_view> tokens =
      signature_tokens(variant.full_signature);
  if (static_signature.num_args < 0 ||
      static_cast<std::size_t>(static_signature.num_args) != tokens.size() ||
      static_signature.arg_type.size() != tokens.size()) {
    compilation_error(
        "THead JIT static signature differs from the artifact signature");
  }

  std::size_t runtime_arguments = 0;
  for (std::size_t index = 0; index < tokens.size(); ++index) {
    const triton_jit::ArgType kind = static_signature.arg_type[index];
    if (kind == triton_jit::ArgType::CONSTEXPR) {
      continue;
    }
    const bool specialized_one =
        !tokens[index].starts_with('*') && tokens[index].ends_with(":1") &&
        (kind == triton_jit::ArgType::SPECIALIZED ||
         kind == triton_jit::ArgType::SPECIALIZED_NO_ALIGNMENT);
    // libtriton_jit specializes optional pointer parameters to Python None.
    // Such parameters have no runtime argument in the compiled CUDA ABI.
    const bool specialized_none = tokens[index] == "nullopt" &&
        (kind == triton_jit::ArgType::SPECIALIZED ||
         kind == triton_jit::ArgType::SPECIALIZED_NO_ALIGNMENT);
    if (!specialized_one && !specialized_none) {
      ++runtime_arguments;
    }
  }
  if (runtime_arguments != variant.arguments.size()) {
    compilation_error(
        "THead JIT runtime argument count differs from the artifact ABI");
  }
}

void launch_jit(const JitFunction& function,
                const KernelVariant& variant,
                CUstream stream,
                LaunchArguments& arguments) {
  if (variant.maxnreg.has_value() ||
      !variant.ppu_compiler_options.empty()) {
    compilation_error(
        "THead CUDA-backend libtriton_jit raw-argument ABI cannot convey "
        "maxnreg or PPU compiler options");
  }
  function.launch_with_raw_args(stream, variant.grid[0], variant.grid[1],
                                variant.grid[2], variant.num_warps,
                                variant.num_stages, variant.full_signature,
                                arguments.data(), arguments.size());
}

struct PreparedLaunch {
  CUfunction function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

PreparedLaunch capture_prepared_launch(const JitFunction& function,
                                       const KernelVariant& variant,
                                       PreparationResources& resources) {
  LaunchArguments arguments(
      variant, [&](const KernelArgument& argument) -> CUdeviceptr {
        if (argument.kind == ArgumentKind::kTensor) {
          return resources.tensor(argument.uid);
        }
        if (resources.workspace() == 0) {
          compilation_error(
              "THead JIT preparation is missing artifact workspace");
        }
        return resources.workspace() + argument.workspace_offset;
      });

  CUgraph graph = nullptr;
  bool capture_active = false;
  try {
    check_preparation(
        cuStreamBeginCapture(resources.stream(), CU_STREAM_CAPTURE_MODE_RELAXED),
        "cuStreamBeginCapture(THead prepared launch)");
    capture_active = true;
    launch_jit(function, variant, resources.stream(), arguments);
    check_preparation(cuStreamEndCapture(resources.stream(), &graph),
                      "cuStreamEndCapture(THead prepared launch)");
    capture_active = false;

    std::size_t node_count = 0;
    check_preparation(cuGraphGetNodes(graph, nullptr, &node_count),
                      "cuGraphGetNodes(THead prepared launch count)");
    if (node_count != 1) {
      compilation_error(
          "one THead execution stage must capture exactly one kernel node");
    }
    CUgraphNode node = nullptr;
    check_preparation(cuGraphGetNodes(graph, &node, &node_count),
                      "cuGraphGetNodes(THead prepared launch)");
    CUgraphNodeType type = CU_GRAPH_NODE_TYPE_EMPTY;
    check_preparation(cuGraphNodeGetType(node, &type),
                      "cuGraphNodeGetType(THead prepared launch)");
    if (type != CU_GRAPH_NODE_TYPE_KERNEL) {
      compilation_error("THead JIT did not capture a kernel launch");
    }

    CUDA_KERNEL_NODE_PARAMS parameters{};
    check_preparation(cuGraphKernelNodeGetParams(node, &parameters),
                      "cuGraphKernelNodeGetParams(THead prepared launch)");
    const PreparedLaunch result{
        parameters.func,
        {parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ},
        {parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ},
        parameters.sharedMemBytes};
    if (result.function == nullptr || result.grid != variant.grid ||
        result.block != variant.block ||
        result.shared_memory != variant.shared_memory) {
      const auto dimensions = [](const std::array<unsigned int, 3>& value) {
        return std::to_string(value[0]) + "x" + std::to_string(value[1]) +
               "x" + std::to_string(value[2]);
      };
      compilation_error(
          "captured THead launch metadata differs from the artifact: "
          "grid=" + dimensions(result.grid) + "/" +
          dimensions(variant.grid) + " block=" +
          dimensions(result.block) + "/" + dimensions(variant.block) +
          " shared=" + std::to_string(result.shared_memory) + "/" +
          std::to_string(variant.shared_memory));
    }
    check_preparation(cuGraphDestroy(graph),
                      "cuGraphDestroy(THead prepared launch)");
    graph = nullptr;
    return result;
  } catch (...) {
    if (capture_active) {
      CUgraph abandoned = nullptr;
      if (cuStreamEndCapture(resources.stream(), &abandoned) == CUDA_SUCCESS &&
          abandoned != nullptr) {
        (void)cuGraphDestroy(abandoned);
      }
    } else if (graph != nullptr) {
      (void)cuGraphDestroy(graph);
    }
    throw;
  }
}

struct BindingRequirement {
  std::int64_t uid = 0;
  std::size_t alignment = 1;
};

struct PreparedStage {
  KernelVariant variant;
  PreparedLaunch launch;
};

void launch_prepared_for_tuning(const PreparedStage& stage,
                                PreparationResources& resources) {
  LaunchArguments arguments(
      stage.variant, [&](const KernelArgument& argument) -> CUdeviceptr {
        if (argument.kind == ArgumentKind::kTensor) {
          return resources.tensor(argument.uid);
        }
        if (resources.workspace() == 0) {
          compilation_error(
              "THead autotune is missing artifact workspace");
        }
        return resources.workspace() + argument.workspace_offset;
      });
  check_preparation(cuLaunchKernel(stage.launch.function,
                                   stage.launch.grid[0],
                                   stage.launch.grid[1],
                                   stage.launch.grid[2],
                                   stage.launch.block[0],
                                   stage.launch.block[1],
                                   stage.launch.block[2],
                                   stage.launch.shared_memory,
                                   resources.stream(),
                                   arguments.data(),
                                   nullptr),
                    "cuLaunchKernel(THead autotune dependency)");
}

void launch_candidate_for_tuning(const JitFunction& function,
                                 const KernelVariant& variant,
                                 PreparationResources& resources) {
  LaunchArguments arguments(
      variant, [&](const KernelArgument& argument) -> CUdeviceptr {
        if (argument.kind == ArgumentKind::kTensor) {
          return resources.tensor(argument.uid);
        }
        if (resources.workspace() == 0) {
          compilation_error(
              "THead autotune is missing artifact workspace");
        }
        return resources.workspace() + argument.workspace_offset;
      });
  launch_jit(function, variant, resources.stream(), arguments);
}

void prepare_candidate(const JitFunction& function,
                       const KernelVariant& variant,
                       PreparationResources& resources,
                       const CandidateDeviceLimits& limits) {
  validate_candidate_resources(variant, limits);
  launch_candidate_for_tuning(function, variant, resources);
  check_preparation(cuStreamSynchronize(resources.stream()),
                    "cuStreamSynchronize(THead autotune candidate prepare)");
}

std::size_t select_candidate(
    const JitFunction& function,
    const ExecutionStage& stage,
    PreparationResources& resources,
    const CandidateDeviceLimits& limits,
    const EngineBuildContext& context,
    const std::vector<PreparedStage>& prepared_prefix) {
  require(stage.autotune && stage.variants.size() >= 2,
          "THead autotune stage is incomplete",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  backend::autotune::SelectionRequest full_request;
  full_request.candidate_identity = stage.candidate_identity;
  full_request.device_identity = context.device_identity;
  full_request.measurement_identity =
      "thead-libtriton-jit-cuda-events-v1-provenance-" +
      std::string(FLAGDNN_THEAD_TRITON_JIT_PROVENANCE_SHA256) + "-stage-" +
      std::to_string(stage.stage_id);
  full_request.cache_path = stage.selection_cache;
  full_request.warmup_milliseconds = stage.warmup;
  full_request.benchmark_milliseconds = stage.repetitions;
  full_request.candidate_ids.reserve(stage.variants.size());
  for (const KernelVariant& variant : stage.variants) {
    full_request.candidate_ids.push_back(variant.variant_id);
  }

  if (const auto cached =
          backend::autotune::find_cached_candidate(full_request)) {
    try {
      prepare_candidate(function, stage.variants.at(*cached), resources,
                        limits);
      return *cached;
    } catch (const CandidateCompatibilityError&) {
      backend::autotune::discard_cached_candidate(full_request);
    }
  }

  std::vector<std::size_t> runnable_indices;
  runnable_indices.reserve(stage.variants.size());
  std::string rejected_candidates;
  for (std::size_t index = 0; index < stage.variants.size(); ++index) {
    try {
      prepare_candidate(function, stage.variants[index], resources, limits);
      runnable_indices.push_back(index);
    } catch (const CandidateCompatibilityError& error) {
      if (!rejected_candidates.empty()) {
        rejected_candidates += "; ";
      }
      rejected_candidates +=
          stage.variants[index].variant_id + ": " + error.what();
    }
  }
  if (runnable_indices.empty()) {
    compilation_error(
        "THead autotune has no resource-compatible candidates" +
        (rejected_candidates.empty()
             ? std::string{}
             : std::string(": ") + rejected_candidates));
  }
  if (runnable_indices.size() == 1) {
    return runnable_indices.front();
  }

  backend::autotune::SelectionRequest request;
  request.candidate_identity = full_request.candidate_identity;
  request.device_identity = full_request.device_identity;
  request.measurement_identity = full_request.measurement_identity;
  request.cache_path = full_request.cache_path;
  request.warmup_milliseconds = full_request.warmup_milliseconds;
  request.benchmark_milliseconds = full_request.benchmark_milliseconds;
  request.candidate_ids.reserve(runnable_indices.size());
  for (const std::size_t index : runnable_indices) {
    request.candidate_ids.push_back(stage.variants[index].variant_id);
  }

  TimingEvents events;
  const auto initialize_candidate_inputs = [&] {
    for (const PreparedStage& dependency : prepared_prefix) {
      launch_prepared_for_tuning(dependency, resources);
    }
  };
  const auto launch_candidate = [&](std::size_t runnable_index) {
    launch_candidate_for_tuning(
        function, stage.variants[runnable_indices.at(runnable_index)],
        resources);
  };
  const backend::autotune::SelectionResult result =
      backend::autotune::select_best_candidate(
          request,
          [&](std::size_t index, unsigned int iterations) {
            initialize_candidate_inputs();
            for (unsigned int iteration = 0; iteration < iterations;
                 ++iteration) {
              launch_candidate(index);
            }
            check_preparation(
                cuStreamSynchronize(resources.stream()),
                "cuStreamSynchronize(THead autotune warmup)");
          },
          [&](std::size_t index, unsigned int iterations) {
            initialize_candidate_inputs();
            check_preparation(cuEventRecord(events.start(), resources.stream()),
                              "cuEventRecord(THead autotune start)");
            for (unsigned int iteration = 0; iteration < iterations;
                 ++iteration) {
              launch_candidate(index);
            }
            check_preparation(cuEventRecord(events.stop(), resources.stream()),
                              "cuEventRecord(THead autotune stop)");
            check_preparation(cuEventSynchronize(events.stop()),
                              "cuEventSynchronize(THead autotune stop)");
            float milliseconds = 0.0F;
            check_preparation(
                cuEventElapsedTime(&milliseconds, events.start(),
                                   events.stop()),
                "cuEventElapsedTime(THead autotune)");
            return milliseconds / static_cast<float>(iterations);
          });
  return runnable_indices.at(result.candidate_index);
}

std::vector<const ExecutionStage*> execution_order(
    const std::vector<ExecutionStage>& stages) {
  std::map<std::size_t, std::size_t> indices;
  for (std::size_t index = 0; index < stages.size(); ++index) {
    indices.emplace(stages[index].stage_id, index);
  }
  std::vector<std::size_t> indegree(stages.size(), 0);
  std::vector<std::vector<std::size_t>> dependents(stages.size());
  for (std::size_t index = 0; index < stages.size(); ++index) {
    indegree[index] = stages[index].dependencies.size();
    for (const std::size_t dependency : stages[index].dependencies) {
      dependents.at(indices.at(dependency)).push_back(index);
    }
  }
  std::vector<std::size_t> ready;
  for (std::size_t index = 0; index < stages.size(); ++index) {
    if (indegree[index] == 0) {
      ready.push_back(index);
    }
  }
  std::vector<const ExecutionStage*> result;
  result.reserve(stages.size());
  for (std::size_t cursor = 0; cursor < ready.size(); ++cursor) {
    const std::size_t index = ready[cursor];
    result.push_back(&stages[index]);
    for (const std::size_t dependent : dependents[index]) {
      if (--indegree[dependent] == 0) {
        ready.push_back(dependent);
      }
    }
  }
  if (result.size() != stages.size()) {
    compilation_error("THead execution stage dependency graph is cyclic");
  }
  return result;
}

class LibTritonJitEngine final : public ExecutionEngine {
 public:
  LibTritonJitEngine(const EngineBuildContext& context,
                     ExecutionProgramArtifact artifact)
      : context_(context),
        workspace_size_(artifact.workspace_size),
        workspace_alignment_(artifact.workspace_alignment) {
    require(context_.context != nullptr,
            "THead executable received a null PPU context",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
    require(artifact.backend == "thead" &&
                artifact.target == context_.target_fingerprint &&
                artifact.engine == "libtriton_jit",
            "THead executable received an incompatible artifact",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    binding_requirements_.reserve(artifact.external_uids.size());
    for (const std::int64_t uid : artifact.external_uids) {
      const auto tensor = std::find_if(
          artifact.tensors.begin(), artifact.tensors.end(),
          [uid](const TensorArtifact& candidate) {
            return candidate.uid == uid;
          });
      if (tensor == artifact.tensors.end() || tensor->is_virtual) {
        compilation_error(
            "THead external binding has no tensor specification");
      }
      binding_requirements_.push_back({uid, tensor->alignment});
    }

    try {
      std::lock_guard<std::mutex> lock(jit_build_mutex);
      ContextGuard guard(context_.context);
      validate_jit_runtime(target_capability(context_.target_fingerprint));
      PreparationResources resources(artifact);
      const CandidateDeviceLimits limits = query_device_limits(context_.device);
      const std::vector<const ExecutionStage*> ordered =
          execution_order(artifact.stages);
      stages_.reserve(ordered.size());
      for (const ExecutionStage* stage : ordered) {
        if (stage == nullptr || stage->source.empty() ||
            stage->function_name.empty() || stage->variants.empty() ||
            (stage->autotune && stage->variants.size() < 2) ||
            (!stage->autotune && stage->variants.size() != 1)) {
          compilation_error("THead JIT execution stage is incomplete");
        }
        const JitFunction& function = JitFunction::get_instance(
            stage->source.string(), stage->function_name);
        for (const KernelVariant& variant : stage->variants) {
          validate_jit_abi(function, variant);
        }
        const std::size_t selected =
            stage->autotune
                ? select_candidate(function, *stage, resources, limits,
                                   context_, stages_)
                : 0;
        const KernelVariant& variant = stage->variants.at(selected);
        validate_candidate_resources(variant, limits);
        stages_.push_back(
            {variant, capture_prepared_launch(function, variant, resources)});
      }
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const TheadError& error) {
      if (error.result() == FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED) {
        throw;
      }
      compilation_error("THead libtriton_jit executable build failed: " +
                        std::string(error.what()));
    } catch (const std::exception& error) {
      compilation_error("THead libtriton_jit executable build failed: " +
                        std::string(error.what()));
    }
  }

  LibTritonJitEngine(const LibTritonJitEngine&) = delete;
  LibTritonJitEngine& operator=(const LibTritonJitEngine&) = delete;

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(CUstream stream,
               const flagdnnBackendBindingV2 bindings[],
               std::size_t binding_count,
               void* workspace,
               std::size_t workspace_size) const override {
    validate_execution_arguments(bindings, binding_count, workspace,
                                 workspace_size);
    CUstreamCaptureStatus capture_status = CU_STREAM_CAPTURE_STATUS_NONE;
    check_driver(cuStreamIsCapturing(stream, &capture_status),
                 "cuStreamIsCapturing(THead execute)");
    require(capture_status != CU_STREAM_CAPTURE_STATUS_INVALIDATED,
            "THead caller stream capture is invalidated");
    try {
      const auto launch_prepared_stages = [&] {
        for (const PreparedStage& stage : stages_) {
          LaunchArguments arguments(
              stage.variant,
              [&](const KernelArgument& argument) -> CUdeviceptr {
                if (argument.kind == ArgumentKind::kTensor) {
                  return binding_pointer(argument.uid, bindings,
                                         binding_count);
                }
                const std::uintptr_t base =
                    reinterpret_cast<std::uintptr_t>(workspace);
                if (argument.workspace_offset >
                    std::numeric_limits<std::uintptr_t>::max() - base) {
                  throw TheadError(
                      FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                      "THead workspace pointer arithmetic overflowed");
                }
                return static_cast<CUdeviceptr>(
                    base + argument.workspace_offset);
              });
          check_driver(cuLaunchKernel(stage.launch.function,
                                      stage.launch.grid[0],
                                      stage.launch.grid[1],
                                      stage.launch.grid[2],
                                      stage.launch.block[0],
                                      stage.launch.block[1],
                                      stage.launch.block[2],
                                      stage.launch.shared_memory,
                                      stream,
                                      arguments.data(),
                                      nullptr),
                       "cuLaunchKernel(THead prepared stage)");
        }
      };

      if (capture_status == CU_STREAM_CAPTURE_STATUS_NONE) {
        CUcontext stream_context = nullptr;
        check_driver(cuStreamGetCtx(stream, &stream_context),
                     "cuStreamGetCtx(THead execute)");
        require(stream_context == context_.context,
                "THead caller stream belongs to another context");
        ContextGuard guard(context_.context);
        launch_prepared_stages();
      } else {
        // HGGC rejects stream-context queries while capture is active.  The
        // prepared function and capture stream are both bound to the retained
        // context, so the launch itself remains the authoritative validation.
        launch_prepared_stages();
      }
    } catch (const TheadError&) {
      throw;
    } catch (const std::exception& error) {
      throw TheadError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                       "THead prepared execution failed: " +
                           std::string(error.what()));
    }
  }

 private:
  void validate_execution_arguments(
      const flagdnnBackendBindingV2* bindings,
      std::size_t binding_count,
      void* workspace,
      std::size_t workspace_size) const {
    require(binding_count == binding_requirements_.size(),
            "binding count does not match THead executable");
    require(binding_count == 0 || bindings != nullptr,
            "THead binding array is null");
    require(workspace_size >= workspace_size_,
            "workspace is smaller than THead executable requirement");
    require(workspace_size_ == 0 || workspace != nullptr,
            "THead executable workspace is null");
    require(workspace_size_ == 0 ||
                reinterpret_cast<std::uintptr_t>(workspace) %
                        workspace_alignment_ ==
                    0,
            "THead workspace does not satisfy artifact alignment");

    for (std::size_t index = 0; index < binding_count; ++index) {
      require(bindings[index].device_pointer != nullptr,
              "THead binding device pointer is null");
      for (std::size_t previous = 0; previous < index; ++previous) {
        require(bindings[previous].uid != bindings[index].uid,
                "THead binding UID is duplicated");
      }
      const auto expected = std::find_if(
          binding_requirements_.begin(), binding_requirements_.end(),
          [&](const BindingRequirement& requirement) {
            return requirement.uid == bindings[index].uid;
          });
      require(expected != binding_requirements_.end(),
              "THead binding UID is not required by the executable");
      require(reinterpret_cast<std::uintptr_t>(
                  bindings[index].device_pointer) %
                      expected->alignment ==
                  0,
              "THead tensor binding does not satisfy alignment");
    }
  }

  static CUdeviceptr binding_pointer(
      std::int64_t uid,
      const flagdnnBackendBindingV2* bindings,
      std::size_t binding_count) {
    for (std::size_t index = 0; index < binding_count; ++index) {
      if (bindings[index].uid == uid) {
        return static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(
            bindings[index].device_pointer));
      }
    }
    throw TheadError(FLAGDNN_BACKEND_RESULT_INVALID_VALUE,
                     "a required THead tensor binding is missing");
  }

  EngineBuildContext context_;
  std::vector<BindingRequirement> binding_requirements_;
  std::vector<PreparedStage> stages_;
  std::size_t workspace_size_ = 0;
  std::size_t workspace_alignment_ = 1;
};

}  // namespace

std::unique_ptr<ExecutionEngine> create_libtriton_jit_engine(
    const EngineBuildContext& context,
    ExecutionProgramArtifact artifact) {
  return std::make_unique<LibTritonJitEngine>(context, std::move(artifact));
}

}  // namespace flagdnn::thead
