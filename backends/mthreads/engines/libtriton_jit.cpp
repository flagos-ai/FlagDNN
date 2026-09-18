/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/engines/engine.hpp"

#include "backends/mthreads/autotune.hpp"
#include "backends/mthreads/error.hpp"
#include "runtime/sha256.hpp"

#include <triton_jit/triton_jit_function.h>
#include <triton_jit/triton_kernel.h>

#include <Python.h>
#include <dlfcn.h>
#include <musa_runtime_api.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#ifndef FLAGDNN_MTHREADS_PYTHONPATH
#define FLAGDNN_MTHREADS_PYTHONPATH ""
#endif

#ifndef FLAGDNN_MTHREADS_ENVIRONMENT_IDENTITY
#define FLAGDNN_MTHREADS_ENVIRONMENT_IDENTITY "unknown"
#endif

#ifndef FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256
#define FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256 "unknown"
#endif

#ifndef FLAGDNN_MTHREADS_JIT_GEN_SSIG_SHA256
#define FLAGDNN_MTHREADS_JIT_GEN_SSIG_SHA256 "unknown"
#endif

#ifndef FLAGDNN_MTHREADS_JIT_STANDALONE_SHA256
#define FLAGDNN_MTHREADS_JIT_STANDALONE_SHA256 "unknown"
#endif

namespace flagdnn::mthreads {
namespace {

using JitFunction = triton_jit::TritonJITFunction;

std::mutex libtriton_jit_build_mutex;
std::once_flag jit_identity_once;
std::once_flag python_path_once;
std::once_flag python_runtime_once;
void* python_global_handle = nullptr;

class ScopedEnvironmentVariable final {
 public:
  ScopedEnvironmentVariable(std::string name, std::string value)
      : name_(std::move(name)) {
    if (const char* previous = std::getenv(name_.c_str());
        previous != nullptr) {
      previous_ = previous;
    }
    require(
        setenv(name_.c_str(), value.c_str(), 1) == 0,
        "cannot configure the mthreads Triton compilation environment",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  }

  ~ScopedEnvironmentVariable() {
    if (previous_.has_value()) {
      static_cast<void>(
          setenv(name_.c_str(), previous_->c_str(), 1));
    } else {
      static_cast<void>(unsetenv(name_.c_str()));
    }
  }

  ScopedEnvironmentVariable(const ScopedEnvironmentVariable&) = delete;
  ScopedEnvironmentVariable& operator=(
      const ScopedEnvironmentVariable&) = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

bool set_active_python_bytecode_disabled(bool disabled) noexcept {
  if (Py_IsInitialized() == 0) {
    return true;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* sys = PyImport_ImportModule("sys");
  const int status =
      sys == nullptr
          ? -1
          : PyObject_SetAttrString(
                sys,
                "dont_write_bytecode",
                disabled ? Py_True : Py_False);
  Py_XDECREF(sys);
  if (PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);
  return status == 0;
}

bool active_python_disables_bytecode() noexcept {
  if (Py_IsInitialized() == 0) {
    return false;
  }
  const PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* sys = PyImport_ImportModule("sys");
  PyObject* value =
      sys == nullptr
          ? nullptr
          : PyObject_GetAttrString(sys, "dont_write_bytecode");
  const int disabled = value == nullptr ? -1 : PyObject_IsTrue(value);
  Py_XDECREF(value);
  Py_XDECREF(sys);
  if (PyErr_Occurred() != nullptr) {
    PyErr_Clear();
  }
  PyGILState_Release(gil);
  return disabled > 0;
}

class ScopedPythonBytecodeSuppression final {
 public:
  ScopedPythonBytecodeSuppression()
      : python_was_initialized_(Py_IsInitialized() != 0),
        previous_global_flag_(Py_DontWriteBytecodeFlag) {
    if (const char* previous = std::getenv("PYTHONDONTWRITEBYTECODE");
        previous != nullptr) {
      previous_environment_ = previous;
    }
    previous_runtime_disabled_ =
        python_was_initialized_
            ? active_python_disables_bytecode()
            : previous_global_flag_ != 0 ||
                  (previous_environment_.has_value() &&
                   !previous_environment_->empty());
    require(
        setenv("PYTHONDONTWRITEBYTECODE", "1", 1) == 0,
        "cannot suppress Python bytecode for mthreads JIT",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    if (python_was_initialized_ &&
        !set_active_python_bytecode_disabled(true)) {
      restore_environment();
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot suppress active Python bytecode for mthreads JIT");
    }
  }

  ~ScopedPythonBytecodeSuppression() {
    if (Py_IsInitialized() != 0) {
      static_cast<void>(set_active_python_bytecode_disabled(
          previous_runtime_disabled_));
    }
    Py_DontWriteBytecodeFlag = previous_global_flag_;
    restore_environment();
  }

  ScopedPythonBytecodeSuppression(
      const ScopedPythonBytecodeSuppression&) = delete;
  ScopedPythonBytecodeSuppression& operator=(
      const ScopedPythonBytecodeSuppression&) = delete;

 private:
  void restore_environment() noexcept {
    if (previous_environment_.has_value()) {
      static_cast<void>(setenv(
          "PYTHONDONTWRITEBYTECODE",
          previous_environment_->c_str(),
          1));
    } else {
      static_cast<void>(unsetenv("PYTHONDONTWRITEBYTECODE"));
    }
  }

  bool python_was_initialized_ = false;
  bool previous_runtime_disabled_ = false;
  int previous_global_flag_ = 0;
  std::optional<std::string> previous_environment_;
};

void verify_mapped_jit_identity() {
  std::call_once(jit_identity_once, [] {
    Dl_info jit_information{};
    const auto jit_address = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(&triton_jit::clear_launch_hooks));
    require(
        dladdr(jit_address, &jit_information) != 0 &&
            jit_information.dli_fname != nullptr,
        "cannot locate the mapped mthreads libtriton_jit image",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    Dl_info plugin_information{};
    const auto plugin_address = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(&verify_mapped_jit_identity));
    require(
        dladdr(plugin_address, &plugin_information) != 0 &&
            plugin_information.dli_fname != nullptr,
        "cannot locate the mapped mthreads backend plugin",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    std::error_code path_error;
    const std::filesystem::path mapped_jit =
        std::filesystem::canonical(jit_information.dli_fname, path_error);
    require(
        !path_error,
        "cannot canonicalize the mapped mthreads libtriton_jit image",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const std::filesystem::path plugin =
        std::filesystem::canonical(plugin_information.dli_fname, path_error);
    require(
        !path_error,
        "cannot canonicalize the mapped mthreads backend plugin",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const std::filesystem::path expected_jit =
        std::filesystem::canonical(
            plugin.parent_path() /
                "flagdnn/mthreads/libtriton_jit.so",
            path_error);
    require(
        !path_error && mapped_jit == expected_jit,
        "mthreads backend mapped libtriton_jit outside its private layout",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(
        flagdnn::native::sha256_file(mapped_jit) ==
            FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256,
        "mapped mthreads libtriton_jit hash differs from configuration",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

    const std::filesystem::path script_directory =
        std::filesystem::canonical(triton_jit::get_script_dir(), path_error);
    const std::filesystem::path expected_script_directory =
        std::filesystem::canonical(
            plugin.parent_path() /
                "flagdnn/share/triton_jit/scripts",
            path_error);
    require(
        !path_error && script_directory == expected_script_directory,
        "mthreads libtriton_jit resolved helper scripts outside its private "
        "layout",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(
        flagdnn::native::sha256_file(script_directory / "gen_ssig.py") ==
                FLAGDNN_MTHREADS_JIT_GEN_SSIG_SHA256 &&
            flagdnn::native::sha256_file(
                script_directory / "standalone_compile.py") ==
                FLAGDNN_MTHREADS_JIT_STANDALONE_SHA256,
        "mthreads libtriton_jit helper script hash differs from "
        "configuration",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  });
}

void promote_python_runtime() {
  std::call_once(python_runtime_once, [] {
    Dl_info information{};
    const auto address = reinterpret_cast<void*>(
        reinterpret_cast<std::uintptr_t>(&Py_IsInitialized));
    if (dladdr(address, &information) == 0 ||
        information.dli_fname == nullptr) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot locate the embedded Python runtime for mthreads");
    }
    python_global_handle = dlopen(
        information.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (python_global_handle == nullptr) {
      python_global_handle =
          dlopen(information.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (python_global_handle == nullptr) {
      const char* detail = dlerror();
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot expose embedded Python symbols to mthreads Triton "
          "extensions: " +
              std::string(
                  detail == nullptr ? "unknown dlopen error" : detail));
    }
  });
}

void configure_python_path() {
  std::call_once(python_path_once, [] {
    const std::string required = FLAGDNN_MTHREADS_PYTHONPATH;
    require(
        !required.empty(),
        "mthreads TritonJIT Python path is not configured",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const char* current_value = std::getenv("PYTHONPATH");
    const std::string current =
        current_value == nullptr ? std::string{} : current_value;
    std::string_view remaining(current);
    while (!remaining.empty()) {
      const std::size_t separator = remaining.find(':');
      if (remaining.substr(0, separator) == required) {
        return;
      }
      if (separator == std::string_view::npos) {
        break;
      }
      remaining.remove_prefix(separator + 1);
    }
    const std::string updated =
        current.empty() ? required : required + ":" + current;
    if (setenv("PYTHONPATH", updated.c_str(), 1) != 0) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "cannot configure PYTHONPATH for mthreads TritonJIT");
    }
  });
}

struct TuningAllocation {
  std::int64_t uid = 0;
  std::size_t size = 0;
  std::string data_type;
  MUdeviceptr pointer = 0;
};

class TuningResources final {
 public:
  TuningResources(
      const StageArtifact& stage, std::size_t workspace_size)
      : workspace_size_(workspace_size) {
    require(
        !stage.variants.empty(),
        "mthreads tuning stage has no variants",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    for (const KernelVariantArtifact& variant : stage.variants) {
      for (const ArgumentSpec& argument : variant.arguments) {
        if (argument.kind != ArgumentKind::kTensor) {
          continue;
        }
        auto existing = std::find_if(
            allocations_.begin(),
            allocations_.end(),
            [&](const TuningAllocation& allocation) {
              return allocation.uid == argument.uid;
            });
        if (existing == allocations_.end()) {
          require(
              argument.storage_size != 0,
              "mthreads tuning tensor has zero storage",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          allocations_.push_back(
              {argument.uid,
               argument.storage_size,
               argument.data_type,
               0});
        } else {
          require(
              existing->data_type == argument.data_type,
              "mthreads tuning tensor UID has inconsistent data types",
              FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
          existing->size =
              std::max(existing->size, argument.storage_size);
        }
      }
    }
    try {
      for (TuningAllocation& allocation : allocations_) {
        check_mu(
            muMemAlloc(&allocation.pointer, allocation.size),
            "muMemAlloc(mthreads autotune tensor)",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      }
      if (workspace_size != 0) {
        check_mu(
            muMemAlloc(&workspace_, workspace_size),
            "muMemAlloc(mthreads autotune workspace)",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      }
      check_mu(
          muStreamCreate(&stream_, MU_STREAM_NON_BLOCKING),
          "muStreamCreate(mthreads autotune)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      check_mu(
          muEventCreate(&start_, MU_EVENT_DEFAULT),
          "muEventCreate(mthreads autotune start)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      check_mu(
          muEventCreate(&stop_, MU_EVENT_DEFAULT),
          "muEventCreate(mthreads autotune stop)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      initialize();
    } catch (...) {
      cleanup();
      throw;
    }
  }

  ~TuningResources() { cleanup(); }

  TuningResources(const TuningResources&) = delete;
  TuningResources& operator=(const TuningResources&) = delete;

  [[nodiscard]] const std::vector<TuningAllocation>& allocations()
      const noexcept {
    return allocations_;
  }
  [[nodiscard]] MUdeviceptr workspace() const noexcept {
    return workspace_;
  }
  [[nodiscard]] MUstream stream() const noexcept { return stream_; }
  [[nodiscard]] MUevent start() const noexcept { return start_; }
  [[nodiscard]] MUevent stop() const noexcept { return stop_; }

  void initialize() const {
    for (const TuningAllocation& allocation : allocations_) {
      if (allocation.data_type == "float32") {
        require(
            allocation.size % sizeof(std::uint32_t) == 0,
            "mthreads float32 tuning tensor has invalid storage",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        check_mu(
            muMemsetD32Async(
                allocation.pointer,
                0x3f800000U,
                allocation.size / sizeof(std::uint32_t),
                stream_),
            "muMemsetD32Async(mthreads float32 autotune tensor)",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      } else if (
          allocation.data_type == "float16" ||
          allocation.data_type == "bfloat16") {
        require(
            allocation.size % sizeof(std::uint16_t) == 0,
            "mthreads 16-bit tuning tensor has invalid storage",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        const unsigned short one =
            allocation.data_type == "float16" ? 0x3c00U : 0x3f80U;
        check_mu(
            muMemsetD16Async(
                allocation.pointer,
                one,
                allocation.size / sizeof(std::uint16_t),
                stream_),
            "muMemsetD16Async(mthreads 16-bit autotune tensor)",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      } else {
        const unsigned char one =
            allocation.data_type == "fp8_e4m3"
                ? 0x38U
                : (allocation.data_type == "fp8_e5m2" ? 0x3cU : 0x01U);
        check_mu(
            muMemsetD8Async(
                allocation.pointer, one, allocation.size, stream_),
            "muMemsetD8Async(mthreads 8-bit autotune tensor)",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      }
    }
    if (workspace_ != 0) {
      check_mu(
          muMemsetD8Async(workspace_, 0, workspace_size_, stream_),
          "muMemsetD8Async(mthreads autotune workspace)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    }
    check_mu(
        muStreamSynchronize(stream_),
        "muStreamSynchronize(mthreads autotune initialize)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  }

 private:
  void cleanup() noexcept {
    if (start_ != nullptr) {
      static_cast<void>(muEventDestroy(start_));
      start_ = nullptr;
    }
    if (stop_ != nullptr) {
      static_cast<void>(muEventDestroy(stop_));
      stop_ = nullptr;
    }
    if (stream_ != nullptr) {
      static_cast<void>(muStreamDestroy(stream_));
      stream_ = nullptr;
    }
    if (workspace_ != 0) {
      static_cast<void>(muMemFree(workspace_));
      workspace_ = 0;
    }
    for (TuningAllocation& allocation : allocations_) {
      if (allocation.pointer != 0) {
        static_cast<void>(muMemFree(allocation.pointer));
        allocation.pointer = 0;
      }
    }
  }

  std::vector<TuningAllocation> allocations_;
  MUdeviceptr workspace_ = 0;
  std::size_t workspace_size_ = 0;
  MUstream stream_ = nullptr;
  MUevent start_ = nullptr;
  MUevent stop_ = nullptr;
};

struct ArgumentValue {
  MUdeviceptr pointer = 0;
  std::uint32_t scalar_bits = 0;
};

struct DescriptorArgumentValue {
  MUtensorDescriptor descriptor{};
  std::array<std::int32_t, 2> shape{};
  std::array<std::int64_t, 2> strides{};
};

class RawArguments final {
 public:
  RawArguments(
      const KernelVariantArtifact& kernel,
      const std::vector<TuningAllocation>& allocations,
      MUdeviceptr global_scratch) {
    initialize(kernel);
    std::size_t physical_index = 0;
    for (std::size_t index = 0;
         index < kernel.arguments.size(); ++index) {
      const ArgumentSpec& argument = kernel.arguments[index];
      if (argument.kind == ArgumentKind::kTensor) {
        const auto allocation = std::find_if(
            allocations.begin(),
            allocations.end(),
            [&](const TuningAllocation& value) {
              return value.uid == argument.uid;
            });
        require(
            allocation != allocations.end(),
            "mthreads autotune allocation is missing",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        append_tensor(
            index,
            argument,
            allocation->pointer,
            physical_index,
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      } else if (argument.kind == ArgumentKind::kWorkspace) {
        require(
            global_scratch != 0,
            "mthreads autotune workspace is missing",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        values_[index].pointer = global_scratch;
        append_parameter(
            &values_[index].pointer,
            physical_index,
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      } else if (
          argument.kind == ArgumentKind::kScalarI32 ||
          argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_bits = argument.scalar_bits;
        append_parameter(
            &values_[index].scalar_bits,
            physical_index,
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      } else {
        throw MthreadsError(
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
            "mthreads autotune argument kind is unsupported");
      }
    }
    finish(physical_index, global_scratch);
  }

  RawArguments(
      const KernelVariantArtifact& kernel,
      const flagdnnBackendBindingV2 bindings[],
      std::size_t binding_count,
      MUdeviceptr global_scratch) {
    initialize(kernel);
    std::size_t physical_index = 0;
    for (std::size_t index = 0;
         index < kernel.arguments.size(); ++index) {
      const ArgumentSpec& argument = kernel.arguments[index];
      if (argument.kind == ArgumentKind::kTensor) {
        const flagdnnBackendBindingV2* found = nullptr;
        for (std::size_t supplied = 0;
             supplied < binding_count; ++supplied) {
          if (bindings[supplied].uid == argument.uid) {
            found = &bindings[supplied];
            break;
          }
        }
        require(found != nullptr, "a required mthreads tensor UID is missing");
        values_[index].pointer = static_cast<MUdeviceptr>(
            reinterpret_cast<std::uintptr_t>(found->device_pointer));
        require(
            values_[index].pointer != 0 &&
            values_[index].pointer % argument.alignment == 0,
            "a mthreads tensor binding is null or misaligned");
        append_tensor(
            index,
            argument,
            values_[index].pointer,
            physical_index,
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      } else if (argument.kind == ArgumentKind::kWorkspace) {
        require(
            global_scratch != 0,
            "mthreads runtime workspace is missing");
        values_[index].pointer = global_scratch;
        append_parameter(
            &values_[index].pointer,
            physical_index,
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      } else if (
          argument.kind == ArgumentKind::kScalarI32 ||
          argument.kind == ArgumentKind::kScalarF32) {
        values_[index].scalar_bits = argument.scalar_bits;
        append_parameter(
            &values_[index].scalar_bits,
            physical_index,
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      } else {
        throw MthreadsError(
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR,
            "mthreads runtime argument kind is unsupported");
      }
    }
    finish(physical_index, global_scratch);
  }

  [[nodiscard]] void** data() noexcept { return parameters_.data(); }
  [[nodiscard]] std::size_t size() const noexcept {
    return parameter_count_;
  }

 private:
  void initialize(const KernelVariantArtifact& kernel) {
    const std::size_t argument_count = kernel.arguments.size();
    require(
        argument_count <= 128,
        "mthreads JIT kernel has too many runtime arguments",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    parameter_count_ = 0;
    parameters_.fill(nullptr);
  }

  void append_parameter(
      void* parameter,
      std::size_t& physical_index,
      flagdnnBackendResult_t failure_result) {
    require(
        physical_index < 128,
        "mthreads JIT kernel has too many physical runtime arguments",
        failure_result);
    parameters_[physical_index++] = parameter;
  }

  void append_tensor(
      std::size_t logical_index,
      const ArgumentSpec& argument,
      MUdeviceptr pointer,
      std::size_t& physical_index,
      flagdnnBackendResult_t failure_result) {
    values_[logical_index].pointer = pointer;
    if (!argument.tensor_descriptor) {
      append_parameter(
          &values_[logical_index].pointer,
          physical_index,
          failure_result);
      return;
    }

    require(
        pointer != 0 && pointer % 16 == 0 &&
            (argument.data_type == "float16" ||
             argument.data_type == "bfloat16") &&
            argument.descriptor_shape[0] > 0 &&
            argument.descriptor_shape[1] > 0 &&
            argument.descriptor_strides[0] > 0 &&
            argument.descriptor_strides[1] == 1 &&
            argument.descriptor_block_shape[0] > 0 &&
            argument.descriptor_block_shape[1] > 0,
        "mthreads tensor descriptor metadata is invalid",
        failure_result);
    constexpr std::uint64_t element_size = 2;
    const std::uint64_t block_elements =
        static_cast<std::uint64_t>(
            argument.descriptor_block_shape[0]) *
        static_cast<std::uint64_t>(
            argument.descriptor_block_shape[1]);
    require(
        block_elements >= 32 / element_size,
        "mthreads tensor descriptor block is smaller than 32 bytes",
        failure_result);

    DescriptorArgumentValue& descriptor =
        descriptors_[logical_index];
    descriptor.shape = argument.descriptor_shape;
    descriptor.strides = argument.descriptor_strides;
    const std::array<muuint64_t, 2> dimensions = {
        static_cast<muuint64_t>(descriptor.shape[1]),
        static_cast<muuint64_t>(descriptor.shape[0])};
    const std::array<muuint64_t, 2> global_strides = {
        dimensions[0] * element_size,
        dimensions[0] * element_size * dimensions[1]};
    check_mu(
        muTensorDescriptorEncode(
            &descriptor.descriptor,
            MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16,
            2,
            reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(pointer)),
            dimensions.data(),
            global_strides.data(),
            MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE,
            0),
        "muTensorDescriptorEncode(mthreads Triton argument)",
        failure_result);

    append_parameter(
        &descriptor.descriptor, physical_index, failure_result);
    for (std::int32_t& dimension : descriptor.shape) {
      append_parameter(&dimension, physical_index, failure_result);
    }
    for (std::int64_t& stride : descriptor.strides) {
      append_parameter(&stride, physical_index, failure_result);
    }
  }

  void finish(
      std::size_t physical_argument_count,
      MUdeviceptr global_scratch) {
    require(
        physical_argument_count <= 128,
        "mthreads JIT kernel physical argument count is invalid",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    global_scratch_ = global_scratch;
    parameters_[physical_argument_count] = &global_scratch_;
    parameters_[physical_argument_count + 1] = &profile_scratch_;
    parameter_count_ = physical_argument_count + 2;
  }

  std::array<ArgumentValue, 128> values_{};
  std::array<DescriptorArgumentValue, 128> descriptors_{};
  std::array<void*, 130> parameters_{};
  std::size_t parameter_count_ = 0;
  MUdeviceptr global_scratch_ = 0;
  MUdeviceptr profile_scratch_ = 0;
};

void launch_jit(
    const JitFunction& function,
    const KernelVariantArtifact& kernel,
    MUstream stream,
    RawArguments& arguments) {
  function.launch_with_raw_args(
      stream,
      kernel.grid[0],
      kernel.grid[1],
      kernel.grid[2],
      kernel.num_warps,
      kernel.num_stages,
      kernel.full_signature,
      arguments.data(),
      arguments.size());
}

struct PreparedLaunch {
  MUfunction function = nullptr;
  std::array<unsigned int, 3> grid = {1, 1, 1};
  std::array<unsigned int, 3> block = {1, 1, 1};
  unsigned int shared_memory = 0;
};

struct TleCompileOnly final {};

struct TleCompileCapture {
  bool entered = false;
  std::string kernel_name;
  unsigned int shared_memory = 0;
};

class ScopedLaunchEnterHook final {
 public:
  ScopedLaunchEnterHook(
      triton_jit::LaunchHook hook)
      : previous_(triton_jit::detail::get_launch_hooks_snapshot()) {
    triton_jit::set_launch_enter_hook(std::move(hook));
    active_ = true;
  }

  ~ScopedLaunchEnterHook() { restore_noexcept(); }

  ScopedLaunchEnterHook(const ScopedLaunchEnterHook&) = delete;
  ScopedLaunchEnterHook& operator=(const ScopedLaunchEnterHook&) = delete;

  void restore() {
    if (!active_) {
      return;
    }
    triton_jit::clear_launch_hooks();
    if (previous_) {
      if (previous_->enter) {
        triton_jit::set_launch_enter_hook(previous_->enter);
      }
      if (previous_->exit) {
        triton_jit::set_launch_exit_hook(previous_->exit);
      }
    }
    active_ = false;
  }

 private:
  void restore_noexcept() noexcept {
    try {
      restore();
    } catch (...) {
    }
  }

  triton_jit::detail::LaunchHooksSnapshot previous_;
  bool active_ = false;
};

struct CachedTleKernel {
  const JitFunction* function = nullptr;
  std::string full_signature;
  unsigned int num_warps = 0;
  unsigned int num_stages = 0;
  MUfunction handle = nullptr;
  unsigned int shared_memory = 0;
};

std::vector<CachedTleKernel> cached_tle_kernels;

PreparedLaunch prepare_tle_launch(
    const JitFunction& function,
    const KernelVariantArtifact& kernel,
    TuningResources& resources) {
  constexpr unsigned int kProducerWarps = 4;
  require(
      kernel.num_warps == 16,
      "mthreads TLE Matmul consumer warp count differs",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  const auto cached = std::find_if(
      cached_tle_kernels.begin(),
      cached_tle_kernels.end(),
      [&](const CachedTleKernel& value) {
        return value.function == &function &&
               value.full_signature == kernel.full_signature &&
               value.num_warps == kernel.num_warps &&
               value.num_stages == kernel.num_stages;
      });
  if (cached != cached_tle_kernels.end()) {
    return {
        cached->handle,
        kernel.grid,
        {(kernel.num_warps + kProducerWarps) * 32U, 1, 1},
        cached->shared_memory};
  }

  std::vector<std::string> previous_module_keys;
  {
    std::lock_guard lock(triton_jit::MusaBackend::cache_mutex_);
    previous_module_keys.reserve(
        triton_jit::MusaBackend::module_cache_.size());
    for (const auto& [key, value] :
         triton_jit::MusaBackend::module_cache_) {
      static_cast<void>(value);
      previous_module_keys.push_back(key);
    }
  }

  TleCompileCapture capture;
  const std::thread::id build_thread = std::this_thread::get_id();
  const auto previous_hooks =
      triton_jit::detail::get_launch_hooks_snapshot();
  ScopedLaunchEnterHook hook(
      [&, build_thread, previous_hooks](
          const triton_jit::LaunchMetadata& metadata) {
        if (std::this_thread::get_id() != build_thread) {
          if (previous_hooks && previous_hooks->enter) {
            previous_hooks->enter(metadata);
          }
          return;
        }
        if (metadata.grid_x != kernel.grid[0] ||
            metadata.grid_y != kernel.grid[1] ||
            metadata.grid_z != kernel.grid[2] ||
            metadata.num_warps !=
                static_cast<int>(kernel.num_warps) ||
            metadata.signature != kernel.full_signature ||
            metadata.stream !=
                reinterpret_cast<void*>(resources.stream())) {
          throw std::runtime_error(
              "mthreads TLE compile-only launch metadata differs");
        }
        capture.entered = true;
        capture.kernel_name = metadata.kernel_name;
        capture.shared_memory = metadata.shared_memory;
        throw TleCompileOnly{};
      });

  RawArguments arguments(
      kernel, resources.allocations(), resources.workspace());
  bool intercepted = false;
  try {
    launch_jit(function, kernel, resources.stream(), arguments);
  } catch (const TleCompileOnly&) {
    intercepted = true;
  }
  hook.restore();
  require(
      intercepted && capture.entered && !capture.kernel_name.empty(),
      "mthreads TLE compile-only launch was not intercepted",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  MUfunction compiled_function = nullptr;
  {
    const std::string suffix = "::" + capture.kernel_name;
    std::lock_guard lock(triton_jit::MusaBackend::cache_mutex_);
    for (const auto& [key, value] :
         triton_jit::MusaBackend::module_cache_) {
      const bool existed =
          std::find(
              previous_module_keys.begin(),
              previous_module_keys.end(),
              key) != previous_module_keys.end();
      if (!existed && key.ends_with(suffix)) {
        require(
            compiled_function == nullptr &&
                value.metadata.shared == capture.shared_memory,
            "mthreads TLE compile-only module metadata differs",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        compiled_function = value.function;
      }
    }
  }
  require(
      compiled_function != nullptr,
      "mthreads TLE compile-only module was not loaded",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  cached_tle_kernels.push_back(
      {&function,
       kernel.full_signature,
       kernel.num_warps,
       kernel.num_stages,
       compiled_function,
       capture.shared_memory});
  return {
      compiled_function,
      kernel.grid,
      {(kernel.num_warps + kProducerWarps) * 32U, 1, 1},
      capture.shared_memory};
}

void launch_prepared(
    const PreparedLaunch& prepared,
    MUstream stream,
    RawArguments& arguments,
    flagdnnBackendResult_t failure_result) {
  check_mu(
      muLaunchKernel(
          prepared.function,
          prepared.grid[0],
          prepared.grid[1],
          prepared.grid[2],
          prepared.block[0],
          prepared.block[1],
          prepared.block[2],
          prepared.shared_memory,
          stream,
          arguments.data(),
          nullptr),
      "muLaunchKernel(mthreads prepared Triton kernel)",
      failure_result);
}

void validate_static_signature(
    const JitFunction& function,
    const StageArtifact& stage) {
  const triton_jit::StaticSignature& signature =
      function.get_static_sig();
  std::size_t expected = 0;
  std::size_t runtime_argument_count = 0;
  bool attention_signature = false;
  if (stage.function_name == "binary_contiguous_kernel") {
    expected = 7;
    runtime_argument_count = 4;
  } else if (stage.function_name == "add_square_tensor_kernel") {
    expected = 7;
    runtime_argument_count = 4;
  } else if (stage.function_name == "conv_bias_relu_2d_kernel") {
    expected = 40;
    runtime_argument_count = 4;
  } else if (stage.function_name == "binary_strided_kernel") {
    expected = 39;
    runtime_argument_count = 4;
  } else if (
      stage.function_name == "unary_pointwise_contiguous_kernel") {
    expected = 13;
    runtime_argument_count = 3;
  } else if (stage.function_name == "unary_pointwise_strided_kernel") {
    expected = 38;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "identity_contiguous_packed_kernel") {
    expected = 6;
    runtime_argument_count = 3;
  } else if (stage.function_name == "identity_contiguous_kernel") {
    expected = 5;
    runtime_argument_count = 3;
  } else if (stage.function_name == "identity_strided_kernel") {
    expected = 28;
    runtime_argument_count = 3;
  } else if (stage.function_name == "reshape_contiguous_kernel") {
    expected = 4;
    runtime_argument_count = 3;
  } else if (stage.function_name == "slice_copy_kernel") {
    expected = 30;
    runtime_argument_count = 3;
  } else if (stage.function_name == "transpose_physical_copy_kernel") {
    expected = 4;
    runtime_argument_count = 3;
  } else if (stage.function_name == "binary_select_tensor_kernel") {
    expected = 6;
    runtime_argument_count = 5;
  } else if (stage.function_name == "binary_select_strided_kernel") {
    expected = 46;
    runtime_argument_count = 5;
  } else if (stage.function_name == "layout_copy_kernel") {
    expected = 37;
    runtime_argument_count = 3;
  } else if (stage.function_name == "reduction_2d_kernel") {
    expected = 9;
    runtime_argument_count = 3;
  } else if (stage.function_name == "reduction_3d_kernel") {
    expected = 11;
    runtime_argument_count = 3;
  } else if (stage.function_name == "reduction_strided_kernel") {
    expected = 32;
    runtime_argument_count = 3;
  } else if (stage.function_name == "matmul_strided_kernel") {
    expected = 42;
    runtime_argument_count = 3;
  } else if (stage.function_name == "matmul_descriptor_kernel") {
    expected = 11;
    runtime_argument_count = 3;
  } else if (stage.function_name == "matmul_tle_kernel") {
    expected = 13;
    runtime_argument_count = 3;
  } else if (stage.function_name == "conv1d_gemm_kernel") {
    expected = 30;
    runtime_argument_count = 4;
  } else if (stage.function_name == "conv2d_spatial_nchw_kernel") {
    expected = 40;
    runtime_argument_count = 4;
  } else if (stage.function_name == "conv3d_spatial_ncdhw_m_kernel") {
    expected = 48;
    runtime_argument_count = 4;
  } else if (
      stage.function_name == "_conv_fprop2d_im2col_kernel") {
    expected = 25;
    runtime_argument_count = 2;
  } else if (
      stage.function_name == "_conv_fprop2d_im2col_mm_kernel") {
    expected = 25;
    runtime_argument_count = 3;
  } else if (stage.function_name == "conv_dgrad_nd_kernel") {
    expected = 45;
    runtime_argument_count = 3;
  } else if (
      stage.function_name ==
      "_conv_dgrad2d_dense_pack_filter_kernel") {
    expected = 10;
    runtime_argument_count = 2;
  } else if (
      stage.function_name == "_conv_dgrad2d_dense_pack_loss_kernel") {
    expected = 13;
    runtime_argument_count = 2;
  } else if (
      stage.function_name == "_conv_dgrad2d_dense_mm_kernel") {
    expected = 20;
    runtime_argument_count = 3;
  } else if (stage.function_name == "conv_wgrad_nd_kernel") {
    expected = 44;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad2d_p5_pack_image_kernel") {
    expected = 12;
    runtime_argument_count = 2;
  } else if (stage.function_name == "_conv_wgrad2d_p5_mm_kernel") {
    expected = 11;
    runtime_argument_count = 3;
  } else if (stage.function_name == "_conv_wgrad2d_im2row_kernel") {
    expected = 24;
    runtime_argument_count = 2;
  } else if (stage.function_name == "_conv_wgrad_nd_im2row_kernel") {
    expected = 31;
    runtime_argument_count = 2;
  } else if (
      stage.function_name == "_conv_wgrad2d_rowmajor_kernel") {
    expected = 22;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad_nd_rowmajor_kernel") {
    expected = 21;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad_nd_reduce_kernel") {
    expected = 6;
    runtime_argument_count = 2;
  } else if (
      stage.function_name == "_conv_wgrad2d_direct_split_kernel") {
    expected = 34;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad2d_1x1_split_kernel") {
    expected = 18;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad2d_stem_split_kernel") {
    expected = 31;
    runtime_argument_count = 3;
  } else if (
      stage.function_name == "_conv_wgrad2d_stem_reduce_kernel") {
    expected = 16;
    runtime_argument_count = 2;
  } else if (stage.function_name == "layer_norm_kernel") {
    expected = 17;
    runtime_argument_count = 7;
  } else if (stage.function_name == "rms_norm_kernel") {
    expected = 14;
    runtime_argument_count = 6;
  } else if (stage.function_name == "batch_norm_nchw_kernel") {
    expected = 21;
    runtime_argument_count = 10;
  } else if (stage.function_name == "batch_norm_kernel") {
    expected = 46;
    runtime_argument_count = 13;
  } else if (
      stage.function_name == "batch_norm_inference_nchw_kernel") {
    expected = 13;
    runtime_argument_count = 6;
  } else if (stage.function_name == "batch_norm_inference_kernel") {
    expected = 39;
    runtime_argument_count = 9;
  } else if (stage.function_name == "_sdpa_fwd_kernel") {
    expected = 48;
    runtime_argument_count = 37;
    attention_signature = true;
  } else if (stage.function_name == "_zero_contiguous_kernel") {
    expected = 3;
    runtime_argument_count = 2;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_bwd_dq_dbias_kernel") {
    expected = 71;
    runtime_argument_count = 15;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_bwd_dkdv_kernel") {
    expected = 59;
    runtime_argument_count = 14;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_bwd_dk_kernel") {
    expected = 56;
    runtime_argument_count = 13;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_bwd_dv_kernel") {
    expected = 46;
    runtime_argument_count = 11;
    attention_signature = true;
  } else if (
      stage.function_name == "_zero_sdpa_fp8_fwd_amax_kernel") {
    expected = 2;
    runtime_argument_count = 2;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_fp8_fwd_kernel") {
    expected = 55;
    runtime_argument_count = 45;
    attention_signature = true;
  } else if (
      stage.function_name == "_zero_sdpa_fp8_bwd_amax_kernel") {
    expected = 4;
    runtime_argument_count = 4;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_fp8_bwd_dq_kernel") {
    expected = 58;
    runtime_argument_count = 22;
    attention_signature = true;
  } else if (stage.function_name == "_sdpa_fp8_bwd_dkdv_kernel") {
    expected = 67;
    runtime_argument_count = 27;
    attention_signature = true;
  } else {
    // The artifact reader has validated the registered source/function pair.
    // New operators use the common runtime/constexpr ABI without duplicating
    // each kernel's Python signature in the execution engine.
    require(!stage.variants.empty(), "mthreads stage has no variants",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const auto& variant = stage.variants.front();
    expected = 1 + static_cast<std::size_t>(
                       std::count(variant.full_signature.begin(),
                                  variant.full_signature.end(), ','));
    runtime_argument_count = variant.arguments.size();
    attention_signature = true;
  }
  if (signature.num_args != static_cast<int>(expected) ||
      signature.arg_type.size() != expected) {
    throw MthreadsError(
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
        "mthreads Triton static signature has the wrong arity");
  }
  std::vector<std::string_view> signature_tokens;
  if (attention_signature) {
    require(
        !stage.variants.empty(),
        "mthreads Attention stage has no variants",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const std::string& full_signature =
        stage.variants.front().full_signature;
    std::size_t offset = 0;
    while (offset <= full_signature.size()) {
      const std::size_t comma = full_signature.find(',', offset);
      const std::size_t end =
          comma == std::string::npos ? full_signature.size() : comma;
      signature_tokens.emplace_back(
          full_signature.data() + offset, end - offset);
      if (comma == std::string::npos) {
        break;
      }
      offset = comma + 1;
    }
    require(
        signature_tokens.size() == expected,
        "mthreads Attention full signature has the wrong arity",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    const std::size_t visible_runtime = static_cast<std::size_t>(
        std::count_if(
            signature_tokens.begin(),
            signature_tokens.end(),
            [](std::string_view token) {
              return token.starts_with('*') || token == "i32" ||
                     token == "fp32";
            }));
    require(
        visible_runtime == runtime_argument_count,
        "mthreads Attention runtime signature count differs",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  }
  for (std::size_t index = 0; index < expected; ++index) {
    const bool is_runtime =
        attention_signature
            ? (signature_tokens[index].starts_with('*') ||
               signature_tokens[index] == "i32" ||
               signature_tokens[index] == "fp32")
            : index < runtime_argument_count;
    const triton_jit::ArgType expected_type =
        is_runtime ? triton_jit::ArgType::SPECIALIZED
                   : triton_jit::ArgType::CONSTEXPR;
    if (signature.arg_type[index] != expected_type) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "mthreads Triton static signature classification differs");
    }
  }
}

PreparedLaunch prepare_launch(
    const JitFunction& function,
    const KernelVariantArtifact& kernel,
    TuningResources& resources) {
  if (kernel.function_name == "matmul_tle_kernel") {
    ScopedEnvironmentVariable tle_optimization(
        "TRITON_MUSA_ENABLE_LLC_OPT", "1");
    return prepare_tle_launch(function, kernel, resources);
  }
  RawArguments arguments(
      kernel,
      resources.allocations(),
      resources.workspace());
  launch_jit(function, kernel, resources.stream(), arguments);
  check_mu(
      muStreamSynchronize(resources.stream()),
      "muStreamSynchronize(mthreads JIT preparation)",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);

  const musaStream_t runtime_stream =
      reinterpret_cast<musaStream_t>(resources.stream());
  musaGraph_t graph = nullptr;
  bool capture_active = false;
  try {
    check_musa(
        musaStreamBeginCapture(
            runtime_stream, musaStreamCaptureModeRelaxed),
        "musaStreamBeginCapture(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    capture_active = true;
    launch_jit(function, kernel, resources.stream(), arguments);
    check_musa(
        musaStreamEndCapture(runtime_stream, &graph),
        "musaStreamEndCapture(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    capture_active = false;

    std::size_t node_count = 0;
    check_musa(
        musaGraphGetNodes(graph, nullptr, &node_count),
        "musaGraphGetNodes(mthreads JIT preparation count)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(
        node_count == 1,
        "mthreads JIT preparation did not capture exactly one kernel",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    musaGraphNode_t node = nullptr;
    check_musa(
        musaGraphGetNodes(graph, &node, &node_count),
        "musaGraphGetNodes(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    musaGraphNodeType node_type = musaGraphNodeTypeCount;
    check_musa(
        musaGraphNodeGetType(node, &node_type),
        "musaGraphNodeGetType(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    require(
        node_type == musaGraphNodeTypeKernel,
        "mthreads JIT preparation captured a non-kernel node",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    // A Triton MUSA kernel is loaded as a driver MUfunction.  MUSA 4.3.5 can
    // capture that launch through the runtime graph API, but its runtime
    // parameter query tries to resolve the function as a runtime symbol and
    // returns musaErrorSymbolNotFound.  Runtime and driver graph handles are
    // the same opaque MUgraphNode_st type, so query the captured driver launch
    // through the driver API and retain the original MUfunction handle.
    MUSA_KERNEL_NODE_PARAMS parameters{};
    check_mu(
        muGraphKernelNodeGetParams(
            reinterpret_cast<MUgraphNode>(node), &parameters),
        "muGraphKernelNodeGetParams(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    PreparedLaunch result{
        parameters.func,
        {
            parameters.gridDimX,
            parameters.gridDimY,
            parameters.gridDimZ,
        },
        {
            parameters.blockDimX,
            parameters.blockDimY,
            parameters.blockDimZ,
        },
        parameters.sharedMemBytes};
    require(
        result.function != nullptr && result.grid == kernel.grid &&
            result.block ==
                std::array<unsigned int, 3>{
                    kernel.num_warps * 32U, 1, 1},
        "mthreads captured launch geometry differs from artifact",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    check_musa(
        musaGraphDestroy(graph),
        "musaGraphDestroy(mthreads JIT preparation)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    graph = nullptr;
    return result;
  } catch (...) {
    if (capture_active) {
      musaGraph_t abandoned = nullptr;
      if (musaStreamEndCapture(runtime_stream, &abandoned) ==
              musaSuccess &&
          abandoned != nullptr) {
        static_cast<void>(musaGraphDestroy(abandoned));
      }
    } else if (graph != nullptr) {
      static_cast<void>(musaGraphDestroy(graph));
    }
    throw;
  }
}

class CapturedLaunchBatch final {
 public:
  template <typename Function>
  CapturedLaunchBatch(MUstream stream,
                      unsigned int execution_count,
                      Function&& launch)
      : execution_count_(execution_count) {
    require(
        stream != nullptr && execution_count_ != 0,
        "mthreads libtriton_jit autotune batch cannot be empty",
        FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);

    check_mu(
        muStreamBeginCapture(stream, MU_STREAM_CAPTURE_MODE_RELAXED),
        "muStreamBeginCapture(mthreads libtriton_jit autotune)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
    bool capture_active = true;
    MUgraph graph = nullptr;
    try {
      for (unsigned int iteration = 0;
           iteration < execution_count_;
           ++iteration) {
        launch();
      }
      check_mu(
          muStreamEndCapture(stream, &graph),
          "muStreamEndCapture(mthreads libtriton_jit autotune)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      capture_active = false;
      require(
          graph != nullptr,
          "mthreads libtriton_jit autotune capture returned a null graph",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      check_mu(
          muGraphInstantiate(&executable_, graph, 0),
          "muGraphInstantiate(mthreads libtriton_jit autotune)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      check_mu(
          muGraphDestroy(graph),
          "muGraphDestroy(mthreads libtriton_jit autotune source)",
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
      graph = nullptr;
    } catch (...) {
      if (capture_active) {
        MUgraph abandoned = nullptr;
        if (muStreamEndCapture(stream, &abandoned) == MUSA_SUCCESS &&
            abandoned != nullptr) {
          static_cast<void>(muGraphDestroy(abandoned));
        }
      } else if (graph != nullptr) {
        static_cast<void>(muGraphDestroy(graph));
      }
      cleanup();
      throw;
    }
  }

  ~CapturedLaunchBatch() { cleanup(); }

  CapturedLaunchBatch(const CapturedLaunchBatch&) = delete;
  CapturedLaunchBatch& operator=(const CapturedLaunchBatch&) = delete;

  void launch(MUstream stream) const {
    check_mu(
        muGraphLaunch(executable_, stream),
        "muGraphLaunch(mthreads libtriton_jit autotune)",
        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  }

  [[nodiscard]] unsigned int execution_count() const noexcept {
    return execution_count_;
  }

 private:
  void cleanup() noexcept {
    if (executable_ != nullptr) {
      static_cast<void>(muGraphExecDestroy(executable_));
      executable_ = nullptr;
    }
  }

  MUgraphExec executable_ = nullptr;
  unsigned int execution_count_ = 0;
};

struct LoadedKernel {
  KernelVariantArtifact specification;
  PreparedLaunch prepared;
  struct DeviceCopy {
    std::int64_t input_uid = 0;
    std::int64_t output_uid = 0;
    std::size_t bytes = 0;
  };
  std::optional<DeviceCopy> device_copy;
};

std::optional<LoadedKernel::DeviceCopy> identity_device_copy(
    const KernelVariantArtifact& kernel) {
  if (kernel.function_name != "identity_contiguous_packed_kernel" &&
      kernel.function_name != "identity_contiguous_kernel" &&
      kernel.function_name != "reshape_contiguous_kernel" &&
      kernel.function_name != "transpose_physical_copy_kernel") {
    return std::nullopt;
  }
  require(
      kernel.arguments.size() == 3 &&
          kernel.arguments[0].kind == ArgumentKind::kTensor &&
          kernel.arguments[0].semantic_name == "input" &&
          kernel.arguments[1].kind == ArgumentKind::kTensor &&
          kernel.arguments[1].semantic_name == "output" &&
          kernel.arguments[0].storage_size != 0 &&
          kernel.arguments[0].storage_size ==
              kernel.arguments[1].storage_size,
      "mthreads dense materialization copy metadata is invalid",
      FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
  return LoadedKernel::DeviceCopy{
      kernel.arguments[0].uid,
      kernel.arguments[1].uid,
      kernel.arguments[0].storage_size};
}

class LibTritonJitEngine final : public ExecutionEngine {
 public:
  LibTritonJitEngine(
      const EngineBuildContext& context, MthreadsArtifact artifact)
      : context_(context),
        binding_uids_(std::move(artifact.binding_uids)),
        workspace_size_(artifact.workspace_size),
        workspace_alignment_(artifact.workspace_alignment) {
    try {
      std::lock_guard lock(libtriton_jit_build_mutex);
      verify_mapped_jit_identity();
      promote_python_runtime();
      configure_python_path();
      ScopedPythonBytecodeSuppression suppress_python_bytecode;
      ContextGuard guard(context_.context, context_.device);
      kernels_.reserve(artifact.stages.size());
      for (const StageArtifact& stage : artifact.stages) {
        require(
            !stage.source.empty() && !stage.function_name.empty() &&
                !stage.variants.empty(),
            "mthreads JIT stage is incomplete",
            FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
        const JitFunction& function = JitFunction::get_instance(
            stage.source.string(), stage.function_name);
        validate_static_signature(function, stage);
        TuningResources resources(stage, workspace_size_);
        std::vector<std::optional<PreparedLaunch>> prepared(
            stage.variants.size());
        const auto prepare = [&](std::size_t index) {
          require(
              index < stage.variants.size(),
              "mthreads JIT preparation index is invalid",
              FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
          if (!prepared[index].has_value()) {
            prepared[index] = prepare_launch(
                function, stage.variants[index], resources);
          }
        };
        std::size_t selected = 0;
        if (stage.autotune) {
          // Match the actual steady-state execution mode and the NVIDIA/Hygon
          // engines: direct driver launches can rank small MUSA kernels very
          // differently from a captured Graph batch.  A direct-launch winner
          // was observed to be materially slower than the non-autotuned
          // default once the executable was replayed through MUSA Graph.
          constexpr unsigned int kGraphBatchSize = 32;
          std::unique_ptr<CapturedLaunchBatch> captured_batch;
          std::size_t captured_candidate = stage.variants.size();
          const auto batch_for = [&](std::size_t index)
              -> CapturedLaunchBatch& {
            prepare(index);
            if (captured_batch == nullptr ||
                captured_candidate != index) {
              captured_batch.reset();
              RawArguments arguments(
                  stage.variants[index],
                  resources.allocations(),
                  resources.workspace());
              captured_batch = std::make_unique<CapturedLaunchBatch>(
                  resources.stream(), kGraphBatchSize, [&] {
                    launch_prepared(
                        *prepared[index],
                        resources.stream(),
                        arguments,
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                  });
              captured_candidate = index;
            }
            return *captured_batch;
          };
          const auto replay_count = [](unsigned int requested,
                                       unsigned int batch_size) {
            return (requested + batch_size - 1U) / batch_size;
          };
          selected = select_mthreads_autotune_candidate(
              context_,
              stage,
              "mthreads-libtriton-jit-captured-mugraph-batch32-v1-"
              "typed-inputs-env-" +
                  std::string(FLAGDNN_MTHREADS_ENVIRONMENT_IDENTITY) +
                  "-jit-" +
                  std::string(FLAGDNN_MTHREADS_JIT_PRIVATE_SHA256) +
                  "-stage-" + std::to_string(stage.id),
              {
                  prepare,
                  [&](std::size_t index, unsigned int iterations) {
                    CapturedLaunchBatch& batch = batch_for(index);
                    resources.initialize();
                    const unsigned int replays = replay_count(
                        iterations, batch.execution_count());
                    for (unsigned int replay = 0;
                         replay < replays;
                         ++replay) {
                      batch.launch(resources.stream());
                    }
                    check_mu(
                        muStreamSynchronize(resources.stream()),
                        "muStreamSynchronize(mthreads autotune warmup)",
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                  },
                  [&](std::size_t index, unsigned int iterations) {
                    CapturedLaunchBatch& batch = batch_for(index);
                    resources.initialize();
                    const unsigned int replays = replay_count(
                        iterations, batch.execution_count());
                    check_mu(
                        muEventRecord(
                            resources.start(), resources.stream()),
                        "muEventRecord(mthreads autotune start)",
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                    for (unsigned int replay = 0;
                         replay < replays;
                         ++replay) {
                      batch.launch(resources.stream());
                    }
                    check_mu(
                        muEventRecord(
                            resources.stop(), resources.stream()),
                        "muEventRecord(mthreads autotune stop)",
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                    check_mu(
                        muEventSynchronize(resources.stop()),
                        "muEventSynchronize(mthreads autotune stop)",
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                    float milliseconds = 0.0F;
                    check_mu(
                        muEventElapsedTime(
                            &milliseconds,
                            resources.start(),
                            resources.stop()),
                        "muEventElapsedTime(mthreads autotune)",
                        FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED);
                    const unsigned int measured_iterations =
                        replays * batch.execution_count();
                    return milliseconds /
                           static_cast<float>(measured_iterations);
                  },
              });
        } else {
          prepare(selected);
        }
        require(
            selected < stage.variants.size() &&
                prepared[selected].has_value(),
            "mthreads JIT selected candidate was not prepared",
            FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR);
        kernels_.push_back(
            {stage.variants[selected],
             *prepared[selected],
             identity_device_copy(stage.variants[selected])});
      }
    } catch (const MthreadsError&) {
      throw;
    } catch (const std::exception& error) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_COMPILATION_FAILED,
          "mthreads libtriton_jit executable build failed: " +
              std::string(error.what()));
    }
  }

  [[nodiscard]] std::size_t workspace_size() const noexcept override {
    return workspace_size_;
  }

  void execute(
      MUstream stream,
      const flagdnnBackendBindingV2 bindings[],
      std::size_t binding_count,
      void* workspace,
      std::size_t workspace_size) const override {
    require(
        workspace_size >= workspace_size_,
        "workspace is smaller than mthreads executable requirement");
    require(
        workspace_size_ == 0 || workspace != nullptr,
        "mthreads executable workspace is null");
    require(
        workspace_size_ == 0 ||
            reinterpret_cast<std::uintptr_t>(workspace) %
                    workspace_alignment_ ==
                0,
        "mthreads executable workspace is misaligned");
    require(
        binding_count == binding_uids_.size(),
        "binding count does not match mthreads executable");
    require(
        binding_count == 0 || bindings != nullptr,
        "mthreads binding array is null");
    for (std::size_t supplied = 0;
         supplied < binding_count; ++supplied) {
      require(
          bindings[supplied].device_pointer != nullptr,
          "mthreads tensor binding pointer is null");
      require(
          std::find(
              binding_uids_.begin(),
              binding_uids_.end(),
              bindings[supplied].uid) != binding_uids_.end(),
          "mthreads executable received an unknown binding UID");
      for (std::size_t previous = 0;
           previous < supplied; ++previous) {
        require(
            bindings[previous].uid != bindings[supplied].uid,
            "mthreads executable received a duplicate binding UID");
      }
    }

    try {
      ContextGuard guard(context_.context, context_.device);
      const MUdeviceptr scratch = static_cast<MUdeviceptr>(
          reinterpret_cast<std::uintptr_t>(workspace));
      for (const LoadedKernel& kernel : kernels_) {
        // Constructing RawArguments is also the runtime ABI validation step.
        // Keep it on the materialization-copy path so that the copy fast path
        // cannot bypass tensor UID, alignment, or argument-contract checks.
        RawArguments arguments(
            kernel.specification,
            bindings,
            binding_count,
            scratch);
        if (kernel.device_copy.has_value()) {
          const auto find_pointer =
              [&](std::int64_t uid) -> MUdeviceptr {
            const auto binding = std::find_if(
                bindings,
                bindings + binding_count,
                [uid](const flagdnnBackendBindingV2& candidate) {
                  return candidate.uid == uid;
                });
            require(
                binding != bindings + binding_count,
                "mthreads materialization copy binding is missing");
            return static_cast<MUdeviceptr>(
                reinterpret_cast<std::uintptr_t>(
                    binding->device_pointer));
          };
          check_mu(
              muMemcpyDtoDAsync(
                  find_pointer(kernel.device_copy->output_uid),
                  find_pointer(kernel.device_copy->input_uid),
                  kernel.device_copy->bytes,
                  stream),
              "muMemcpyDtoDAsync(mthreads dense materialization)",
              FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
          continue;
        }
        launch_prepared(
            kernel.prepared,
            stream,
            arguments,
            FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
      }
    } catch (const MthreadsError&) {
      throw;
    } catch (const std::exception& error) {
      throw MthreadsError(
          FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
          "mthreads libtriton_jit kernel launch failed: " +
              std::string(error.what()));
    }
  }

 private:
  EngineBuildContext context_;
  std::vector<std::int64_t> binding_uids_;
  std::vector<LoadedKernel> kernels_;
  std::size_t workspace_size_ = 0;
  std::size_t workspace_alignment_ = 1;
};

}  // namespace

std::unique_ptr<ExecutionEngine> create_libtriton_jit_engine(
    const EngineBuildContext& context, MthreadsArtifact artifact) {
  return std::make_unique<LibTritonJitEngine>(
      context, std::move(artifact));
}

}  // namespace flagdnn::mthreads
