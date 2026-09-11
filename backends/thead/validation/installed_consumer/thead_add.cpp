// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <acdnn.h>
#include <acdnn_backend.h>
#include <cuda.h>
#include <flagdnn/flagdnn.hpp>
#include <flagdnn_frontend.h>

#include <dlfcn.h>
#include <link.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace fe = ::flagdnn_frontend;

constexpr std::size_t kElementCount = 256;

void require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  }
}

std::filesystem::path required_environment_path(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    throw std::runtime_error(std::string(name) + " is missing");
  }
  const std::filesystem::path path(value);
  if (!path.is_absolute() || !std::filesystem::exists(path)) {
    throw std::runtime_error(std::string(name) +
                             " is not an existing absolute path");
  }
  return std::filesystem::canonical(path);
}

bool is_within(const std::filesystem::path& root,
               const std::filesystem::path& candidate) {
  const std::filesystem::path relative = candidate.lexically_relative(root);
  return !relative.empty() && !relative.is_absolute() &&
         *relative.begin() != "..";
}

void require_clean_environment() {
  // LD_LIBRARY_PATH is deliberately supplied by the installed-consumer
  // contract with only pinned PPU SDK directories.  The vendor ppu-llc tool
  // dynamically loads targets/x86_64-linux/lib/libhggcrt1.so and has no ELF
  // search path for that target runtime.
  constexpr std::array<const char*, 16> forbidden = {
      "FLAGDNN_BACKEND",
      "FLAGDNN_BACKEND_PATH",
      "FLAGDNN_BACKEND_ROOT",
      "FLAGDNN_KERNEL_SOURCE_ROOT",
      "FLAGDNN_TUNING_ROOT",
      "FLAGDNN_THEAD_RESOURCE_ROOT",
      "FLAGDNN_THEAD_TRITON_JIT_ROOT",
      "FLAGDNN_THEAD_TRITON_JIT_DIR",
      "FLAGDNN_THEAD_TRITON_JIT_LIBRARY",
      "FLAGDNN_THEAD_TRITON_JIT_INCLUDE_DIR",
      "FLAGDNN_THEAD_TRITON_JIT_SCRIPT_DIR",
      "FLAGDNN_CODEGEN_COMPILER",
      "FLAGDNN_COMPILER",
      "FLAGDNN_COMPILER_EXECUTABLE",
      "FLAGDNN_EXECUTION_ENGINE",
      "PYTHONHOME"};
  for (const char* name : forbidden) {
    if (std::getenv(name) != nullptr) {
      throw std::runtime_error(std::string("inherited override was not unset: ") +
                               name);
    }
  }
}

void check_driver(CUresult status, std::string_view operation) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* detail = nullptr;
  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &detail);
  throw std::runtime_error(
      std::string(operation) + " failed (" +
      (name == nullptr ? "HGGC_ERROR_UNKNOWN" : name) + "): " +
      (detail == nullptr ? "driver error description unavailable" : detail));
}

void check_acdnn(acdnnStatus_t status, std::string_view operation) {
  if (status == ACDNN_STATUS_SUCCESS) {
    return;
  }
  const char* detail = acdnnGetErrorString(status);
  throw std::runtime_error(
      std::string(operation) + " failed: " +
      (detail == nullptr ? "acDNN error description unavailable" : detail));
}

void check_frontend(const fe::error_t& status, std::string_view operation) {
  if (status.is_bad()) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             status.get_message());
  }
}

class PrimaryContext final {
 public:
  explicit PrimaryContext(CUdevice device) : device_(device) {
    check_driver(cuDevicePrimaryCtxRetain(&context_, device_),
                 "cuDevicePrimaryCtxRetain");
    CUcontext current = nullptr;
    check_driver(cuCtxGetCurrent(&current), "cuCtxGetCurrent");
    if (current != context_) {
      check_driver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~PrimaryContext() {
    if (pushed_) {
      CUcontext ignored = nullptr;
      (void)cuCtxPopCurrent(&ignored);
    }
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  PrimaryContext(const PrimaryContext&) = delete;
  PrimaryContext& operator=(const PrimaryContext&) = delete;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  bool pushed_ = false;
};

class Stream final {
 public:
  Stream() {
    check_driver(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate(non-default)");
    require(value_ != nullptr, "driver returned a null non-default stream");
  }

  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] CUstream get() const noexcept { return value_; }
  [[nodiscard]] flagdnnStream_t opaque() const noexcept {
    return reinterpret_cast<flagdnnStream_t>(value_);
  }

 private:
  CUstream value_ = nullptr;
};

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0) {
      check_driver(cuMemAlloc(&value_, bytes_), "cuMemAlloc");
      require(value_ != 0, "driver returned a null allocation");
    }
  }

  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* data() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value_));
  }

  [[nodiscard]] CUdeviceptr address() const noexcept { return value_; }

  void copy_from(const void* source, std::size_t bytes, CUstream stream) const {
    require(bytes <= bytes_, "host-to-device copy exceeds allocation");
    check_driver(cuMemcpyHtoDAsync(value_, source, bytes, stream),
                 "cuMemcpyHtoDAsync");
  }

  void copy_to(void* destination, std::size_t bytes, CUstream stream) const {
    require(bytes <= bytes_, "device-to-host copy exceeds allocation");
    check_driver(cuMemcpyDtoHAsync(destination, value_, bytes, stream),
                 "cuMemcpyDtoHAsync");
  }

 private:
  CUdeviceptr value_ = 0;
  std::size_t bytes_ = 0;
};

class AcdnnPointwise final {
 public:
  AcdnnPointwise(acdnnOpTensorOp_t operation, std::string primitive,
                 float right_coefficient)
      : primitive_(std::move(primitive)),
        right_coefficient_(right_coefficient) {
    check_acdnn(acdnnCreate(&handle_), "acdnnCreate");
    try {
      constexpr int dimensions[] = {1, 1,
                                    static_cast<int>(kElementCount)};
      constexpr int strides[] = {static_cast<int>(kElementCount),
                                 static_cast<int>(kElementCount), 1};
      for (auto& descriptor : tensors_) {
        check_acdnn(acdnnCreateTensorDescriptor(&descriptor),
                    "acdnnCreateTensorDescriptor");
        check_acdnn(acdnnSetTensorNdDescriptor(
                        descriptor, ACDNN_DATA_FLOAT, 3, dimensions, strides),
                    "acdnnSetTensorNdDescriptor");
      }
      check_acdnn(acdnnCreateOpTensorDescriptor(&operation_),
                  "acdnnCreateOpTensorDescriptor");
      check_acdnn(acdnnSetOpTensorDescriptor(
                      operation_, operation, ACDNN_DATA_FLOAT,
                      ACDNN_PROPAGATE_NAN),
                  "acdnnSetOpTensorDescriptor");
    } catch (...) {
      release();
      throw;
    }
  }

  ~AcdnnPointwise() { release(); }

  AcdnnPointwise(const AcdnnPointwise&) = delete;
  AcdnnPointwise& operator=(const AcdnnPointwise&) = delete;

  void execute(CUstream stream, void* left, void* right, void* output) {
    check_acdnn(acdnnSetStream(handle_, reinterpret_cast<hggcStream_t>(stream)),
                "acdnnSetStream");
    constexpr float alpha_left = 1.0F;
    const float alpha_right = right_coefficient_;
    constexpr float zero = 0.0F;
    check_acdnn(acdnnOpTensor(handle_, operation_, &alpha_left, tensors_[0],
                              left, &alpha_right, tensors_[1], right, &zero,
                              tensors_[2], output),
                primitive_);
  }

 private:
  void release() noexcept {
    if (operation_ != nullptr) {
      (void)acdnnDestroyOpTensorDescriptor(operation_);
      operation_ = nullptr;
    }
    for (auto& descriptor : tensors_) {
      if (descriptor != nullptr) {
        (void)acdnnDestroyTensorDescriptor(descriptor);
        descriptor = nullptr;
      }
    }
    if (handle_ != nullptr) {
      (void)acdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  std::string primitive_;
  float right_coefficient_ = 1.0F;
  acdnnHandle_t handle_ = nullptr;
  std::array<acdnnTensorDescriptor_t, 3> tensors_ = {nullptr, nullptr, nullptr};
  acdnnOpTensorDescriptor_t operation_ = nullptr;
};

class AcdnnActivation final {
 public:
  AcdnnActivation(acdnnActivationMode_t mode, std::string primitive,
                  double coefficient)
      : primitive_(std::move(primitive)),
        coefficient_(coefficient),
        uses_transform_(mode == ACDNN_ACTIVATION_IDENTITY) {
    check_acdnn(acdnnCreate(&handle_), "acdnnCreate");
    try {
      constexpr int dimensions[] = {1, 1,
                                    static_cast<int>(kElementCount)};
      constexpr int strides[] = {static_cast<int>(kElementCount),
                                 static_cast<int>(kElementCount), 1};
      for (auto& descriptor : tensors_) {
        check_acdnn(acdnnCreateTensorDescriptor(&descriptor),
                    "acdnnCreateTensorDescriptor");
        check_acdnn(acdnnSetTensorNdDescriptor(
                        descriptor, ACDNN_DATA_FLOAT, 3, dimensions, strides),
                    "acdnnSetTensorNdDescriptor");
      }
      if (!uses_transform_) {
        check_acdnn(acdnnCreateActivationDescriptor(&activation_),
                    "acdnnCreateActivationDescriptor");
        check_acdnn(acdnnSetActivationDescriptor(
                        activation_, mode, ACDNN_NOT_PROPAGATE_NAN,
                        coefficient),
                    "acdnnSetActivationDescriptor(NOT_PROPAGATE_NAN)");
      }
    } catch (...) {
      release();
      throw;
    }
  }

  ~AcdnnActivation() { release(); }

  AcdnnActivation(const AcdnnActivation&) = delete;
  AcdnnActivation& operator=(const AcdnnActivation&) = delete;

  void execute(CUstream stream, void* input, void* output) {
    check_acdnn(acdnnSetStream(handle_, reinterpret_cast<hggcStream_t>(stream)),
                "acdnnSetStream");
    const float alpha = uses_transform_ ? static_cast<float>(coefficient_)
                                        : 1.0F;
    constexpr float beta = 0.0F;
    const acdnnStatus_t status =
        uses_transform_
            ? acdnnTransformTensor(handle_, &alpha, tensors_[0], input,
                                   &beta, tensors_[1], output)
            : acdnnActivationForward(handle_, activation_, &alpha,
                                     tensors_[0], input, &beta, tensors_[1],
                                     output);
    check_acdnn(status, primitive_);
  }

 private:
  void release() noexcept {
    if (activation_ != nullptr) {
      (void)acdnnDestroyActivationDescriptor(activation_);
      activation_ = nullptr;
    }
    for (auto& descriptor : tensors_) {
      if (descriptor != nullptr) {
        (void)acdnnDestroyTensorDescriptor(descriptor);
        descriptor = nullptr;
      }
    }
    if (handle_ != nullptr) {
      (void)acdnnDestroy(handle_);
      handle_ = nullptr;
    }
  }

  std::string primitive_;
  double coefficient_ = 0.0;
  bool uses_transform_ = false;
  acdnnHandle_t handle_ = nullptr;
  std::array<acdnnTensorDescriptor_t, 2> tensors_ = {nullptr, nullptr};
  acdnnActivationDescriptor_t activation_ = nullptr;
};

class AcdnnBackendDescriptor final {
 public:
  explicit AcdnnBackendDescriptor(acdnnBackendDescriptorType_t type) {
    check_acdnn(acdnnBackendCreateDescriptor(type, &value_),
                "acdnnBackendCreateDescriptor");
  }

  ~AcdnnBackendDescriptor() {
    if (value_ != nullptr) {
      (void)acdnnBackendDestroyDescriptor(value_);
    }
  }

  AcdnnBackendDescriptor(const AcdnnBackendDescriptor&) = delete;
  AcdnnBackendDescriptor& operator=(const AcdnnBackendDescriptor&) = delete;

  [[nodiscard]] acdnnBackendDescriptor_t get() const noexcept {
    return value_;
  }

  void set(acdnnBackendAttributeName_t name,
           acdnnBackendAttributeType_t type, std::int64_t count,
           const void* values, std::string_view description) {
    check_acdnn(acdnnBackendSetAttribute(value_, name, type, count, values),
                description);
  }

  void finalize(std::string_view description) {
    check_acdnn(acdnnBackendFinalize(value_), description);
  }

 private:
  acdnnBackendDescriptor_t value_ = nullptr;
};

class AcdnnBackendHandle final {
 public:
  AcdnnBackendHandle() {
    check_acdnn(acdnnCreate(&value_), "acdnnCreate(backend pointwise)");
  }

  ~AcdnnBackendHandle() {
    if (value_ != nullptr) {
      (void)acdnnDestroy(value_);
    }
  }

  AcdnnBackendHandle(const AcdnnBackendHandle&) = delete;
  AcdnnBackendHandle& operator=(const AcdnnBackendHandle&) = delete;

  [[nodiscard]] acdnnHandle_t get() const noexcept { return value_; }

 private:
  acdnnHandle_t value_ = nullptr;
};

class AcdnnBackendPointwise final {
 public:
  AcdnnBackendPointwise(acdnnPointwiseMode_t mode, std::string primitive,
                        bool binary, double parameter = 1.0,
                        bool constant_one_numerator = false,
                        bool comparison = false)
      : primitive_(std::move(primitive)),
        binary_(binary),
        constant_one_numerator_(constant_one_numerator),
        comparison_(comparison),
        pointwise_(ACDNN_BACKEND_POINTWISE_DESCRIPTOR),
        input_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        second_input_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        output_(ACDNN_BACKEND_TENSOR_DESCRIPTOR),
        operation_(ACDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR),
        graph_(ACDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR),
        heuristics_(ACDNN_BACKEND_ENGINEHEUR_DESCRIPTOR),
        engine_config_(ACDNN_BACKEND_ENGINECFG_DESCRIPTOR),
        plan_(ACDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR) {
    build_tensor(input_, constant_one_numerator_ ? kConstantUid : kInputUid,
                 ACDNN_DATA_FLOAT);
    build_tensor(second_input_,
                 constant_one_numerator_ ? kInputUid : kSecondInputUid,
                 ACDNN_DATA_FLOAT);
    build_tensor(output_, kOutputUid,
                 comparison_ ? ACDNN_DATA_BOOL : ACDNN_DATA_FLOAT);
    if (constant_one_numerator_) {
      require(mode == ACDNN_POINTWISE_DIV && !binary_,
              "installed reciprocal reference configuration is invalid");
      constant_numerator_ =
          std::make_unique<DeviceBuffer>(kElementCount * sizeof(float));
      check_driver(cuMemsetD32(constant_numerator_->address(), 0x3f800000U,
                              kElementCount),
                   "cuMemsetD32(installed reciprocal numerator)");
    }

    const acdnnDataType_t math_precision =
        comparison_ ? ACDNN_DATA_BOOL : ACDNN_DATA_FLOAT;
    const acdnnNanPropagation_t nan_policy = ACDNN_NOT_PROPAGATE_NAN;
    pointwise_.set(ACDNN_ATTR_POINTWISE_MODE, ACDNN_TYPE_POINTWISE_MODE, 1,
                   &mode, "acdnnBackendSetAttribute(pointwise mode)");
    pointwise_.set(ACDNN_ATTR_POINTWISE_MATH_PREC, ACDNN_TYPE_DATA_TYPE, 1,
                   &math_precision,
                   "acdnnBackendSetAttribute(pointwise math precision)");
    pointwise_.set(ACDNN_ATTR_POINTWISE_NAN_PROPAGATION,
                   ACDNN_TYPE_NAN_PROPOGATION, 1, &nan_policy,
                   "acdnnBackendSetAttribute(pointwise nan policy)");
    if (mode == ACDNN_POINTWISE_SOFTPLUS_FWD) {
      pointwise_.set(ACDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                     ACDNN_TYPE_DOUBLE, 1, &parameter,
                     "acdnnBackendSetAttribute(pointwise Softplus beta)");
    } else if (mode == ACDNN_POINTWISE_SWISH_FWD) {
      pointwise_.set(ACDNN_ATTR_POINTWISE_SWISH_BETA,
                     ACDNN_TYPE_DOUBLE, 1, &parameter,
                     "acdnnBackendSetAttribute(pointwise Swish beta)");
    }
    pointwise_.finalize("acdnnBackendFinalize(pointwise)");

    acdnnBackendDescriptor_t pointwise = pointwise_.get();
    acdnnBackendDescriptor_t input = input_.get();
    acdnnBackendDescriptor_t second_input = second_input_.get();
    acdnnBackendDescriptor_t output = output_.get();
    constexpr float alpha = 1.0F;
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR,
                   ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &pointwise,
                   "acdnnBackendSetAttribute(operation pointwise)");
    if (mode == ACDNN_POINTWISE_SIGMOID_BWD) {
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &second_input,
                     "acdnnBackendSetAttribute(operation backward input)");
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_DYDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input,
                     "acdnnBackendSetAttribute(operation upstream gradient)");
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_DXDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output,
                     "acdnnBackendSetAttribute(operation input gradient)");
    } else {
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_XDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input,
                     "acdnnBackendSetAttribute(operation input)");
      if (binary_ || constant_one_numerator_) {
        operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_BDESC,
                       ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &second_input,
                       "acdnnBackendSetAttribute(operation second input)");
      }
      operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_YDESC,
                     ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output,
                     "acdnnBackendSetAttribute(operation output)");
    }
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA1, ACDNN_TYPE_FLOAT,
                   1, &alpha,
                   "acdnnBackendSetAttribute(operation alpha1)");
    operation_.set(ACDNN_ATTR_OPERATION_POINTWISE_ALPHA2, ACDNN_TYPE_FLOAT,
                   1, &alpha,
                   "acdnnBackendSetAttribute(operation alpha2)");
    operation_.finalize("acdnnBackendFinalize(pointwise operation)");

    acdnnBackendDescriptor_t operation = operation_.get();
    acdnnHandle_t handle = handle_.get();
    graph_.set(ACDNN_ATTR_OPERATIONGRAPH_OPS,
               ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation,
               "acdnnBackendSetAttribute(operation graph ops)");
    graph_.set(ACDNN_ATTR_OPERATIONGRAPH_HANDLE, ACDNN_TYPE_HANDLE, 1,
               &handle,
               "acdnnBackendSetAttribute(operation graph handle)");
    graph_.finalize("acdnnBackendFinalize(operation graph)");

    acdnnBackendDescriptor_t graph = graph_.get();
    const acdnnBackendHeurMode_t heuristics_mode = ACDNN_HEUR_MODE_A;
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph,
                    "acdnnBackendSetAttribute(heuristics graph)");
    heuristics_.set(ACDNN_ATTR_ENGINEHEUR_MODE, ACDNN_TYPE_HEUR_MODE, 1,
                    &heuristics_mode,
                    "acdnnBackendSetAttribute(heuristics mode)");
    heuristics_.finalize("acdnnBackendFinalize(heuristics)");

    std::int64_t advertised_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 0, &advertised_count,
                    nullptr),
                "acdnnBackendGetAttribute(heuristics count)");
    require(advertised_count > 0,
            "acDNN backend heuristics returned no engine config");
    acdnnBackendDescriptor_t engine_config = engine_config_.get();
    std::int64_t returned_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    heuristics_.get(), ACDNN_ATTR_ENGINEHEUR_RESULTS,
                    ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned_count,
                    &engine_config),
                "acdnnBackendGetAttribute(heuristics result)");
    require(returned_count > 0,
            "acDNN backend heuristics returned no usable engine config");

    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_HANDLE, ACDNN_TYPE_HANDLE, 1,
              &handle, "acdnnBackendSetAttribute(plan handle)");
    plan_.set(ACDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
              ACDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_config,
              "acdnnBackendSetAttribute(plan engine config)");
    plan_.finalize("acdnnBackendFinalize(execution plan)");

    std::int64_t workspace_bytes = 0;
    std::int64_t workspace_count = 0;
    check_acdnn(acdnnBackendGetAttribute(
                    plan_.get(), ACDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                    ACDNN_TYPE_INT64, 1, &workspace_count, &workspace_bytes),
                "acdnnBackendGetAttribute(plan workspace)");
    require(workspace_count == 1 && workspace_bytes >= 0,
            "acDNN backend pointwise workspace is invalid");
    workspace_ = std::make_unique<DeviceBuffer>(
        static_cast<std::size_t>(workspace_bytes));
    workspace_bytes_ = workspace_bytes;
  }

  void execute(CUstream stream, void* input, void* second_input,
               void* output) {
    check_acdnn(acdnnSetStream(handle_.get(),
                               reinterpret_cast<hggcStream_t>(stream)),
                "acdnnSetStream(backend pointwise)");
    std::vector<std::int64_t> uids = {
        constant_one_numerator_ ? kConstantUid : kInputUid};
    std::vector<void*> pointers = {
        constant_one_numerator_ ? constant_numerator_->data() : input};
    if (constant_one_numerator_) {
      uids.push_back(kInputUid);
      pointers.push_back(input);
    }
    if (binary_) {
      uids.push_back(kSecondInputUid);
      pointers.push_back(second_input);
    }
    uids.push_back(kOutputUid);
    pointers.push_back(output);
    void* workspace = workspace_->data();
    AcdnnBackendDescriptor pack(ACDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    pack.set(ACDNN_ATTR_VARIANT_PACK_DATA_POINTERS, ACDNN_TYPE_VOID_PTR,
             static_cast<std::int64_t>(pointers.size()), pointers.data(),
             "acdnnBackendSetAttribute(variant pointers)");
    pack.set(ACDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, ACDNN_TYPE_INT64,
             static_cast<std::int64_t>(uids.size()), uids.data(),
             "acdnnBackendSetAttribute(variant uids)");
    pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE, ACDNN_TYPE_VOID_PTR, 1,
             &workspace,
             "acdnnBackendSetAttribute(variant workspace)");
    pack.set(ACDNN_ATTR_VARIANT_PACK_WORKSPACE_SIZE, ACDNN_TYPE_INT64, 1,
             &workspace_bytes_,
             "acdnnBackendSetAttribute(variant workspace size)");
    pack.finalize("acdnnBackendFinalize(variant pack)");
    check_acdnn(acdnnBackendExecute(handle_.get(), plan_.get(), pack.get()),
                primitive_);
    check_driver(cuStreamSynchronize(stream),
                 "cuStreamSynchronize(acDNN backend pointwise)");
  }

 private:
  static constexpr std::int64_t kInputUid = 7001;
  static constexpr std::int64_t kSecondInputUid = 7002;
  static constexpr std::int64_t kOutputUid = 7003;
  static constexpr std::int64_t kConstantUid = 7004;

  static void build_tensor(AcdnnBackendDescriptor& descriptor,
                           std::int64_t uid,
                           acdnnDataType_t data_type) {
    constexpr std::array<std::int64_t, 3> dimensions = {
        1, 1, static_cast<std::int64_t>(kElementCount)};
    constexpr std::array<std::int64_t, 3> strides = {
        static_cast<std::int64_t>(kElementCount),
        static_cast<std::int64_t>(kElementCount), 1};
    constexpr std::int64_t alignment = 16;
    descriptor.set(ACDNN_ATTR_TENSOR_DATA_TYPE, ACDNN_TYPE_DATA_TYPE, 1,
                   &data_type,
                   "acdnnBackendSetAttribute(tensor data type)");
    descriptor.set(ACDNN_ATTR_TENSOR_DIMENSIONS, ACDNN_TYPE_INT64,
                   static_cast<std::int64_t>(dimensions.size()),
                   dimensions.data(),
                   "acdnnBackendSetAttribute(tensor dimensions)");
    descriptor.set(ACDNN_ATTR_TENSOR_STRIDES, ACDNN_TYPE_INT64,
                   static_cast<std::int64_t>(strides.size()), strides.data(),
                   "acdnnBackendSetAttribute(tensor strides)");
    descriptor.set(ACDNN_ATTR_TENSOR_UNIQUE_ID, ACDNN_TYPE_INT64, 1, &uid,
                   "acdnnBackendSetAttribute(tensor uid)");
    descriptor.set(ACDNN_ATTR_TENSOR_BYTE_ALIGNMENT, ACDNN_TYPE_INT64, 1,
                   &alignment,
                   "acdnnBackendSetAttribute(tensor alignment)");
    descriptor.finalize("acdnnBackendFinalize(tensor)");
  }

  std::string primitive_;
  bool binary_ = false;
  bool constant_one_numerator_ = false;
  bool comparison_ = false;
  AcdnnBackendHandle handle_;
  AcdnnBackendDescriptor pointwise_;
  AcdnnBackendDescriptor input_;
  AcdnnBackendDescriptor second_input_;
  AcdnnBackendDescriptor output_;
  AcdnnBackendDescriptor operation_;
  AcdnnBackendDescriptor graph_;
  AcdnnBackendDescriptor heuristics_;
  AcdnnBackendDescriptor engine_config_;
  AcdnnBackendDescriptor plan_;
  std::unique_ptr<DeviceBuffer> workspace_;
  std::unique_ptr<DeviceBuffer> constant_numerator_;
  std::int64_t workspace_bytes_ = 0;
};

std::filesystem::path mapped_image(const std::filesystem::path& expected) {
  void* handle = dlopen(expected.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) {
    const char* detail = dlerror();
    throw std::runtime_error(
        "expected image is not mapped: " + expected.string() + ": " +
        (detail == nullptr ? "unknown loader error" : detail));
  }
  struct link_map* mapping = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &mapping) != 0 || mapping == nullptr ||
      mapping->l_name == nullptr || mapping->l_name[0] == '\0') {
    (void)dlclose(handle);
    throw std::runtime_error("cannot inspect mapped image: " +
                             expected.string());
  }
  const std::filesystem::path result =
      std::filesystem::canonical(mapping->l_name);
  (void)dlclose(handle);
  return result;
}

template <typename Function>
std::filesystem::path image_containing(Function* function,
                                       std::string_view description) {
  Dl_info information{};
  const auto address = reinterpret_cast<void*>(
      reinterpret_cast<std::uintptr_t>(function));
  if (dladdr(address, &information) == 0 || information.dli_fname == nullptr) {
    throw std::runtime_error("cannot locate " + std::string(description));
  }
  return std::filesystem::canonical(information.dli_fname);
}

using CacheSnapshot = std::map<std::string, std::string>;

CacheSnapshot snapshot_cache(const std::filesystem::path& root) {
  CacheSnapshot result;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
    if (entry.is_regular_file()) {
      std::ifstream input(entry.path(), std::ios::binary);
      if (!input) {
        throw std::runtime_error("cannot read installed THead cache file " +
                                 entry.path().string());
      }
      std::string contents{std::istreambuf_iterator<char>(input),
                           std::istreambuf_iterator<char>()};
      if (input.bad()) {
        throw std::runtime_error("cannot snapshot installed THead cache file " +
                                 entry.path().string());
      }
      result.emplace(entry.path().lexically_relative(root).generic_string(),
                     std::move(contents));
    }
  }
  return result;
}

struct PointwiseOperation {
  const char* name;
  fe::PointwiseMode_t frontend_mode;
  acdnnOpTensorOp_t acdnn_mode;
  acdnnActivationMode_t acdnn_activation_mode;
  const char* acdnn_primitive;
  float acdnn_right_coefficient;
  double acdnn_activation_coefficient;
  bool unary;
  bool comparison = false;
};

bool is_comparison_mode(fe::PointwiseMode_t mode) {
  return mode == fe::PointwiseMode_t::CMP_EQ ||
         mode == fe::PointwiseMode_t::CMP_NEQ ||
         mode == fe::PointwiseMode_t::CMP_GT ||
         mode == fe::PointwiseMode_t::CMP_GE ||
         mode == fe::PointwiseMode_t::CMP_LT ||
         mode == fe::PointwiseMode_t::CMP_LE;
}

bool uses_backend_reference(fe::PointwiseMode_t mode) {
  return mode == fe::PointwiseMode_t::CEIL ||
         mode == fe::PointwiseMode_t::FLOOR ||
         mode == fe::PointwiseMode_t::EXP ||
         mode == fe::PointwiseMode_t::LOG ||
         mode == fe::PointwiseMode_t::COS ||
         mode == fe::PointwiseMode_t::RSQRT ||
         mode == fe::PointwiseMode_t::SIN ||
         mode == fe::PointwiseMode_t::TAN ||
         mode == fe::PointwiseMode_t::SOFTPLUS_FWD ||
         mode == fe::PointwiseMode_t::SWISH_FWD ||
         mode == fe::PointwiseMode_t::GELU_APPROX_TANH_FWD ||
         mode == fe::PointwiseMode_t::DIV ||
         mode == fe::PointwiseMode_t::POW ||
         mode == fe::PointwiseMode_t::MOD ||
         mode == fe::PointwiseMode_t::SIGMOID_BWD ||
         mode == fe::PointwiseMode_t::RECIPROCAL ||
         is_comparison_mode(mode);
}

acdnnPointwiseMode_t backend_reference_mode(fe::PointwiseMode_t mode) {
  switch (mode) {
    case fe::PointwiseMode_t::CEIL:
      return ACDNN_POINTWISE_CEIL;
    case fe::PointwiseMode_t::FLOOR:
      return ACDNN_POINTWISE_FLOOR;
    case fe::PointwiseMode_t::EXP:
      return ACDNN_POINTWISE_EXP;
    case fe::PointwiseMode_t::LOG:
      return ACDNN_POINTWISE_LOG;
    case fe::PointwiseMode_t::COS:
      return ACDNN_POINTWISE_COS;
    case fe::PointwiseMode_t::RSQRT:
      return ACDNN_POINTWISE_RSQRT;
    case fe::PointwiseMode_t::SIN:
      return ACDNN_POINTWISE_SIN;
    case fe::PointwiseMode_t::TAN:
      return ACDNN_POINTWISE_TAN;
    case fe::PointwiseMode_t::SOFTPLUS_FWD:
      return ACDNN_POINTWISE_SOFTPLUS_FWD;
    case fe::PointwiseMode_t::SWISH_FWD:
      return ACDNN_POINTWISE_SWISH_FWD;
    case fe::PointwiseMode_t::GELU_APPROX_TANH_FWD:
      return ACDNN_POINTWISE_GELU_APPROX_TANH_FWD;
    case fe::PointwiseMode_t::DIV:
      return ACDNN_POINTWISE_DIV;
    case fe::PointwiseMode_t::POW:
      return ACDNN_POINTWISE_POW;
    case fe::PointwiseMode_t::MOD:
      return ACDNN_POINTWISE_MOD;
    case fe::PointwiseMode_t::SIGMOID_BWD:
      return ACDNN_POINTWISE_SIGMOID_BWD;
    case fe::PointwiseMode_t::RECIPROCAL:
      return ACDNN_POINTWISE_DIV;
    case fe::PointwiseMode_t::CMP_EQ:
      return ACDNN_POINTWISE_CMP_EQ;
    case fe::PointwiseMode_t::CMP_NEQ:
      return ACDNN_POINTWISE_CMP_NEQ;
    case fe::PointwiseMode_t::CMP_GT:
      return ACDNN_POINTWISE_CMP_GT;
    case fe::PointwiseMode_t::CMP_GE:
      return ACDNN_POINTWISE_CMP_GE;
    case fe::PointwiseMode_t::CMP_LT:
      return ACDNN_POINTWISE_CMP_LT;
    case fe::PointwiseMode_t::CMP_LE:
      return ACDNN_POINTWISE_CMP_LE;
    default:
      throw std::invalid_argument(
          "installed THead backend reference mode is not qualified");
  }
}

fe::graph::Graph make_graph(const PointwiseOperation& operation) {
  fe::graph::Graph graph;
  graph.set_name(std::string("installed_thead_") + operation.name)
      .set_io_data_type(fe::DataType_t::FLOAT)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT)
      .set_autotune(true);
  const auto make_tensor = [&graph](const char* name, std::int64_t uid) {
    return graph.tensor(fe::graph::Tensor_attributes()
                            .set_name(name)
                            .set_uid(uid)
                            .set_data_type(fe::DataType_t::FLOAT)
                            .set_dim({1, 1,
                                      static_cast<std::int64_t>(kElementCount)})
                            .set_stride({static_cast<std::int64_t>(kElementCount),
                                         static_cast<std::int64_t>(kElementCount),
                                         1}));
  };
  const auto left = make_tensor("left", 1);
  fe::graph::Pointwise_attributes attributes;
  attributes.set_name(operation.name)
      .set_mode(operation.frontend_mode)
      .set_compute_data_type(operation.comparison
                                 ? fe::DataType_t::BOOLEAN
                                 : fe::DataType_t::FLOAT);
  if (operation.acdnn_activation_mode == ACDNN_ACTIVATION_ELU) {
    attributes.set_elu_alpha(1.0F);
  } else if (operation.frontend_mode ==
             fe::PointwiseMode_t::SOFTPLUS_FWD) {
    attributes.set_softplus_beta(1.0F);
  } else if (operation.frontend_mode == fe::PointwiseMode_t::SWISH_FWD) {
    attributes.set_swish_beta(1.25F);
  }
  fe::graph::Graph::Tensor output;
  if (std::string_view(operation.name) == "add_square") {
    const auto right = make_tensor("right", 2);
    const auto square = graph.pointwise(
        right, right,
        fe::graph::Pointwise_attributes()
            .set_name("square")
            .set_mode(fe::PointwiseMode_t::MUL)
            .set_compute_data_type(fe::DataType_t::FLOAT));
    square->set_name("square")
        .set_uid(4)
        .set_data_type(fe::DataType_t::FLOAT)
        .set_dim({1, 1, static_cast<std::int64_t>(kElementCount)})
        .set_stride({static_cast<std::int64_t>(kElementCount),
                     static_cast<std::int64_t>(kElementCount), 1})
        .set_is_virtual(true);
    output = graph.pointwise(
        left, square,
        fe::graph::Pointwise_attributes()
            .set_name("add_square")
            .set_mode(fe::PointwiseMode_t::ADD)
            .set_compute_data_type(fe::DataType_t::FLOAT));
  } else if (operation.unary) {
    output = graph.pointwise(left, attributes);
  } else {
    const auto right = make_tensor("right", 2);
    output = graph.pointwise(left, right, attributes);
  }
  output->set_name("output")
      .set_uid(3)
      .set_data_type(operation.comparison
                         ? fe::DataType_t::BOOLEAN
                         : fe::DataType_t::FLOAT)
      .set_dim({1, 1, static_cast<std::int64_t>(kElementCount)})
      .set_stride({static_cast<std::int64_t>(kElementCount),
                   static_cast<std::int64_t>(kElementCount), 1})
      .set_output(true);
  return graph;
}

void execute_graph(fe::graph::Graph& graph, flagdnn::Handle& handle,
                   Stream& stream, DeviceBuffer& left, DeviceBuffer& right,
                   DeviceBuffer& output, bool unary) {
  const std::int64_t workspace_size = graph.get_workspace_size();
  require(workspace_size >= 0, "FlagDNN returned negative workspace size");
  DeviceBuffer workspace(static_cast<std::size_t>(workspace_size));
  std::vector<flagdnnBinding_t> bindings = {{1, left.data()}};
  if (!unary) {
    bindings.push_back({2, right.data()});
  }
  bindings.push_back({3, output.data()});
  check_frontend(graph.execute(handle, bindings, workspace.data(),
                               static_cast<std::size_t>(workspace_size),
                               stream.opaque()),
                 "installed THead pointwise execute");
}

}  // namespace

int main() {
  try {
    require_clean_environment();
    const std::filesystem::path sdk =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_SDK");
    const std::filesystem::path expected_plugin =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_PLUGIN");
    const std::filesystem::path expected_jit =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_JIT");
    const std::filesystem::path expected_triton =
        required_environment_path("FLAGDNN_INSTALLED_EXPECTED_TRITON");
    const std::filesystem::path ppu_sdk =
        required_environment_path("FLAGDNN_THEAD_PPU_SDK_ROOT");
    const std::filesystem::path cache =
        required_environment_path("FLAGDNN_CACHE_DIRECTORY");
    require(is_within(sdk, expected_plugin) && is_within(sdk, expected_jit),
            "installed plugin/JIT escaped the isolated SDK");
    require(!is_within(sdk, expected_triton),
            "Triton must remain an external installed prerequisite");

    check_driver(cuInit(0), "cuInit");
    CUdevice device = 0;
    check_driver(cuDeviceGet(&device, 0), "cuDeviceGet");
    PrimaryContext primary(device);
    Stream stream;
    flagdnn::Handle handle;
    require(handle.backend_name() == "thead",
            "default installed Handle did not select THead");

    std::array<float, kElementCount> host_left{};
    std::array<float, kElementCount> host_right{};
    std::array<float, kElementCount> host_flagdnn{};
    std::array<float, kElementCount> host_acdnn{};
    std::array<std::uint8_t, kElementCount> host_flagdnn_bool{};
    std::array<std::uint8_t, kElementCount> host_acdnn_bool{};
    for (std::size_t index = 0; index < kElementCount; ++index) {
      host_left[index] =
          static_cast<float>(static_cast<int>(index) - 128) * 0.25F;
      host_right[index] = 7.0F - static_cast<float>(index) * 0.125F;
    }
    constexpr std::size_t tensor_bytes = sizeof(host_left);
    DeviceBuffer left(tensor_bytes);
    DeviceBuffer right(tensor_bytes);
    DeviceBuffer flagdnn_output(tensor_bytes);
    DeviceBuffer acdnn_output(tensor_bytes);
    DeviceBuffer acdnn_intermediate(tensor_bytes);
    left.copy_from(host_left.data(), tensor_bytes, stream.get());
    right.copy_from(host_right.data(), tensor_bytes, stream.get());

    const std::array<PointwiseOperation, 37> operations = {{
        {"add", fe::PointwiseMode_t::ADD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(ADD)", 1.0F, 0.0, false},
        {"add_square", fe::PointwiseMode_t::ADD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(MUL)+acdnnOpTensor(ADD)", 1.0F, 0.0, false},
        {"sub", fe::PointwiseMode_t::SUB, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(ADD,alpha_right=-alpha)", -1.0F, 0.0, false},
        {"mul", fe::PointwiseMode_t::MUL, ACDNN_OP_TENSOR_MUL,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(MUL)", 1.0F, 0.0, false},
        {"min", fe::PointwiseMode_t::MIN, ACDNN_OP_TENSOR_MIN,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(MIN)", 1.0F, 0.0, false},
        {"max", fe::PointwiseMode_t::MAX, ACDNN_OP_TENSOR_MAX,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(MAX)", 1.0F, 0.0, false},
        {"scale", fe::PointwiseMode_t::MUL, ACDNN_OP_TENSOR_MUL,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnOpTensor(MUL)", 1.0F, 0.0, false},
        {"relu", fe::PointwiseMode_t::RELU_FWD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_RELU,
         "acdnnActivationForward(RELU,NOT_PROPAGATE_NAN)", 1.0F, 0.0,
         true},
        {"sigmoid", fe::PointwiseMode_t::SIGMOID_FWD,
         ACDNN_OP_TENSOR_ADD, ACDNN_ACTIVATION_SIGMOID,
         "acdnnActivationForward(SIGMOID,NOT_PROPAGATE_NAN)", 1.0F, 0.0,
         true},
        {"tanh", fe::PointwiseMode_t::TANH_FWD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_TANH,
         "acdnnActivationForward(TANH,NOT_PROPAGATE_NAN)", 1.0F, 0.0,
         true},
        {"elu", fe::PointwiseMode_t::ELU_FWD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_ELU,
         "acdnnActivationForward(ELU,NOT_PROPAGATE_NAN,alpha=1)", 1.0F,
         1.0, true},
        {"identity", fe::PointwiseMode_t::IDENTITY, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_IDENTITY,
         "acdnnTransformTensor(alpha=1,beta=0)", 1.0F, 1.0, true},
        {"gelu", fe::PointwiseMode_t::GELU_FWD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_GELU,
         "acdnnActivationForward(GELU,NOT_PROPAGATE_NAN)", 1.0F, 0.0,
         true},
        {"abs", fe::PointwiseMode_t::ABS, ACDNN_OP_TENSOR_MAX,
         ACDNN_ACTIVATION_IDENTITY,
         "acdnnTransformTensor(alpha=-1)+acdnnOpTensor(MAX)", 1.0F,
         -1.0, true},
        {"neg", fe::PointwiseMode_t::NEG, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_IDENTITY,
         "acdnnTransformTensor(alpha=-1,beta=0)", 1.0F, -1.0, true},
        {"sqrt", fe::PointwiseMode_t::SQRT, ACDNN_OP_TENSOR_SQRT,
         ACDNN_ACTIVATION_SIGMOID, "acdnnOpTensor(SQRT)", 1.0F, 0.0,
         true},
        {"ceil", fe::PointwiseMode_t::CEIL, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CEIL)", 1.0F, 0.0, true},
        {"floor", fe::PointwiseMode_t::FLOOR, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_FLOOR)", 1.0F, 0.0, true},
        {"exp", fe::PointwiseMode_t::EXP, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_EXP)", 1.0F, 0.0, true},
        {"log", fe::PointwiseMode_t::LOG, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_LOG)", 1.0F, 0.0, true},
        {"cos", fe::PointwiseMode_t::COS, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_COS)", 1.0F, 0.0, true},
        {"rsqrt", fe::PointwiseMode_t::RSQRT, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_RSQRT)", 1.0F, 0.0, true},
        {"sin", fe::PointwiseMode_t::SIN, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_SIN)", 1.0F, 0.0, true},
        {"tan", fe::PointwiseMode_t::TAN, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_TAN)", 1.0F, 0.0, true},
        {"softplus", fe::PointwiseMode_t::SOFTPLUS_FWD,
         ACDNN_OP_TENSOR_ADD, ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_SOFTPLUS_FWD,beta=1)", 1.0F,
         1.0, true},
        {"swish", fe::PointwiseMode_t::SWISH_FWD, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_SWISH_FWD,beta=1.25)", 1.0F,
         1.25, true},
        {"gelu_approx_tanh", fe::PointwiseMode_t::GELU_APPROX_TANH_FWD,
         ACDNN_OP_TENSOR_ADD, ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_GELU_APPROX_TANH_FWD)", 1.0F,
         0.0, true},
        {"div", fe::PointwiseMode_t::DIV, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_DIV)", 1.0F, 0.0, false},
        {"pow", fe::PointwiseMode_t::POW, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_POW)", 1.0F, 0.0, false},
        {"sigmoid_backward", fe::PointwiseMode_t::SIGMOID_BWD,
         ACDNN_OP_TENSOR_ADD, ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_SIGMOID_BWD)", 1.0F, 0.0,
         false},
        {"reciprocal", fe::PointwiseMode_t::RECIPROCAL,
         ACDNN_OP_TENSOR_ADD, ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_DIV,numerator=1)", 1.0F, 0.0,
         true},
        {"cmp_eq", fe::PointwiseMode_t::CMP_EQ, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_EQ)", 1.0F, 0.0, false,
         true},
        {"cmp_neq", fe::PointwiseMode_t::CMP_NEQ, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_NEQ)", 1.0F, 0.0, false,
         true},
        {"cmp_gt", fe::PointwiseMode_t::CMP_GT, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_GT)", 1.0F, 0.0, false,
         true},
        {"cmp_ge", fe::PointwiseMode_t::CMP_GE, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_GE)", 1.0F, 0.0, false,
         true},
        {"cmp_lt", fe::PointwiseMode_t::CMP_LT, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_LT)", 1.0F, 0.0, false,
         true},
        {"cmp_le", fe::PointwiseMode_t::CMP_LE, ACDNN_OP_TENSOR_ADD,
         ACDNN_ACTIVATION_SIGMOID,
         "acdnnBackendExecute(POINTWISE_CMP_LE)", 1.0F, 0.0, false,
         true},
    }};
    bool mapped_images_checked = false;
    for (const PointwiseOperation& operation : operations) {
      if (operation.frontend_mode == fe::PointwiseMode_t::DIV ||
          operation.frontend_mode == fe::PointwiseMode_t::MOD) {
        for (std::size_t index = 0; index < kElementCount; ++index) {
          host_left[index] = static_cast<float>((index % 41) + 1) / 13.0F;
          host_right[index] = static_cast<float>((index % 23) + 1) / 7.0F;
        }
        left.copy_from(host_left.data(), tensor_bytes, stream.get());
        right.copy_from(host_right.data(), tensor_bytes, stream.get());
      } else if (operation.frontend_mode == fe::PointwiseMode_t::POW) {
        for (std::size_t index = 0; index < kElementCount; ++index) {
          host_left[index] = static_cast<float>((index % 31) + 1) / 16.0F;
          host_right[index] =
              static_cast<float>(static_cast<int>(index % 9) - 4) / 8.0F;
        }
        left.copy_from(host_left.data(), tensor_bytes, stream.get());
        right.copy_from(host_right.data(), tensor_bytes, stream.get());
      } else if (operation.frontend_mode == fe::PointwiseMode_t::SQRT ||
          operation.frontend_mode == fe::PointwiseMode_t::LOG ||
          operation.frontend_mode == fe::PointwiseMode_t::RSQRT ||
          operation.frontend_mode == fe::PointwiseMode_t::RECIPROCAL) {
        for (std::size_t index = 0; index < kElementCount; ++index) {
          host_left[index] = static_cast<float>((index % 41) + 1) / 13.0F;
        }
        left.copy_from(host_left.data(), tensor_bytes, stream.get());
      } else if (operation.frontend_mode == fe::PointwiseMode_t::EXP ||
                 operation.frontend_mode == fe::PointwiseMode_t::TAN) {
        for (std::size_t index = 0; index < kElementCount; ++index) {
          host_left[index] =
              static_cast<float>(static_cast<int>(index % 31) - 15) / 8.0F;
        }
        left.copy_from(host_left.data(), tensor_bytes, stream.get());
      } else {
        for (std::size_t index = 0; index < kElementCount; ++index) {
          host_left[index] =
              static_cast<float>(static_cast<int>(index) - 128) * 0.25F;
        }
        left.copy_from(host_left.data(), tensor_bytes, stream.get());
      }
      fe::graph::Graph graph = make_graph(operation);
      check_frontend(graph.build(handle, {fe::HeurMode_t::A}),
                     std::string("installed THead ") + operation.name +
                         " graph build");
      if (!mapped_images_checked) {
        require(mapped_image(expected_plugin) == expected_plugin,
                "installed consumer mapped a different THead plugin");
        require(mapped_image(expected_jit) == expected_jit,
                "installed consumer mapped libtriton_jit outside its SDK");
        const auto expected_cuda = std::filesystem::canonical(
            ppu_sdk / "CUDA_SDK/lib64/libcuda.so.1");
        require(image_containing(&cuInit, "libcuda.so.1") == expected_cuda,
                "installed consumer mapped an unexpected libcuda.so.1");
        const auto expected_acdnn =
            std::filesystem::canonical(ppu_sdk / "lib/libacdnn.so");
        require(image_containing(&acdnnCreate, "libacdnn.so") ==
                    expected_acdnn,
                "installed consumer mapped an unexpected libacdnn.so");
        mapped_images_checked = true;
      }

      if (std::string_view(operation.name) == "add_square") {
        AcdnnPointwise multiply(ACDNN_OP_TENSOR_MUL,
                                "acdnnOpTensor(MUL)", 1.0F);
        multiply.execute(stream.get(), right.data(), right.data(),
                         acdnn_intermediate.data());
        AcdnnPointwise add(ACDNN_OP_TENSOR_ADD,
                           "acdnnOpTensor(ADD)", 1.0F);
        add.execute(stream.get(), left.data(), acdnn_intermediate.data(),
                    acdnn_output.data());
        check_driver(cuStreamSynchronize(stream.get()),
                     "cuStreamSynchronize(acDNN AddSquare reference)");
      } else if (uses_backend_reference(operation.frontend_mode)) {
        AcdnnBackendPointwise reference(
            backend_reference_mode(operation.frontend_mode),
            operation.acdnn_primitive,
            !operation.unary,
            operation.acdnn_activation_coefficient,
            operation.frontend_mode == fe::PointwiseMode_t::RECIPROCAL,
            operation.comparison);
        reference.execute(stream.get(), left.data(), right.data(),
                          acdnn_output.data());
      } else if (operation.frontend_mode == fe::PointwiseMode_t::ABS) {
        AcdnnActivation negate(ACDNN_ACTIVATION_IDENTITY,
                               "acdnnTransformTensor(alpha=-1,beta=0)",
                               -1.0);
        negate.execute(stream.get(), left.data(),
                       acdnn_intermediate.data());
        AcdnnPointwise maximum(ACDNN_OP_TENSOR_MAX,
                               "acdnnOpTensor(MAX)", 1.0F);
        maximum.execute(stream.get(), left.data(),
                        acdnn_intermediate.data(), acdnn_output.data());
        check_driver(cuStreamSynchronize(stream.get()),
                     "cuStreamSynchronize(acDNN Abs reference)");
      } else if (operation.unary &&
          operation.frontend_mode != fe::PointwiseMode_t::SQRT) {
        AcdnnActivation reference(operation.acdnn_activation_mode,
                                  operation.acdnn_primitive,
                                  operation.acdnn_activation_coefficient);
        reference.execute(stream.get(), left.data(), acdnn_output.data());
        check_driver(cuStreamSynchronize(stream.get()),
                     "cuStreamSynchronize(acDNN activation reference)");
      } else {
        AcdnnPointwise reference(operation.acdnn_mode,
                                 operation.acdnn_primitive,
                                 operation.acdnn_right_coefficient);
        reference.execute(stream.get(), left.data(),
                          operation.unary ? left.data() : right.data(),
                          acdnn_output.data());
        check_driver(cuStreamSynchronize(stream.get()),
                     "cuStreamSynchronize(acDNN OpTensor reference)");
      }
      execute_graph(graph, handle, stream, left, right, flagdnn_output,
                    operation.unary);
      flagdnn_output.copy_to(
          operation.comparison
              ? static_cast<void*>(host_flagdnn_bool.data())
              : static_cast<void*>(host_flagdnn.data()),
          operation.comparison ? sizeof(host_flagdnn_bool) : tensor_bytes,
          stream.get());
      acdnn_output.copy_to(
          operation.comparison ? static_cast<void*>(host_acdnn_bool.data())
                               : static_cast<void*>(host_acdnn.data()),
          operation.comparison ? sizeof(host_acdnn_bool) : tensor_bytes,
          stream.get());
      check_driver(cuStreamSynchronize(stream.get()), "cuStreamSynchronize");
      const bool approximate =
          operation.frontend_mode == fe::PointwiseMode_t::SIGMOID_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::TANH_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::ELU_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::GELU_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::SQRT ||
          operation.frontend_mode == fe::PointwiseMode_t::EXP ||
          operation.frontend_mode == fe::PointwiseMode_t::LOG ||
          operation.frontend_mode == fe::PointwiseMode_t::COS ||
          operation.frontend_mode == fe::PointwiseMode_t::RSQRT ||
          operation.frontend_mode == fe::PointwiseMode_t::SIN ||
          operation.frontend_mode == fe::PointwiseMode_t::TAN ||
          operation.frontend_mode == fe::PointwiseMode_t::SOFTPLUS_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::SWISH_FWD ||
          operation.frontend_mode ==
              fe::PointwiseMode_t::GELU_APPROX_TANH_FWD ||
          operation.frontend_mode == fe::PointwiseMode_t::DIV ||
          operation.frontend_mode == fe::PointwiseMode_t::POW ||
          operation.frontend_mode == fe::PointwiseMode_t::MOD ||
          operation.frontend_mode == fe::PointwiseMode_t::SIGMOID_BWD ||
          operation.frontend_mode == fe::PointwiseMode_t::RECIPROCAL;
      for (std::size_t index = 0; index < kElementCount; ++index) {
        if (operation.comparison) {
          if (host_flagdnn_bool[index] != host_acdnn_bool[index]) {
            throw std::runtime_error(
                std::string("installed THead ") + operation.name +
                " differs from acDNN at index " + std::to_string(index));
          }
          continue;
        }
        const float absolute =
            std::abs(host_flagdnn[index] - host_acdnn[index]);
        const float relative =
            absolute /
            std::max({std::abs(host_flagdnn[index]),
                      std::abs(host_acdnn[index]), 1.0e-30F});
        if (!std::isfinite(absolute) ||
            (absolute > (approximate ? 2.0e-5F : 0.0F) &&
             relative > (approximate ? 1.0e-5F : 0.0F))) {
          throw std::runtime_error(
              std::string("installed THead ") + operation.name +
              " differs from acDNN at index " + std::to_string(index) +
              ": FlagDNN=" + std::to_string(host_flagdnn[index]) +
              " acDNN=" + std::to_string(host_acdnn[index]));
        }
      }

      const CacheSnapshot first_snapshot = snapshot_cache(cache);
      require(!first_snapshot.empty(), "installed THead cache is empty");
      fe::graph::Graph replay_graph = make_graph(operation);
      check_frontend(replay_graph.build(handle, {fe::HeurMode_t::A}),
                     std::string("installed cached THead ") + operation.name +
                         " graph build");
      execute_graph(replay_graph, handle, stream, left, right, flagdnn_output,
                    operation.unary);
      flagdnn_output.copy_to(
          operation.comparison
              ? static_cast<void*>(host_flagdnn_bool.data())
              : static_cast<void*>(host_flagdnn.data()),
          operation.comparison ? sizeof(host_flagdnn_bool) : tensor_bytes,
          stream.get());
      check_driver(cuStreamSynchronize(stream.get()),
                   "cuStreamSynchronize(cached execute)");
      for (std::size_t index = 0; index < kElementCount; ++index) {
        if (operation.comparison) {
          if (host_flagdnn_bool[index] != host_acdnn_bool[index]) {
            throw std::runtime_error(
                std::string("installed cached THead ") + operation.name +
                " differs from acDNN at index " + std::to_string(index));
          }
          continue;
        }
        const float absolute =
            std::abs(host_flagdnn[index] - host_acdnn[index]);
        const float relative =
            absolute /
            std::max({std::abs(host_flagdnn[index]),
                      std::abs(host_acdnn[index]), 1.0e-30F});
        if (!std::isfinite(absolute) ||
            (absolute > (approximate ? 2.0e-5F : 0.0F) &&
             relative > (approximate ? 1.0e-5F : 0.0F))) {
          throw std::runtime_error(
              std::string("installed cached THead ") + operation.name +
              " differs from acDNN at index " + std::to_string(index) +
              ": FlagDNN=" + std::to_string(host_flagdnn[index]) +
              " acDNN=" + std::to_string(host_acdnn[index]));
        }
      }
      require(snapshot_cache(cache) == first_snapshot,
              "installed cached build/execute changed compiler or autotune "
              "cache contents");
      std::cout << "PASS installed THead " << operation.name
                << " Graph vs " << operation.acdnn_primitive << '\n';
    }

    std::cout << "PASS installed FlagDNN Graph "
                 "pointwise and AddSquare operations -> "
                 "CUDA-backend "
                 "libtriton_jit -> PPU-aware Triton -> autotune; "
                 "reference=acDNN OpTensor Add/Sub/Mul/Min/Max/Scale/Sqrt + "
                 "Activation Relu/Sigmoid/Tanh/Elu/Gelu + Transform Identity/Neg + "
                 "Transform/Max Abs + qualified backend pointwise descriptors\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
