// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_PPU_DRIVER_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_PPU_DRIVER_HPP_

#include <cuda.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::validation::thead {

inline void check_driver(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* detail = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &detail);
  std::string message = std::string(operation) + " failed";
  if (name != nullptr) {
    message += " (" + std::string(name) + ")";
  }
  if (detail != nullptr) {
    message += ": " + std::string(detail);
  }
  throw std::runtime_error(std::move(message));
}

struct PrimaryContextState {
  unsigned int flags = 0;
  int active = 0;

  [[nodiscard]] bool operator==(const PrimaryContextState&) const = default;
};

inline PrimaryContextState primary_context_state(CUdevice device) {
  PrimaryContextState state;
  check_driver(cuDevicePrimaryCtxGetState(device, &state.flags, &state.active),
               "cuDevicePrimaryCtxGetState");
  return state;
}

inline CUcontext current_context() {
  CUcontext context = nullptr;
  check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
  return context;
}

class PrimaryContext final {
 public:
  explicit PrimaryContext(CUdevice device) : device_(device) {
    check_driver(cuDevicePrimaryCtxRetain(&context_, device_),
                 "cuDevicePrimaryCtxRetain");
  }

  ~PrimaryContext() {
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  PrimaryContext(const PrimaryContext&) = delete;
  PrimaryContext& operator=(const PrimaryContext&) = delete;

  [[nodiscard]] CUcontext get() const noexcept { return context_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

class ScopedCurrentContext final {
 public:
  explicit ScopedCurrentContext(CUcontext context) {
    previous_ = current_context();
    if (previous_ != context) {
      check_driver(cuCtxPushCurrent(context), "cuCtxPushCurrent");
      pushed_ = true;
    }
  }

  ~ScopedCurrentContext() {
    if (pushed_) {
      CUcontext popped = nullptr;
      (void)cuCtxPopCurrent(&popped);
    }
  }

  ScopedCurrentContext(const ScopedCurrentContext&) = delete;
  ScopedCurrentContext& operator=(const ScopedCurrentContext&) = delete;

  [[nodiscard]] CUcontext previous() const noexcept { return previous_; }

 private:
  CUcontext previous_ = nullptr;
  bool pushed_ = false;
};

class Stream final {
 public:
  Stream() {
    check_driver(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
                 "cuStreamCreate(non-default)");
    if (stream_ == nullptr) {
      throw std::runtime_error("PPU driver returned a null non-default stream");
    }
  }

  ~Stream() {
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] CUstream get() const noexcept { return stream_; }
  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(stream_);
  }

  [[nodiscard]] CUcontext context() const {
    CUcontext result = nullptr;
    check_driver(cuStreamGetCtx(stream_, &result), "cuStreamGetCtx");
    return result;
  }

 private:
  CUstream stream_ = nullptr;
};

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_PPU_DRIVER_HPP_
