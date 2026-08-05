/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_NVIDIA_VALIDATION_CUDA_DRIVER_HPP_
#define FLAGDNN_BACKENDS_NVIDIA_VALIDATION_CUDA_DRIVER_HPP_

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>

namespace flagdnn::validation::nvidia {

inline void check_cuda(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  const char* message = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &message);
  std::ostringstream output;
  output << operation << " failed";
  if (name != nullptr) {
    output << " (" << name << ")";
  }
  if (message != nullptr) {
    output << ": " << message;
  }
  throw std::runtime_error(output.str());
}

class DriverContext {
 public:
  explicit DriverContext(int device_ordinal = 0) {
    check_cuda(cuInit(0), "cuInit");
    check_cuda(cuDeviceGet(&device_, device_ordinal), "cuDeviceGet");
    check_cuda(cuDevicePrimaryCtxRetain(&context_, device_),
               "cuDevicePrimaryCtxRetain");
    check_cuda(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
  }

  ~DriverContext() {
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  DriverContext(const DriverContext&) = delete;
  DriverContext& operator=(const DriverContext&) = delete;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

class Stream {
 public:
  Stream() {
    check_cuda(cuStreamCreate(&value_, CU_STREAM_NON_BLOCKING),
               "cuStreamCreate");
  }

  ~Stream() {
    if (value_ != nullptr) {
      (void)cuStreamDestroy(value_);
    }
  }

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] CUstream get() const noexcept { return value_; }
  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(value_);
  }

  void synchronize() const {
    check_cuda(cuStreamSynchronize(value_), "cuStreamSynchronize");
  }

 private:
  CUstream value_ = nullptr;
};

class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t bytes)
      : bytes_(std::max<std::size_t>(bytes, 1)) {
    check_cuda(cuMemAlloc(&value_, bytes_), "cuMemAlloc");
  }

  ~DeviceBuffer() {
    if (value_ != 0) {
      (void)cuMemFree(value_);
    }
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(static_cast<std::uintptr_t>(value_));
  }

  [[nodiscard]] void* opaque_at(std::size_t byte_offset) const {
    if (byte_offset >= bytes_) {
      throw std::invalid_argument("device pointer offset exceeds buffer");
    }
    return reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(value_ + byte_offset));
  }

  void copy_from_host(const void* source,
                      std::size_t bytes,
                      CUstream stream) const {
    if (bytes > bytes_) {
      throw std::invalid_argument("host-to-device copy exceeds buffer");
    }
    check_cuda(cuMemcpyHtoDAsync(value_, source, bytes, stream),
               "cuMemcpyHtoDAsync");
  }

  void copy_from_host_at(const void* source,
                         std::size_t bytes,
                         std::size_t byte_offset,
                         CUstream stream) const {
    if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("host-to-device offset copy exceeds buffer");
    }
    check_cuda(cuMemcpyHtoDAsync(
                   value_ + byte_offset, source, bytes, stream),
               "cuMemcpyHtoDAsync(offset)");
  }

  void copy_to_host(void* destination,
                    std::size_t bytes,
                    CUstream stream) const {
    if (bytes > bytes_) {
      throw std::invalid_argument("device-to-host copy exceeds buffer");
    }
    check_cuda(cuMemcpyDtoHAsync(destination, value_, bytes, stream),
               "cuMemcpyDtoHAsync");
  }

  void copy_to_host_at(void* destination,
                       std::size_t bytes,
                       std::size_t byte_offset,
                       CUstream stream) const {
    if (byte_offset > bytes_ || bytes > bytes_ - byte_offset) {
      throw std::invalid_argument("device-to-host offset copy exceeds buffer");
    }
    check_cuda(cuMemcpyDtoHAsync(
                   destination, value_ + byte_offset, bytes, stream),
               "cuMemcpyDtoHAsync(offset)");
  }

 private:
  CUdeviceptr value_ = 0;
  std::size_t bytes_ = 0;
};

class EventTimer {
 public:
  EventTimer() {
    check_cuda(cuEventCreate(&start_, CU_EVENT_DEFAULT),
               "cuEventCreate(start)");
    try {
      check_cuda(cuEventCreate(&stop_, CU_EVENT_DEFAULT),
                 "cuEventCreate(stop)");
    } catch (...) {
      (void)cuEventDestroy(start_);
      start_ = nullptr;
      throw;
    }
  }

  ~EventTimer() {
    if (stop_ != nullptr) {
      (void)cuEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cuEventDestroy(start_);
    }
  }

  EventTimer(const EventTimer&) = delete;
  EventTimer& operator=(const EventTimer&) = delete;

  template <typename Function>
  double measure_microseconds(CUstream stream,
                              int iterations,
                              Function&& function) {
    if (iterations <= 0) {
      throw std::invalid_argument("benchmark iterations must be positive");
    }
    check_cuda(cuEventRecord(start_, stream), "cuEventRecord(start)");
    for (int index = 0; index < iterations; ++index) {
      function();
    }
    check_cuda(cuEventRecord(stop_, stream), "cuEventRecord(stop)");
    check_cuda(cuEventSynchronize(stop_), "cuEventSynchronize(stop)");
    float milliseconds = 0.0F;
    check_cuda(cuEventElapsedTime(&milliseconds, start_, stop_),
               "cuEventElapsedTime");
    return static_cast<double>(milliseconds) * 1000.0 /
           static_cast<double>(iterations);
  }

 private:
  CUevent start_ = nullptr;
  CUevent stop_ = nullptr;
};

}  // namespace flagdnn::validation::nvidia


namespace flagdnn::testing {
using validation::nvidia::check_cuda;
using validation::nvidia::DeviceBuffer;
using validation::nvidia::DriverContext;
using validation::nvidia::EventTimer;
using validation::nvidia::Stream;
}  // namespace flagdnn::testing

namespace flagdnn::benchmarking {
using validation::nvidia::check_cuda;
using validation::nvidia::DeviceBuffer;
using validation::nvidia::DriverContext;
using validation::nvidia::EventTimer;
using validation::nvidia::Stream;
}  // namespace flagdnn::benchmarking

#endif  // FLAGDNN_BACKENDS_NVIDIA_VALIDATION_CUDA_DRIVER_HPP_
