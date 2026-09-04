/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/validation/musa_driver.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace flagdnn::validation::mthreads {

void check_musa(musaError_t status, std::string_view operation) {
  if (status == musaSuccess) {
    return;
  }
  const char* name = musaGetErrorName(status);
  const char* description = musaGetErrorString(status);
  throw std::runtime_error(
      std::string(operation) + " failed with MUSA status " +
      std::to_string(static_cast<int>(status)) + " (" +
      (name == nullptr ? "unknown" : name) + "): " +
      (description == nullptr ? "unknown" : description));
}

void check_mudnn(
    musa::dnn::Status status, std::string_view operation) {
  if (status == musa::dnn::Status::SUCCESS) {
    return;
  }
  throw std::runtime_error(
      std::string(operation) + " failed with muDNN status " +
      std::to_string(static_cast<int>(status)));
}

Stream::Stream() {
  check_musa(
      musaStreamCreateWithFlags(&stream_, musaStreamNonBlocking),
      "musaStreamCreateWithFlags");
}

Stream::~Stream() {
  if (stream_ != nullptr) {
    static_cast<void>(musaStreamDestroy(stream_));
  }
}

void Stream::synchronize() const {
  check_musa(musaStreamSynchronize(stream_), "musaStreamSynchronize");
}

DeviceBuffer::DeviceBuffer(std::size_t size, std::size_t alignment)
    : size_(size) {
  if (alignment == 0 ||
      (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument(
        "MUSA allocation alignment must be a power of two");
  }
  if (size_ == 0) {
    return;
  }
  if (size_ > std::numeric_limits<std::size_t>::max() -
                  (alignment - 1)) {
    throw std::overflow_error("aligned MUSA allocation size overflows");
  }
  check_musa(
      musaMalloc(&allocation_, size_ + alignment - 1), "musaMalloc");
  const std::uintptr_t base =
      reinterpret_cast<std::uintptr_t>(allocation_);
  if (base > std::numeric_limits<std::uintptr_t>::max() -
                 (alignment - 1)) {
    static_cast<void>(musaFree(allocation_));
    allocation_ = nullptr;
    throw std::overflow_error("aligned MUSA pointer overflows");
  }
  const std::uintptr_t aligned =
      (base + alignment - 1) & ~(alignment - 1);
  pointer_ = reinterpret_cast<void*>(aligned);
}

DeviceBuffer::~DeviceBuffer() {
  if (allocation_ != nullptr) {
    static_cast<void>(musaFree(allocation_));
  }
}

void* DeviceBuffer::opaque_at(std::size_t offset) const {
  if (offset > size_) {
    throw std::out_of_range("MUSA allocation offset is out of range");
  }
  if (pointer_ == nullptr) {
    if (offset != 0) {
      throw std::out_of_range("null MUSA allocation has a nonzero offset");
    }
    return nullptr;
  }
  return static_cast<void*>(static_cast<std::byte*>(pointer_) + offset);
}

void DeviceBuffer::copy_from_host(
    const void* source, std::size_t size, musaStream_t stream) {
  copy_from_host_at(source, size, 0, stream);
}

void DeviceBuffer::copy_from_host_at(
    const void* source,
    std::size_t size,
    std::size_t offset,
    musaStream_t stream) {
  if (offset > size_ || size > size_ - offset ||
      (size != 0 && source == nullptr)) {
    throw std::invalid_argument(
        "host-to-device copy exceeds the MUSA allocation");
  }
  if (size != 0) {
    check_musa(
        musaMemcpyAsync(
            opaque_at(offset),
            source,
            size,
            musaMemcpyHostToDevice,
            stream),
        "musaMemcpyAsync(host-to-device)");
  }
}

void DeviceBuffer::copy_to_host(
    void* destination, std::size_t size, musaStream_t stream) const {
  copy_to_host_at(destination, size, 0, stream);
}

void DeviceBuffer::copy_to_host_at(
    void* destination,
    std::size_t size,
    std::size_t offset,
    musaStream_t stream) const {
  if (offset > size_ || size > size_ - offset ||
      (size != 0 && destination == nullptr)) {
    throw std::invalid_argument(
        "device-to-host copy exceeds the MUSA allocation");
  }
  if (size != 0) {
    check_musa(
        musaMemcpyAsync(
            destination,
            opaque_at(offset),
            size,
            musaMemcpyDeviceToHost,
            stream),
        "musaMemcpyAsync(device-to-host)");
  }
}

void DeviceBuffer::clear(musaStream_t stream) {
  if (size_ != 0) {
    check_musa(
        musaMemsetAsync(pointer_, 0, size_, stream),
        "musaMemsetAsync");
  }
}

}  // namespace flagdnn::validation::mthreads
