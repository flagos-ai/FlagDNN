// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "tensor_io.hpp"

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {

void validate_transfer_bounds(std::size_t allocation_bytes,
                              std::size_t offset_bytes,
                              std::size_t transfer_bytes) {
  if (offset_bytes > allocation_bytes ||
      transfer_bytes > allocation_bytes - offset_bytes) {
    throw std::out_of_range("PPU transfer exceeds device allocation");
  }
}

DeviceBuffer::DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
  if (bytes_ != 0) {
    check_driver(cuMemAlloc(&address_, bytes_), "cuMemAlloc");
    if (address_ == 0) {
      throw std::runtime_error("PPU driver returned a null device allocation");
    }
  }
}

DeviceBuffer::~DeviceBuffer() { release(); }

DeviceBuffer::DeviceBuffer(DeviceBuffer &&other) noexcept
    : address_(std::exchange(other.address_, 0)),
      bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer &DeviceBuffer::operator=(DeviceBuffer &&other) noexcept {
  if (this != &other) {
    release();
    address_ = std::exchange(other.address_, 0);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

void *DeviceBuffer::data() const noexcept {
  return reinterpret_cast<void *>(static_cast<std::uintptr_t>(address_));
}

void *DeviceBuffer::at(std::size_t byte_offset) const {
  validate_transfer_bounds(bytes_, byte_offset, 0);
  if (address_ == 0) {
    if (byte_offset == 0 && bytes_ == 0) {
      return nullptr;
    }
    throw std::invalid_argument("device allocation is not live");
  }
  return reinterpret_cast<void *>(
      static_cast<std::uintptr_t>(address_ + byte_offset));
}

void DeviceBuffer::release() noexcept {
  if (address_ != 0) {
    (void)cuMemFree(address_);
    address_ = 0;
  }
  bytes_ = 0;
}

DeviceEvent::DeviceEvent() {
  check_driver(cuEventCreate(&event_, CU_EVENT_DEFAULT), "cuEventCreate");
  if (event_ == nullptr) {
    throw std::runtime_error("PPU driver returned a null event");
  }
}

DeviceEvent::~DeviceEvent() { release(); }

DeviceEvent::DeviceEvent(DeviceEvent &&other) noexcept
    : event_(std::exchange(other.event_, nullptr)) {}

DeviceEvent &DeviceEvent::operator=(DeviceEvent &&other) noexcept {
  if (this != &other) {
    release();
    event_ = std::exchange(other.event_, nullptr);
  }
  return *this;
}

void DeviceEvent::record(CUstream stream) const {
  if (event_ == nullptr || stream == nullptr) {
    throw std::invalid_argument("event recording requires live resources");
  }
  check_driver(cuEventRecord(event_, stream), "cuEventRecord");
}

void DeviceEvent::synchronize() const {
  if (event_ == nullptr) {
    throw std::invalid_argument("event synchronization requires a live event");
  }
  check_driver(cuEventSynchronize(event_), "cuEventSynchronize");
}

double DeviceEvent::elapsed_microseconds_to(const DeviceEvent &stop) const {
  if (event_ == nullptr || stop.event_ == nullptr) {
    throw std::invalid_argument("event timing requires live events");
  }
  float milliseconds = 0.0F;
  check_driver(cuEventElapsedTime(&milliseconds, event_, stop.event_),
               "cuEventElapsedTime");
  const double microseconds = static_cast<double>(milliseconds) * 1000.0;
  if (!std::isfinite(microseconds) || microseconds <= 0.0) {
    throw std::runtime_error("PPU event duration is not positive and finite");
  }
  return microseconds;
}

void DeviceEvent::release() noexcept {
  if (event_ != nullptr) {
    (void)cuEventDestroy(event_);
    event_ = nullptr;
  }
}

void copy_to_device_async(DeviceBuffer &destination,
                          std::span<const std::byte> source,
                          std::size_t destination_offset,
                          CUstream stream) {
  validate_transfer_bounds(destination.size(), destination_offset,
                           source.size_bytes());
  if (source.empty()) {
    return;
  }
  if (stream == nullptr || destination.address() == 0) {
    throw std::invalid_argument("device upload requires live resources");
  }
  check_driver(cuMemcpyHtoDAsync(destination.address() + destination_offset,
                                 source.data(), source.size_bytes(), stream),
               "cuMemcpyHtoDAsync");
}

void copy_from_device_async(std::span<std::byte> destination,
                            const DeviceBuffer &source,
                            std::size_t source_offset,
                            CUstream stream) {
  validate_transfer_bounds(source.size(), source_offset,
                           destination.size_bytes());
  if (destination.empty()) {
    return;
  }
  if (stream == nullptr || source.address() == 0) {
    throw std::invalid_argument("device download requires live resources");
  }
  check_driver(cuMemcpyDtoHAsync(destination.data(),
                                 source.address() + source_offset,
                                 destination.size_bytes(), stream),
               "cuMemcpyDtoHAsync");
}

}  // namespace flagdnn::validation::thead
