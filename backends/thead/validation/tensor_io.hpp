// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_TENSOR_IO_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_TENSOR_IO_HPP_

#include "ppu_driver.hpp"

#include <cuda.h>

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

namespace flagdnn::validation::thead {

using DeviceStream = Stream;

void validate_transfer_bounds(std::size_t allocation_bytes,
                              std::size_t offset_bytes,
                              std::size_t transfer_bytes);

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t bytes = 0);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept;
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept;

  [[nodiscard]] CUdeviceptr address() const noexcept { return address_; }
  [[nodiscard]] void *data() const noexcept;
  [[nodiscard]] void *at(std::size_t byte_offset) const;
  [[nodiscard]] std::size_t size() const noexcept { return bytes_; }

 private:
  void release() noexcept;

  CUdeviceptr address_ = 0;
  std::size_t bytes_ = 0;
};

class DeviceEvent final {
 public:
  DeviceEvent();
  ~DeviceEvent();

  DeviceEvent(const DeviceEvent &) = delete;
  DeviceEvent &operator=(const DeviceEvent &) = delete;
  DeviceEvent(DeviceEvent &&other) noexcept;
  DeviceEvent &operator=(DeviceEvent &&other) noexcept;

  [[nodiscard]] CUevent get() const noexcept { return event_; }
  void record(CUstream stream) const;
  void synchronize() const;
  [[nodiscard]] double elapsed_microseconds_to(
      const DeviceEvent &stop) const;

 private:
  void release() noexcept;

  CUevent event_ = nullptr;
};

void copy_to_device_async(DeviceBuffer &destination,
                          std::span<const std::byte> source,
                          std::size_t destination_offset,
                          CUstream stream);
void copy_from_device_async(std::span<std::byte> destination,
                            const DeviceBuffer &source,
                            std::size_t source_offset,
                            CUstream stream);

template <typename T>
void copy_to_device_async(DeviceBuffer &destination,
                          std::span<const T> source,
                          std::size_t destination_element_offset,
                          CUstream stream) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (destination_element_offset >
      std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::out_of_range("typed PPU upload byte offset overflows");
  }
  copy_to_device_async(destination, std::as_bytes(source),
                       destination_element_offset * sizeof(T), stream);
}

template <typename T>
void copy_from_device_async(std::span<T> destination,
                            const DeviceBuffer &source,
                            std::size_t source_element_offset,
                            CUstream stream) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (source_element_offset >
      std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::out_of_range("typed PPU download byte offset overflows");
  }
  copy_from_device_async(std::as_writable_bytes(destination), source,
                         source_element_offset * sizeof(T), stream);
}

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_TENSOR_IO_HPP_
