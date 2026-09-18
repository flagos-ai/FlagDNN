/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUSA_DRIVER_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUSA_DRIVER_HPP_

#include <mudnn.h>
#include <musa_runtime_api.h>

#include <cstddef>
#include <string_view>

#include "backends/mthreads/validation/case_status.hpp"

namespace flagdnn::validation::mthreads {

void check_musa(musaError_t status, std::string_view operation);
void check_mudnn(musa::dnn::Status status, std::string_view operation);

class Stream final {
 public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  [[nodiscard]] musaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] void* opaque() const noexcept {
    return reinterpret_cast<void*>(stream_);
  }
  void synchronize() const;

 private:
  musaStream_t stream_ = nullptr;
};

class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t size, std::size_t alignment = 1);
  ~DeviceBuffer();

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] void* opaque() const noexcept { return pointer_; }
  [[nodiscard]] void* opaque_at(std::size_t offset) const;
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  void copy_from_host(
      const void* source, std::size_t size, musaStream_t stream);
  void copy_from_host_at(const void* source,
                         std::size_t size,
                         std::size_t offset,
                         musaStream_t stream);
  void copy_to_host(
      void* destination, std::size_t size, musaStream_t stream) const;
  void copy_to_host_at(void* destination,
                       std::size_t size,
                       std::size_t offset,
                       musaStream_t stream) const;
  void clear(musaStream_t stream);

 private:
  void* allocation_ = nullptr;
  void* pointer_ = nullptr;
  std::size_t size_ = 0;
};

}  // namespace flagdnn::validation::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_VALIDATION_MUSA_DRIVER_HPP_
