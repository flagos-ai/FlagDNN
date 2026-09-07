// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REFERENCE_HPP_
#define FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REFERENCE_HPP_

#include "capability.hpp"
#include "tensor_io.hpp"

#include <acdnn.h>

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace flagdnn::validation::thead {

class AcdnnStatusError final : public std::runtime_error {
 public:
  AcdnnStatusError(acdnnStatus_t status, std::string_view operation);

  [[nodiscard]] acdnnStatus_t status() const noexcept { return status_; }

 private:
  acdnnStatus_t status_;
};

void check_acdnn(acdnnStatus_t status, std::string_view operation);

class AcdnnHandle final {
 public:
  AcdnnHandle();
  ~AcdnnHandle();

  AcdnnHandle(const AcdnnHandle &) = delete;
  AcdnnHandle &operator=(const AcdnnHandle &) = delete;
  AcdnnHandle(AcdnnHandle &&other) noexcept;
  AcdnnHandle &operator=(AcdnnHandle &&other) noexcept;

  [[nodiscard]] acdnnHandle_t get() const noexcept { return handle_; }
  void set_stream(hggcStream_t stream);

 private:
  void release() noexcept;

  acdnnHandle_t handle_ = nullptr;
};

class AcdnnTensorDescriptor final {
 public:
  AcdnnTensorDescriptor();
  ~AcdnnTensorDescriptor();

  AcdnnTensorDescriptor(const AcdnnTensorDescriptor &) = delete;
  AcdnnTensorDescriptor &operator=(const AcdnnTensorDescriptor &) = delete;
  AcdnnTensorDescriptor(AcdnnTensorDescriptor &&other) noexcept;
  AcdnnTensorDescriptor &operator=(AcdnnTensorDescriptor &&other) noexcept;

  [[nodiscard]] acdnnTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }
  void set(acdnnDataType_t data_type, std::span<const int> dimensions,
           std::span<const int> strides);

 private:
  void release() noexcept;

  acdnnTensorDescriptor_t descriptor_ = nullptr;
};

class AcdnnOpTensorDescriptor final {
 public:
  AcdnnOpTensorDescriptor();
  ~AcdnnOpTensorDescriptor();

  AcdnnOpTensorDescriptor(const AcdnnOpTensorDescriptor &) = delete;
  AcdnnOpTensorDescriptor &operator=(const AcdnnOpTensorDescriptor &) = delete;
  AcdnnOpTensorDescriptor(AcdnnOpTensorDescriptor &&other) noexcept;
  AcdnnOpTensorDescriptor &operator=(AcdnnOpTensorDescriptor &&other) noexcept;

  [[nodiscard]] acdnnOpTensorDescriptor_t get() const noexcept {
    return descriptor_;
  }
  void set(acdnnOpTensorOp_t operation, acdnnDataType_t compute_type,
           acdnnNanPropagation_t nan_policy);

 private:
  void release() noexcept;

  acdnnOpTensorDescriptor_t descriptor_ = nullptr;
};

class AcdnnActivationDescriptor final {
 public:
  AcdnnActivationDescriptor();
  ~AcdnnActivationDescriptor();

  AcdnnActivationDescriptor(const AcdnnActivationDescriptor &) = delete;
  AcdnnActivationDescriptor &operator=(const AcdnnActivationDescriptor &) =
      delete;
  AcdnnActivationDescriptor(AcdnnActivationDescriptor &&other) noexcept;
  AcdnnActivationDescriptor &operator=(
      AcdnnActivationDescriptor &&other) noexcept;

  [[nodiscard]] acdnnActivationDescriptor_t get() const noexcept {
    return descriptor_;
  }
  void set(acdnnActivationMode_t mode, acdnnNanPropagation_t nan_policy,
           double coefficient);

 private:
  void release() noexcept;

  acdnnActivationDescriptor_t descriptor_ = nullptr;
};

class AcdnnWorkspace final {
 public:
  explicit AcdnnWorkspace(std::size_t bytes) : storage_(bytes) {}

  [[nodiscard]] void *data() const noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

 private:
  DeviceBuffer storage_;
};

struct ReferencePlan {
  CapabilityStatus qualification = CapabilityStatus::kProbeRequired;
  ReferencePath path = ReferencePath::kNone;
  std::vector<std::string> primitives;
  CapabilityConstraints constraints;
};

struct UnsupportedCapability {
  std::string reason_code;
  std::string detail;
};

using ReferenceSelection =
    std::variant<ReferencePlan, UnsupportedCapability>;

[[nodiscard]] ReferenceSelection
select_reference(const CapabilityRecord &record);
void require_reference_status(const CapabilityRecord &record,
                              acdnnStatus_t status,
                              std::string_view operation);

}  // namespace flagdnn::validation::thead

#endif  // FLAGDNN_BACKENDS_THEAD_VALIDATION_ACDNN_REFERENCE_HPP_
