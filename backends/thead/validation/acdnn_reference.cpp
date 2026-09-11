// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "acdnn_reference.hpp"

#include <stdexcept>
#include <utility>

namespace flagdnn::validation::thead {
namespace {

std::string status_message(acdnnStatus_t status,
                           std::string_view operation) {
  std::string message(operation);
  message += " failed with acDNN status ";
  message += std::to_string(static_cast<int>(status));
  const char *detail = acdnnGetErrorString(status);
  if (detail != nullptr && *detail != '\0') {
    message += ": ";
    message += detail;
  }
  return message;
}

}  // namespace

AcdnnStatusError::AcdnnStatusError(acdnnStatus_t status,
                                   std::string_view operation)
    : std::runtime_error(status_message(status, operation)), status_(status) {}

void check_acdnn(acdnnStatus_t status, std::string_view operation) {
  if (status != ACDNN_STATUS_SUCCESS) {
    throw AcdnnStatusError(status, operation);
  }
}

AcdnnHandle::AcdnnHandle() {
  check_acdnn(acdnnCreate(&handle_), "acdnnCreate");
  if (handle_ == nullptr) {
    throw std::runtime_error("acdnnCreate returned a null handle");
  }
}

AcdnnHandle::~AcdnnHandle() { release(); }

AcdnnHandle::AcdnnHandle(AcdnnHandle &&other) noexcept
    : handle_(std::exchange(other.handle_, nullptr)) {}

AcdnnHandle &AcdnnHandle::operator=(AcdnnHandle &&other) noexcept {
  if (this != &other) {
    release();
    handle_ = std::exchange(other.handle_, nullptr);
  }
  return *this;
}

void AcdnnHandle::set_stream(hggcStream_t stream) {
  if (handle_ == nullptr || stream == nullptr) {
    throw std::invalid_argument("acDNN stream binding requires live resources");
  }
  check_acdnn(acdnnSetStream(handle_, stream), "acdnnSetStream");
}

void AcdnnHandle::release() noexcept {
  if (handle_ != nullptr) {
    (void)acdnnDestroy(handle_);
    handle_ = nullptr;
  }
}

AcdnnTensorDescriptor::AcdnnTensorDescriptor() {
  check_acdnn(acdnnCreateTensorDescriptor(&descriptor_),
              "acdnnCreateTensorDescriptor");
  if (descriptor_ == nullptr) {
    throw std::runtime_error(
        "acdnnCreateTensorDescriptor returned a null descriptor");
  }
}

AcdnnTensorDescriptor::~AcdnnTensorDescriptor() { release(); }

AcdnnTensorDescriptor::AcdnnTensorDescriptor(
    AcdnnTensorDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

AcdnnTensorDescriptor &
AcdnnTensorDescriptor::operator=(AcdnnTensorDescriptor &&other) noexcept {
  if (this != &other) {
    release();
    descriptor_ = std::exchange(other.descriptor_, nullptr);
  }
  return *this;
}

void AcdnnTensorDescriptor::set(acdnnDataType_t data_type,
                                std::span<const int> dimensions,
                                std::span<const int> strides) {
  if (descriptor_ == nullptr || dimensions.empty() ||
      dimensions.size() != strides.size()) {
    throw std::invalid_argument("invalid acDNN tensor descriptor shape");
  }
  for (std::size_t axis = 0; axis < dimensions.size(); ++axis) {
    if (dimensions[axis] <= 0 || strides[axis] <= 0) {
      throw std::invalid_argument(
          "acDNN tensor dimensions and strides must be positive");
    }
  }
  check_acdnn(acdnnSetTensorNdDescriptor(
                  descriptor_, data_type, static_cast<int>(dimensions.size()),
                  dimensions.data(), strides.data()),
              "acdnnSetTensorNdDescriptor");
}

void AcdnnTensorDescriptor::release() noexcept {
  if (descriptor_ != nullptr) {
    (void)acdnnDestroyTensorDescriptor(descriptor_);
    descriptor_ = nullptr;
  }
}

AcdnnOpTensorDescriptor::AcdnnOpTensorDescriptor() {
  check_acdnn(acdnnCreateOpTensorDescriptor(&descriptor_),
              "acdnnCreateOpTensorDescriptor");
  if (descriptor_ == nullptr) {
    throw std::runtime_error(
        "acdnnCreateOpTensorDescriptor returned a null descriptor");
  }
}

AcdnnOpTensorDescriptor::~AcdnnOpTensorDescriptor() { release(); }

AcdnnOpTensorDescriptor::AcdnnOpTensorDescriptor(
    AcdnnOpTensorDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

AcdnnOpTensorDescriptor &AcdnnOpTensorDescriptor::operator=(
    AcdnnOpTensorDescriptor &&other) noexcept {
  if (this != &other) {
    release();
    descriptor_ = std::exchange(other.descriptor_, nullptr);
  }
  return *this;
}

void AcdnnOpTensorDescriptor::set(acdnnOpTensorOp_t operation,
                                  acdnnDataType_t compute_type,
                                  acdnnNanPropagation_t nan_policy) {
  if (descriptor_ == nullptr) {
    throw std::invalid_argument("acDNN operation descriptor is not live");
  }
  check_acdnn(acdnnSetOpTensorDescriptor(descriptor_, operation, compute_type,
                                          nan_policy),
              "acdnnSetOpTensorDescriptor");
}

void AcdnnOpTensorDescriptor::release() noexcept {
  if (descriptor_ != nullptr) {
    (void)acdnnDestroyOpTensorDescriptor(descriptor_);
    descriptor_ = nullptr;
  }
}

AcdnnActivationDescriptor::AcdnnActivationDescriptor() {
  check_acdnn(acdnnCreateActivationDescriptor(&descriptor_),
              "acdnnCreateActivationDescriptor");
  if (descriptor_ == nullptr) {
    throw std::runtime_error(
        "acdnnCreateActivationDescriptor returned a null descriptor");
  }
}

AcdnnActivationDescriptor::~AcdnnActivationDescriptor() { release(); }

AcdnnActivationDescriptor::AcdnnActivationDescriptor(
    AcdnnActivationDescriptor &&other) noexcept
    : descriptor_(std::exchange(other.descriptor_, nullptr)) {}

AcdnnActivationDescriptor &AcdnnActivationDescriptor::operator=(
    AcdnnActivationDescriptor &&other) noexcept {
  if (this != &other) {
    release();
    descriptor_ = std::exchange(other.descriptor_, nullptr);
  }
  return *this;
}

void AcdnnActivationDescriptor::set(
    acdnnActivationMode_t mode, acdnnNanPropagation_t nan_policy,
    double coefficient) {
  if (descriptor_ == nullptr) {
    throw std::invalid_argument("acDNN activation descriptor is not live");
  }
  check_acdnn(acdnnSetActivationDescriptor(descriptor_, mode, nan_policy,
                                            coefficient),
              "acdnnSetActivationDescriptor");
}

void AcdnnActivationDescriptor::release() noexcept {
  if (descriptor_ != nullptr) {
    (void)acdnnDestroyActivationDescriptor(descriptor_);
    descriptor_ = nullptr;
  }
}

ReferenceSelection select_reference(const CapabilityRecord &record) {
  if (record.status == CapabilityStatus::kUnsupported) {
    return UnsupportedCapability{record.reason_code, record.detail};
  }
  if (!record.constraints.has_value() ||
      record.path == ReferencePath::kNone || record.reference_plan.empty()) {
    throw std::invalid_argument("executable acDNN capability is malformed");
  }
  return ReferencePlan{record.status, record.path, record.reference_plan,
                       *record.constraints};
}

void require_reference_status(const CapabilityRecord &record,
                              acdnnStatus_t status,
                              std::string_view operation) {
  if (record.status == CapabilityStatus::kUnsupported) {
    throw std::invalid_argument(
        "cannot execute a statically unsupported acDNN capability");
  }
  check_acdnn(status, operation);
}

}  // namespace flagdnn::validation::thead
