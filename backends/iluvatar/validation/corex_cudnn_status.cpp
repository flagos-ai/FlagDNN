// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "corex_cudnn_status.hpp"

#include <string>

namespace flagdnn::iluvatar::validation {
namespace {

std::string status_message(cudnnStatus_t status, const char *operation) {
  const char *detail = cudnnGetErrorString(status);
  return std::string(operation == nullptr ? "CoreX cuDNN operation"
                                          : operation) +
         " failed: " + (detail == nullptr ? "unknown cuDNN status" : detail) +
         " (status=" + std::to_string(static_cast<int>(status)) + ")";
}

} // namespace

CorexCudnnStatusError::CorexCudnnStatusError(cudnnStatus_t status,
                                             const char *operation)
    : std::runtime_error(status_message(status, operation)), status_(status) {}

void check_cudnn(cudnnStatus_t status, const char *operation) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw CorexCudnnStatusError(status, operation);
  }
}

bool cudnn_status_is_runtime_capability(cudnnStatus_t status) noexcept {
  return status == CUDNN_STATUS_NOT_SUPPORTED;
}

} // namespace flagdnn::iluvatar::validation
