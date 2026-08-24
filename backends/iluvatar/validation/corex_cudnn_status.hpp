// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_STATUS_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_STATUS_HPP_

#include <cudnn.h>

#include <stdexcept>

namespace flagdnn::iluvatar::validation {

class CorexCudnnStatusError final : public std::runtime_error {
public:
  CorexCudnnStatusError(cudnnStatus_t status, const char *operation);

  [[nodiscard]] cudnnStatus_t status() const noexcept { return status_; }

private:
  cudnnStatus_t status_;
};

void check_cudnn(cudnnStatus_t status, const char *operation);
[[nodiscard]] bool
cudnn_status_is_runtime_capability(cudnnStatus_t status) noexcept;

} // namespace flagdnn::iluvatar::validation

#endif // FLAGDNN_BACKENDS_ILUVATAR_VALIDATION_COREX_CUDNN_STATUS_HPP_
