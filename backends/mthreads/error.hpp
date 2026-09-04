/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_MTHREADS_ERROR_HPP_
#define FLAGDNN_BACKENDS_MTHREADS_ERROR_HPP_

#include "backends/backend_api.h"

#include <musa.h>
#include <musa_runtime_api.h>

#include <stdexcept>
#include <string>
#include <string_view>

namespace flagdnn::mthreads {

class MthreadsError final : public std::runtime_error {
 public:
  MthreadsError(flagdnnBackendResult_t result, std::string message);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;

 private:
  flagdnnBackendResult_t result_;
};

[[nodiscard]] std::string musa_error(
    musaError_t status, std::string_view operation);
[[nodiscard]] std::string mu_error(
    MUresult status, std::string_view operation);

void check_musa(
    musaError_t status,
    std::string_view operation,
    flagdnnBackendResult_t failure_result =
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
void check_mu(
    MUresult status,
    std::string_view operation,
    flagdnnBackendResult_t failure_result =
        FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR);
void require(
    bool condition,
    std::string message,
    flagdnnBackendResult_t result =
        FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

}  // namespace flagdnn::mthreads

#endif  // FLAGDNN_BACKENDS_MTHREADS_ERROR_HPP_
