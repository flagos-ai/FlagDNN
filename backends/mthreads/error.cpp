/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/mthreads/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::mthreads {
namespace {

flagdnnBackendResult_t map_runtime_status(
    musaError_t status,
    flagdnnBackendResult_t fallback) {
  if (status == musaErrorMemoryAllocation) {
    return FLAGDNN_BACKEND_RESULT_ALLOC_FAILED;
  }
  if (status == musaErrorInvalidValue ||
      status == musaErrorInvalidDevice ||
      status == musaErrorInvalidResourceHandle) {
    return FLAGDNN_BACKEND_RESULT_INVALID_VALUE;
  }
  return fallback;
}

flagdnnBackendResult_t map_driver_status(
    MUresult status,
    flagdnnBackendResult_t fallback) {
  if (status == MUSA_ERROR_OUT_OF_MEMORY) {
    return FLAGDNN_BACKEND_RESULT_ALLOC_FAILED;
  }
  if (status == MUSA_ERROR_INVALID_VALUE ||
      status == MUSA_ERROR_INVALID_DEVICE ||
      status == MUSA_ERROR_INVALID_HANDLE) {
    return FLAGDNN_BACKEND_RESULT_INVALID_VALUE;
  }
  return fallback;
}

}  // namespace

MthreadsError::MthreadsError(
    flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t MthreadsError::result() const noexcept {
  return result_;
}

std::string musa_error(
    musaError_t status, std::string_view operation) {
  const char* name = musaGetErrorName(status);
  const char* description = musaGetErrorString(status);
  std::ostringstream output;
  output << operation << " failed with MUSA runtime status "
         << static_cast<int>(status);
  if (name != nullptr) {
    output << " (" << name << ')';
  }
  if (description != nullptr) {
    output << ": " << description;
  }
  return output.str();
}

std::string mu_error(
    MUresult status, std::string_view operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  static_cast<void>(muGetErrorName(status, &name));
  static_cast<void>(muGetErrorString(status, &description));
  std::ostringstream output;
  output << operation << " failed with MUSA driver status "
         << static_cast<int>(status);
  if (name != nullptr) {
    output << " (" << name << ')';
  }
  if (description != nullptr) {
    output << ": " << description;
  }
  return output.str();
}

void check_musa(
    musaError_t status,
    std::string_view operation,
    flagdnnBackendResult_t failure_result) {
  if (status != musaSuccess) {
    throw MthreadsError(
        map_runtime_status(status, failure_result),
        musa_error(status, operation));
  }
}

void check_mu(
    MUresult status,
    std::string_view operation,
    flagdnnBackendResult_t failure_result) {
  if (status != MUSA_SUCCESS) {
    throw MthreadsError(
        map_driver_status(status, failure_result),
        mu_error(status, operation));
  }
}

void require(
    bool condition,
    std::string message,
    flagdnnBackendResult_t result) {
  if (!condition) {
    throw MthreadsError(result, std::move(message));
  }
}

}  // namespace flagdnn::mthreads
