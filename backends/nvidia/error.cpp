/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/nvidia/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::cuda {

CudaError::CudaError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t CudaError::result() const noexcept {
  return result_;
}

std::string cuda_error(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  std::ostringstream output;
  output << operation << " failed";
  if (name != nullptr) {
    output << " (" << name << ')';
  }
  if (description != nullptr) {
    output << ": " << description;
  }
  return output.str();
}

void check_cuda(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw CudaError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                    cuda_error(result, operation));
  }
}

void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw CudaError(result, message);
  }
}

}  // namespace flagdnn::cuda
