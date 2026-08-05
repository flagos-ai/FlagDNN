/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_CUDA_ERROR_HPP_
#define FLAGDNN_BACKENDS_CUDA_ERROR_HPP_

#include "backends/backend_api.h"

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace flagdnn::cuda {

class CudaError : public std::runtime_error {
 public:
  CudaError(flagdnnBackendResult_t result, std::string message);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;

 private:
  flagdnnBackendResult_t result_;
};

[[nodiscard]] std::string cuda_error(CUresult result,
                                     const char* operation);
void check_cuda(CUresult result, const char* operation);
void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result =
                 FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

}  // namespace flagdnn::cuda

#endif  // FLAGDNN_BACKENDS_CUDA_ERROR_HPP_
