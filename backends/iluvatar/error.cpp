/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "backends/iluvatar/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::iluvatar {

IluvatarError::IluvatarError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t IluvatarError::result() const noexcept {
  return result_;
}

std::string corex_error(CUresult result, const char *operation) {
  const char *name = nullptr;
  const char *description = nullptr;
  const CUresult name_result = cuGetErrorName(result, &name);
  const CUresult description_result = cuGetErrorString(result, &description);

  std::ostringstream output;
  output << (operation == nullptr ? "CoreX Driver operation" : operation)
         << " failed with CUresult " << static_cast<int>(result);
  if (name_result == CUDA_SUCCESS && name != nullptr) {
    output << " (" << name << ')';
  } else {
    output << " (error name unavailable)";
  }
  if (description_result == CUDA_SUCCESS && description != nullptr) {
    output << ": " << description;
  } else {
    output << ": error description unavailable";
  }
  return output.str();
}

void check_corex(CUresult result, const char *operation) {
  if (result != CUDA_SUCCESS) {
    throw IluvatarError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                        corex_error(result, operation));
  }
}

void require(bool condition, const char *message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw IluvatarError(result, message == nullptr ? "Iluvatar contract failed"
                                                   : message);
  }
}

} // namespace flagdnn::iluvatar
