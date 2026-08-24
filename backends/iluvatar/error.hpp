/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_BACKENDS_ILUVATAR_ERROR_HPP_
#define FLAGDNN_BACKENDS_ILUVATAR_ERROR_HPP_

#include "backends/backend_api.h"

#include <cuda.h>

#include <stdexcept>
#include <string>

namespace flagdnn::iluvatar {

class IluvatarError final : public std::runtime_error {
public:
  IluvatarError(flagdnnBackendResult_t result, std::string message);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;

private:
  flagdnnBackendResult_t result_;
};

[[nodiscard]] std::string corex_error(CUresult result, const char *operation);
void check_corex(CUresult result, const char *operation);
void require(
    bool condition, const char *message,
    flagdnnBackendResult_t result = FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

} // namespace flagdnn::iluvatar

#endif // FLAGDNN_BACKENDS_ILUVATAR_ERROR_HPP_
