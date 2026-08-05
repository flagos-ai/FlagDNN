/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#ifndef FLAGDNN_ERROR_HPP_
#define FLAGDNN_ERROR_HPP_

#include <flagdnn/flagdnn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::native {

class ApiError : public std::runtime_error {
 public:
  ApiError(flagdnnStatus_t status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  [[nodiscard]] flagdnnStatus_t status() const noexcept { return status_; }

 private:
  flagdnnStatus_t status_;
};

void clear_last_error() noexcept;
void set_last_error(std::string message) noexcept;
const char* last_error() noexcept;

}  // namespace flagdnn::native

#endif  // FLAGDNN_ERROR_HPP_
