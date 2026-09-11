// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#ifndef FLAGDNN_BACKENDS_THEAD_ERROR_HPP_
#define FLAGDNN_BACKENDS_THEAD_ERROR_HPP_

#include "backends/backend_api.h"

#include <cuda.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

namespace flagdnn::thead {

class TheadError final : public std::runtime_error {
 public:
  TheadError(flagdnnBackendResult_t result, std::string message);

  [[nodiscard]] flagdnnBackendResult_t result() const noexcept;

 private:
  flagdnnBackendResult_t result_;
};

void clear_last_error() noexcept;
void set_last_error(const char* message) noexcept;
[[nodiscard]] const char* last_error() noexcept;

[[nodiscard]] std::string driver_error(CUresult result,
                                       const char* operation);
void check_driver(CUresult result, const char* operation);
void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result =
                 FLAGDNN_BACKEND_RESULT_INVALID_VALUE);

template <typename Function>
flagdnnBackendResult_t plugin_call(Function&& function) noexcept {
  clear_last_error();
  try {
    std::forward<Function>(function)();
    return FLAGDNN_BACKEND_RESULT_SUCCESS;
  } catch (const TheadError& error) {
    set_last_error(error.what());
    return error.result();
  } catch (const std::bad_alloc&) {
    set_last_error("host memory allocation failed");
    return FLAGDNN_BACKEND_RESULT_ALLOC_FAILED;
  } catch (const std::exception& error) {
    set_last_error(error.what());
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  } catch (...) {
    set_last_error("unknown THead backend error");
    return FLAGDNN_BACKEND_RESULT_INTERNAL_ERROR;
  }
}

}  // namespace flagdnn::thead

#endif  // FLAGDNN_BACKENDS_THEAD_ERROR_HPP_
