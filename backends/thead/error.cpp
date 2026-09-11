// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "backends/thead/error.hpp"

#include <sstream>
#include <utility>

namespace flagdnn::thead {
namespace {

thread_local std::string current_error;

}  // namespace

TheadError::TheadError(flagdnnBackendResult_t result, std::string message)
    : std::runtime_error(std::move(message)), result_(result) {}

flagdnnBackendResult_t TheadError::result() const noexcept {
  return result_;
}

void clear_last_error() noexcept {
  current_error.clear();
}

void set_last_error(const char* message) noexcept {
  try {
    current_error = message == nullptr ? "" : message;
  } catch (...) {
    current_error.clear();
  }
}

const char* last_error() noexcept {
  return current_error.c_str();
}

std::string driver_error(CUresult result, const char* operation) {
  const char* name = nullptr;
  const char* description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  std::ostringstream output;
  output << "PPU CUDA compatibility operation " << operation << " failed";
  if (name != nullptr) {
    output << " (" << name << ')';
  }
  if (description != nullptr) {
    output << ": " << description;
  }
  return output.str();
}

void check_driver(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    throw TheadError(FLAGDNN_BACKEND_RESULT_RUNTIME_ERROR,
                     driver_error(result, operation));
  }
}

void require(bool condition,
             const char* message,
             flagdnnBackendResult_t result) {
  if (!condition) {
    throw TheadError(result, message);
  }
}

}  // namespace flagdnn::thead
