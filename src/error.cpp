/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */

#include "error.hpp"

#include <string>
#include <utility>

namespace flagdnn::native {
namespace {

thread_local std::string current_last_error;

}  // namespace

void clear_last_error() noexcept { current_last_error.clear(); }

void set_last_error(std::string message) noexcept {
  try {
    current_last_error = std::move(message);
  } catch (...) {
    current_last_error.clear();
  }
}

const char* last_error() noexcept { return current_last_error.c_str(); }

}  // namespace flagdnn::native
