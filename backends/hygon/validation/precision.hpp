/* Copyright (c) 2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string_view>
namespace flagdnn::validation::hygon {
inline int selected_input_precision() {
  const char *value = std::getenv("FLAGDNN_INPUT_PRECISION");
  if (!value || std::string_view(value) == "0")
    return 0;
  if (std::string_view(value) == "1")
    return 1;
  if (std::string_view(value) == "2")
    return 2;
  throw std::invalid_argument("FLAGDNN_INPUT_PRECISION must be 0, 1 or 2");
}
} // namespace flagdnn::validation::hygon
