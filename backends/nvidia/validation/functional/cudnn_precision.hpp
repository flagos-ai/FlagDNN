/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#ifndef FLAGDNN_NVIDIA_CUDNN_PRECISION_HPP_
#define FLAGDNN_NVIDIA_CUDNN_PRECISION_HPP_
#include <cstdlib>
#include <stdexcept>
#include <string_view>
namespace flagdnn::testing::cuda {
inline int selected_input_precision() {
  const char* value = std::getenv("FLAGDNN_INPUT_PRECISION");
  if (!value || std::string_view(value) == "0") return 0;
  if (std::string_view(value) == "1") return 1;
  if (std::string_view(value) == "2") return 2;
  throw std::invalid_argument("FLAGDNN_INPUT_PRECISION must be 0, 1 or 2");
}
inline void require_cudnn_precision_environment(int precision) {
  const char* override_value = std::getenv("NVIDIA_TF32_OVERRIDE");
  if (precision == 1 &&
      (!override_value || std::string_view(override_value) != "0"))
    throw std::invalid_argument(
        "IEEE cuDNN comparisons require NVIDIA_TF32_OVERRIDE=0 before process "
        "startup");
  if (precision == 2 && override_value &&
      std::string_view(override_value) == "0")
    throw std::invalid_argument(
        "TF32 cuDNN comparisons require TF32 to be enabled at process startup");
}
}  // namespace flagdnn::testing::cuda
#endif
