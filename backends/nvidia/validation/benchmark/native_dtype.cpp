/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include <iostream>
#include <stdexcept>

#include "common/dtype_runner.hpp"
namespace flagdnn::testing {
int run_cudnn_boolean_benchmark(int, char**, std::string_view);
int run_native_copy_benchmark(int, char**, std::string_view);
int run_native_precision_benchmark(int, char**, std::string_view);
int run_native_reduction_benchmark(int, char**);
int run_native_dtype_benchmark(int argc, char** argv,
                               std::string_view operation,
                               std::string_view category) {
  try {
    if (category == "boolean")
      return run_cudnn_boolean_benchmark(argc, argv, operation);
    if (category == "copy")
      return run_native_copy_benchmark(argc, argv, operation);
    if (category == "precision")
      return run_native_precision_benchmark(argc, argv, operation);
    if (category == "fp32_output")
      return run_native_reduction_benchmark(argc, argv);
    throw std::invalid_argument("unknown dtype benchmark category");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
}  // namespace flagdnn::testing
