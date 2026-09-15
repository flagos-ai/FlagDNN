/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/layout.hpp"
#include "common/pointwise.hpp"
#include "validation/functional/layout_runner.hpp"
#include "validation/functional/pointwise_runner.hpp"
namespace flagdnn::testing {
int run_native_copy_benchmark(int argc, char** argv,
                              std::string_view operation) {
  if (operation == "identity") {
    const auto cases = make_unary_pointwise_cases(
        {.operation_name = "identity", .mode = FLAGDNN_POINTWISE_IDENTITY});
    return run_cudnn_pointwise_tests(argc, argv, cases,
                                     "FLAGDNN_IDENTITY_BENCHMARK", true);
  }
  const auto mode = operation == "reshape"     ? LayoutOperation::kReshape
                    : operation == "transpose" ? LayoutOperation::kTranspose
                                               : LayoutOperation::kSlice;
  const auto cases = make_layout_cases(mode);
  return run_cudnn_layout_tests(argc, argv, cases, true);
}
}  // namespace flagdnn::testing
