/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/moe_matmul.hpp"
namespace flagdnn::testing {
int run_moe_matmul_functional_test(int argc, char** argv,
                                   std::span<const MoeMatmulTestCase> cases,
                                   bool benchmark) {
  return mthreads::run_paired_cases(
      argc, argv, cases, "FLAGDNN_MOE_MATMUL_CASE",
      cases.front().backward ? "moe_grouped_matmul_bwd" : "moe_grouped_matmul",
      build_flagdnn_moe_matmul, moe_matmul_inputs,
      [](const auto& c) { return mthreads::reference(c); },
      [](const MoeMatmulTestCase& test_case, std::size_t) {
        const auto type = test_case.outputs[0].data_type;
        const auto tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 4.0e-3
                               : type == FLAGDNN_DATA_FLOAT16 ? 5.0e-4
                                                              : 1.0e-3;
        // Keep the accuracy thresholds aligned with the NVIDIA matrix,
        // including its contraction-size bound for scaled FP8 GEMM.
        const double contraction =
            static_cast<double>(test_case.inputs[0].dimensions.back());
        const double absolute =
            type == FLAGDNN_DATA_FLOAT32
                ? 2.0e-3 * std::sqrt(std::max(1.0, contraction / 128.0))
                : tolerance;
        return mthreads::Tolerance{absolute, tolerance};
      },
      // muDNN 3.1.5 GroupedMatMul graph replay can hang or produce zeros.
      // Keep both providers on the same direct MUSA Event timing path.
      benchmark, false);
}
}  // namespace flagdnn::testing
