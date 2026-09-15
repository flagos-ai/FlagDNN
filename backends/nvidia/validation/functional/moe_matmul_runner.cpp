/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/moe_matmul.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_moe_matmul_functional_test(int argc, char** argv,
                                   std::span<const MoeMatmulTestCase> cases,
                                   bool benchmark) {
  return cuda::run_paired_cases(
      argc, argv, cases, "FLAGDNN_MOE_MATMUL_CASE", build_flagdnn_moe_matmul,
      moe_matmul_inputs, build_cudnn_moe_matmul,
      [](const MoeMatmulTestCase& test_case, std::size_t) {
        const auto type = test_case.outputs[0].data_type;
        const auto tolerance = type == FLAGDNN_DATA_BFLOAT16  ? 4.0e-3
                               : type == FLAGDNN_DATA_FLOAT16 ? 5.0e-4
                                                              : 1.0e-3;
        // cuDNN grouped FP8 accumulation differs from cuDNN's standalone
        // matmul even for identical quantized inputs. Cross-checking those
        // two cuDNN paths over K=16..512 establishes this absolute bound.
        const double contraction =
            static_cast<double>(test_case.inputs[0].dimensions.back());
        const double absolute =
            type == FLAGDNN_DATA_FLOAT32
                ? 2.0e-3 * std::sqrt(std::max(1.0, contraction / 128.0))
                : tolerance;
        return cuda::PairedTolerance{absolute, tolerance};
      },
      benchmark);
}
}  // namespace flagdnn::testing
