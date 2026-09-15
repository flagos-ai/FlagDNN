/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "common/fp8_matmul.hpp"
#include "validation/functional/cudnn_extended.hpp"
#include "validation/functional/paired.hpp"
namespace flagdnn::testing {
int run_fp8_matmul_functional_test(int argc, char** argv,
                                   std::span<const Fp8MatmulTestCase> cases,
                                   bool benchmark) {
  const char* api_value = std::getenv("FLAGDNN_FP8_MATMUL_API");
  const std::string_view api = api_value ? api_value : "matmul_fp8";
  if (api != "matmul" && api != "matmul_fp8")
    throw std::invalid_argument("unknown FP8 matmul API selection");
  std::vector<Fp8MatmulTestCase> selected;
  for (const auto& test_case : cases)
    if (test_case.plain_matmul == (api == "matmul"))
      selected.push_back(test_case);
  return cuda::run_paired_cases(
      argc, argv, std::span<const Fp8MatmulTestCase>(selected),
      "FLAGDNN_FP8_MATMUL_CASE", build_flagdnn_fp8_matmul, fp8_matmul_inputs,
      build_cudnn_fp8_matmul,
      [](const Fp8MatmulTestCase& test_case, std::size_t) {
        const auto type = test_case.outputs[0].data_type;
        // cuDNN's fused FP8 descale plan has lower accumulation accuracy than
        // its unscaled matmul plan on SM90. Independent cuDNN-vs-cuDNN runs
        // over all four input formats and K=32..512 establish this bound.
        const double contraction =
            static_cast<double>(test_case.inputs[0].dimensions.back());
        const double absolute =
            test_case.scale_mode == 1
                ? 2.0e-3 * std::sqrt(std::max(1.0, contraction / 128.0))
                : 2.0e-5;
        return cuda::PairedTolerance{
            absolute, type == FLAGDNN_DATA_BFLOAT16 ? 8.0e-3 : 1.0e-3};
      },
      benchmark);
}
}  // namespace flagdnn::testing
