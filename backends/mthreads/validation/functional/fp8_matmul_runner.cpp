/* Copyright (c) 2025-2026 BAAI. SPDX-License-Identifier: Apache-2.0 */
#include "backends/mthreads/validation/functional/mudnn_extended.hpp"
#include "backends/mthreads/validation/functional/paired.hpp"
#include "common/fp8_matmul.hpp"
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
  return mthreads::run_paired_cases(
      argc, argv, std::span<const Fp8MatmulTestCase>(selected),
      "FLAGDNN_FP8_MATMUL_CASE", std::string(api), build_flagdnn_fp8_matmul,
      fp8_matmul_inputs, [](const auto& c) { return mthreads::reference(c); },
      [](const Fp8MatmulTestCase& test_case, std::size_t) {
        const auto type = test_case.outputs[0].data_type;
        // Independent double-precision accumulation of the public FP8
        // inputs bounds muDNN 3.1.5 RunLt error by 5.8e-4 for K <= 512.
        // Preserve NVIDIA's relative and scaled-GEMM bounds; unscaled FP8
        // needs a 1e-3 absolute bound on this platform, including near zero.
        const double contraction =
            static_cast<double>(test_case.inputs[0].dimensions.back());
        const double absolute =
            test_case.scale_mode == 1
                ? 2.0e-3 * std::sqrt(std::max(1.0, contraction / 128.0))
                : 1.0e-3;
        return mthreads::Tolerance{
            absolute, type == FLAGDNN_DATA_BFLOAT16 ? 8.0e-3 : 1.0e-3};
      },
      benchmark);
}
}  // namespace flagdnn::testing
